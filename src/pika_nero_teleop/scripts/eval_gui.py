#!/usr/bin/env python3
"""Browser UI for running policy evaluation trials.

Why this exists: evaluating a policy means the same four actions fifty times --
send the arm to READY, start, stop, write down whether it worked -- and doing
that from a terminal means three windows, a text file, and a miscount at trial
37. This puts the loop on one page and keeps the score itself.

    ros2 launch pika_nero_teleop eval_gui.launch.py
    # then open http://localhost:8081

It drives the SAME services you would call by hand, so nothing here is a
private back door:

    /arm_ready    std_srvs/Trigger   arm_pose_manager, blocks until it arrives
    /policy/start std_srvs/Trigger   policy_client
    /policy/stop  std_srvs/Trigger   policy_client

and reads /policy/status, which policy_client publishes at 4 Hz.

Results are appended to disk after every single trial, not held in memory and
written at the end. A browser refresh, a crash, or a container restart costs you
nothing but the trial in progress.
"""
import csv
import json
import math
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

STATUS_STALE_AFTER = 3.0     # seconds without /policy/status before it reads dead
SERVICE_TIMEOUT = 40.0       # a READY move is allowed 25 s by arm_pose_manager

# Preview budget. These previews run WHILE a policy drives the arm at 30 Hz, so
# they are deliberately cheap: downscaled, rate-capped, and encoded only when a
# browser is actually attached. Full-resolution frames still go to the policy --
# nothing here touches the observation path.
PREVIEW_WIDTH = 320
PREVIEW_FPS = 8
JPEG_QUALITY = 70
CAMERA_STALE_AFTER = 2.0

# label=topic pairs, SEMICOLON separated. Semicolon rather than comma so a label
# is free to contain one; a comma separator silently tore "scene (ego, base_0_rgb)"
# into two broken entries.
#
# The default is exactly what the policy sees: the scene D435 (-> base_0_rgb) and
# the gripper D405 colour stream (-> left_wrist_0_rgb). Anything else (depth, the
# fisheye) is off in cameras.launch.py, so listing it would only show a dead panel.
#
# Keep labels short and URL-safe -- the label is the stream's query parameter.
DEFAULT_CAMERAS = ("scene=/scene/camera/color/image_raw;"
                   "gripper=/gripper/camera/color/image_raw")

OUTCOMES = ("success", "failure", "discard")


def wilson(successes, n, z=1.96):
    """95% confidence interval for a success rate.

    Reported instead of the bare fraction because the bare fraction invites
    over-reading a small sample: 8/10 and 40/50 are both "80%", but the first
    is consistent with anything from 49% to 94%. Wilson rather than the normal
    approximation because it stays sane at 0/n and n/n, which is exactly where
    an early evaluation sits.
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


class EvalNode(Node):

    def __init__(self, out_dir, checkpoint, prompt, cameras):
        super().__init__("eval_gui")
        group = ReentrantCallbackGroup()

        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.frames = {}        # label -> (jpeg bytes, wall time)
        self.viewers = {}       # label -> count of attached browsers
        self.cameras = []
        for entry in cameras.split(";"):
            if "=" not in entry:
                continue
            label, topic = entry.split("=", 1)
            label, topic = label.strip(), topic.strip()
            if not label or not topic:
                continue
            self.cameras.append({"label": label, "topic": topic})
            self.create_subscription(
                Image, topic,
                (lambda m, l=label: self._on_image(l, m)),
                qos_profile_sensor_data, callback_group=group)
            self.get_logger().info(f"preview {label} <- {topic}")

        self.cli_ready = self.create_client(Trigger, "/arm_ready", callback_group=group)
        self.cli_start = self.create_client(Trigger, "/policy/start", callback_group=group)
        self.cli_stop = self.create_client(Trigger, "/policy/stop", callback_group=group)
        self.create_subscription(String, "/policy/status", self._status, 10,
                                 callback_group=group)

        self.policy = {}
        self.policy_seen = 0.0
        self.lock = threading.Lock()

        # phase drives the whole UI:
        #   idle     nothing running, ready to reset or start
        #   moving   an /arm_ready call is in flight
        #   running  the policy is driving the arm
        #   scoring  the trial ended and is waiting for a verdict
        self.phase = "idle"
        self.message = "ready"
        self.busy = False

        self.checkpoint = checkpoint
        self.prompt = prompt
        self.trials = []
        self.current = None

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = os.path.join(out_dir, f"session-{stamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.session_dir, "trials.jsonl")
        self.csv_path = os.path.join(self.session_dir, "results.csv")
        self._write_meta()
        self.get_logger().info(f"session -> {self.session_dir}")

    # --- cameras -----------------------------------------------------------

    def _on_image(self, label, msg):
        """Encode at most PREVIEW_FPS, and only while someone is watching."""
        now = time.time()
        with self.frame_lock:
            prev = self.frames.get(label)
            watching = self.viewers.get(label, 0) > 0
            # Always keep a recent timestamp so the UI can report the stream as
            # alive even with no browser attached -- that costs nothing. Only
            # the JPEG encode is gated.
            if prev and now - prev[1] < 1.0 / PREVIEW_FPS:
                return
            if not watching:
                self.frames[label] = (prev[0] if prev else None, now)
                return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w = img.shape[:2]
            if w > PREVIEW_WIDTH:
                img = cv2.resize(img, (PREVIEW_WIDTH, int(h * PREVIEW_WIDTH / w)),
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        except Exception:
            return
        if ok:
            with self.frame_lock:
                self.frames[label] = (buf.tobytes(), now)

    def latest_frame(self, label):
        with self.frame_lock:
            entry = self.frames.get(label)
            return entry[0] if entry else None

    def camera_health(self):
        now = time.time()
        with self.frame_lock:
            out = []
            for cam in self.cameras:
                entry = self.frames.get(cam["label"])
                out.append({"label": cam["label"], "topic": cam["topic"],
                            "alive": bool(entry) and (now - entry[1]) < CAMERA_STALE_AFTER})
            return out

    def add_viewer(self, label, delta):
        with self.frame_lock:
            self.viewers[label] = max(0, self.viewers.get(label, 0) + delta)

    # --- ROS ---------------------------------------------------------------

    def _status(self, msg):
        try:
            with self.lock:
                self.policy = json.loads(msg.data)
                self.policy_seen = time.time()
        except json.JSONDecodeError:
            pass

    def policy_alive(self):
        return time.time() - self.policy_seen < STATUS_STALE_AFTER

    def _call(self, client, label):
        """Call a Trigger service and wait. Returns (ok, message).

        Timed and logged: a call that takes seconds is the difference between
        "the UI is broken" and "the service on the other end is slow", and
        without the number that is guesswork.
        """
        t0 = time.time()
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f"{label}: service not available "
                                       f"(waited {time.time() - t0:.1f}s)")
                return False, f"{label}: service not available"
        future = client.call_async(Trigger.Request())
        deadline = time.time() + SERVICE_TIMEOUT
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        dt = time.time() - t0
        if not future.done():
            self.get_logger().error(f"{label}: timed out after {dt:.1f}s")
            return False, f"{label}: timed out after {dt:.0f}s"
        res = future.result()
        if dt > 1.0:
            self.get_logger().warn(f"{label}: took {dt:.1f}s")
        else:
            self.get_logger().info(f"{label}: {dt * 1000:.0f}ms -> {res.message}")
        return bool(res.success), res.message or label

    # --- trial lifecycle ---------------------------------------------------

    def reset(self):
        """Send the arm to READY. Blocks -- /arm_ready does not return early."""
        with self.lock:
            if self.busy or self.phase == "running":
                return False, "busy"
            self.busy, self.phase, self.message = True, "moving", "moving to READY ..."
        try:
            ok, msg = self._call(self.cli_ready, "arm_ready")
        finally:
            with self.lock:
                self.busy = False
                self.phase = "idle" if self.phase == "moving" else self.phase
        with self.lock:
            self.message = msg if ok else f"reset failed: {msg}"
        return ok, self.message

    def start(self):
        with self.lock:
            if self.busy or self.phase == "running":
                return False, "already running"
            self.busy = True
        try:
            ok, msg = self._call(self.cli_start, "policy/start")
        finally:
            with self.lock:
                self.busy = False
        with self.lock:
            if ok:
                self.phase = "running"
                self.current = {
                    "trial": len(self.trials) + 1,
                    "started": datetime.now().isoformat(timespec="seconds"),
                    "t0": time.time(),
                }
            self.message = msg
        return ok, msg

    def stop(self):
        with self.lock:
            if self.busy:
                return False, "busy"
            self.busy = True
        try:
            ok, msg = self._call(self.cli_stop, "policy/stop")
        finally:
            with self.lock:
                self.busy = False
        with self.lock:
            self.message = msg
            if self.current is not None:
                # Snapshot what the run actually did, so the verdict is stored
                # next to the numbers rather than alone.
                self.current.update({
                    "duration_s": round(time.time() - self.current["t0"], 1),
                    "ticks": self.policy.get("ticks", 0),
                    "clamped_pct": round(self.policy.get("clamped_pct", 0.0), 2),
                    "infer_ms": self.policy.get("infer_ms", 0.0),
                })
                self.phase = "scoring"
            else:
                self.phase = "idle"
        return ok, msg

    def score(self, outcome, note):
        """Record a verdict for the trial that just ended."""
        if outcome not in OUTCOMES:
            return False, f"unknown outcome {outcome}"
        with self.lock:
            if self.current is None:
                return False, "no trial to score"
            row = dict(self.current)
            row.pop("t0", None)
            row.update({"outcome": outcome, "note": note or "",
                        "checkpoint": self.checkpoint, "prompt": self.prompt})
            self.trials.append(row)
            self.current = None
            self.phase = "idle"
            self.message = f"trial {row['trial']} recorded as {outcome}"
        self._append(row)
        return True, self.message

    def undo(self):
        """Drop the most recent verdict. Mis-clicks happen at trial 37."""
        with self.lock:
            if not self.trials:
                return False, "nothing to undo"
            row = self.trials.pop()
            self.message = f"removed trial {row['trial']}"
        self._rewrite()
        return True, self.message

    # --- persistence -------------------------------------------------------

    def _write_meta(self):
        with open(os.path.join(self.session_dir, "meta.json"), "w") as fh:
            json.dump({"checkpoint": self.checkpoint, "prompt": self.prompt,
                       "started": datetime.now().isoformat(timespec="seconds")}, fh, indent=2)

    def _append(self, row):
        with open(self.jsonl_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        self._write_csv()

    def _rewrite(self):
        with open(self.jsonl_path, "w") as fh:
            for row in self.trials:
                fh.write(json.dumps(row) + "\n")
        self._write_csv()

    def _write_csv(self):
        cols = ["trial", "outcome", "duration_s", "ticks", "clamped_pct",
                "infer_ms", "started", "note", "checkpoint", "prompt"]
        with open(self.csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in self.trials:
                w.writerow(row)

    # --- view --------------------------------------------------------------

    def snapshot(self):
        with self.lock:
            scored = [t for t in self.trials if t["outcome"] != "discard"]
            succ = sum(1 for t in scored if t["outcome"] == "success")
            rate, lo, hi = wilson(succ, len(scored))
            elapsed = (time.time() - self.current["t0"]) if (
                self.current and "t0" in self.current and self.phase == "running") else 0.0
            return {
                "phase": self.phase,
                "message": self.message,
                "checkpoint": self.checkpoint,
                "prompt": self.prompt,
                "session_dir": self.session_dir,
                "elapsed": round(elapsed, 1),
                "next_trial": len(self.trials) + 1,
                "counts": {"total": len(self.trials), "scored": len(scored),
                           "success": succ,
                           "failure": sum(1 for t in scored if t["outcome"] == "failure"),
                           "discard": len(self.trials) - len(scored)},
                "rate": {"p": round(rate * 100, 1), "lo": round(lo * 100, 1),
                         "hi": round(hi * 100, 1)},
                "policy": self.policy if self.policy_alive() else {},
                "policy_alive": self.policy_alive(),
                "cameras": self.camera_health(),
                "trials": list(reversed(self.trials[-25:])),
            }


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Policy evaluation</title>
<style>
:root{--bg:#14161a;--panel:#1c1f26;--line:#2c313b;--fg:#e6e9ef;--dim:#8b93a3;
      --ok:#3fb950;--bad:#f85149;--warn:#d29922;--go:#2f81f7}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:20px}
h1{font-size:18px;margin:0 0 2px}
.sub{color:var(--dim);font-size:12px;margin-bottom:16px;word-break:break-all}
.grid{display:grid;grid-template-columns:1fr 320px;gap:16px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
.phase{font-size:26px;font-weight:600;letter-spacing:.3px;margin-bottom:4px}
.msg{color:var(--dim);font-size:13px;min-height:20px}
.btns{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}
button{flex:1;min-width:110px;padding:16px 12px;font-size:15px;font-weight:600;
       border-radius:8px;border:1px solid var(--line);background:#252a33;color:var(--fg);cursor:pointer}
button:hover:not(:disabled){border-color:#3d4553}
button:disabled{opacity:.35;cursor:not-allowed}
button.go{background:var(--go);border-color:var(--go);color:#fff}
button.stop{background:var(--bad);border-color:var(--bad);color:#fff}
button.ok{background:var(--ok);border-color:var(--ok);color:#04260c}
button.no{background:var(--bad);border-color:var(--bad);color:#fff}
kbd{font:11px ui-monospace,monospace;background:#0f1115;border:1px solid var(--line);
    border-radius:4px;padding:1px 5px;color:var(--dim);margin-left:6px}
.big{font-size:32px;font-weight:600}
.rate{font-size:12px;color:var(--dim)}
table{width:100%;border-collapse:collapse;font:12px ui-monospace,monospace}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;font-weight:600}
.p-success{background:rgba(63,185,80,.15);color:var(--ok)}
.p-failure{background:rgba(248,81,73,.15);color:var(--bad)}
.p-discard{background:rgba(139,147,163,.15);color:var(--dim)}
.row{display:flex;justify-content:space-between;padding:4px 0;font-size:13px}
.row span:last-child{font-family:ui-monospace,monospace}
.dead{color:var(--bad)}.alive{color:var(--ok)}.warnc{color:var(--warn)}
input[type=text]{width:100%;padding:9px;border-radius:6px;border:1px solid var(--line);
                 background:#0f1115;color:var(--fg);font-size:13px;margin-top:10px}
.cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.cam{background:#0f1115;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.cam img{display:block;width:100%;background:#000;aspect-ratio:4/3;object-fit:cover}
.cam .lab{display:flex;justify-content:space-between;align-items:center;
          padding:6px 9px;font-size:11px;color:var(--dim)}
.hint{font-size:12px;color:var(--warn);margin-top:10px;min-height:16px}
</style></head><body><div class="wrap">
<h1>Policy evaluation</h1>
<div class="sub" id="sub">-</div>
<div class="grid">
 <div>
  <div class="card">
    <div class="phase" id="phase">-</div>
    <div class="msg" id="msg"></div>
    <div class="btns" id="mainbtns">
      <button id="b-reset" onclick="post('reset')">Reset to READY<kbd>R</kbd></button>
      <button id="b-start" class="go" onclick="post('start')">Start<kbd>space</kbd></button>
      <button id="b-stop" class="stop" onclick="post('stop')">Stop<kbd>space</kbd></button>
    </div>
    <div class="btns" id="scorebtns" style="display:none">
      <button class="ok" onclick="score('success')">Success<kbd>S</kbd></button>
      <button class="no" onclick="score('failure')">Failure<kbd>F</kbd></button>
      <button onclick="score('discard')">Discard<kbd>D</kbd></button>
    </div>
    <input type="text" id="note" placeholder="note for this trial (optional) - saved with the verdict">
    <div class="hint" id="hint"></div>
  </div>
  <div class="card">
    <div class="cams" id="cams"></div>
  </div>
  <div class="card">
    <table><thead><tr><th>#</th><th>outcome</th><th>dur</th><th>clamp</th><th>note</th></tr></thead>
    <tbody id="rows"></tbody></table>
  </div>
 </div>
 <div>
  <div class="card">
    <div class="big" id="rate">-</div>
    <div class="rate" id="ci"></div>
    <div style="margin-top:12px">
      <div class="row"><span>success</span><span id="c-s">0</span></div>
      <div class="row"><span>failure</span><span id="c-f">0</span></div>
      <div class="row"><span>discarded</span><span id="c-d">0</span></div>
    </div>
  </div>
  <div class="card">
    <div class="row"><span>client</span><span id="p-alive">-</span></div>
    <div class="row"><span>observations</span><span id="p-obs">-</span></div>
    <div class="row"><span>elapsed</span><span id="p-el">-</span></div>
    <div class="row"><span>steps</span><span id="p-ticks">-</span></div>
    <div class="row"><span>clamped</span><span id="p-clamp">-</span></div>
    <div class="row"><span>inference</span><span id="p-inf">-</span></div>
    <div class="row"><span>gripper</span><span id="p-jaw">-</span></div>
    <div class="row"><span>policy server</span><span id="p-srv">-</span></div>
    <button style="margin-top:12px" onclick="post('undo')">Undo last verdict</button>
  </div>
 </div>
</div></div>
<script>
var S={};
function post(a,body){
  return fetch('/api/'+a,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body||{})}).then(function(r){return r.json()}).then(tick)
}
function score(o){
  var n=document.getElementById('note');
  post('score',{outcome:o,note:n.value}).then(function(){n.value=''})
}
function fmt(x,d){return (x===undefined||x===null)?'-':Number(x).toFixed(d===undefined?0:d)}
function tick(){
  return fetch('/api/status').then(function(r){return r.json()}).then(function(s){
    S=s;
    document.getElementById('sub').textContent =
      s.checkpoint+'  |  "'+s.prompt+'"  |  '+s.session_dir;
    var ph=document.getElementById('phase');
    var label={idle:'READY FOR TRIAL '+s.next_trial,moving:'MOVING TO READY',
               running:'RUNNING - TRIAL '+(s.counts.total+1),scoring:'SCORE THIS TRIAL'}[s.phase];
    ph.textContent=label||s.phase;
    ph.style.color = s.phase==='running' ? 'var(--go)'
                   : s.phase==='scoring' ? 'var(--warn)' : 'var(--fg)';
    document.getElementById('msg').textContent=s.message;

    var scoring = s.phase==='scoring';
    document.getElementById('mainbtns').style.display = scoring?'none':'flex';
    document.getElementById('scorebtns').style.display = scoring?'flex':'none';
    document.getElementById('b-start').style.display = s.phase==='running'?'none':'';
    document.getElementById('b-stop').style.display  = s.phase==='running'?'':'none';
    document.getElementById('b-reset').disabled = (s.phase==='running'||s.phase==='moving');
    var why='';
    if(!s.policy_alive) why='Start is disabled: policy_client is not running. '
                          + 'Launch policy.launch.py first.';
    else if(s.policy&&s.policy.teleop_live) why='Start is disabled: teleop is armed '
                          + '(/delta_pose streaming). Disarm before running a policy.';
    else if(s.policy&&s.policy.missing&&s.policy.missing.length)
      why='Start is disabled: no '+s.policy.missing.join(', ')+'.';
    else if(s.phase==='moving') why='Waiting for the arm to reach READY ...';
    else if(s.policy&&s.policy.server_connected===false)
      why='Waiting for a policy server on the configured port. '
         + 'Start one with ./training/serve_checkpoint.sh <key>.';
    else if(s.policy&&s.policy.jaw_driven===false)
      why='Gripper will NOT move: jaw commands go to '+(s.policy.jaw_topic||'?')
         + ' with no subscriber. Relaunch with drive_gripper:=true to grasp.';
    document.getElementById('hint').textContent = (s.phase==='idle'||s.phase==='moving')?why:'';
    document.getElementById('b-start').disabled =
      (s.phase!=='idle') || !s.policy_alive || !(s.policy&&s.policy.ready);

    document.getElementById('rate').textContent =
      s.counts.scored ? s.rate.p+'%' : '-';
    document.getElementById('ci').textContent = s.counts.scored
      ? s.counts.success+'/'+s.counts.scored+' scored, 95% CI '+s.rate.lo+'-'+s.rate.hi+'%'
      : 'no scored trials yet';
    document.getElementById('c-s').textContent=s.counts.success;
    document.getElementById('c-f').textContent=s.counts.failure;
    document.getElementById('c-d').textContent=s.counts.discard;

    var p=s.policy||{};
    var al=document.getElementById('p-alive');
    al.textContent = s.policy_alive?'connected':'NOT RUNNING';
    al.className = s.policy_alive?'alive':'dead';
    var ob=document.getElementById('p-obs');
    if(!s.policy_alive){ob.textContent='-';ob.className='';}
    else if(p.teleop_live){ob.textContent='TELEOP ARMED';ob.className='dead';}
    else if(p.missing&&p.missing.length){ob.textContent='missing: '+p.missing.join(', ');ob.className='dead';}
    else {ob.textContent='all present';ob.className='alive';}
    document.getElementById('p-el').textContent = s.phase==='running'?fmt(s.elapsed,1)+' s':'-';
    document.getElementById('p-ticks').textContent = p.ticks!==undefined?p.ticks:'-';
    var cl=document.getElementById('p-clamp');
    cl.textContent = p.clamped_pct!==undefined?fmt(p.clamped_pct,1)+'%':'-';
    cl.className = (p.clamped_pct>5)?'warnc':'';
    document.getElementById('p-inf').textContent = p.infer_ms?fmt(p.infer_ms,0)+' ms':'-';
    var sv=document.getElementById('p-srv');
    if(p.server_connected===undefined){sv.textContent='-';sv.className='';}
    else if(p.server_connected){sv.textContent='connected';sv.className='alive';}
    else {sv.textContent='RECONNECTING';sv.className='dead';}
    var jw=document.getElementById('p-jaw');
    if(p.jaw_driven===undefined){jw.textContent='-';jw.className='';}
    else if(p.jaw_driven){jw.textContent='driven';jw.className='alive';}
    else {jw.textContent='NOT DRIVEN';jw.className='dead';
          jw.title='jaw commands go to '+(p.jaw_topic||'?')+' and nothing subscribes';}

    var cams=s.cameras||[];
    var box=document.getElementById('cams');
    if(box.children.length!==cams.length){          // build once, never re-set src
      var ch='';                                    // (resetting src restarts the stream)
      cams.forEach(function(c){
        ch+='<div class="cam"><img title="'+c.topic+'" src="/api/stream?cam='
          + encodeURIComponent(c.label)+'">'
          + '<div class="lab"><span>'+c.label+'</span><span id="cam-'
          + encodeURIComponent(c.label)+'"></span></div></div>';
      });
      box.innerHTML=ch;
    }
    cams.forEach(function(c){
      var el=document.getElementById('cam-'+encodeURIComponent(c.label));
      if(el){el.textContent=c.alive?'live':'NO SIGNAL';el.className=c.alive?'alive':'dead';}
    });

    var html='';
    (s.trials||[]).forEach(function(t){
      html+='<tr><td>'+t.trial+'</td><td><span class="pill p-'+t.outcome+'">'+t.outcome
          +'</span></td><td>'+fmt(t.duration_s,1)+'s</td><td>'+fmt(t.clamped_pct,1)+'%</td><td>'
          +(t.note||'').replace(/[<>&]/g,'')+'</td></tr>';
    });
    document.getElementById('rows').innerHTML=html||'<tr><td colspan="5">no trials yet</td></tr>';
  })
}
document.addEventListener('keydown',function(e){
  if(e.target.tagName==='INPUT')return;          // typing a note, not driving
  var k=e.key.toLowerCase();
  if(k===' '||e.code==='Space'){
    e.preventDefault();
    // Space also activates whichever button has focus, so without this the
    // previous click's button fires again and we send two POSTs for one press.
    if(document.activeElement&&document.activeElement.blur)document.activeElement.blur();
    if(S.phase==='running')post('stop');
    else if(S.phase==='idle'&&S.policy_alive&&S.policy&&S.policy.ready)post('start');
    return}
  if(S.phase==='scoring'){
    if(k==='s')score('success'); else if(k==='f')score('failure'); else if(k==='d')score('discard');
    return}
  if(k==='r'&&S.phase!=='running')post('reset');
});
tick();setInterval(tick,400);
</script></body></html>"""


def make_handler(node):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def _send(self, code, body, ctype="application/json"):
            # The browser may be gone by the time we reply: it navigated, was
            # reloaded, or gave up on a slow request. Writing to that socket
            # raises BrokenPipeError, and socketserver then prints a traceback
            # per dropped request, which buries the log during an eval session.
            #
            # THE ACTION HAS ALREADY HAPPENED by the time we get here -- do_POST
            # calls the node first and formats the reply second. So this
            # discards a reply, never a command. The page polls 4x a second, so
            # it picks up the real state immediately afterwards either way.
            payload = body.encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _stream(self):
            """MJPEG via multipart/x-mixed-replace -- every browser renders this
            in a plain <img>, with no JS and no extra dependency."""
            from urllib.parse import parse_qs, urlparse
            label = (parse_qs(urlparse(self.path).query).get("cam") or [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            node.add_viewer(label, +1)
            last = None
            try:
                while True:
                    frame = node.latest_frame(label)
                    if frame is not None and frame is not last:
                        last = frame
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(frame)).encode()
                                         + b"\r\n\r\n" + frame + b"\r\n")
                    time.sleep(1.0 / PREVIEW_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass    # viewer closed the tab
            finally:
                node.add_viewer(label, -1)

        def do_GET(self):
            if self.path.startswith("/api/stream"):
                return self._stream()
            if self.path.startswith("/api/status"):
                self._send(200, json.dumps(node.snapshot()))
            elif self.path.startswith("/results.csv"):
                try:
                    with open(node.csv_path) as fh:
                        self._send(200, fh.read(), "text/csv")
                except OSError:
                    self._send(404, "no results yet", "text/plain")
            else:
                self._send(200, PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or "{}")
            # Each of these blocks -- ThreadingHTTPServer gives every request its
            # own thread, and the page polls on a separate one, so the UI keeps
            # updating while a READY move is in flight.
            if self.path.startswith("/api/reset"):
                ok, msg = node.reset()
            elif self.path.startswith("/api/start"):
                ok, msg = node.start()
            elif self.path.startswith("/api/stop"):
                ok, msg = node.stop()
            elif self.path.startswith("/api/score"):
                ok, msg = node.score(body.get("outcome", ""), body.get("note", ""))
            elif self.path.startswith("/api/undo"):
                ok, msg = node.undo()
            else:
                return self._send(404, json.dumps({"ok": False}))
            self._send(200, json.dumps({"ok": ok, "message": msg}))
    return Handler


def main():
    rclpy.init()
    out_dir = os.environ.get("PIKA_EVAL_DIR", "/root/pika_ros/eval")
    port = int(os.environ.get("PIKA_EVAL_PORT", "8081"))
    checkpoint = os.environ.get("PIKA_EVAL_CHECKPOINT", "unnamed-checkpoint")
    prompt = os.environ.get("PIKA_EVAL_PROMPT", "pick up the fruits and put into the basket")
    # A launch arg left at its empty default must mean "use the defaults",
    # not "no cameras" -- os.environ.get would return the empty string.
    cameras = os.environ.get("PIKA_EVAL_CAMERAS") or DEFAULT_CAMERAS

    node = EvalNode(out_dir, checkpoint, prompt, cameras)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    node.get_logger().info(f"Evaluation UI on http://localhost:{port}")
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(node))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
