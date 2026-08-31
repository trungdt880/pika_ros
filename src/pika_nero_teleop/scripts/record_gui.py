#!/usr/bin/env python3
"""Browser UI for recording NERO + Pika teleop episodes.

Why this exists: recording from the terminal means watching `ros2 topic hz` in
one window, remembering which episodeIndex you are on, and typing an
instruction string per take. This puts the three things that actually matter --
is every stream alive, what am I about to record, start/stop -- on one page.

It drives data_tools' capture SERVICE rather than spawning a capture process
per episode, so the node stays up between takes:

    start = {start: True,  end: False}
    stop  = {start: False, end: True }
    toggle= {start: True,  end: True }   <- what the Pika Sense button sends

Topic health is read from the same YAML the recorder uses, so the page can
never disagree with what will actually be written to disk.

    ros2 launch pika_nero_teleop record_gui.launch.py
    # then open http://localhost:8080
"""
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image

from data_msgs.srv import CaptureService
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters

RATE_WINDOW = 30          # messages kept per topic for rate estimation
STALE_AFTER = 2.0         # seconds without a message before a topic reads dead
PREVIEW_WIDTH = 320       # preview downscale; full frames still go to disk
PREVIEW_FPS = 10          # encode cap, so previews cannot starve the recorder
JPEG_QUALITY = 70


def load_topics(params_file):
    """Pull every recorded topic out of the capture YAML.

    Walks dataInfo.<group>.<kind>.topics so the page covers exactly what the
    recorder covers, including anything added to the YAML later.
    """
    with open(params_file) as fh:
        cfg = yaml.safe_load(fh)
    info = cfg.get("/**", {}).get("ros__parameters", {}).get("dataInfo", {})
    out = []
    for group, kinds in (info or {}).items():
        if not isinstance(kinds, dict):
            continue
        for kind, spec in kinds.items():
            if not isinstance(spec, dict):
                continue
            names = spec.get("names") or []
            topics = spec.get("topics") or []
            for i, topic in enumerate(topics):
                label = names[i] if i < len(names) else topic
                out.append({"group": f"{group}.{kind}", "label": label, "topic": topic})
    return out


def load_cameras(params_file):
    """Camera streams to preview, from the same YAML.

    Reading the config rather than hardcoding means a second Pika Gripper (or
    any camera) added for bimanual work shows up in the UI automatically.
    """
    with open(params_file) as fh:
        cfg = yaml.safe_load(fh)
    cam = (cfg.get("/**", {}).get("ros__parameters", {})
              .get("dataInfo", {}).get("camera", {})) or {}
    out = []
    for kind in ("color", "depth"):
        spec = cam.get(kind) or {}
        names = spec.get("names") or []
        for i, topic in enumerate(spec.get("topics") or []):
            label = names[i] if i < len(names) else topic
            out.append({"label": label, "kind": kind, "topic": topic})
    return out


class RecorderNode(Node):
    def __init__(self, params_file, dataset_dir):
        super().__init__("pika_record_gui")
        self.params_file = params_file
        self.dataset_dir = dataset_dir
        self.topics = load_topics(params_file)
        self.cameras = load_cameras(params_file)
        self.stamps = {t["topic"]: deque(maxlen=RATE_WINDOW) for t in self.topics}
        self.subs = {}
        self.bridge = CvBridge()
        self.frames = {}        # topic -> latest encoded JPEG
        self.frame_lock = threading.Lock()
        self._last_encode = {}
        self.recording = False
        self.started_at = None
        self.last_message = ""
        # Start from the next FREE index, not 0. The capture node opens each
        # episode with `rm -rf episodeDir`, so handing it an index that already
        # has data destroys that take. The box in the page used to be hardcoded
        # to 0, which meant every page reload or GUI restart aimed the next
        # Start straight at episode0.
        self.episode_index = self.next_free_episode_index()

        self.cli = self.create_client(
            CaptureService, "/data_tools_dataCapture/capture_service")
        # The instruction has to reach the capture node as a PARAMETER, not
        # just in the start request. An episode begun with the Pika Sense
        # double-click never passes through this GUI -- the Sense builds its
        # own request with a hardcoded "[null]" -- so the only way the text you
        # type here can reach that episode is by setting the recorder's
        # parameter, which it re-reads at the start of every take.
        # NOTE the two different names. dataCapture.cpp creates its node as
        # "data_capture" but advertises the capture service under a hardcoded
        # ABSOLUTE name, /data_tools_dataCapture/capture_service. Parameter
        # services follow the node name, so they are under /data_capture/.
        # (A /data_tools_dataCapture/set_parameters may also appear in
        # `ros2 service list` -- that is stale discovery, and no client can
        # reach it.)
        self.param_cli = self.create_client(
            SetParameters, os.environ.get("PIKA_CAPTURE_NODE", "/data_capture")
                           + "/set_parameters")
        self.instruction = ""        # wrapped form sent to the node
        self.instruction_text = ""   # what the operator typed
        # Topic types are only known once publishers exist, so resolving
        # subscriptions is retried rather than done once at startup.
        self.create_timer(2.0, self._resubscribe)
        self._resubscribe()

    def next_free_episode_index(self):
        """First episode index whose directory is absent or empty."""
        i = 0
        while True:
            d = os.path.join(self.dataset_dir, f"episode{i}")
            if not (os.path.isdir(d) and os.listdir(d)):
                return i
            i += 1

    def _resubscribe(self):
        available = dict(self.get_topic_names_and_types())
        for entry in self.topics:
            topic = entry["topic"]
            if topic in self.subs or topic not in available:
                continue
            try:
                msg_type = get_message(available[topic][0])
            except Exception:
                continue
            # BEST_EFFORT deliberately: a RELIABLE subscriber will not match a
            # BEST_EFFORT publisher, and every sensor topic here (all the
            # cameras) publishes with sensor QoS. Best-effort matches both.
            self.subs[topic] = self.create_subscription(
                msg_type, topic,
                lambda _m, t=topic: self.stamps[t].append(time.monotonic()),
                qos_profile_sensor_data)

        for cam in self.cameras:
            topic = cam["topic"]
            key = ("preview", topic)
            if key in self.subs or topic not in available:
                continue
            self.subs[key] = self.create_subscription(
                Image, topic,
                lambda m, t=topic, k=cam["kind"]: self._on_image(m, t, k),
                qos_profile_sensor_data)

    def _on_image(self, msg, topic, kind):
        # Encoding is capped well below camera rate: the preview must never
        # compete with the recorder for CPU.
        now = time.monotonic()
        if now - self._last_encode.get(topic, 0.0) < 1.0 / PREVIEW_FPS:
            return
        self._last_encode[topic] = now
        try:
            if kind == "depth":
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                depth = np.nan_to_num(np.asarray(depth, dtype=np.float32))
                far = np.percentile(depth[depth > 0], 95) if np.any(depth > 0) else 1.0
                scaled = np.clip(depth / max(far, 1e-6), 0, 1)
                img = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                img[depth <= 0] = 0            # no return -> black, not red
            else:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            h, w = img.shape[:2]
            if w > PREVIEW_WIDTH:
                img = cv2.resize(img, (PREVIEW_WIDTH, int(h * PREVIEW_WIDTH / w)))
            ok, buf = cv2.imencode(".jpg", img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with self.frame_lock:
                    self.frames[topic] = buf.tobytes()
        except Exception as exc:
            self.get_logger().warn(f"preview {topic}: {exc}", throttle_duration_sec=10.0)

    def latest_frame(self, topic):
        with self.frame_lock:
            return self.frames.get(topic)

    def rates(self):
        now = time.monotonic()
        out = []
        for entry in self.topics:
            stamps = self.stamps[entry["topic"]]
            hz, alive = 0.0, False
            if len(stamps) >= 2:
                span = stamps[-1] - stamps[0]
                if span > 0:
                    hz = (len(stamps) - 1) / span
                alive = (now - stamps[-1]) < STALE_AFTER
            out.append({**entry,
                        "hz": round(hz, 1),
                        "alive": alive,
                        "connected": entry["topic"] in self.subs})
        return out

    @staticmethod
    def wrap_instruction(text):
        """dataCapture's parser wants \\[text\\]: it checks for a leading
        backslash and strips two characters from each end. Plain text and the
        default "[null]" both fail that check, so nothing is written and every
        frame ends up with the task string "null"."""
        text = (text or "").strip()
        if not text:
            return ""
        return text if text.startswith("\\[") else "\\[" + text + "\\]"

    def set_instruction(self, text):
        """Push the instruction onto the capture node so ANY start path uses it."""
        wrapped = self.wrap_instruction(text)
        if wrapped == self.instruction:
            return True
        if not self.param_cli.wait_for_service(timeout_sec=5.0):
            self.last_message = "capture node not reachable to set the instruction"
            return False
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name="instructions",
            value=ParameterValue(type=ParameterType.PARAMETER_STRING,
                                 string_value=wrapped))]
        fut = self.param_cli.call_async(req)
        end = time.time() + 3.0
        while not fut.done() and time.time() < end:
            time.sleep(0.02)
        if fut.done() and fut.result() and all(r.successful for r in fut.result().results):
            self.instruction = wrapped
            self.instruction_text = (text or "").strip()
            return True
        self.last_message = "failed to set the instruction on the capture node"
        return False

    def _call(self, start, end, episode_index, instructions):
        if not self.cli.wait_for_service(timeout_sec=3.0):
            return False, "capture service not available - is data_tools_dataCapture running?"
        req = CaptureService.Request()
        req.start, req.end = start, end
        instructions = self.wrap_instruction(instructions)

        req.episode_index = int(episode_index)
        req.dataset_dir = self.dataset_dir
        req.instructions = instructions or "[null]"
        future = self.cli.call_async(req)
        deadline = time.monotonic() + 15.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return False, "capture service timed out"
        res = future.result()
        return bool(res.success), res.message or ""

    def start_episode(self, episode_index, instructions):
        ok, msg = self._call(True, False, episode_index, instructions)
        if ok:
            self.recording = True
            self.started_at = time.monotonic()
            self.episode_index = int(episode_index)
        self.last_message = msg or ("recording started" if ok else "start failed")
        return ok

    def stop_episode(self):
        ok, msg = self._call(False, True, -1, "")
        # The write happens on stop; treat the episode as closed either way so
        # the UI cannot get stuck showing "recording".
        self.recording = False
        self.started_at = None
        self.last_message = msg or ("episode saved" if ok else "stop reported a failure")
        return ok

    def episodes(self):
        try:
            names = [d for d in os.listdir(self.dataset_dir) if d.startswith("episode")]
        except OSError:
            return []
        def idx(n):
            try:
                return int(n.replace("episode", ""))
            except ValueError:
                return -1
        return sorted(names, key=idx)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Pika Recorder</title><style>
:root{color-scheme:light dark}
body{font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px;max-width:900px}
h1{font-size:18px;margin:0 0 16px}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin-bottom:16px}
label{display:block;font-size:12px;opacity:.7;margin-bottom:4px}
input{font:inherit;padding:6px 8px;border:1px solid #8888;border-radius:6px;background:transparent;color:inherit}
button{font:inherit;padding:9px 18px;border-radius:6px;border:0;cursor:pointer;color:#fff}
#go{background:#1a7f37}#go[disabled]{opacity:.4;cursor:not-allowed}
#stop{background:#b42318}
table{border-collapse:collapse;width:100%;margin-top:8px}
td,th{text-align:left;padding:5px 8px;border-bottom:1px solid #8883;font-size:13px}
.ok{color:#1a7f37;font-weight:600}.bad{color:#b42318;font-weight:600}
.state{padding:10px 14px;border-radius:8px;margin-bottom:16px;font-weight:600}
.idle{background:#8882}.rec{background:#b4231822;color:#b42318}
small{opacity:.65}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
.cam{border:1px solid #8884;border-radius:8px;overflow:hidden;background:#0002}
.cam img{width:100%;display:block;aspect-ratio:4/3;object-fit:cover;background:#0004}
.cam .cap{padding:6px 8px;font-size:12px;display:flex;justify-content:space-between;gap:8px}
.cam .cap b{font-weight:600}
</style></head><body>
<h1>Pika Recorder &mdash; NERO + Pika Gripper</h1>
<div id="state" class="state idle">idle</div>
<div class="row">
  <div><label>episode index (-1 = auto)</label><input id="ep" type="number" style="width:150px"></div>
  <div style="flex:1"><label>instruction</label><input id="ins" style="width:100%" placeholder="pick up the red block"></div>
  <button id="go">Start</button><button id="stop">Stop</button>
</div>
<div id="msg"><small></small></div>
<h3 style="font-size:14px;margin:18px 0 8px">Cameras</h3>
<div id="cams" class="grid"></div>
<h3 style="font-size:14px;margin:18px 0 0">Recorded streams</h3>
<table><thead><tr><th>stream</th><th>topic</th><th>Hz</th><th>status</th></tr></thead>
<tbody id="tb"></tbody></table>
<h3 style="font-size:14px;margin:18px 0 0">Episodes on disk</h3>
<div id="eps"><small>&mdash;</small></div>
<script>
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return r.json()}
go.onclick=async()=>{const r=await post('/api/start',{episode_index:+ep.value,instructions:ins.value});if(r.ok)ep.value=(+ep.value<0)?-1:(+ep.value+1);refresh()}
stop.onclick=async()=>{await post('/api/stop',{});refresh()}
async function refresh(){
  const s=await (await fetch('/api/status')).json();
  const st=document.getElementById('state');
  st.className='state '+(s.recording?'rec':'idle');
  if(ep.value===''||(!s.recording&&+ep.value<s.episode_index))ep.value=s.episode_index;
  const want=ins.value.trim();
  if(!s.recording&&want!==(s.instruction_text||''))post('/api/instruction',{instructions:want});
  st.textContent=s.recording?('RECORDING episode '+s.episode_index+'  ·  '+s.elapsed+'s'):'idle';
  go.disabled=s.recording;
  document.querySelector('#msg small').textContent=s.message||'';
  let dead=0;
  tb.innerHTML=s.topics.map(t=>{
    const good=t.alive&&t.hz>=1; if(!good)dead++;
    return `<tr><td>${t.label}<br><small>${t.group}</small></td><td><small>${t.topic}</small></td>
    <td>${t.hz.toFixed(1)}</td><td class="${good?'ok':'bad'}">${good?'ok':(t.connected?'stalled':'no publisher')}</td></tr>`}).join('');
  if(dead&&!s.recording)document.querySelector('#msg small').textContent=dead+' stream(s) not healthy — recording will abort if any falls below the hz threshold.';
  eps.innerHTML=s.episodes.length?s.episodes.map(e=>`<code>${e}</code>`).join(' &nbsp; '):'<small>none yet</small>';
  if(!cams.dataset.built){
    cams.dataset.built='1';
    cams.innerHTML=s.cameras.map(c=>`<div class="cam">
      <img alt="${c.label}" src="/api/stream?topic=${encodeURIComponent(c.topic)}">
      <div class="cap"><b>${c.label}</b><span>${c.kind}</span></div></div>`).join('');
  }
}
refresh();setInterval(refresh,1000);
</script></body></html>"""


def make_handler(node):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def _send(self, code, body, ctype="application/json"):
            payload = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.startswith("/api/stream"):
                return self._stream()
            if self.path.startswith("/api/status"):
                elapsed = int(time.monotonic() - node.started_at) if node.started_at else 0
                self._send(200, json.dumps({
                    "recording": node.recording,
                    "episode_index": node.episode_index,
                    "instruction": node.instruction,
                    "instruction_text": node.instruction_text,
                    "elapsed": elapsed,
                    "message": node.last_message,
                    "topics": node.rates(),
                    "cameras": node.cameras,
                    "episodes": node.episodes(),
                }))
            else:
                self._send(200, PAGE, "text/html; charset=utf-8")

        def _stream(self):
            """MJPEG: multipart/x-mixed-replace, which every browser renders in
            a plain <img> with no JS and no extra dependency."""
            from urllib.parse import parse_qs, urlparse
            topic = (parse_qs(urlparse(self.path).query).get("topic") or [""])[0]
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            try:
                while True:
                    frame = node.latest_frame(topic)
                    if frame is not None and frame is not last:
                        last = frame
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(frame)).encode()
                                         + b"\r\n\r\n" + frame + b"\r\n")
                    time.sleep(1.0 / PREVIEW_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass   # viewer closed the tab

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or "{}")
            if self.path.startswith("/api/start"):
                ok = node.start_episode(body.get("episode_index", -1),
                                        body.get("instructions", ""))
            elif self.path.startswith("/api/instruction"):
                ok = node.set_instruction(body.get("instructions", ""))
            elif self.path.startswith("/api/stop"):
                ok = node.stop_episode()
            else:
                return self._send(404, json.dumps({"ok": False}))
            self._send(200, json.dumps({"ok": ok, "message": node.last_message}))
    return Handler


def main():
    rclpy.init()
    params_file = os.environ.get("PIKA_CAPTURE_PARAMS") or os.path.join(
        get_package_share_directory("data_tools"),
        "config", "nero_pika_teleop_data_params.yaml")
    dataset_dir = os.environ.get("PIKA_DATASET_DIR", "/root/pika_ros/data")
    port = int(os.environ.get("PIKA_GUI_PORT", "8080"))

    node = RecorderNode(params_file, dataset_dir)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    node.get_logger().info(f"Recorder UI on http://localhost:{port}  (data -> {dataset_dir})")
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
