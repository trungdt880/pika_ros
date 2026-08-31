#!/usr/bin/env python3
"""Drive the NERO from a policy server. The robot half of inference.

Collects an observation, asks a server for actions, executes them on the arm,
repeats. The server is interchangeable because both ends speak openpi's
websocket protocol:

    # a trained policy
    uv run scripts/serve_policy.py policy:checkpoint \\
        --policy.config=pi05_nero_pika --policy.dir=<checkpoint>
    # or a recorded episode, no model needed
    ros2 run pika_nero_teleop replay_policy_server.py --episode <dir>

    ros2 run pika_nero_teleop policy_client.py --host localhost --port 8000

Nothing moves until you say so. Start it, check the printed observation summary
looks sane, then:

    ros2 service call /policy/start std_srvs/srv/Trigger   # go
    ros2 service call /policy/stop  std_srvs/srv/Trigger   # halt, arm holds

WHAT IT SENDS, matching what the policy was trained on (training/DATASET_CARD.md):

    observation/image        egoCamera   (scene D435)   uint8 HWC
    observation/wrist_image  gripper D405 colour        uint8 HWC
    observation/state        [7 joint radians, jaw opening m]
    prompt                   the task string

and receives {"actions": (horizon, 8)}, consumed one row per control tick.

SAFETY, and why each piece is here:

  * It refuses to run while /delta_pose is streaming. That means teleop is
    armed and arm_ik_pose_node is already publishing /control/joint_states --
    two command sources on one arm is how you break something.
  * Commanded joint steps are CLAMPED to --max-joint-speed, default 143 deg/s.
    The recorded data contained single-frame command spikes of 200-409 deg/s
    from tracker dropouts, and a policy trained on that can reproduce them.
    Clean demonstrations peak at 75-115 deg/s, so the clamp passes real motion
    and catches the spikes. It reports what fraction it clamped -- a high
    number means the policy is asking for motion the demonstrations never
    contained, which is worth knowing before you blame the hardware.
  * It starts from READY, because every training episode does. A first
    observation from anywhere else is out of distribution.
  * Stop, disconnect, or an exhausted chunk all leave the arm holding position
    rather than falling back to anything.
"""
import argparse
import json
import sys
import threading
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from data_msgs.msg import Gripper

try:
    from openpi_client import websocket_client_policy
except ImportError:
    sys.exit("openpi-client is not installed:\n"
             "  pip install -e <openpi>/packages/openpi-client")

JOINTS = [f"joint{i}" for i in range(1, 8)]


class PolicyClient(Node):

    def __init__(self, args):
        super().__init__("policy_client")
        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.scene = None
        self.wrist = None
        self.joints = None
        self.jaw = None
        self.delta_seen = 0.0
        self.running = False
        self.last_cmd = None
        # Counters live on the node rather than in main()'s loop so that both
        # the status publisher and /policy/start (which zeroes them per trial)
        # can reach them. eval_gui.py reads these to score a run.
        self.ticks = 0
        self.clamped = 0
        self.infer_ms = 0.0
        self.started_at = 0.0
        self.discard_chunk = False
        self.server_connected = False
        self.last_message = "idle"
        group = ReentrantCallbackGroup()

        self.create_subscription(Image, args.scene_topic,
                                 lambda m: self._img(m, "scene"), qos_profile_sensor_data,
                                 callback_group=group)
        self.create_subscription(Image, args.wrist_topic,
                                 lambda m: self._img(m, "wrist"), qos_profile_sensor_data,
                                 callback_group=group)
        self.create_subscription(JointState, args.joint_topic, self._joints, 10,
                                 callback_group=group)
        self.create_subscription(Gripper, args.jaw_topic, self._jaw, 10,
                                 callback_group=group)
        # Only used to detect that teleop is live; see the safety note above.
        self.create_subscription(PoseStamped, "/delta_pose",
                                 lambda m: setattr(self, "delta_seen", time.time()), 10,
                                 callback_group=group)

        self.pub_arm = self.create_publisher(JointState, args.arm_command_topic, 10)
        self.pub_jaw = self.create_publisher(JointState, args.jaw_command_topic, 10)

        self.create_service(Trigger, "/policy/start", self._start, callback_group=group)
        self.create_service(Trigger, "/policy/stop", self._stop, callback_group=group)

        # Status as a JSON string rather than a custom message: it keeps this
        # node free of a new interface package, and the only consumer is
        # eval_gui.py, which wants JSON anyway. 4 Hz is enough for a UI and
        # cannot interfere with the 30 Hz control loop.
        self.pub_status = self.create_publisher(String, "/policy/status", 10)
        self.create_timer(0.25, self._publish_status, callback_group=group)

    # --- observation ------------------------------------------------------

    def _img(self, msg, which):
        # BGR, deliberately -- ask cv_bridge for the same byte order the model
        # was trained on, NOT for "true" colour.
        #
        # The training images went through this chain:
        #   dataCapture.cpp   cv_bridge -> BGR8 -> cv::imwrite  (jpg is correct colour)
        #   data_to_hdf5.py   cv2.imread(IMREAD_UNCHANGED)      -> BGR array into the HDF5
        #   hdf5_to_lerobot.py                                  -> handed to LeRobot as "RGB"
        #
        # That last step never swaps the channels back: the cvtColor(BGR2RGB)
        # call in hdf5_to_lerobot.py is commented out. So the dataset's mp4s
        # hold B,G,R bytes labelled RGB, and the model learned on images where
        # red and blue are exchanged. Verified 2026-08-31 by matching frame 0 of
        # lerobot episode_000000.mp4 against its source jpg: MSE 8.1 as-is (pure
        # compression noise) against 813.7 with the channels reversed.
        #
        # This node therefore reproduces the swap, so that inference sees what
        # training saw.
        #
        # Honest caveat on how much this matters: measured, almost nothing.
        # Replaying held-out episode7 through pi05_nero_pika/fruits_full/14999
        # with the channels reversed scored 1.98 deg mean joint error against
        # 1.99 deg correct -- indistinguishable. The checkpoint turns out to be
        # insensitive to R/B exchange, presumably leaning on geometry rather
        # than hue. So this is not a bug fix, it is removing a variable: it
        # costs nothing, and it means a future checkpoint that IS colour-
        # sensitive will not quietly misbehave here.
        #
        # If the dataset is ever rebuilt with that cvtColor restored, flip this
        # back to "rgb8" in the same commit -- the two must change together.
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        with self.lock:
            setattr(self, which, img)

    def _joints(self, msg):
        d = dict(zip(msg.name, msg.position))
        try:
            q = [d[n] for n in JOINTS]
        except KeyError:
            return
        with self.lock:
            self.joints = q

    def _jaw(self, msg):
        with self.lock:
            self.jaw = float(msg.distance)

    def observation(self):
        with self.lock:
            if self.scene is None or self.wrist is None or self.joints is None or self.jaw is None:
                return None
            state = np.asarray(self.joints + [self.jaw], dtype=np.float32)
            return {
                "observation/image": self.scene,
                "observation/wrist_image": self.wrist,
                "observation/state": state,
                "prompt": self.args.prompt,
            }

    def missing(self):
        with self.lock:
            return [n for n, v in (("scene image", self.scene), ("wrist image", self.wrist),
                                   ("joint states", self.joints), ("gripper", self.jaw))
                    if v is None]

    def teleop_live(self):
        return time.time() - self.delta_seen < 1.0

    # --- control ----------------------------------------------------------

    def _start(self, req, res):
        miss = self.missing()
        if miss:
            res.success, res.message = False, "no " + ", ".join(miss)
        elif self.teleop_live():
            res.success = False
            res.message = ("teleop is armed (/delta_pose streaming) -- double-click "
                           "the Sense to disarm before running a policy")
        else:
            # Zero the counters so "clamped %" describes THIS trial, not every
            # trial since the node started. Fifty evaluations in a row is the
            # normal case, not the exception.
            self.ticks = 0
            self.clamped = 0
            self.started_at = time.time()
            # Seed the clamp reference from where the arm IS, so the first
            # command of the trial is rate-limited like every other one.
            with self.lock:
                self.last_cmd = np.asarray(self.joints + [self.jaw], dtype=np.float32)
            # Throw away anything left over from the previous trial. The action
            # chunk lives in main()'s loop and used to survive a stop/start
            # cycle: after Stop -> Reset -> Start the arm resumed a chunk
            # computed for the MIDDLE of the previous trial, from the READY pose,
            # and lunged for it. Combined with the unclamped first step above,
            # that was a full-speed jump across the workspace.
            self.discard_chunk = True
            self.running = True
            res.success, res.message = True, "running"
        self.last_message = res.message
        self.get_logger().info(res.message)
        return res

    def _stop(self, req, res):
        self.running = False
        self.last_cmd = None
        self.discard_chunk = True
        res.success, res.message = True, "stopped, arm holding"
        self.last_message = res.message
        self.get_logger().info(res.message)
        return res

    def _publish_status(self):
        miss = self.missing()
        self.pub_status.publish(String(data=json.dumps({
            "running": self.running,
            "ticks": self.ticks,
            "clamped": self.clamped,
            "clamped_pct": (100.0 * self.clamped / self.ticks) if self.ticks else 0.0,
            "elapsed": (time.time() - self.started_at) if self.running else 0.0,
            "infer_ms": round(self.infer_ms, 1),
            "missing": miss,
            # Whether anything is actually listening to our jaw commands.
            # drive_gripper:=false points them at /unused/jaw, which looks
            # identical to working right up until the policy tries to grasp.
            "server_connected": self.server_connected,
            "jaw_driven": self.pub_jaw.get_subscription_count() > 0,
            "jaw_topic": self.args.jaw_command_topic,
            "ready": not miss and not self.teleop_live(),
            "teleop_live": self.teleop_live(),
            "message": getattr(self, "last_message", ""),
        })))

    def clamp(self, target, dt):
        """Limit the per-tick joint step. Returns (clamped, was_clamped).

        There is deliberately NO unclamped path. This used to return `target`
        untouched whenever `last_cmd` was None -- which is exactly the state
        /policy/stop leaves behind -- so the first command of every trial after
        the first was published with no rate limit at all. Seed from the
        measured arm position instead: the first step is then limited relative
        to where the arm actually is, which is the only reference that is true
        at that moment.
        """
        if self.last_cmd is None:
            with self.lock:
                if self.joints is None or self.jaw is None:
                    return target, False        # nothing measured yet; caller gates on missing()
                self.last_cmd = np.asarray(self.joints + [self.jaw], dtype=np.float32)
        limit = self.args.max_joint_speed * dt
        delta = np.clip(target[:7] - self.last_cmd[:7], -limit, limit)
        out = np.concatenate([self.last_cmd[:7] + delta, target[7:]])
        return out, bool(np.any(np.abs(target[:7] - self.last_cmd[:7]) > limit + 1e-9))

    def publish(self, action):
        now = self.get_clock().now().to_msg()
        arm = JointState()
        arm.header.stamp = now
        arm.name = list(JOINTS)
        arm.position = [float(v) for v in action[:7]]
        self.pub_arm.publish(arm)

        jaw = JointState()
        jaw.header.stamp = now
        jaw.name = [self.args.jaw_joint_name]
        jaw.position = [float(action[7])]
        self.pub_jaw.publish(jaw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default="pick up the fruits and put into the basket")
    ap.add_argument("--rate", type=float, default=30.0,
                    help="control rate; must match the dataset fps (30)")
    ap.add_argument("--action-horizon", type=int, default=30,
                    help="actions per server reply; must match the policy config")
    ap.add_argument("--max-joint-speed", type=float, default=2.5,
                    help="rad/s per joint, default 2.5 (143 deg/s). Chosen from the "
                         "recorded data: clean episodes peak around 75-115 deg/s, so "
                         "this passes real demonstrations untouched, while the "
                         "single-frame spikes that tracker dropouts produced were "
                         "200-409 deg/s and get caught. The arm's own URDF limit is "
                         "5.0 rad/s (286 deg/s)")
    ap.add_argument("--scene-topic", default="/scene/camera/color/image_raw")
    ap.add_argument("--wrist-topic", default="/gripper/camera/color/image_raw")
    ap.add_argument("--joint-topic", default="/feedback/joint_states")
    ap.add_argument("--jaw-topic", default="/gripper/gripper/data")
    ap.add_argument("--arm-command-topic", default="/control/joint_states")
    # The follower gripper listens on the topic the Pika Sense normally
    # publishes (see the remap in pika_devices.launch.py). Commanding it
    # autonomously means publishing there, so the Sense's serial node must NOT
    # be running at the same time or the two will fight over the jaws.
    ap.add_argument("--jaw-command-topic", default="/sensor/gripper/joint_state")
    ap.add_argument("--jaw-joint-name", default="gripper")
    ap.add_argument("--observation-timeout", type=float, default=0.0,
                    help="seconds to wait for every observation source before giving up. "
                         "0 (the default) means wait forever, which is what you want when "
                         "this is launched next to the cameras. Set a positive value only "
                         "for scripted runs that must fail rather than hang.")
    ap.add_argument("--autostart", action="store_true",
                    help="begin immediately instead of waiting for /policy/start")
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = PolicyClient(args)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    log = node.get_logger()
    def connect():
        """Block until a server answers. The constructor retries every 5 s."""
        p = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
        log.info(f"server metadata: {p.get_server_metadata()}")
        node.server_connected = True
        return p

    log.info(f"connecting to ws://{args.host}:{args.port} ...")
    policy = connect()

    # WAIT, do not exit. This node is launched alongside the cameras it depends
    # on, and cameras.launch.py staggers the two RealSense starts -- so a fixed
    # deadline here is a race against our own launch file. It used to be 20 s
    # and lost: the node died with exit code 1 about 22 s in, leaving the arm,
    # gripper and cameras up with nothing driving them, and the evaluation UI
    # correctly reporting "policy_client is not running".
    #
    # Waiting costs nothing. Nothing moves until /policy/start, and that service
    # refuses while any source is missing, so a client sitting here is idle and
    # harmless -- and it still publishes /policy/status, so the UI can show
    # exactly which stream is late instead of showing an absent node.
    waited = 0.0
    while node.missing():
        if args.observation_timeout and waited >= args.observation_timeout:
            log.error("missing observation sources after "
                      f"{waited:.0f}s: " + ", ".join(node.missing()))
            rclpy.try_shutdown()
            return 1
        if waited and waited % 5 < 0.2:
            log.warn(f"waiting {waited:.0f}s for: " + ", ".join(node.missing()))
        time.sleep(0.2)
        waited += 0.2
    if waited:
        log.info(f"all observation sources up after {waited:.0f}s")

    obs = node.observation()
    log.info(f"observation ready: images {obs['observation/image'].shape} / "
             f"{obs['observation/wrist_image'].shape}, "
             f"state {np.round(obs['observation/state'], 4).tolist()}")
    # The single most expensive silent misconfiguration available here: the arm
    # tracks the policy perfectly, reaches the object, and never closes, because
    # the jaw commands are going to a topic nobody subscribes to. Say so.
    time.sleep(0.5)                      # let discovery settle before counting
    if node.pub_jaw.get_subscription_count() == 0:
        log.warn("=" * 68)
        log.warn(f"NOTHING SUBSCRIBES TO {args.jaw_command_topic} -- "
                 "the gripper will NOT move.")
        if "unused" in args.jaw_command_topic:
            log.warn("That is drive_gripper:=false (the default). For a task that "
                     "needs grasping, relaunch with drive_gripper:=true.")
        else:
            log.warn("Check that the gripper's serial node is running.")
        log.warn("=" * 68)
    else:
        log.info(f"jaw commands -> {args.jaw_command_topic} "
                 f"({node.pub_jaw.get_subscription_count()} subscriber)")

    if args.autostart:
        node.running = True
        log.warn("autostart: the arm will move now")
    else:
        log.info("idle. start with: ros2 service call /policy/start std_srvs/srv/Trigger")

    period = 1.0 / args.rate
    chunk, idx = None, 0
    try:
        while rclpy.ok():
            t0 = time.time()
            if not node.running:
                time.sleep(period)
                continue
            if node.teleop_live():
                log.error("teleop went live -- stopping")
                node.running = False
                continue
            if node.discard_chunk:
                chunk, idx = None, 0
                node.discard_chunk = False
            obs = node.observation()
            if obs is None:
                time.sleep(period)
                continue
            if chunk is None or idx >= len(chunk):
                t_infer = time.time()
                try:
                    result = policy.infer(obs)
                except Exception as exc:            # noqa: BLE001 -- see below
                    # Swapping checkpoints means stopping one server and starting
                    # another, and the old one says goodbye with a websocket 1001
                    # ("going away"). This used to propagate out of main() and
                    # kill the process, taking the client node out of a running
                    # launch -- so changing checkpoint meant restarting the arm,
                    # gripper and cameras too. Reconnect instead.
                    #
                    # DO NOT auto-resume. The arm holds where it is and waits for
                    # an explicit /policy/start, same as every other way a trial
                    # can end. Silently carrying on across a server swap would
                    # mean driving the arm with a policy nobody chose.
                    log.error(f"server connection lost ({type(exc).__name__}: {exc}) "
                              "-- stopping and reconnecting")
                    node.running = False
                    node.last_cmd = None
                    node.discard_chunk = True
                    node.server_connected = False
                    policy = connect()
                    log.warn("reconnected. Press start again when ready.")
                    continue
                node.infer_ms = (time.time() - t_infer) * 1000.0
                chunk = np.asarray(result["actions"], dtype=np.float32)
                if chunk.ndim == 1:
                    chunk = chunk[None, :]
                idx = 0
            action, hit = node.clamp(chunk[idx], period)
            node.clamped += hit
            node.last_cmd = action
            node.publish(action)
            idx += 1
            node.ticks += 1
            if node.ticks % (int(args.rate) * 5) == 0:
                log.info(f"{node.ticks} steps, {node.clamped} clamped "
                         f"({100*node.clamped/node.ticks:.1f}%)")
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        log.info(f"stopped after {node.ticks} steps, {node.clamped} clamped")
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
