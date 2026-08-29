#!/usr/bin/env python3
"""Puts the NERO in a known pose around every teleop session.

Without this node the arm simply sits wherever it was left when the driver
came up, teleop starts from that arbitrary pose, and it stays wherever the
operator stopped moving. Three problems with that: episodes are not comparable
(every take starts from a different arm configuration), the operator has to
re-find the workspace by hand each time, and shutting down leaves the arm in a
pose that may be expensive when the motors de-energise -- a de-powered NERO
goes limp and drops.

So this node owns three poses and the transitions between them:

    READY ("pose A")  the pose every episode starts and ends in
    REST              compact and low, where the arm can safely go limp
    (teleop)          wherever the operator drives it

    startup ................................ move to READY
    double-click #1 (record starts) ........ arm teleop, from READY
    double-click #2 (record stops) ......... disarm teleop, move back to READY
    park ................................... move to REST

Services (all std_srvs/Trigger):

    /teleop_trigger   toggle teleop, with the READY moves around it. Answers
                      immediately -- the Sense fires it and never reads back.
    /arm_ready        go to READY.  Answers when the arm is actually there.
    /arm_park         go to REST.   Answers when the arm is actually there.

/arm_park blocking is what makes docker/stop_teleop.sh work: it can park, know
the arm arrived, and only then stop the launch. That two-step matters because
Ctrl-C signals every process in the terminal's group at once -- the arm driver
stops serving /control/move_j within milliseconds, long before anything can
move the arm. This node still tries on the way out (park_on_shutdown), and
says so when it loses that race.


Why it sits in front of the trigger service
-------------------------------------------
The Pika Sense's double-click calls `/teleop_trigger` (see
sensor_tools/src/serial_gripper_imu.cpp) which is normally served by
pub_delta_pose.py directly. This node takes that name over and forwards to
pub_delta_pose's service under a private name (`arm_teleop_service`), so it can
sequence a pose move around the arm/disarm. teleop_nero_pika.launch.py wires
the rename; nothing on the Sense side changes.

The same double-click also toggles data_tools' capture service, directly and
independently. That is what makes recording line up with the two clicks: the
episode starts on the first and stops on the second, and the return-to-READY
move happens after the recorder has already stopped, so it never lands in the
data.

Motion is commanded on `/control/move_j`, the driver's interpolated joint move,
not on `/control/joint_states` (which is teleop's own channel, and in fast_mode
goes to the unsmoothed move_js interface). Arrival is judged from
`/feedback/joint_states` rather than the driver's motion_status flag, because
that flag is also set by teleop's streaming commands and cannot distinguish
"finished my move_j" from "keeping up with the operator".

Only ever moves the arm while teleop is disarmed. pub_delta_pose stops
publishing /delta_pose the moment it is disarmed, and arm_ik_pose_node is
purely event-driven on that topic, so /control/joint_states goes quiet and
there is no second command source to fight with.
"""
import signal
import threading
import time
from queue import Empty, Queue

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from data_msgs.msg import TeleopStatus

# Sense LED colours are driven from TeleopStatus (serial_gripper_imu.cpp
# teleopStatusHandler): fail -> yellow, else quit -> off, else green.
LED_ARMED = TeleopStatus(fail=False, quit=False)   # green: teleop live
LED_BUSY = TeleopStatus(fail=True, quit=False)     # yellow: arm is moving, wait
LED_IDLE = TeleopStatus(fail=False, quit=True)     # off: parked at a fixed pose


class ArmPoseManager(Node):

    def __init__(self):
        super().__init__("arm_pose_manager")

        self.declare_parameter("joint_names", [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"])
        # Fallbacks only -- the launch passes config/arm_poses.nero_pika.yaml,
        # which is where these are explained and where you should change them.
        # They are URDF forward kinematics for a table-mounted NERO, NOT a
        # measured configuration: look at them first with render_poses.py, and
        # re-capture your own with capture_pose.py.
        self.declare_parameter("ready_pose", [0.0, -0.3491, 0.0873, 2.0944, 0.0, 0.0, 0.8727])
        self.declare_parameter("rest_pose", [-0.0423, -0.7252, 0.0555, 2.1380, -0.0224, -0.1002, 1.5688])

        self.declare_parameter("move_to_ready_on_start", True)
        self.declare_parameter("return_to_ready_after_teleop", True)
        # Best effort only -- see park_on_shutdown() for why the reliable way
        # to park is to call the service before stopping the launch.
        self.declare_parameter("park_on_shutdown", True)

        self.declare_parameter("feedback_joint_topic", "/feedback/joint_states")
        self.declare_parameter("move_j_topic", "/control/move_j")
        self.declare_parameter("teleop_status_topic", "/teleop_status")
        # Served by us -- this is the name the Sense's double-click calls.
        self.declare_parameter("teleop_trigger_service", "/teleop_trigger")
        # pub_delta_pose.py's own trigger, renamed out of the way in the launch.
        self.declare_parameter("arm_teleop_service", "/teleop_trigger_arm")
        self.declare_parameter("park_service", "/arm_park")
        self.declare_parameter("ready_service", "/arm_ready")

        # A joint counts as arrived once it is inside the tolerance and has
        # stayed there for the settle time -- the arm overshoots slightly and
        # a single in-tolerance sample fires far too early.
        self.declare_parameter("arrival_tolerance", 0.05)
        self.declare_parameter("arrival_settle_time", 0.4)
        self.declare_parameter("move_timeout", 25.0)
        self.declare_parameter("shutdown_move_timeout", 8.0)
        # How long to wait at startup for the arm driver to publish feedback.
        self.declare_parameter("feedback_timeout", 60.0)
        # Grace period between feedback appearing and the startup move. The
        # arm energises and then moves to READY on its own a few seconds after
        # the launch -- this is the window to stand clear or Ctrl-C.
        self.declare_parameter("startup_delay", 3.0)

        self.joint_names = [str(n) for n in self.get_parameter("joint_names").value]
        self.ready_pose = self._pose_param("ready_pose")
        self.rest_pose = self._pose_param("rest_pose")

        self.move_to_ready_on_start = bool(self.get_parameter("move_to_ready_on_start").value)
        self.return_to_ready = bool(self.get_parameter("return_to_ready_after_teleop").value)
        self.park_on_exit = bool(self.get_parameter("park_on_shutdown").value)

        self.arrival_tolerance = float(self.get_parameter("arrival_tolerance").value)
        self.arrival_settle_time = float(self.get_parameter("arrival_settle_time").value)
        self.move_timeout = float(self.get_parameter("move_timeout").value)
        self.shutdown_move_timeout = float(self.get_parameter("shutdown_move_timeout").value)
        self.feedback_timeout = float(self.get_parameter("feedback_timeout").value)
        self.startup_delay = float(self.get_parameter("startup_delay").value)

        group = ReentrantCallbackGroup()

        self._feedback = None                 # {joint name: angle}, latest
        self._feedback_event = threading.Event()
        self._lock = threading.Lock()
        self.teleop_active = False

        self.pub_move_j = self.create_publisher(
            JointState, str(self.get_parameter("move_j_topic").value), 10)
        self.pub_teleop_status = self.create_publisher(
            TeleopStatus, str(self.get_parameter("teleop_status_topic").value), 1)

        self.create_subscription(
            JointState, str(self.get_parameter("feedback_joint_topic").value),
            self._feedback_callback, 10, callback_group=group)

        self.teleop_client = self.create_client(
            Trigger, str(self.get_parameter("arm_teleop_service").value),
            callback_group=group)

        self.create_service(
            Trigger, str(self.get_parameter("teleop_trigger_service").value),
            self._teleop_trigger_callback, callback_group=group)
        self.create_service(
            Trigger, str(self.get_parameter("park_service").value),
            self._park_callback, callback_group=group)
        self.create_service(
            Trigger, str(self.get_parameter("ready_service").value),
            self._ready_callback, callback_group=group)

        # Everything that moves the arm runs on this one worker, so a move can
        # never overlap another move or an arm/disarm.
        #
        # /teleop_trigger only enqueues and answers immediately: the Sense
        # fires it asynchronously and never reads the response, and holding a
        # service handler open for a 5-second joint move would buy nothing.
        # /arm_park and /arm_ready wait for the move to finish instead, so a
        # shutdown script can call one and know the arm is actually there
        # before it kills the launch.
        self._jobs = Queue()
        self._shutting_down = threading.Event()
        self._worker = threading.Thread(target=self._run_jobs, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f"arm_pose_manager ready. READY={self._fmt(self.ready_pose)} "
            f"REST={self._fmt(self.rest_pose)}")

        if self.move_to_ready_on_start:
            self._jobs.put(("startup", None, None))

    # --- parameters -------------------------------------------------------

    def _pose_param(self, name):
        pose = [float(v) for v in self.get_parameter(name).value]
        if len(pose) != len(self.joint_names):
            raise ValueError(
                f"{name} has {len(pose)} values but joint_names has "
                f"{len(self.joint_names)}: {pose}")
        return pose

    @staticmethod
    def _fmt(pose):
        return "[" + ", ".join(f"{v:.3f}" for v in pose) + "]"

    # --- feedback ---------------------------------------------------------

    def _feedback_callback(self, msg: JointState):
        with self._lock:
            self._feedback = dict(zip(msg.name, msg.position))
        self._feedback_event.set()

    def _joint_error(self, target):
        """Largest per-joint deviation from `target`, or None if no feedback.

        None also covers a feedback message that is missing one of our joints,
        which would otherwise silently compare against a default of zero.
        """
        with self._lock:
            feedback = self._feedback
        if feedback is None:
            return None
        try:
            return max(abs(feedback[name] - value)
                       for name, value in zip(self.joint_names, target))
        except KeyError:
            return None

    def _wait_for_feedback(self, timeout):
        deadline = self.get_clock().now().nanoseconds * 1e-9 + timeout
        while rclpy.ok():
            if self._joint_error(self.ready_pose) is not None:
                return True
            remaining = deadline - self.get_clock().now().nanoseconds * 1e-9
            if remaining <= 0:
                return False
            self._feedback_event.clear()
            self._feedback_event.wait(min(remaining, 0.5))
        return False

    # --- motion -----------------------------------------------------------

    def _move_to(self, target, label, timeout=None):
        """Command a joint move and block until the arm gets there.

        Returns True on arrival. A move_j is a one-shot goal, so a command sent
        before the driver's subscription has matched is simply lost -- hence
        the match wait, and one retry if nothing has moved after a couple of
        seconds.
        """
        timeout = self.move_timeout if timeout is None else timeout

        error = self._joint_error(target)
        if error is None:
            self.get_logger().warn(
                f"no arm feedback, refusing to move to {label}")
            return False
        if error <= self.arrival_tolerance:
            self.get_logger().info(f"already at {label}")
            return True

        self._set_led(LED_BUSY)
        self.get_logger().info(
            f"moving to {label} {self._fmt(target)} (max joint error {error:.3f} rad)")

        msg = JointState()
        msg.name = list(self.joint_names)
        msg.position = list(target)

        if not self._wait_for_subscriber(self.pub_move_j, timeout=5.0):
            self.get_logger().warn(
                f"nothing subscribed to {self.pub_move_j.topic_name} -- is the "
                f"arm driver running?")
            return False

        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_move_j.publish(msg)
        # Clock the timeout from the command, not from the subscription wait.
        start = self.get_clock().now().nanoseconds * 1e-9

        retried = False
        in_tolerance_since = None
        start_error = error
        while rclpy.ok():
            now = self.get_clock().now().nanoseconds * 1e-9
            error = self._joint_error(target)
            if error is not None and error <= self.arrival_tolerance:
                if in_tolerance_since is None:
                    in_tolerance_since = now
                elif now - in_tolerance_since >= self.arrival_settle_time:
                    self.get_logger().info(f"reached {label}")
                    return True
            else:
                in_tolerance_since = None

            elapsed = now - start
            # Nothing happened at all: the command was probably dropped by the
            # driver's warm-up gate (control_ready) or by an unmatched
            # subscription. One resend, then give up and say so.
            if (not retried and elapsed > 2.0 and error is not None
                    and error > start_error - 0.01):
                self.get_logger().warn(f"no motion towards {label}, resending move_j")
                msg.header.stamp = self.get_clock().now().to_msg()
                self.pub_move_j.publish(msg)
                retried = True

            if elapsed > timeout:
                self.get_logger().error(
                    f"timed out after {timeout:.0f}s moving to {label} "
                    f"(max joint error {error if error is None else round(error, 3)} rad)")
                return False

            time.sleep(0.02)
        return False

    def _wait_for_subscriber(self, publisher, timeout):
        deadline = self.get_clock().now().nanoseconds * 1e-9 + timeout
        while rclpy.ok():
            if publisher.get_subscription_count() > 0:
                return True
            if self.get_clock().now().nanoseconds * 1e-9 > deadline:
                return False
            time.sleep(0.05)
        return False

    def _set_led(self, status):
        self.pub_teleop_status.publish(status)

    # --- teleop arm/disarm ------------------------------------------------

    def _call_teleop_trigger(self, what):
        """Toggle pub_delta_pose's teleop state and report what it said."""
        if not self.teleop_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                f"{self.teleop_client.srv_name} not available, cannot {what} teleop")
            return False
        future = self.teleop_client.call_async(Trigger.Request())
        # A ReentrantCallbackGroup on a MultiThreadedExecutor lets the response
        # land on another thread while this worker waits.
        if not self._wait_for_future(future, timeout=5.0):
            self.get_logger().error(f"{what} teleop: no response")
            return False
        result = future.result()
        if not result.success:
            self.get_logger().error(f"{what} teleop refused: {result.message}")
            return False
        return True

    def _wait_for_future(self, future, timeout):
        deadline = self.get_clock().now().nanoseconds * 1e-9 + timeout
        while rclpy.ok() and not future.done():
            if self.get_clock().now().nanoseconds * 1e-9 > deadline:
                return False
            time.sleep(0.02)
        return future.done()

    # --- jobs -------------------------------------------------------------

    def _run_jobs(self):
        while rclpy.ok():
            try:
                job, done, result = self._jobs.get(timeout=0.2)
            except Empty:
                continue
            if self._shutting_down.is_set():
                # The shutdown park owns the arm from here on; anything still
                # queued would fight it for the same move_j topic.
                if result is not None:
                    result["success"], result["message"] = False, "shutting down"
                if done is not None:
                    done.set()
                continue
            try:
                handler = {
                    "startup": self._do_startup,
                    "toggle": self._do_toggle,
                    "park": self._do_park,
                    "ready": self._do_ready,
                }[job]
                ok, message = handler()
            except Exception as exc:                      # noqa: BLE001
                self.get_logger().error(f"{job} failed: {exc}")
                self._set_led(LED_BUSY)
                ok, message = False, str(exc)
            finally:
                if result is not None:
                    result["success"], result["message"] = ok, message
                if done is not None:
                    done.set()

    def _do_startup(self):
        self.get_logger().info("waiting for arm feedback before moving to READY...")
        if not self._wait_for_feedback(self.feedback_timeout):
            message = (f"no {self.get_parameter('feedback_joint_topic').value} after "
                       f"{self.feedback_timeout:.0f}s -- staying put. Arm driver up?")
            self.get_logger().error(message)
            return False, message
        if self.startup_delay > 0.0:
            self.get_logger().warn(
                f"ARM WILL MOVE to READY in {self.startup_delay:.0f}s -- stand clear")
            time.sleep(self.startup_delay)
        if not self._move_to(self.ready_pose, "READY"):
            return False, "could not reach READY"
        self._set_led(LED_IDLE)
        return True, "at READY"

    def _do_toggle(self):
        if self.teleop_active:
            # Stop first, then move: pub_delta_pose must be quiet before we
            # command a pose, and the recorder has already been stopped by the
            # same double-click, so the return trip is not recorded.
            if not self._call_teleop_trigger("stop"):
                return False, "pub_delta_pose would not stop"
            self.teleop_active = False
            self.get_logger().info("teleop disarmed")
            if self.return_to_ready and not self._move_to(self.ready_pose, "READY"):
                return False, "teleop stopped but the arm is not at READY"
            self._set_led(LED_IDLE)
            return True, "teleop stopped, at READY"

        # Start every episode from the same place. Normally a no-op -- the arm
        # is already there from startup or from the end of the last episode.
        if self.return_to_ready and not self._move_to(self.ready_pose, "READY"):
            self.get_logger().error("not at READY, refusing to arm teleop")
            self._set_led(LED_BUSY)
            return False, "not at READY, teleop not armed"
        if not self._call_teleop_trigger("start"):
            return False, "pub_delta_pose would not start"
        self.teleop_active = True
        self.get_logger().info("teleop armed from READY")
        self._set_led(LED_ARMED)
        return True, "teleop armed from READY"

    def _goto(self, pose, label):
        if self.teleop_active:
            if not self._call_teleop_trigger("stop"):
                return False, "pub_delta_pose would not stop"
            self.teleop_active = False
        if not self._move_to(pose, label):
            return False, f"could not reach {label}"
        self._set_led(LED_IDLE)
        return True, f"at {label}"

    def _do_park(self):
        return self._goto(self.rest_pose, "REST")

    def _do_ready(self):
        return self._goto(self.ready_pose, "READY")

    # --- services ---------------------------------------------------------

    def _teleop_trigger_callback(self, request, response):
        response.success = True
        response.message = ("stopping teleop and returning to READY"
                            if self.teleop_active else "arming teleop from READY")
        self._jobs.put(("toggle", None, None))
        return response

    def _blocking_job(self, job, response):
        """Run a job and answer only once the arm has actually got there."""
        done = threading.Event()
        result = {"success": False, "message": "timed out"}
        self._jobs.put((job, done, result))
        # Generous: the job may be queued behind a move that is already
        # running, so allow for two of them plus the trigger round-trips.
        done.wait(2 * self.move_timeout + 15.0)
        response.success = bool(result["success"])
        response.message = str(result["message"])
        return response

    def _park_callback(self, request, response):
        return self._blocking_job("park", response)

    def _ready_callback(self, request, response):
        return self._blocking_job("ready", response)

    # --- shutdown ---------------------------------------------------------

    def park_on_shutdown(self):
        """Try to park before exiting. Only works if the driver outlives us.

        Ctrl-C in a terminal goes to every process in the foreground group at
        once, so `ros2 launch` and the arm driver get SIGINT at the same
        instant we do and the driver stops serving /control/move_j within
        milliseconds. This attempt therefore usually fails, loudly, and that is
        fine -- it costs a few seconds and occasionally saves the arm.

        The reliable order is: park, THEN stop the launch. docker/stop_teleop.sh
        does exactly that.
        """
        if not self.park_on_exit:
            return
        self._shutting_down.set()
        self.get_logger().info("shutdown: parking arm at REST")
        if self.teleop_active:
            self._call_teleop_trigger("stop")
            self.teleop_active = False
        if not self._move_to(self.rest_pose, "REST", timeout=self.shutdown_move_timeout):
            self.get_logger().warn(
                "could not park on shutdown -- the arm driver is probably "
                "already gone. Park BEFORE stopping the launch: "
                "ros2 service call /arm_park std_srvs/srv/Trigger, or "
                "docker/stop_teleop.sh")
        self._set_led(LED_IDLE)


def main(args=None):
    rclpy.init(args=args)
    node = None
    stop = threading.Event()

    # Replace rclpy's SIGINT handler, which tears the context down immediately
    # and would leave us unable to publish anything on the way out.
    def on_sigint(signum, frame):
        stop.set()

    try:
        node = ArmPoseManager()
        signal.signal(signal.SIGINT, on_sigint)
        signal.signal(signal.SIGTERM, on_sigint)

        executor = MultiThreadedExecutor()
        executor.add_node(node)
        spin = threading.Thread(target=executor.spin, daemon=True)
        spin.start()
        while not stop.wait(0.2):
            if not spin.is_alive():
                break
        node.park_on_shutdown()
        executor.shutdown(timeout_sec=2.0)
    except Exception as exc:                              # noqa: BLE001
        print(f"arm_pose_manager: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
