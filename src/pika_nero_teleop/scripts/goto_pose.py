#!/usr/bin/env python3
"""Send the arm to READY, REST, or any joint pose, and wait until it gets there.

    ros2 run pika_nero_teleop goto_pose.py ready
    ros2 run pika_nero_teleop goto_pose.py rest
    ros2 run pika_nero_teleop goto_pose.py --joints 0 0.3 0 1.4 -1.5708 0.1 0

This is the manual counterpart to arm_pose_manager, for trying a pose out
before committing it to the config, and for the times you want the arm parked
without stopping teleop.

Two ways to get there, picked automatically:

  via arm_pose_manager  (/arm_ready, /arm_park)
      Preferred when the manager is running. It disarms teleop first, so it is
      safe to call mid-session, and it is the same code path the Sense's
      double-click takes.

  direct on /control/move_j
      Used when the manager is not running, and always for --joints. Refuses
      to move while /delta_pose is live, because that means teleop is armed
      and the IK chain is already commanding the arm -- two command sources
      fighting over one bus is how you break something. Double-click to disarm
      first, or pass --force if you know better.

Arrival is judged from /feedback/joint_states, same as arm_pose_manager: every
joint inside --tolerance for --settle seconds.
"""
import argparse
import os
import sys
import time

import rclpy
import yaml
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

DEFAULT_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
MANAGER_SERVICE = {"ready": "/arm_ready", "rest": "/arm_park"}


def default_poses_file():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("pika_nero_teleop"),
                            "config", "arm_poses.nero_pika.yaml")
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "config", "arm_poses.nero_pika.yaml")


def load_poses(path):
    with open(path) as fh:
        params = yaml.safe_load(fh).get("/**", {}).get("ros__parameters", {})
    return (params.get("joint_names", DEFAULT_JOINTS),
            {"ready": params["ready_pose"], "rest": params["rest_pose"]})


class GotoPose(Node):

    def __init__(self, joint_names):
        super().__init__("goto_pose")
        self.joint_names = joint_names
        self.feedback = None
        self.delta_pose_seen = 0.0
        self.pub = self.create_publisher(JointState, "/control/move_j", 10)
        self.create_subscription(JointState, "/feedback/joint_states", self._joints, 10)
        # Only used as a "is teleop actually streaming right now" flag.
        self.create_subscription(PoseStamped, "/delta_pose", self._delta, 10)

    def _joints(self, msg):
        self.feedback = dict(zip(msg.name, msg.position))

    def _delta(self, msg):
        self.delta_pose_seen = time.time()

    def spin(self, seconds):
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def error(self, target):
        if self.feedback is None:
            return None
        try:
            return max(abs(self.feedback[n] - v)
                       for n, v in zip(self.joint_names, target))
        except KeyError:
            return None

    def teleop_streaming(self):
        return time.time() - self.delta_pose_seen < 1.0

    def call(self, service, timeout):
        client = self.create_client(Trigger, service)
        if not client.wait_for_service(timeout_sec=2.0):
            return None
        future = client.call_async(Trigger.Request())
        end = time.time() + timeout
        while rclpy.ok() and not future.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return future.result() if future.done() else None

    def move_direct(self, target, tolerance, settle, timeout):
        msg = JointState()
        msg.name = list(self.joint_names)
        msg.position = [float(v) for v in target]

        end = time.time() + 5.0
        while rclpy.ok() and self.pub.get_subscription_count() == 0 and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.pub.get_subscription_count() == 0:
            print("nothing subscribed to /control/move_j -- is the arm driver running?",
                  file=sys.stderr)
            return False

        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)

        start = time.time()
        start_error = self.error(target) or 0.0
        in_tolerance_since = None
        retried = False
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            now = time.time()
            err = self.error(target)
            if err is not None and err <= tolerance:
                in_tolerance_since = in_tolerance_since or now
                if now - in_tolerance_since >= settle:
                    return True
            else:
                in_tolerance_since = None
            # One resend: a move_j published before the driver's warm-up gate
            # opened is simply dropped, and nothing tells you so.
            if (not retried and now - start > 2.0 and err is not None
                    and err > start_error - 0.01):
                print("no motion yet, resending move_j")
                msg.header.stamp = self.get_clock().now().to_msg()
                self.pub.publish(msg)
                retried = True
            if now - start > timeout:
                print(f"timed out after {timeout:.0f}s "
                      f"(max joint error {err if err is None else round(err, 3)} rad)",
                      file=sys.stderr)
                return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pose", nargs="?", choices=["ready", "rest"], default=None)
    parser.add_argument("--joints", nargs="+", type=float, default=None,
                        help="explicit joint angles, radians; implies --direct")
    parser.add_argument("--poses", default=None, help="arm_poses YAML")
    parser.add_argument("--direct", action="store_true",
                        help="publish /control/move_j even if arm_pose_manager is up")
    parser.add_argument("--force", action="store_true",
                        help="move even while teleop is streaming /delta_pose")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--settle", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=30.0)
    args, ros_args = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    if not args.pose and not args.joints:
        parser.error("give a pose (ready|rest) or --joints")

    names, poses = load_poses(args.poses or default_poses_file())
    if args.joints:
        target, label = args.joints, "the given joints"
        if len(target) != len(names):
            parser.error(f"--joints needs {len(names)} values, got {len(target)}")
    else:
        target, label = poses[args.pose], args.pose.upper()

    rclpy.init(args=ros_args)
    node = GotoPose(names)
    try:
        node.spin(0.5)          # let feedback and /delta_pose arrive

        use_manager = args.pose and not args.joints and not args.direct
        if use_manager:
            service = MANAGER_SERVICE[args.pose]
            print(f"calling {service} ...")
            # Blocks in arm_pose_manager until the arm has arrived.
            result = node.call(service, args.timeout + 20.0)
            if result is not None:
                print(f"{'ok' if result.success else 'FAILED'}: {result.message}")
                return 0 if result.success else 1
            print(f"{service} did not answer -- falling back to /control/move_j")

        if node.teleop_streaming() and not args.force:
            print("teleop is live (/delta_pose is streaming). Double-click the Sense "
                  "to disarm, or pass --force.", file=sys.stderr)
            return 1

        print(f"moving to {label}: [" + ", ".join(f"{v:.4f}" for v in target) + "]")
        ok = node.move_direct(target, args.tolerance, args.settle, args.timeout)
        print("arrived" if ok else "did not arrive")
        return 0 if ok else 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
