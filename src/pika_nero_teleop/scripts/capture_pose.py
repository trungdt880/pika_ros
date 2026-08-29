#!/usr/bin/env python3
"""Print the arm's current joint angles in the form arm_poses.nero_pika.yaml wants.

The READY and REST poses shipped in that file are forward kinematics on the
URDF and nothing else -- they know nothing about your table, fixturing or the
Pika Gripper's body. The honest way to set them is to put the arm where you
want it and read the joints back.

    # 1. Drag-teach the arm into the pose (web UI -> WEB -> Drag Teaching ->
    #    Add, hand on the arm, then Pause), or drive it there with teleop.
    # 2. With the arm driver running:
    ros2 run pika_nero_teleop capture_pose.py

Then paste the printed line into config/arm_poses.nero_pika.yaml as ready_pose
or rest_pose.

Reads /feedback/joint_states, which the driver publishes whether or not the
motors are enabled, so this works in drag-teaching mode.
"""
import argparse
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

DEFAULT_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]


class PoseCapture(Node):

    def __init__(self, topic, joints, samples):
        super().__init__("capture_pose")
        self.joints = joints
        self.samples = samples
        self.collected = []
        self.create_subscription(JointState, topic, self._callback, 10)

    def _callback(self, msg: JointState):
        angles = dict(zip(msg.name, msg.position))
        try:
            self.collected.append([angles[name] for name in self.joints])
        except KeyError as exc:
            self.get_logger().warn(
                f"joint {exc} not in feedback (has: {', '.join(msg.name)})")

    def done(self):
        return len(self.collected) >= self.samples

    def average(self):
        # Averaging a handful of samples takes the encoder jitter out of the
        # last digit, so the same physical pose captures to the same numbers.
        n = len(self.collected)
        return [sum(s[i] for s in self.collected) / n for i in range(len(self.joints))]


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/feedback/joint_states")
    parser.add_argument("--joints", nargs="+", default=DEFAULT_JOINTS)
    parser.add_argument("--samples", type=int, default=20,
                        help="feedback messages to average (default 20)")
    parser.add_argument("--timeout", type=float, default=10.0)
    parsed, ros_args = parser.parse_known_args(args if args is not None else sys.argv[1:])

    rclpy.init(args=ros_args)
    node = PoseCapture(parsed.topic, parsed.joints, parsed.samples)
    try:
        start = node.get_clock().now()
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.1)
            if (node.get_clock().now() - start).nanoseconds * 1e-9 > parsed.timeout:
                break

        if not node.collected:
            print(f"no messages on {parsed.topic} -- is the arm driver running?",
                  file=sys.stderr)
            return 1

        pose = node.average()
        print()
        print("    # captured " + " ".join(f"{n}={v:+.3f}" for n, v in zip(parsed.joints, pose)))
        print("    ready_pose: [" + ", ".join(f"{v:.3f}" for v in pose) + "]")
        print()
        return 0
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
