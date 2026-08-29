"""Arm side of Pika Sense -> NERO teleoperation, with a Pika Gripper on the flange.

Adapted from PikaAnyArm's teleop_single_nero.launch.py. Two deliberate changes:

  effector_type := none   (upstream: agx_gripper)
      The Pika Gripper is a USB Type-C device on its own serial link; its XT30
      CAN pins are documented as "reserved" and unused. It is NOT on the arm's
      end-effector CAN bus, so the arm driver must not try to drive a gripper.
      The jaws are commanded by pika_devices.launch.py instead.

  tcp_offset  := configurable, default zeros   (upstream: [0.1755, 0, -0.0235])
      That upstream number is the AgileX two-finger gripper's offset. See the
      TOOL OFFSET note in config/arm_ik_pose_node.nero_pika.yaml -- measure
      yours, then set it in BOTH places (they must agree).

Chain:  /pika_pose --(pub_delta_pose)--> /delta_pose --(arm_ik_pose_node)-->
        /control/joint_states --(agx_arm_ctrl)--> CAN --> NERO
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('pika_nero_teleop')
    agx_arm_ctrl_dir = get_package_share_directory('agx_arm_ctrl')

    declared = [
        DeclareLaunchArgument('can_port', default_value='can0'),
        # Firmware profile. Empty means auto-detect, which DOES NOT WORK on
        # this arm: it broadcasts state happily but never answers the 0x4AF
        # firmware query, so the driver exits with "Failed to get firmware
        # version". v104 is this arm's actual software version, read from the
        # web UI (Upgrade -> 1.04 / RM_PATCH_20260116_03); it resolves to the
        # NeroFW "default" profile.
        # Re-check after any controller software upgrade: the profiles change
        # command encoding (v112 overrides enable/disable, v121 overrides
        # move_mit/move_cpv_vel), so a stale value can mis-encode commands.
        DeclareLaunchArgument('fw_version', default_value='v104'),
        DeclareLaunchArgument('auto_enable', default_value='true'),
        # fast_mode routes /control/joint_states to the SDK's unsmoothed
        # move_js interface. Upstream enables it for NERO teleop: the IK node
        # already runs at the Sense's rate, and the extra interpolation just
        # adds lag. Set false if the arm feels jerky.
        DeclareLaunchArgument('fast_mode', default_value='true'),
        DeclareLaunchArgument(
            'tcp_offset', default_value='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]',
            description='Flange->TCP [x,y,z,rx,ry,rz]. MUST match '
                        'tool_translation_xyz in the IK params file.'),
        DeclareLaunchArgument(
            'ik_params_file',
            default_value=os.path.join(pkg_dir, 'config', 'arm_ik_pose_node.nero_pika.yaml')),
        # Mounting rotation applied to the Sense pose before differencing. The
        # -1.57 roll is upstream's value for holding the Sense the standard way
        # relative to the NERO base; adjust if the arm moves along the wrong
        # axis when you move the Sense.
        DeclareLaunchArgument('handle_pose_roll', default_value='-1.57'),
        DeclareLaunchArgument('handle_pose_pitch', default_value='0.0'),
        DeclareLaunchArgument('handle_pose_yaw', default_value='0.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
    ]

    # Arm driver + robot_state_publisher + RViz. control:=false so RViz only
    # follows /feedback/joint_states -- the IK node is the single source of
    # /control/joint_states.
    arm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(agx_arm_ctrl_dir, 'launch', 'start_single_agx_arm_rviz.launch.py')),
        launch_arguments={
            'can_port': LaunchConfiguration('can_port'),
            'fw_version': LaunchConfiguration('fw_version'),
            'arm_type': 'nero',
            'auto_enable': LaunchConfiguration('auto_enable'),
            'effector_type': 'none',
            'tcp_offset': LaunchConfiguration('tcp_offset'),
            'control': 'false',
            'follow': 'true',
            'fast_mode': LaunchConfiguration('fast_mode'),
        }.items(),
    )

    arm_ik_pose_node = Node(
        package='pika_remote_agx_arm',
        executable='arm_ik_pose_node.py',
        name='arm_ik_pose_node',
        output='screen',
        parameters=[LaunchConfiguration('ik_params_file')],
    )

    # Serves /teleop_trigger (double-click on the Sense) and turns absolute
    # Sense poses into arm-frame targets relative to where teleop was armed.
    pub_delta_pose_node = Node(
        package='pika_remote_agx_arm',
        executable='pub_delta_pose.py',
        name='pub_delta_pose_node',
        output='screen',
        parameters=[{
            'handle_pose_roll': LaunchConfiguration('handle_pose_roll'),
            'handle_pose_pitch': LaunchConfiguration('handle_pose_pitch'),
            'handle_pose_yaw': LaunchConfiguration('handle_pose_yaw'),
        }],
    )

    return LaunchDescription(declared + [
        arm_launch,
        arm_ik_pose_node,
        pub_delta_pose_node,
    ])
