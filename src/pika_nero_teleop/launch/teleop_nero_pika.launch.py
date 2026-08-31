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

arm_pose_manager sits in front of that chain rather than inside it. It owns the
Sense's /teleop_trigger service and forwards to pub_delta_pose under
/teleop_trigger_arm, so it can put the arm in a fixed READY pose before every
episode and bring it back afterwards, and park it at REST before power-off. It
commands /control/move_j, a different topic from the teleop chain's
/control/joint_states, and only ever while teleop is disarmed.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
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
        # false drops the arm's RViz. Worth doing while recording: two RViz
        # instances cost ~3.5 cores, and this file DECLARED this argument
        # without ever using it, so it silently did nothing until now.
        #
        # It also drops robot_state_publisher, because upstream bundles the two
        # in start_single_agx_arm_rviz.launch.py and neither it nor
        # display.launch.py exposes a switch. Nothing in the teleop or
        # recording path needs it -- arm_ik_pose_node loads the URDF itself and
        # no /tf is recorded -- but you lose /tf and /tf_static, so anything
        # you add later that wants them needs rviz:=true.
        #
        # The LOCATOR's RViz (vive.rviz) is separate and cannot be switched off
        # at all: pika_locator's launch adds it unconditionally and the node is
        # closed source. Kill it after bringup if you want the cores back:
        #     pkill -f 'rviz2.*vive.rviz'
        DeclareLaunchArgument('rviz', default_value='true'),
        # READY / REST poses and the timings around them. Defaults are FK on
        # the URDF only -- capture your own with capture_pose.py. See the file.
        DeclareLaunchArgument(
            'arm_poses_file',
            default_value=os.path.join(pkg_dir, 'config', 'arm_poses.nero_pika.yaml')),
        # false gives the plain upstream behaviour: no startup move, teleop
        # arms wherever the arm happens to be, nothing parks it on the way out.
        DeclareLaunchArgument('pose_manager', default_value='true'),
    ]

    # Arm driver + robot_state_publisher + RViz. control:=false so RViz only
    # follows /feedback/joint_states -- the IK node is the single source of
    # /control/joint_states.
    _arm_args = {
        'can_port': LaunchConfiguration('can_port'),
        'fw_version': LaunchConfiguration('fw_version'),
        'arm_type': 'nero',
        'auto_enable': LaunchConfiguration('auto_enable'),
        'effector_type': 'none',
        'tcp_offset': LaunchConfiguration('tcp_offset'),
        'fast_mode': LaunchConfiguration('fast_mode'),
    }
    # With RViz: the _rviz variant, which also brings robot_state_publisher.
    # control:=false so RViz only follows /feedback/joint_states -- the IK node
    # is the single source of /control/joint_states.
    arm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(agx_arm_ctrl_dir, 'launch', 'start_single_agx_arm_rviz.launch.py')),
        condition=IfCondition(LaunchConfiguration('rviz')),
        launch_arguments=dict(_arm_args, control='false', follow='true').items(),
    )
    # Without: the plain driver launch. It takes no control/follow arguments,
    # which are display-side only.
    arm_launch_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(agx_arm_ctrl_dir, 'launch', 'start_single_agx_arm.launch.py')),
        condition=UnlessCondition(LaunchConfiguration('rviz')),
        launch_arguments=_arm_args.items(),
    )

    arm_ik_pose_node = Node(
        package='pika_remote_agx_arm',
        executable='arm_ik_pose_node.py',
        name='arm_ik_pose_node',
        output='screen',
        parameters=[LaunchConfiguration('ik_params_file')],
    )

    # Turns absolute Sense poses into arm-frame targets relative to where
    # teleop was armed. Its trigger service is moved off /teleop_trigger:
    # with pose_manager:=true that name belongs to arm_pose_manager, which
    # forwards here after it has the arm in the right pose. With
    # pose_manager:=false nothing serves /teleop_trigger and the Sense's
    # double-click would go nowhere, so the name is chosen by the same
    # condition.
    pub_delta_pose_node = Node(
        package='pika_remote_agx_arm',
        executable='pub_delta_pose.py',
        name='pub_delta_pose_node',
        output='screen',
        parameters=[{
            'handle_pose_roll': LaunchConfiguration('handle_pose_roll'),
            'handle_pose_pitch': LaunchConfiguration('handle_pose_pitch'),
            'handle_pose_yaw': LaunchConfiguration('handle_pose_yaw'),
            'teleop_trigger_service': PythonExpression([
                "'/teleop_trigger_arm' if '", LaunchConfiguration('pose_manager'),
                "' in ('true', 'True', '1') else '/teleop_trigger'",
            ]),
        }],
    )

    # Serves /teleop_trigger (the Sense's double-click) and sequences the
    # READY / REST poses around each teleop session.
    arm_pose_manager_node = Node(
        package='pika_nero_teleop',
        executable='arm_pose_manager.py',
        name='arm_pose_manager',
        output='screen',
        emulate_tty=True,
        parameters=[LaunchConfiguration('arm_poses_file')],
        condition=IfCondition(LaunchConfiguration('pose_manager')),
    )

    return LaunchDescription(declared + [
        arm_launch,
        arm_launch_headless,
        arm_ik_pose_node,
        pub_delta_pose_node,
        arm_pose_manager_node,
    ])
