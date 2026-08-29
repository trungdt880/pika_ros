"""Pika Sense (leader) + Pika Gripper (follower) device bringup.

This is the half of the system that does not involve the arm at all. Cameras
are deliberately NOT here -- they live in cameras.launch.py, so that a camera
problem can never stop you from driving the arm.

Gripper teleop never touches a ROS control topic: the second serial_gripper_imu
node has its `/gripper/joint_state_ctrl` input remapped to the FIRST node's
`/gripper/joint_state` output, so the Pika Gripper's jaws mirror the Sense's
trigger directly at the serial driver level. That is also why upstream's
pub_delta_pose.py has its gripper publishing commented out -- it would be a
second, competing command source.

The Sense's button also lives here: on a double-click the serial node calls the
`/teleop_trigger` Trigger service, which is what arms and disarms arm teleop in
pub_delta_pose.py.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declared = [
        # Serial ports. These are the udev symlinks created by
        # sensor_tools/scripts/setup_sensor_gripper.bash -- but that script
        # hardcodes AgileX's own USB bus paths (3-4.4.4:1.0 etc). Run
        # setup_device.py on your machine to generate rules for YOUR ports,
        # or just point these at the raw /dev/ttyUSBn names.
        DeclareLaunchArgument('sense_serial_port', default_value='/dev/ttyUSB50'),
        DeclareLaunchArgument('gripper_serial_port', default_value='/dev/ttyUSB60'),

        DeclareLaunchArgument('sense_joint_name', default_value='sensor_gripper_center_joint'),
        DeclareLaunchArgument('gripper_joint_name', default_value='gripper_gripper_center_joint'),

        # Gripper motor limits, as used by upstream open_sensor_gripper.launch.py
        DeclareLaunchArgument('motor_current_limit', default_value='1000.0'),
        DeclareLaunchArgument('motor_current_redundancy', default_value='500.0'),
        DeclareLaunchArgument('mit_mode', default_value='true'),
        DeclareLaunchArgument('ctrl_rate', default_value='50.0'),

        # RViz inside the locator launch is useful for checking tracking, but
        # it is another window competing with the arm's RViz.
        DeclareLaunchArgument('locator', default_value='true'),
    ]

    # pika_locator turns Vive base-station data into /pika_pose. Prebuilt,
    # closed source; comes out of pika_ros/source/install.zip.
    locator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('pika_locator'),
                         'launch', 'pika_single_locator.launch.py')),
        condition=IfCondition(LaunchConfiguration('locator')),
    )

    # --- Pika Sense: pose trigger + jaw opening measurement -----------------
    sense_node = Node(
        package='sensor_tools',
        executable='serial_gripper_imu',
        name='sense_serial_gripper_imu',
        parameters=[{
            'serial_port': LaunchConfiguration('sense_serial_port'),
            'joint_name': LaunchConfiguration('sense_joint_name'),
        }],
        remappings=[
            ('/imu/data', '/sensor/imu/data'),
            ('/gripper/data', '/sensor/gripper/data'),
            ('/gripper/ctrl', '/sensor/gripper/ctrl'),
            # This is the measurement the follower gripper tracks.
            ('/gripper/joint_state', '/sensor/gripper/joint_state'),
            ('/gripper/joint_state_ctrl', '/sensor/gripper/joint_state_ctrl'),
            ('/joint_state_info', '/joint_states'),
            ('/joint_state_gripper', '/joint_states_gripper'),
            ('/data_capture_status', '/data_tools_dataCapture/status'),
            ('/teleop_status', '/teleop_status'),
            ('/localization_status', '/pika_localization_status'),
            ('/arm_control_status', '/arm_control_status'),
            # Left unremapped on purpose: the double-click must reach the
            # /teleop_trigger service served by pub_delta_pose.py.
        ],
        respawn=True,
        output='screen',
    )

    # --- Pika Gripper on the NERO flange: follows the Sense -----------------
    gripper_node = Node(
        package='sensor_tools',
        executable='serial_gripper_imu',
        name='gripper_serial_gripper_imu',
        parameters=[{
            'serial_port': LaunchConfiguration('gripper_serial_port'),
            'joint_name': LaunchConfiguration('gripper_joint_name'),
            'motor_current_limit': LaunchConfiguration('motor_current_limit'),
            'motor_current_redundancy': LaunchConfiguration('motor_current_redundancy'),
            'mit_mode': LaunchConfiguration('mit_mode'),
            'ctrl_rate': LaunchConfiguration('ctrl_rate'),
        }],
        remappings=[
            ('/imu/data', '/gripper/imu/data'),
            ('/gripper/data', '/gripper/gripper/data'),
            ('/gripper/ctrl', '/gripper/gripper/ctrl'),
            ('/gripper/joint_state', '/gripper/gripper/joint_state'),
            # THE teleop link: follower jaw command <- leader jaw measurement.
            ('/gripper/joint_state_ctrl', '/sensor/gripper/joint_state'),
            ('/joint_state_info', '/joint_states_single'),
            ('/joint_state_gripper', '/joint_states_single_gripper'),
        ],
        respawn=True,
        output='screen',
    )

    return LaunchDescription(declared + [
        locator_launch,
        sense_node,
        gripper_node,
    ])
