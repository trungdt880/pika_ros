"""Everything: Pika devices + NERO arm teleop, in one command.

    ros2 launch pika_nero_teleop bringup.launch.py can_port:=can0 \
        sense_serial_port:=/dev/ttyUSB50 gripper_serial_port:=/dev/ttyUSB60

Bring the two halves up in separate terminals instead (pika_devices.launch.py,
then teleop_nero_pika.launch.py) when you want to see which side is
misbehaving -- six nodes logging into one terminal is hard to read.

Arguments are re-declared and forwarded explicitly rather than relying on
scope inheritance, so `ros2 launch ... --show-args` lists them here.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# (name, default) forwarded to pika_devices.launch.py
_DEVICE_ARGS = [
    ('sense_serial_port', '/dev/ttyUSB50'),
    ('gripper_serial_port', '/dev/ttyUSB60'),
    ('locator', 'true'),
]

# Cameras are opt-in: teleop does not need them, and starting them here would
# let a camera fault take the arm down with it.
_CAMERA_ARGS = [
    ('wrist', 'true'),
    ('scene', 'true'),
    ('sense_cameras', 'false'),
]

# (name, default) forwarded to teleop_nero_pika.launch.py
_ARM_ARGS = [
    ('can_port', 'can0'),
    ('auto_enable', 'true'),
    ('fast_mode', 'true'),
    ('tcp_offset', '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'),
    ('handle_pose_roll', '-1.57'),
    ('handle_pose_pitch', '0.0'),
    ('handle_pose_yaw', '0.0'),
]


def generate_launch_description():
    launch_dir = os.path.join(get_package_share_directory('pika_nero_teleop'), 'launch')

    declared = [DeclareLaunchArgument(n, default_value=d)
                for n, d in _DEVICE_ARGS + _ARM_ARGS + _CAMERA_ARGS]
    declared.append(DeclareLaunchArgument(
        'cameras', default_value='false',
        description='also start the recording cameras (cameras.launch.py)'))

    devices = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'pika_devices.launch.py')),
        launch_arguments={n: LaunchConfiguration(n) for n, _ in _DEVICE_ARGS}.items(),
    )
    arm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'teleop_nero_pika.launch.py')),
        launch_arguments={n: LaunchConfiguration(n) for n, _ in _ARM_ARGS}.items(),
    )

    cameras = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'cameras.launch.py')),
        condition=IfCondition(LaunchConfiguration('cameras')),
        launch_arguments={n: LaunchConfiguration(n) for n, _ in _CAMERA_ARGS}.items(),
    )

    return LaunchDescription(declared + [devices, arm, cameras])
