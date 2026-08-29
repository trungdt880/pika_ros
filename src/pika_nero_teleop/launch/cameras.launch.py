"""Cameras for recording. Separate from teleop on purpose -- teleop needs none
of these, so a camera problem must never stop you driving the arm.

What this starts, matching what is actually wanted for NERO teleop recording:

  * the two WRIST cameras on the end effector (Pika Gripper): its fisheye and
    its D405 depth camera
  * the SCENE / third-person camera (D435)

Deliberately NOT started: the Pika Sense's own fisheye and D405. On the leader
device they are only useful for UMI-style handheld collection, and skipping
them frees real USB bandwidth. Set sense_cameras:=true when doing UMI.

NOTE on serial numbers: realsense2_camera needs a LEADING UNDERSCORE on
serial_no or it silently matches nothing (verified: `serial_no:=254622075792`
starts no topics, `serial_no:=_254622075792` works). This file adds the
underscore for you -- pass the bare serial.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Serials detected on this rig, 2026-08-28. Re-check with `rs-enumerate-devices`
# if you swap hardware.
GRIPPER_D405 = '419122270401'   # on the Pika Gripper (wrist), USB 3.2
SCENE_D435   = '254622075792'   # third-person view, USB 3.2
SENSE_D405   = '419122271385'   # on the Pika Sense; currently USB 2.1 only


def _rs(name, namespace, serial_arg, profile, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'),
                         'launch', 'rs_launch.py')),
        condition=condition,
        launch_arguments={
            # leading '_' is required by the wrapper, see module docstring
            'serial_no': ['_', LaunchConfiguration(serial_arg)],
            'camera_namespace': namespace,
            'camera_name': name,
            'rgb_camera.color_profile': LaunchConfiguration(profile),
            'depth_module.depth_profile': LaunchConfiguration(profile),
            # data_tools records .../aligned_depth_to_color/..., which the
            # wrapper only publishes when alignment is switched on.
            'align_depth.enable': 'true',
        }.items(),
    )


def generate_launch_description():
    declared = [
        DeclareLaunchArgument('gripper_depth_serial', default_value=GRIPPER_D405),
        DeclareLaunchArgument('scene_depth_serial', default_value=SCENE_D435),
        DeclareLaunchArgument('sense_depth_serial', default_value=SENSE_D405),

        # /dev/video60 and /dev/video50 come from docker/host_setup.sh
        DeclareLaunchArgument('gripper_fisheye_port', default_value='60'),
        DeclareLaunchArgument('sense_fisheye_port', default_value='50'),

        DeclareLaunchArgument('camera_fps', default_value='30'),
        DeclareLaunchArgument('camera_width', default_value='640'),
        DeclareLaunchArgument('camera_height', default_value='480'),
        DeclareLaunchArgument('camera_profile', default_value='640x480x30'),

        DeclareLaunchArgument('wrist', default_value='true',
                              description='Pika Gripper fisheye + D405'),
        DeclareLaunchArgument('scene', default_value='true',
                              description='third-person D435'),
        DeclareLaunchArgument('sense_cameras', default_value='false',
                              description='Pika Sense fisheye + D405; only for UMI collection'),
    ]

    wrist = IfCondition(LaunchConfiguration('wrist'))
    sense = IfCondition(LaunchConfiguration('sense_cameras'))

    gripper_fisheye = Node(
        package='sensor_tools', executable='usb_camera.py',
        name='gripper_camera_fisheye', condition=wrist,
        parameters=[{
            'camera_port': LaunchConfiguration('gripper_fisheye_port'),
            'camera_fps': LaunchConfiguration('camera_fps'),
            'camera_height': LaunchConfiguration('camera_height'),
            'camera_width': LaunchConfiguration('camera_width'),
            'camera_frame_id': 'gripper/camera_fisheye_link',
        }],
        remappings=[
            ('/camera_rgb/color/image_raw', '/gripper/camera_fisheye/color/image_raw'),
            ('/camera_rgb/color/camera_info', '/gripper/camera_fisheye/color/camera_info'),
        ],
        respawn=True, output='screen',
    )

    sense_fisheye = Node(
        package='sensor_tools', executable='usb_camera.py',
        name='sense_camera_fisheye', condition=sense,
        parameters=[{
            'camera_port': LaunchConfiguration('sense_fisheye_port'),
            'camera_fps': LaunchConfiguration('camera_fps'),
            'camera_height': LaunchConfiguration('camera_height'),
            'camera_width': LaunchConfiguration('camera_width'),
            'camera_frame_id': 'sensor/camera_fisheye_link',
        }],
        remappings=[
            ('/camera_rgb/color/image_raw', '/sensor/camera_fisheye/color/image_raw'),
            ('/camera_rgb/color/camera_info', '/sensor/camera_fisheye/color/camera_info'),
        ],
        respawn=True, output='screen',
    )

    return LaunchDescription(declared + [
        gripper_fisheye,
        sense_fisheye,
        _rs('camera', 'gripper', 'gripper_depth_serial', 'camera_profile', wrist),
        _rs('camera', 'scene',   'scene_depth_serial',   'camera_profile',
            IfCondition(LaunchConfiguration('scene'))),
        _rs('camera', 'sensor',  'sense_depth_serial',   'camera_profile', sense),
    ])
