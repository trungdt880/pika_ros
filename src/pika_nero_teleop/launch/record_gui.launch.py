"""Recording with a browser UI instead of a terminal.

Starts data_tools' capture node in SERVICE mode (so it stays resident between
takes) plus a small web server that drives it.

    ros2 launch pika_nero_teleop record_gui.launch.py
    # open http://localhost:8080

The Pika Sense button also calls that same capture service, so a double-click
starts/stops a recording as well -- note it toggles teleop at the same time.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('data_tools'),
        'config', 'nero_pika_teleop_data_params.yaml')

    declared = [
        DeclareLaunchArgument('datasetDir', default_value='/root/pika_ros/data'),
        DeclareLaunchArgument('port', default_value='8080'),
        # 10, not 20: the cameras measure 18-21 Hz on this rig despite a
        # 30 fps profile, so a threshold of 20 aborts recordings. See
        # record.launch.py for the numbers.
        DeclareLaunchArgument('hz', default_value='10'),
        DeclareLaunchArgument('timeout', default_value='5'),
        DeclareLaunchArgument('cropTime', default_value='1.0'),
    ]

    capture = Node(
        package='data_tools',
        executable='data_tools_dataCapture',
        parameters=[params, {
            'useService': True,          # stay resident; start/stop by service
            'datasetDir': LaunchConfiguration('datasetDir'),
            'episodeIndex': 0,
            'instructions': '[null]',
            'hz': LaunchConfiguration('hz'),
            'timeout': LaunchConfiguration('timeout'),
            'cropTime': LaunchConfiguration('cropTime'),
        }],
        output='screen',
        emulate_tty=True,
    )

    gui = Node(
        package='pika_nero_teleop',
        executable='record_gui.py',
        name='pika_record_gui',
        output='screen',
        emulate_tty=True,
        additional_env={
            'PIKA_DATASET_DIR': LaunchConfiguration('datasetDir'),
            'PIKA_GUI_PORT': LaunchConfiguration('port'),
            'PIKA_CAPTURE_PARAMS': params,
        },
    )

    return LaunchDescription(declared + [capture, gui])
