"""Record an episode of NERO + Pika teleop.

Wraps data_tools' capture node. A separate launch exists because
data_tools/launch/run_data_capture.launch.py sets

    prefix='gnome-terminal -- bash -c ...'

which cannot work in the container (no gnome-terminal, no desktop session).
This runs the node directly in the current terminal instead.

    ros2 launch pika_nero_teleop record.launch.py \
        datasetDir:=/root/pika_ros/data episodeIndex:=0 \
        instructions:='pick up the red block'

Press ENTER in this terminal to stop the episode and flush it to disk. Writing
takes a moment; wait for "Done".
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # The config has to live in data_tools' share dir: the capture node, and
    # also data_to_hdf5.py / hdf5_to_lerobot.py, all resolve
    # config/<type>_data_params.yaml relative to data_tools.
    params = os.path.join(
        get_package_share_directory('data_tools'),
        'config', 'nero_pika_teleop_data_params.yaml')

    declared = [
        DeclareLaunchArgument('datasetDir', default_value='/root/pika_ros/data'),
        DeclareLaunchArgument('episodeIndex', default_value='0'),
        DeclareLaunchArgument('instructions', default_value='[null]'),
        # Capture aborts as soon as any topic drops below this. With the Fast
        # DDS large-data profile in place (docker/fastdds_large_data.xml) all
        # four image streams measure 29.9 Hz with zero loss, so 20 is safe.
        # Without that profile images arrive at ~9 Hz and this must be lowered.
        DeclareLaunchArgument('hz', default_value='20'),
        DeclareLaunchArgument('timeout', default_value='5'),
        DeclareLaunchArgument('cropTime', default_value='1.0'),
        # false = press ENTER here to stop. true = start/stop via the
        # /data_tools_dataCapture/capture_service, which the Pika Sense button
        # calls -- note that same button also toggles teleop.
        DeclareLaunchArgument('useService', default_value='false'),
    ]

    return LaunchDescription(declared + [
        Node(
            package='data_tools',
            executable='data_tools_dataCapture',
            parameters=[params, {
                'useService': LaunchConfiguration('useService'),
                'datasetDir': LaunchConfiguration('datasetDir'),
                'episodeIndex': LaunchConfiguration('episodeIndex'),
                'instructions': LaunchConfiguration('instructions'),
                'hz': LaunchConfiguration('hz'),
                'timeout': LaunchConfiguration('timeout'),
                'cropTime': LaunchConfiguration('cropTime'),
            }],
            output='screen',
            emulate_tty=True,
        )
    ])
