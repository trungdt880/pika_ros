"""Everything needed to run a policy on the arm, in one command.

    # replay a recorded episode (no model, no GPU) -- the smoke test
    ros2 launch pika_nero_teleop policy.launch.py \
        replay_episode:=/root/pika_ros/data/episode0

    # a real policy server, here or on another machine
    ros2 launch pika_nero_teleop policy.launch.py server_host:=192.168.1.50

Then, when the client reports the observation looks right:

    ros2 service call /policy/start std_srvs/srv/Trigger
    ros2 service call /policy/stop  std_srvs/srv/Trigger

This is the AUTONOMOUS counterpart to bringup.launch.py, and the differences
from it are deliberate:

  sense:=false      The Pika Sense is NOT started. A policy commands the jaws on
                    /sensor/gripper/joint_state, which is the topic the Sense
                    itself publishes and the follower gripper listens to -- with
                    both running they fight over the jaws. The gripper node
                    stays up, because a policy still needs the jaw opening for
                    observation.state[7].

  locator:=false    No Vive tracking. Nothing produces /pika_pose, so teleop can
                    never arm, so /delta_pose stays silent -- which is what the
                    client's safety check wants. Two command sources on one arm
                    is how you break something.

  rviz:=false       ~3.5 cores between the two RViz instances, and nobody is
                    watching them during a rollout.

The arm still moves to its READY pose on startup (arm_pose_manager), which is
where every training episode begins. A policy asked to start from anywhere else
is being asked to generalise in a direction it has no data for.
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
    launch_dir = os.path.join(get_package_share_directory('pika_nero_teleop'), 'launch')

    declared = [
        DeclareLaunchArgument('can_port', default_value='can0'),
        DeclareLaunchArgument('gripper_serial_port', default_value='/dev/ttyUSB60'),

        DeclareLaunchArgument(
            'replay_episode', default_value='',
            description='a recorded episode directory. Non-empty starts the replay '
                        'server locally and points the client at it.'),
        DeclareLaunchArgument('server_host', default_value='localhost'),
        DeclareLaunchArgument('server_port', default_value='8000'),

        DeclareLaunchArgument('prompt',
                              default_value='pick up the fruits and put into the basket'),
        DeclareLaunchArgument('action_horizon', default_value='30'),
        DeclareLaunchArgument('rate', default_value='30.0'),
        DeclareLaunchArgument('max_joint_speed', default_value='2.5'),

        # OFF by default: on a first run you want to see the arm behave before
        # letting anything close on an object.
        DeclareLaunchArgument(
            'drive_gripper', default_value='false',
            description='false sends jaw commands to a dead topic, so only the arm moves'),
        DeclareLaunchArgument('cameras', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
    ]

    arm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'teleop_nero_pika.launch.py')),
        launch_arguments={'can_port': LaunchConfiguration('can_port'),
                          'rviz': LaunchConfiguration('rviz')}.items(),
    )
    gripper = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'pika_devices.launch.py')),
        launch_arguments={'sense': 'false', 'locator': 'false',
                          'gripper_serial_port': LaunchConfiguration('gripper_serial_port')}.items(),
    )
    cameras = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'cameras.launch.py')),
        condition=IfCondition(LaunchConfiguration('cameras')),
    )

    replay_server = Node(
        package='pika_nero_teleop', executable='replay_policy_server.py',
        name='replay_policy_server', output='screen', emulate_tty=True,
        condition=UnlessCondition(PythonExpression(
            ["'", LaunchConfiguration('replay_episode'), "' == ''"])),
        arguments=['--episode', LaunchConfiguration('replay_episode'),
                   '--port', LaunchConfiguration('server_port'),
                   '--action-horizon', LaunchConfiguration('action_horizon')],
    )

    client = Node(
        package='pika_nero_teleop', executable='policy_client.py',
        name='policy_client', output='screen', emulate_tty=True,
        arguments=[
            '--host', LaunchConfiguration('server_host'),
            '--port', LaunchConfiguration('server_port'),
            '--prompt', LaunchConfiguration('prompt'),
            '--rate', LaunchConfiguration('rate'),
            '--action-horizon', LaunchConfiguration('action_horizon'),
            '--max-joint-speed', LaunchConfiguration('max_joint_speed'),
            '--jaw-command-topic', PythonExpression(
                ["'/sensor/gripper/joint_state' if '", LaunchConfiguration('drive_gripper'),
                 "' in ('true','True','1') else '/unused/jaw'"]),
        ],
    )

    return LaunchDescription(declared + [arm, gripper, cameras, replay_server, client])
