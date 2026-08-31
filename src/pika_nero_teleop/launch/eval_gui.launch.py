"""Evaluation UI for policy trials.

Starts only the web UI -- the arm, cameras, client and policy server are
whatever you already have running. That separation is deliberate: a browser
refresh or a UI restart must never be able to interrupt a trial or move the arm.

    ros2 launch pika_nero_teleop policy.launch.py          # terminal 1
    ros2 launch pika_nero_teleop eval_gui.launch.py \\
        checkpoint:=full95                                 # terminal 2
    # open http://localhost:8081

`checkpoint` is a label only. It is written into every row of results.csv so a
session can be attributed later; it does not select or load anything.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declared = [
        DeclareLaunchArgument('evalDir', default_value='/root/pika_ros/eval'),
        # 8080 is the recording UI, so this one sits next to it and the two can
        # run at the same time.
        DeclareLaunchArgument('port', default_value='8081'),
        DeclareLaunchArgument('checkpoint', default_value='unnamed-checkpoint',
                              description='label recorded with every trial'),
        DeclareLaunchArgument('prompt',
                              default_value='pick up the fruits and put into the basket'),
        # Semicolon-separated label=topic pairs, e.g.
        #   cameras:='scene=/scene/camera/color/image_raw;wrist=/gripper/camera/color/image_raw'
        # Defaults to the two streams the policy actually consumes.
        DeclareLaunchArgument('cameras', default_value=''),
    ]

    gui = Node(
        package='pika_nero_teleop',
        executable='eval_gui.py',
        name='pika_eval_gui',
        output='screen',
        emulate_tty=True,
        additional_env={
            'PIKA_EVAL_DIR': LaunchConfiguration('evalDir'),
            'PIKA_EVAL_PORT': LaunchConfiguration('port'),
            'PIKA_EVAL_CHECKPOINT': LaunchConfiguration('checkpoint'),
            'PIKA_EVAL_PROMPT': LaunchConfiguration('prompt'),
            'PIKA_EVAL_CAMERAS': LaunchConfiguration('cameras'),
        },
    )

    return LaunchDescription(declared + [gui])
