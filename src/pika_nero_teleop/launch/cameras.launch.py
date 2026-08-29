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

NOTE on start order: devices are STAGGERED, one every `camera_start_delay`
seconds. Brought up all at once they were seen to collide:

    [gripper.camera] get_xu(id=) failed! Last Error: Device or resource busy
    [gripper.camera] Error starting device: depth_module.enable_auto_exposure
    [scene.camera]   Hardware Notification:Depth stream start failure

Those errors are a symptom of CPU starvation at startup, NOT of the camera --
see the depth note below for the evidence. On a quiet machine a clean single
launch produces none of them and every stream runs 29.99 Hz. The delay is kept
because it costs only startup time and it makes bring-up robust when the
machine is busy, which during recording it usually is.

Cost: the last camera is up at slot * camera_start_delay seconds -- 16 s for
the default wrist+scene set, 24 s with sense_cameras:=true. If you have culled
the load and want faster starts, this is the knob to lower.


DEPTH, and why it is off by default
-----------------------------------
Because the policies being trained here take colour only, and depth is bytes,
CPU and disk spent on something nothing reads. That is the whole reason.

It is NOT because depth is broken. An earlier version of this file claimed the
gripper D405's depth was the slow stream and blamed its firmware; both were
wrong. Measured properly on 2026-08-29: with the machine idle the stock
configuration gives 29.99 Hz on colour, raw depth AND aligned depth on both
cameras simultaneously, with alignment on and no errors.

What actually caused every slow number and every "Device or resource busy":
CPU starvation. The machine was at loadavg 20 with 2.7% idle, from the arm
bringup's two rviz2 instances plus TWO duplicate cameras.launch.py instances
whose fisheye nodes were fighting over /dev/video60 in a respawn loop burning
280% CPU. Same camera, same config, only the load differs:

    loaded (loadavg 20)   colour 26.2   raw depth 30.1   aligned 26.3   6.0% frames lost
    idle   (loadavg  1.9) colour 29.99  raw depth 30.00  aligned 29.99  0.0% lost

Note which stream was slow: COLOUR. Raw depth was the fastest thing running.
Ruled out by measurement, not argument: firmware (raw UVC via v4l2-ctl hit
29.99 fps on the same firmware), alignment cost (turning align off dropped node
CPU from 10% to 1.9% and made frame loss slightly WORSE), USB bandwidth (raw
UVC ran colour and depth concurrently at 30 each; halving the rate to 15 still
lost ~6%), and depth exposure -- the realsense-ros #2486 mechanism is real but
quantises to 30/15/6 Hz, so it cannot produce 24-28.

So: if you want depth back, `depth:=true` is expected to just work. Re-enable
the camera.depth block in
data_tools/config/nero_pika_teleop_data_params.yaml at the same time, and keep
an eye on machine load rather than on the camera.

DON'T RUN TWO OF THESE AT ONCE. Duplicate instances put two usb_camera nodes on
the same /dev/video60; both fail, both respawn, and the resulting CPU storm is
what makes the RealSense nodes collide at startup. Check with
`pgrep -af cameras.launch` before launching.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

# Serials detected on this rig, 2026-08-28. Re-check with `rs-enumerate-devices`
# if you swap hardware.
GRIPPER_D405 = '419122270401'   # on the Pika Gripper (wrist), USB 3.2
SCENE_D435   = '254622075792'   # third-person view, USB 3.2
SENSE_D405   = '419122271385'   # on the Pika Sense; currently USB 2.1 only


def _rs(name, namespace, serial_arg, profile, condition=None,
        color_param='rgb_camera.color_profile'):
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
            # Which sensor carries Color differs by model, and the wrapper
            # silently IGNORES a profile aimed at a sensor the camera does not
            # have -- no warning, it just runs at the default resolution.
            # Verified with `rs-enumerate-devices -c` on this rig:
            #   D405  no RGB Camera at all; Color is on the Stereo Module,
            #         so the parameter is depth_module.color_profile
            #   D435  separate RGB Camera; rgb_camera.color_profile
            # Getting this wrong left the gripper's colour at the 848x480
            # default instead of 640x480 -- ~33% more bandwidth than asked for,
            # and measurably more fragile under CPU load.
            color_param: LaunchConfiguration(profile),
            'depth_module.depth_profile': LaunchConfiguration(profile),
            # Depth is OFF by default -- see the `depth` argument below.
            # data_tools records .../aligned_depth_to_color/..., which the
            # wrapper only publishes when alignment is switched on, so the two
            # follow the same flag.
            'enable_depth': LaunchConfiguration('depth'),
            'align_depth.enable': LaunchConfiguration('depth'),
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
        # The gripper D405 is asked for 60 and allowed to lose frames down to
        # ~55, rather than asked for 30 and delivering 28.
        #
        # It drops ~8% of frames in USB transit. The camera says so itself:
        #
        #     XXX Hardware Notification: Incomplete video frame detected!
        #
        # and the frames are gone at the DEVICE frame counter -- 41 of 41 gaps
        # in a 25 s sample skipped it -- so nothing on the host is discarding
        # them. Measured and ruled out: CPU (node at 2.5%, box 68% idle, PSI
        # cpu full=0.00, no thread over 2%), disk (0% util against 89k small
        # writes/s of headroom), depth alignment (depth is off entirely),
        # global_time polling, and the fisheye sharing the Pika hub. What does
        # correlate is bandwidth: 848x480 -> 26.8 Hz, 640x480 -> ~28,
        # 480x270 -> 29.995. RGB8 at 640x480x30 is 27.6 MB/s, and this camera
        # sits at the end of the arm's cable run behind the gripper's hub,
        # unlike the directly-connected D435, which never drops a frame.
        #
        # Oversampling sidesteps it. Measured with this setting: gripper 57.98
        # Hz delivered (8.1% lost from 60), scene 30.06 Hz. That makes the D435
        # the slowest stream, and data_sync.py matches every frame to the
        # slowest -- so the dataset comes out a uniform 30 rather than being
        # dragged to 28. Cross-camera alignment improves too: a gripper frame
        # is now within ~17 ms of every scene frame instead of ~33.
        #
        # NOT TRIED, and the better fix if you want one: depth_module.color_format
        # YUYV instead of RGB8 cuts the wire bandwidth by a third (18.4 vs
        # 27.6 MB/s at 30 Hz), which may remove the transport errors outright.
        # It changes the published image encoding, so check data_tools and
        # record_gui still convert it before trusting a recording.
        #
        # The D405's Stereo Module offers 90/60/30/15/5 at 640x480 in RGB8,
        # BGR8, RGBA8, BGRA8 and YUYV (`rs-enumerate-devices -c`).
        DeclareLaunchArgument('gripper_profile', default_value='640x480x60'),

        DeclareLaunchArgument('wrist', default_value='true',
                              description='Pika Gripper D405'),
        # The gripper's fisheye. OFF by default -- not currently used for
        # training, and it is a second camera on the Pika hub competing for the
        # same CPU that the D405 needs to hold 30 Hz. Turn it on with
        # fisheye:=true, and re-add pikaGripperFisheyeCamera to the
        # camera.color lists in nero_pika_teleop_data_params.yaml at the same
        # time, or the recorder will wait for a topic nobody publishes.
        DeclareLaunchArgument('fisheye', default_value='false',
                              description='Pika Gripper fisheye camera'),
        DeclareLaunchArgument('scene', default_value='true',
                              description='third-person D435'),
        DeclareLaunchArgument('sense_cameras', default_value='false',
                              description='Pika Sense fisheye + D405; only for UMI collection'),

        # Seconds between RealSense node starts. 8 is what was measured to
        # work; see the start-order note in the module docstring. Raise it if a
        # camera still reports "Device or resource busy" at startup.
        DeclareLaunchArgument('camera_start_delay', default_value='8.0'),

        # Depth streams. OFF by default: the policies being trained here take
        # colour only, and the D405 depth path is where all the trouble is --
        # see the depth note in the module docstring. Turning it off also
        # removes the slowest stream from data_sync.py, which is what was
        # dragging the synced dataset below the colour rate.
        #
        # Switching this on REQUIRES re-enabling the camera.depth block in
        # data_tools/config/nero_pika_teleop_data_params.yaml, or the recorder
        # will wait for topics that are not being published.
        DeclareLaunchArgument('depth', default_value='false'),
    ]

    delay = LaunchConfiguration('camera_start_delay')

    def staggered(slot, action):
        """Start `action` slot*camera_start_delay seconds in."""
        if slot == 0:
            return action
        return TimerAction(period=PythonExpression([str(slot), ' * ', delay]),
                           actions=[action])

    wrist = IfCondition(LaunchConfiguration('wrist'))
    sense = IfCondition(LaunchConfiguration('sense_cameras'))
    fisheye = IfCondition(LaunchConfiguration('fisheye'))

    gripper_fisheye = Node(
        package='sensor_tools', executable='usb_camera.py',
        name='gripper_camera_fisheye', condition=fisheye,
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

    # One device per slot, in increasing order of how badly it behaves.
    #
    # The fisheyes go first even though they are plain V4L2 and share no code
    # with librealsense: they sit on the same Pika hubs, and bringing them up
    # in the same instant as the D435 was enough to make the D435 log
    # "Depth stream start failure" (measured -- with the fisheyes started
    # separately the D435 comes up clean). They settle in well under a second.
    #
    # Then the healthy D435, then the D405s, which are the ones that misbehave.
    return LaunchDescription(declared + [
        staggered(0, gripper_fisheye),
        staggered(0, sense_fisheye),
        staggered(1, _rs('camera', 'scene', 'scene_depth_serial', 'camera_profile',
                         IfCondition(LaunchConfiguration('scene')),
                         color_param='rgb_camera.color_profile')),      # D435
        staggered(2, _rs('camera', 'gripper', 'gripper_depth_serial',
                         'gripper_profile', wrist,
                         color_param='depth_module.color_profile')),    # D405
        staggered(3, _rs('camera', 'sensor', 'sense_depth_serial',
                         'camera_profile', sense,
                         color_param='depth_module.color_profile')),    # D405
    ])
