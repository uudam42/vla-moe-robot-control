"""README "Launch file": starts the full simulated ROS2 pipeline (MuJoCo
bridge node + VLA policy node) with checkpoint/policy-type/device/
instruction as launch arguments -- never hardcoded local paths.

    ros2 launch vla_robot_control mujoco_vla.launch.py \
        checkpoint:=outputs/training/temporal_dense_vla_run_001/best.pt \
        policy_type:=temporal \
        instruction:="Pick up the red cube."
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    checkpoint_arg = DeclareLaunchArgument(
        "checkpoint", description="Path to a trained VLA checkpoint (.pt) -- required, no default."
    )
    policy_type_arg = DeclareLaunchArgument(
        "policy_type", default_value="temporal", description="dense | moe | temporal | dagger"
    )
    device_arg = DeclareLaunchArgument("device", default_value="cpu", description="cpu | cuda | mps")
    instruction_arg = DeclareLaunchArgument("instruction", default_value="Pick up the red cube.")
    control_frequency_arg = DeclareLaunchArgument("control_frequency_hz", default_value="30.0")
    command_timeout_arg = DeclareLaunchArgument("command_timeout_sec", default_value="0.5")
    stale_timeout_arg = DeclareLaunchArgument("stale_timeout_sec", default_value="0.5")

    bridge_node = Node(
        package="vla_robot_control",
        executable="mujoco_bridge_node",
        name="mujoco_bridge_node",
        output="screen",
        parameters=[
            {
                "publish_rate_hz": LaunchConfiguration("control_frequency_hz"),
                "command_timeout_sec": LaunchConfiguration("command_timeout_sec"),
            }
        ],
    )

    policy_node = Node(
        package="vla_robot_control",
        executable="vla_policy_node",
        name="vla_policy_node",
        output="screen",
        parameters=[
            {
                "checkpoint": LaunchConfiguration("checkpoint"),
                "policy_type": LaunchConfiguration("policy_type"),
                "device": LaunchConfiguration("device"),
                "instruction": LaunchConfiguration("instruction"),
                "control_frequency_hz": LaunchConfiguration("control_frequency_hz"),
                "stale_timeout_sec": LaunchConfiguration("stale_timeout_sec"),
            }
        ],
    )

    return LaunchDescription(
        [
            checkpoint_arg, policy_type_arg, device_arg, instruction_arg,
            control_frequency_arg, command_timeout_arg, stale_timeout_arg,
            bridge_node, policy_node,
        ]
    )
