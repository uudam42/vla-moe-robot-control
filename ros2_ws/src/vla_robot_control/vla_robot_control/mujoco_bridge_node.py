"""ROS2 node: MuJoCo bridge (README "MuJoCo Bridge Node").

The ONLY ROS2-layer node allowed to know MuJoCo/``SimulationEnvironment``
exists -- via ``robot_backend.mujoco_backend.MuJoCoBackend``, never
directly. Publishes camera RGB + robot state on a timer; subscribes to
``VLARobotAction`` commands, validates and executes them through
``ros_integration.bridge_node_core.MuJoCoBridgeNodeCore`` (command
validation + watchdog logic, fully unit-tested without rclpy -- see
``tests/test_ros_bridge_node_core.py``), and serves ``/reset_episode``.

Requires this repository's root on ``PYTHONPATH`` (README "Packaging") so
``robot_backend``/``ros_integration``/``observations``/``control``/
``simulation`` are importable. This file itself contains no ML/physics
logic -- only ROS2 transport wiring around ``MuJoCoBridgeNodeCore``.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from vla_robot_control_msgs.msg import VLARobotAction
from vla_robot_control_msgs.srv import ResetEpisode

from robot_backend.mujoco_backend import MuJoCoBackend
from ros_integration.bridge_node_core import MuJoCoBridgeNodeCore
from ros_integration.command_validator import CommandValidator
from ros_integration.serialization import rgb_to_image_fields, state_to_joint_state_fields
from ros_integration.watchdog import CommandWatchdog

# QoS (README "ROS2 QoS" -- explicit, never left at implicit defaults).
# Sensors: best-effort + depth 1 -- an occasional dropped frame should
# never block delivery of the newest one; the policy node's staleness
# check already handles gaps (README "Stale data handling").
SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
# Commands: reliable + small queue -- a dropped action command is worse
# than a dropped camera frame; the watchdog already bounds staleness if
# the publisher stalls (README "Commands: reliable, small queue").
COMMAND_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=5)


class MuJoCoBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("mujoco_bridge_node")

        self.declare_parameter("camera_topic", "/vla/camera/image")
        self.declare_parameter("state_topic", "/vla/robot/state")
        self.declare_parameter("action_topic", "/vla/action")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("command_timeout_sec", 0.5)
        self.declare_parameter("max_joint_delta", 0.5)

        camera_topic = self.get_parameter("camera_topic").value
        state_topic = self.get_parameter("state_topic").value
        action_topic = self.get_parameter("action_topic").value
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)
        max_joint_delta = float(self.get_parameter("max_joint_delta").value)

        # MuJoCo control_substeps stays at SimulationEnvironment's own
        # default (10) -- the ROS2 layer never touches physics stepping
        # cadence (README "Low-level physics frequency").
        backend = MuJoCoBackend()
        validator = CommandValidator(max_joint_delta=max_joint_delta)
        watchdog = CommandWatchdog(timeout_sec=command_timeout_sec)
        self.core = MuJoCoBridgeNodeCore(backend, validator=validator, watchdog=watchdog)

        self.image_pub = self.create_publisher(Image, camera_topic, SENSOR_QOS)
        self.state_pub = self.create_publisher(JointState, state_topic, SENSOR_QOS)
        self.action_sub = self.create_subscription(VLARobotAction, action_topic, self._on_action, COMMAND_QOS)
        self.reset_srv = self.create_service(ResetEpisode, "reset_episode", self._on_reset)

        # Separate timers: sensor publishing and command execution are
        # decoupled from each other and from any subscriber callback
        # timing (README "Control loop": timer-driven, not callback-driven).
        self.create_timer(1.0 / publish_rate_hz, self._publish_observation)
        self.create_timer(1.0 / publish_rate_hz, self._execute_tick)

        self.get_logger().info(
            f"mujoco_bridge_node ready: camera={camera_topic} state={state_topic} action={action_topic} "
            f"rate={publish_rate_hz}Hz command_timeout={command_timeout_sec}s max_joint_delta={max_joint_delta}"
        )

    def _publish_observation(self) -> None:
        observation = self.core.backend.get_observation()
        now = self.get_clock().now().to_msg()

        image_fields = rgb_to_image_fields(observation.rgb)
        image_msg = Image()
        image_msg.header.stamp = now
        image_msg.height = image_fields["height"]
        image_msg.width = image_fields["width"]
        image_msg.encoding = image_fields["encoding"]
        image_msg.is_bigendian = image_fields["is_bigendian"]
        image_msg.step = image_fields["step"]
        image_msg.data = image_fields["data"]
        self.image_pub.publish(image_msg)

        state_fields = state_to_joint_state_fields(observation.state, observation.timestamp)
        state_msg = JointState()
        state_msg.header.stamp = now
        state_msg.name = state_fields["name"]
        state_msg.position = state_fields["position"]
        self.state_pub.publish(state_msg)

    def _on_action(self, msg: VLARobotAction) -> None:
        now = time.monotonic()
        result = self.core.on_action_received(list(msg.joint_targets), msg.gripper_target, now)
        if not result.valid:
            self.get_logger().warn(f"action rejected: {result.reason}")

    def _execute_tick(self) -> None:
        _, timed_out = self.core.tick(time.monotonic())
        if timed_out:
            self.get_logger().warn("command watchdog timeout -- holding last safe action")

    def _on_reset(self, request, response):
        self.core.reset()
        response.success = True
        response.message = "backend and watchdog reset"
        self.get_logger().info("episode reset via /reset_episode")
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MuJoCoBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.core.backend.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
