"""ROS2 node: VLA policy inference (README "VLA Policy Node").

Never imports MuJoCo -- verified for the shared control-loop logic by
``ros_integration/policy_node_core.py``'s own no-MuJoCo test
(``tests/test_ros_policy_node_core.py``); this file only adds rclpy
transport around it. Subscribes to camera/state/instruction topics,
synchronizes them (``ros_integration.sync``), and publishes a
``VLARobotAction`` on a TIMER at the configured control frequency (README
"Control loop": timer-driven, not subscriber-callback-driven).

Requires this repository's root on ``PYTHONPATH`` (README "Packaging").
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from vla_robot_control_msgs.msg import VLARobotAction
from vla_robot_control_msgs.srv import ResetEpisode

from robot_backend.policy_factory import build_policy
from ros_integration.policy_node_core import VLAPolicyNodeCore
from ros_integration.serialization import image_fields_to_rgb, joint_state_fields_to_state
from ros_integration.sync import LatestMessageSynchronizer, StalenessChecker
from training.config import resolve_device

SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
COMMAND_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
# Instruction: reliable + transient-local -- a node that (re)starts after
# the last instruction change should still see it, not wait for the next
# one (README "Instruction: reliable, transient-local optional" -- taken).
INSTRUCTION_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST, depth=1,
)


class VLAPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("vla_policy_node")

        self.declare_parameter("checkpoint", "")
        self.declare_parameter("policy_type", "temporal")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("instruction", "Pick up the red cube.")
        self.declare_parameter("control_frequency_hz", 30.0)
        self.declare_parameter("camera_topic", "/vla/camera/image")
        self.declare_parameter("state_topic", "/vla/robot/state")
        self.declare_parameter("instruction_topic", "/task_instruction")
        self.declare_parameter("action_topic", "/vla/action")
        self.declare_parameter("sync_max_delta_sec", 0.1)
        self.declare_parameter("stale_timeout_sec", 0.5)

        checkpoint = self.get_parameter("checkpoint").value
        if not checkpoint:
            # README "Do not hardcode local absolute paths" -- the
            # checkpoint MUST be supplied via launch argument/parameter.
            raise ValueError("The 'checkpoint' parameter is required (pass via launch argument).")
        policy_type = self.get_parameter("policy_type").value
        device_name = self.get_parameter("device").value
        control_frequency_hz = float(self.get_parameter("control_frequency_hz").value)

        device = resolve_device(device_name)
        policy = build_policy(policy_type, checkpoint, device=device)
        self.get_logger().info(f"checkpoint loaded: {checkpoint} (policy_type={policy_type}, device={device})")

        synchronizer = LatestMessageSynchronizer(
            max_sync_delta_sec=float(self.get_parameter("sync_max_delta_sec").value)
        )
        staleness = StalenessChecker(max_age_sec=float(self.get_parameter("stale_timeout_sec").value))
        self.core = VLAPolicyNodeCore(policy, synchronizer=synchronizer, staleness_checker=staleness)
        self.core.instructions.update(self.get_parameter("instruction").value)

        camera_topic = self.get_parameter("camera_topic").value
        state_topic = self.get_parameter("state_topic").value
        instruction_topic = self.get_parameter("instruction_topic").value
        action_topic = self.get_parameter("action_topic").value

        self.create_subscription(Image, camera_topic, self._on_image, SENSOR_QOS)
        self.create_subscription(JointState, state_topic, self._on_state, SENSOR_QOS)
        self.create_subscription(String, instruction_topic, self._on_instruction, INSTRUCTION_QOS)
        self.action_pub = self.create_publisher(VLARobotAction, action_topic, COMMAND_QOS)
        self.reset_client = self.create_client(ResetEpisode, "reset_episode")

        self.create_timer(1.0 / control_frequency_hz, self._on_control_tick)
        self.get_logger().info(
            f"vla_policy_node ready: policy={policy_type} rate={control_frequency_hz}Hz "
            f"instruction={self.core.instructions.get()!r}"
        )

    def _on_image(self, msg: Image) -> None:
        fields = {
            "height": msg.height, "width": msg.width, "encoding": msg.encoding,
            "is_bigendian": msg.is_bigendian, "step": msg.step, "data": bytes(msg.data),
        }
        rgb = image_fields_to_rgb(fields)
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.core.on_image(rgb, timestamp)

    def _on_state(self, msg: JointState) -> None:
        fields = {"name": list(msg.name), "position": list(msg.position)}
        state = joint_state_fields_to_state(fields)
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.core.on_state(state, timestamp)

    def _on_instruction(self, msg: String) -> None:
        if self.core.on_instruction(msg.data):
            self.get_logger().info(f"instruction updated: {msg.data!r}")

    def _now(self) -> float:
        """Seconds, same clock/epoch as message header stamps -- see
        mujoco_bridge_node.py::MuJoCoBridgeNode._now() for why this must
        NOT be time.monotonic()."""
        return self.get_clock().now().nanoseconds / 1e9

    def _on_control_tick(self) -> None:
        action = self.core.tick(self._now())
        if action is None:
            self.get_logger().warn(
                "observation stale or not yet synchronized -- skipping policy tick", throttle_duration_sec=2.0
            )
            return

        msg = VLARobotAction()
        msg.stamp = self.get_clock().now().to_msg()
        msg.joint_targets = action.joint_targets.tolist()
        msg.gripper_target = float(action.gripper_target)
        self.action_pub.publish(msg)

    def reset_episode(self) -> None:
        """Local (non-service) reset -- clears this node's own
        synchronizer + policy history. The bridge's ``/reset_episode``
        service call (backend + watchdog) is separate; a launch-level
        orchestrator or operator calls both (README "Document ordering")."""
        self.core.reset()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VLAPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
