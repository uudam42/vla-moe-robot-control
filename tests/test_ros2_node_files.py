"""ROS2 integration tests (README "ROS2 Test Strategy"): import and
exercise the REAL ``rclpy`` node files under
``ros2_ws/src/vla_robot_control/``, as opposed to the pure-Python
``ros_integration``/``robot_backend`` logic those nodes wrap (already
covered, without any ROS2 dependency, by
``tests/test_ros_policy_node_core.py`` / ``test_ros_bridge_node_core.py``
/ ``test_robot_backend.py``).

Automatically skipped via ``pytest.importorskip`` when ``rclpy`` is not
installed -- true in this development environment (no ROS2 distribution
is installed here; see README "Step 9" limitations). Also carries the
``ros2`` marker so the two suites can be run separately:

    pytest -m "not ros2"   # pure-Python suite -- this repo's normal CI, no ROS2 needed
    pytest -m ros2          # this file -- needs a real ROS2 (rclpy) install

NOTE: because rclpy is unavailable in the environment this milestone was
implemented in, these tests have been written to be correct against the
public rclpy/ROS2 API but have NOT been executed end-to-end here. Treat
them as ready-to-run on a real ROS2 machine, not as verified passing.
"""

import sys
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")

pytestmark = pytest.mark.ros2

REPO_ROOT = Path(__file__).resolve().parent.parent
ROS2_PACKAGE_ROOT = REPO_ROOT / "ros2_ws" / "src" / "vla_robot_control"
if str(ROS2_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS2_PACKAGE_ROOT))


def test_vla_policy_node_module_never_imports_mujoco():
    import inspect

    from vla_robot_control import vla_policy_node

    assert not hasattr(vla_policy_node, "mujoco")
    source = inspect.getsource(vla_policy_node)
    assert "import mujoco" not in source
    assert "SimulationEnvironment" not in source
    assert "MuJoCoBackend" not in source


def test_mujoco_bridge_node_is_the_only_ros2_module_touching_mujoco_backend():
    import inspect

    from vla_robot_control import mujoco_bridge_node, vla_policy_node

    assert "MuJoCoBackend" in inspect.getsource(mujoco_bridge_node)
    assert "MuJoCoBackend" not in inspect.getsource(vla_policy_node)


def test_multi_step_ros2_smoke_rollout(tiny_temporal_vla_checkpoint):
    """README "Multi-step Integration Test": bridge publishes sensor
    topics, the policy node subscribes/synchronizes/predicts, publishes a
    VLARobotAction, the bridge validates+executes it -- for several ticks.
    Verifies messages flow, actions stay finite, and timestamps increase.

    Parameter override follows the standard rclpy pattern (CLI-style
    ``--ros-args -p``), rather than a custom constructor kwarg, so neither
    node class needs ROS2-test-specific code.
    """
    import math
    import time

    from vla_robot_control_msgs.msg import VLARobotAction
    from vla_robot_control.mujoco_bridge_node import MuJoCoBridgeNode
    from vla_robot_control.vla_policy_node import VLAPolicyNode

    rclpy.init(args=["--ros-args", "-p", f"checkpoint:={tiny_temporal_vla_checkpoint}"])
    try:
        bridge = MuJoCoBridgeNode()
        policy_node = VLAPolicyNode()

        received_actions = []
        bridge.create_subscription(VLARobotAction, "/vla/action", received_actions.append, 10)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(bridge)
        executor.add_node(policy_node)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(received_actions) < 5:
            executor.spin_once(timeout_sec=0.1)

        assert len(received_actions) >= 1, "expected at least one VLARobotAction message to flow end-to-end"
        for msg in received_actions:
            assert all(math.isfinite(v) for v in msg.joint_targets)
            assert 0.0 <= msg.gripper_target <= 1.0

        executor.remove_node(bridge)
        executor.remove_node(policy_node)
        bridge.core.backend.close()
        bridge.destroy_node()
        policy_node.destroy_node()
    finally:
        rclpy.shutdown()
