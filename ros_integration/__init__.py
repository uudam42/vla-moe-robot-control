"""Step 9: ROS2-adjacent logic with NO ``rclpy`` dependency.

README "Dependency isolation": the non-ROS2 training/evaluation code must
keep working on a machine without ROS2 installed. Every module in this
package is plain Python/NumPy -- serialization, command validation, the
command watchdog, and observation synchronization -- fully unit-testable
here. The actual ``rclpy`` ``Node`` subclasses that wire these into real
ROS2 topics/services live under ``ros2_ws/src/vla_robot_control/`` and are
thin wrappers around this package.
"""
