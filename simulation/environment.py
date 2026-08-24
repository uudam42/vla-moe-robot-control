"""MuJoCo simulation wrapper producing simulator-agnostic Observations.

This module is the only place in the codebase allowed to know about
MuJoCo's MjModel/MjData/Renderer types, joint/actuator/site IDs, and
control units. Everything it hands back to callers -- Observation,
RobotState, RobotAction handling, Jacobians as plain arrays -- is
simulator-agnostic. In particular, callers (the scripted controller, a
future VLA) never touch ``data.ctrl`` or MuJoCo IDs directly; they go
through ``get_robot_state()`` / ``get_jacobian()`` / ``step(action)``.
"""

from pathlib import Path

import mujoco
import numpy as np

from control.action import RobotAction
from observations.observation import Observation
from observations.robot_state import RobotState

DEFAULT_SCENE_PATH = Path(__file__).parent / "scene.xml"
DEFAULT_CAMERA_NAME = "main_camera"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_CONTROL_SUBSTEPS = 10  # physics steps advanced per env.step() call
DEFAULT_KEYFRAME_NAME = "home"

# Names centralized here rather than scattered as string literals elsewhere.
# The scripted controller never needs to know these -- it only sees
# RobotState / RobotAction / plain numpy arrays.
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATOR_NAMES = [f"actuator{i}" for i in range(1, 8)]
GRIPPER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]
GRIPPER_ACTUATOR_NAME = "actuator8"
END_EFFECTOR_SITE_NAME = "end_effector"
CUBE_BODY_NAME = "red_cube"

# actuator8's ctrlrange in assets/franka_panda/panda.xml: 0 -> fingers
# closed, 255 -> fingers fully open. gripper_target is normalized [0, 1]
# (0=closed, 1=open); this is the only place that mapping is known.
GRIPPER_CTRL_MIN = 0.0
GRIPPER_CTRL_MAX = 255.0


class SimulationEnvironment:
    """Owns a MuJoCo model/data/renderer triple and exposes Observations.

    Args:
        scene_path: Path to the MuJoCo scene XML file.
        camera_name: Name of the camera defined in the scene to render from.
        width: Render width in pixels.
        height: Render height in pixels.
        control_substeps: Number of ``mj_step`` calls per ``step(action)``.
    """

    def __init__(
        self,
        scene_path: Path = DEFAULT_SCENE_PATH,
        camera_name: str = DEFAULT_CAMERA_NAME,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        control_substeps: int = DEFAULT_CONTROL_SUBSTEPS,
    ) -> None:
        scene_path = Path(scene_path)
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene XML not found: {scene_path}")

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.control_substeps = control_substeps

        self.camera_name = camera_name
        _require_id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        try:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        except Exception as exc:  # pragma: no cover - depends on local GL setup
            raise RuntimeError(
                "Failed to initialize MuJoCo renderer. If running headless, "
                "set MUJOCO_GL=egl (Linux) or MUJOCO_GL=glfw/cgl (macOS) "
                "before starting Python."
            ) from exc

        self._arm_joint_ids = [
            _require_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ARM_JOINT_NAMES
        ]
        self._gripper_joint_ids = [
            _require_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in GRIPPER_JOINT_NAMES
        ]
        self._arm_actuator_ids = [
            _require_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ARM_ACTUATOR_NAMES
        ]
        self._gripper_actuator_id = _require_id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME
        )
        self._ee_site_id = _require_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE_NAME
        )
        self._cube_body_id = _require_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY_NAME
        )
        self._arm_dof_adr = [self.model.jnt_dofadr[j] for j in self._arm_joint_ids]
        self._arm_qpos_adr = [self.model.jnt_qposadr[j] for j in self._arm_joint_ids]
        self._gripper_qpos_adr = [
            self.model.jnt_qposadr[j] for j in self._gripper_joint_ids
        ]
        self._arm_joint_range = self.model.jnt_range[self._arm_joint_ids].copy()

        self._load_initial_state()
        mujoco.mj_forward(self.model, self.data)

        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()
        self._initial_ctrl = self.data.ctrl.copy()

    def _load_initial_state(self) -> None:
        """Set qpos/qvel/ctrl to the robot's home keyframe + resting cube."""
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, DEFAULT_KEYFRAME_NAME
        )
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # The "home" keyframe only defines the robot's 9 DoF; the cube's
        # free joint isn't part of it, so pull the cube's rest pose from
        # qpos0 (the scene's compiled reference configuration, i.e. the
        # <body pos=.../> given in scene.xml) instead of leaving it zeroed.
        cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint"
        )
        cube_qpos_adr = self.model.jnt_qposadr[cube_joint_id]
        self.data.qpos[cube_qpos_adr : cube_qpos_adr + 7] = self.model.qpos0[
            cube_qpos_adr : cube_qpos_adr + 7
        ]

    def get_robot_state(self) -> RobotState:
        """Extract structured proprioception using named joint/site IDs.

        Reads only the robot's own DoFs -- the cube's free joint is never
        included here, even though it shares the same qpos/qvel arrays.
        """
        mujoco.mj_forward(self.model, self.data)

        joint_positions = np.array(
            [self.data.qpos[adr] for adr in self._arm_qpos_adr]
        )
        joint_velocities = np.array(
            [self.data.qvel[adr] for adr in self._arm_dof_adr]
        )
        end_effector_position = self.data.site_xpos[self._ee_site_id].copy()
        end_effector_orientation = np.empty(4)
        mujoco.mju_mat2Quat(
            end_effector_orientation, self.data.site_xmat[self._ee_site_id]
        )
        gripper_position = np.array(
            [self.data.qpos[adr] for adr in self._gripper_qpos_adr]
        )

        return RobotState(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            end_effector_position=end_effector_position,
            end_effector_orientation=end_effector_orientation,
            gripper_position=gripper_position,
        )

    def get_observation(self) -> Observation:
        """Render the camera and collect robot state into an Observation.

        ``Observation.state`` is the robot's proprioception only -- the red
        cube's ground-truth position is deliberately excluded. A future VLA
        must infer object state from ``rgb``, not read it off ``state``.
        Use :meth:`get_object_position` for privileged access instead.
        """
        mujoco.mj_forward(self.model, self.data)

        self.renderer.update_scene(self.data, camera=self.camera_name)
        rgb = self.renderer.render()

        if rgb is None or rgb.size == 0:
            raise RuntimeError("Renderer produced an empty frame")
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

        state = self.get_robot_state().as_vector()
        timestamp = float(self.data.time)

        return Observation(rgb=rgb, state=state, timestamp=timestamp)

    def get_jacobian(self) -> np.ndarray:
        """Translational end-effector Jacobian w.r.t. the 7 arm joints.

        Returns:
            Array of shape ``(3, 7)``, columns ordered to match
            ``RobotState.joint_positions`` / ``RobotAction.joint_targets``.
        """
        jacobian_position, _ = self._site_jacobian()
        return jacobian_position

    def get_end_effector_jacobian(self) -> tuple:
        """Full (translational, rotational) end-effector Jacobians.

        Returns:
            ``(J_pos, J_rot)``, each shape ``(3, 7)``, columns ordered to
            match ``RobotState.joint_positions`` / ``RobotAction.joint_targets``.
            Only the 7 arm-joint DoF columns are included -- the cube's free
            joint never contaminates these.
        """
        return self._site_jacobian()

    def _site_jacobian(self) -> tuple:
        mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site_id)
        jacobian_position = jacp[:, self._arm_dof_adr]
        jacobian_rotation = jacr[:, self._arm_dof_adr]

        if jacobian_position.shape != (3, 7) or jacobian_rotation.shape != (3, 7):
            raise RuntimeError(
                "Unexpected Jacobian shape: "
                f"pos={jacobian_position.shape}, rot={jacobian_rotation.shape}"
            )
        if not (np.all(np.isfinite(jacobian_position)) and np.all(np.isfinite(jacobian_rotation))):
            raise RuntimeError("Jacobian contains non-finite values")

        return jacobian_position, jacobian_rotation

    def get_object_position(self, body_name: str) -> np.ndarray:
        """Ground-truth world position of a named body, e.g. ``"red_cube"``.

        Privileged simulator state: only the scripted expert controller (and
        tests) may call this. A learned VLA must infer object position from
        ``Observation.rgb`` instead -- this is intentionally kept separate
        from ``Observation.state``.
        """
        body_id = _require_id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[body_id].copy()

    def set_object_position(self, body_name: str, position: np.ndarray) -> None:
        """Overwrite a free-jointed body's position (privileged; eval/test use only).

        Used by the reliability evaluation to apply small randomized cube
        offsets between episodes (call after :meth:`reset`). This is not
        part of the RobotAction control interface -- neither the scripted
        controller nor a future VLA calls this during an episode.
        """
        body_id = _require_id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_id = self.model.body_jntadr[body_id]
        if joint_id == -1 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Body '{body_name}' does not have a free joint")
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {position.shape}")

        qpos_adr = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qpos_adr : qpos_adr + 3] = position
        mujoco.mj_forward(self.model, self.data)

    def step(self, action: RobotAction) -> None:
        """Apply a RobotAction and advance physics.

        This is the only place ``data.ctrl`` is written. Joint targets are
        clamped to each joint's physical range and the gripper's normalized
        target is mapped to the actuator's control units here, so callers
        never need to know MuJoCo actuator limits.
        """
        if not isinstance(action, RobotAction):
            raise TypeError(f"step() requires a RobotAction, got {type(action)}")

        clamped_targets = np.clip(
            action.joint_targets,
            self._arm_joint_range[:, 0],
            self._arm_joint_range[:, 1],
        )
        for actuator_id, target in zip(self._arm_actuator_ids, clamped_targets):
            self.data.ctrl[actuator_id] = target

        gripper_ctrl = GRIPPER_CTRL_MIN + action.gripper_target * (
            GRIPPER_CTRL_MAX - GRIPPER_CTRL_MIN
        )
        self.data.ctrl[self._gripper_actuator_id] = gripper_ctrl

        for _ in range(self.control_substeps):
            mujoco.mj_step(self.model, self.data)

    def reset(self) -> None:
        """Restore the robot, gripper, and cube to their initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = self._initial_qvel
        self.data.ctrl[:] = self._initial_ctrl
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        """Release renderer resources."""
        self.renderer.close()

    def __enter__(self) -> "SimulationEnvironment":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _require_id(model: "mujoco.MjModel", obj_type, name: str) -> int:
    """Look up a named MuJoCo object's ID, raising a clear error if missing."""
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id == -1:
        raise ValueError(f"{obj_type} '{name}' not found in scene")
    return obj_id
