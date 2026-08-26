"""``policy_type`` -> policy class lookup, shared by the direct-backend
runner and the ROS2 VLA policy node so neither duplicates policy internals
(README "Do not duplicate policy internals inside the ROS2 node").

``"dagger"`` intentionally maps to the same ``TemporalDenseVLAPolicy``
class as ``"temporal"`` -- Step 8's DAgger-fine-tuned checkpoint has the
identical architecture/config shape as the Step 7 Temporal checkpoint (see
README "DAgger policy class"); only the checkpoint weights differ.
"""

import torch

from models.moe_policy import MoEVLAPolicy
from models.policy import DenseVLAPolicy
from models.temporal_policy import TemporalDenseVLAPolicy

POLICY_CLASSES = {
    "dense": DenseVLAPolicy,
    "moe": MoEVLAPolicy,
    "temporal": TemporalDenseVLAPolicy,
    "dagger": TemporalDenseVLAPolicy,
}


def build_policy(policy_type: str, checkpoint_path: str, device: torch.device = None):
    """Load a policy of the given type from a checkpoint.

    Args:
        policy_type: One of ``POLICY_CLASSES`` (``"dense"``, ``"moe"``,
            ``"temporal"``, ``"dagger"``).
        checkpoint_path: Path to a ``.pt`` checkpoint saved by the matching
            training script.
        device: Inference device; defaults to CPU.
    """
    if policy_type not in POLICY_CLASSES:
        raise ValueError(f"Unknown policy_type {policy_type!r}; expected one of {sorted(POLICY_CLASSES)}")
    policy_class = POLICY_CLASSES[policy_type]
    return policy_class.from_checkpoint(checkpoint_path, device=device)
