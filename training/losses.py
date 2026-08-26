"""Behavior-cloning loss: MSE on normalized joint targets + BCE on gripper logit.

L = L_joint (MSE, 7D normalized joint targets)
  + GRIPPER_LOSS_WEIGHT * L_gripper (BCEWithLogitsLoss)

Expert gripper behavior is mostly binary open/closed (see
``control/scripted_controller.py``'s GRIPPER_OPEN/GRIPPER_CLOSED constants),
so BCE on a raw logit is a better fit than MSE on the [0,1] value -- the
model's action head outputs an unbounded logit
(``DenseVLA.forward()['gripper_logit']``) and ``sigmoid`` is applied only
at inference (``models/policy.py``), never inside the loss.
"""

import torch
import torch.nn.functional as F

DEFAULT_GRIPPER_LOSS_WEIGHT = 1.0


def compute_loss(prediction: dict, batch: dict, gripper_loss_weight: float = DEFAULT_GRIPPER_LOSS_WEIGHT) -> tuple:
    """Returns ``(total_loss, metrics_dict)``. ``metrics_dict`` values are plain floats."""
    joint_loss = F.mse_loss(prediction["joint_targets_normalized"], batch["joint_targets_normalized"])
    gripper_loss = F.binary_cross_entropy_with_logits(
        prediction["gripper_logit"].squeeze(-1), batch["gripper_target"]
    )
    total_loss = joint_loss + gripper_loss_weight * gripper_loss

    return total_loss, {
        "loss": float(total_loss.item()),
        "joint_loss": float(joint_loss.item()),
        "gripper_loss": float(gripper_loss.item()),
    }
