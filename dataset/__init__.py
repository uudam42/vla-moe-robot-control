"""Demonstration dataset generation, recording, loading, and validation.

Step 3: converts scripted-expert rollouts (Observation + expert RobotAction
at every control timestep) into an on-disk dataset for future behavior
cloning. Nothing in this package trains a model or touches PyTorch --
that's Step 4.
"""

DATASET_VERSION = "1.0"

# Semantically identical instruction variants (see dataset/generate_dataset.py)
# -- language diversity to discourage the future model from memorizing one
# exact string, without changing task semantics.
INSTRUCTION_VARIANTS = (
    "Pick up the red cube.",
    "Grasp the red cube.",
    "Lift the red cube.",
    "Pick up the red block.",
)
