"""Shape and input-contract tests for models/dense_vla.py."""

import inspect

import pytest
import torch

from models.dense_vla import DenseVLA, DenseVLAConfig, count_parameters
from models.language_encoder import load_tokenizer

BATCH_SIZE = 3


@pytest.fixture(scope="module")
def config():
    return DenseVLAConfig()


@pytest.fixture(scope="module")
def model(config):
    m = DenseVLA(config)
    m.eval()
    return m


@pytest.fixture(scope="module")
def tokenizer(config):
    return load_tokenizer(config.language_backbone)


def make_batch(config, tokenizer, batch_size=BATCH_SIZE):
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    state = torch.randn(batch_size, config.state_dim)
    tokenized = tokenizer(
        ["Pick up the red cube."] * batch_size,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.max_instruction_length,
    )
    return pixel_values, state, tokenized["input_ids"], tokenized["attention_mask"]


def test_forward_output_shapes(config, model, tokenizer):
    pixel_values, state, input_ids, attention_mask = make_batch(config, tokenizer)
    with torch.no_grad():
        output = model(pixel_values, input_ids, attention_mask, state)

    assert output["joint_targets_normalized"].shape == (BATCH_SIZE, 7)
    assert output["gripper_logit"].shape == (BATCH_SIZE, 1)
    assert torch.all(torch.isfinite(output["joint_targets_normalized"]))
    assert torch.all(torch.isfinite(output["gripper_logit"]))


def test_batch_size_one_works(config, model, tokenizer):
    pixel_values, state, input_ids, attention_mask = make_batch(config, tokenizer, batch_size=1)
    with torch.no_grad():
        output = model(pixel_values, input_ids, attention_mask, state)
    assert output["joint_targets_normalized"].shape == (1, 7)
    assert output["gripper_logit"].shape == (1, 1)


def test_parameter_counts_are_reported_and_sane(model):
    total, trainable = count_parameters(model)
    assert total > 0
    assert trainable > 0
    assert trainable <= total


def test_frozen_encoders_have_no_trainable_parameters_by_default(config):
    assert config.train_vision_encoder is False
    assert config.train_language_encoder is False
    m = DenseVLA(config)
    assert all(not p.requires_grad for p in m.vision_encoder.parameters())
    assert all(not p.requires_grad for p in m.language_encoder.parameters())
    # Everything else (state encoder, fusion, transformer, action head) must be trainable.
    assert any(p.requires_grad for p in m.state_encoder.parameters())
    assert any(p.requires_grad for p in m.fusion.parameters())
    assert any(p.requires_grad for p in m.transformer.parameters())
    assert any(p.requires_grad for p in m.action_head.parameters())


def test_model_input_contract_excludes_privileged_information(model):
    """Section 47: the model must not be able to receive cube XYZ, Jacobian,
    controller stage, etc. -- verified by inspecting forward()'s signature.
    """
    signature = inspect.signature(model.forward)
    allowed = {"pixel_values", "input_ids", "attention_mask", "state"}
    assert set(signature.parameters) == allowed


def test_config_round_trips_through_dict():
    config = DenseVLAConfig(hidden_dim=128, num_layers=2)
    restored = DenseVLAConfig.from_dict(config.to_dict())
    assert restored == config
