"""MoEVLA forward-shape tests and Dense->MoE conversion functional-equivalence test."""

import torch
import pytest

from models.dense_vla import DenseVLA, DenseVLAConfig
from models.language_encoder import load_tokenizer
from models.moe_vla import MoEVLA, MoEVLAConfig, convert_dense_to_moe, parameter_accounting

BATCH_SIZE = 3


@pytest.fixture(scope="module")
def dense_config():
    return DenseVLAConfig(hidden_dim=32, num_layers=4, num_heads=4, ffn_dim=64)


@pytest.fixture(scope="module")
def dense_model(dense_config):
    m = DenseVLA(dense_config)
    m.eval()
    return m


@pytest.fixture(scope="module")
def moe_config(dense_config):
    return MoEVLAConfig(
        hidden_dim=dense_config.hidden_dim, num_layers=dense_config.num_layers,
        num_heads=dense_config.num_heads, ffn_dim=dense_config.ffn_dim,
        num_experts=4, top_k=1, moe_layers=(1, 3),
    )


@pytest.fixture(scope="module")
def tokenizer(dense_config):
    return load_tokenizer(dense_config.language_backbone)


def make_batch(config, tokenizer, batch_size=BATCH_SIZE):
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    state = torch.randn(batch_size, config.state_dim)
    tokenized = tokenizer(
        ["Pick up the red cube."] * batch_size, return_tensors="pt", padding="max_length",
        truncation=True, max_length=config.max_instruction_length,
    )
    return pixel_values, state, tokenized["input_ids"], tokenized["attention_mask"]


def test_moe_output_shapes_match_dense(dense_config, moe_config, tokenizer):
    model = MoEVLA(moe_config)
    model.eval()
    pixel_values, state, input_ids, attention_mask = make_batch(dense_config, tokenizer)

    with torch.no_grad():
        output = model(pixel_values, input_ids, attention_mask, state)

    assert output["joint_targets_normalized"].shape == (BATCH_SIZE, 7)
    assert output["gripper_logit"].shape == (BATCH_SIZE, 1)
    assert torch.all(torch.isfinite(output["joint_targets_normalized"]))
    assert torch.all(torch.isfinite(output["gripper_logit"]))
    assert torch.isfinite(output["router_aux_loss"])


def test_collect_diagnostics_populates_layer_diagnostics(dense_config, moe_config, tokenizer):
    model = MoEVLA(moe_config)
    model.eval()
    pixel_values, state, input_ids, attention_mask = make_batch(dense_config, tokenizer)

    with torch.no_grad():
        output = model(pixel_values, input_ids, attention_mask, state, collect_diagnostics=True)

    assert set(output["layer_diagnostics"].keys()) == {1, 3}
    for diagnostics in output["layer_diagnostics"].values():
        assert diagnostics["probs"].shape == (BATCH_SIZE, 4, 4)  # (B, T=4 tokens, E=4 experts)


def test_dense_to_moe_conversion_is_functionally_equivalent(dense_model, moe_config, tokenizer, dense_config):
    moe_model = convert_dense_to_moe(dense_model, moe_config)
    moe_model.eval()

    pixel_values, state, input_ids, attention_mask = make_batch(dense_config, tokenizer)

    with torch.no_grad():
        dense_out = dense_model(pixel_values, input_ids, attention_mask, state)
        moe_out = moe_model(pixel_values, input_ids, attention_mask, state)

    joint_diff = (dense_out["joint_targets_normalized"] - moe_out["joint_targets_normalized"]).abs().max().item()
    gripper_diff = (dense_out["gripper_logit"] - moe_out["gripper_logit"]).abs().max().item()

    # Every expert starts as an exact copy of the Dense FFN and top-1 output
    # is unweighted, so the converted model's output should match Dense to
    # float32 precision regardless of which (identical) expert gets picked.
    assert joint_diff < 1e-4
    assert gripper_diff < 1e-4


def test_converted_experts_are_exact_copies_of_dense_ffn(dense_model, moe_config):
    moe_model = convert_dense_to_moe(dense_model, moe_config)

    for layer_index in moe_config.moe_layers:
        dense_layer = dense_model.transformer.layers[layer_index]
        moe_block = moe_model.transformer.blocks[layer_index]
        for expert in moe_block.ffn.experts:
            assert torch.equal(expert.linear1.weight, dense_layer.linear1.weight)
            assert torch.equal(expert.linear2.weight, dense_layer.linear2.weight)


def test_conversion_rejects_mismatched_architecture(dense_model):
    bad_config = MoEVLAConfig(hidden_dim=999, num_layers=dense_model.config.num_layers)
    with pytest.raises(ValueError):
        convert_dense_to_moe(dense_model, bad_config)


def test_parameter_accounting_active_params_close_to_dense(dense_model, moe_config):
    from models.dense_vla import count_parameters

    moe_model = convert_dense_to_moe(dense_model, moe_config)
    dense_total, _ = count_parameters(dense_model)
    accounting = parameter_accounting(moe_model)

    assert accounting["total_parameters"] > dense_total  # extra expert copies exist
    # Active params per token should be very close to Dense (same-size FFN
    # actually used per token, plus tiny router overhead).
    assert accounting["active_parameters_per_token"] >= dense_total
    assert accounting["active_parameters_per_token"] < dense_total + 10_000
    assert accounting["num_moe_layers"] == 2
