"""TemporalDenseVLA forward-shape tests and Dense->Temporal conversion checks."""

import pytest
import torch

from models.dense_vla import DenseVLA, DenseVLAConfig, count_parameters
from models.language_encoder import load_tokenizer
from models.temporal_vla import TemporalDenseVLA, TemporalDenseVLAConfig, convert_dense_to_temporal

BATCH_SIZE = 3
HISTORY_LENGTH = 4


@pytest.fixture(scope="module")
def dense_config():
    return DenseVLAConfig(hidden_dim=32, num_layers=4, num_heads=4, ffn_dim=64)


@pytest.fixture(scope="module")
def dense_model(dense_config):
    m = DenseVLA(dense_config)
    m.eval()
    return m


@pytest.fixture(scope="module")
def temporal_config(dense_config):
    return TemporalDenseVLAConfig(
        hidden_dim=dense_config.hidden_dim, num_layers=dense_config.num_layers,
        num_heads=dense_config.num_heads, ffn_dim=dense_config.ffn_dim, history_length=HISTORY_LENGTH,
    )


@pytest.fixture(scope="module")
def tokenizer(dense_config):
    return load_tokenizer(dense_config.language_backbone)


def make_batch(config, tokenizer, history_length=HISTORY_LENGTH, batch_size=BATCH_SIZE):
    pixel_values = torch.randn(batch_size, history_length, 3, 224, 224)
    states = torch.randn(batch_size, history_length, config.state_dim)
    previous_actions = torch.randn(batch_size, history_length, 8)
    tokenized = tokenizer(
        ["Pick up the red cube."] * batch_size, return_tensors="pt", padding="max_length",
        truncation=True, max_length=config.max_instruction_length,
    )
    return pixel_values, states, previous_actions, tokenized["input_ids"], tokenized["attention_mask"]


def test_forward_output_shapes(temporal_config, tokenizer):
    model = TemporalDenseVLA(temporal_config)
    model.eval()
    pixel_values, states, previous_actions, input_ids, attention_mask = make_batch(temporal_config, tokenizer)

    with torch.no_grad():
        output = model(pixel_values, states, previous_actions, input_ids, attention_mask)

    assert output["joint_targets_normalized"].shape == (BATCH_SIZE, 7)
    assert output["gripper_logit"].shape == (BATCH_SIZE, 1)
    assert torch.all(torch.isfinite(output["joint_targets_normalized"]))
    assert torch.all(torch.isfinite(output["gripper_logit"]))


def test_different_history_lengths_produce_correct_shapes(dense_config, tokenizer):
    for history_length in (1, 2, 4, 6):
        config = TemporalDenseVLAConfig(
            hidden_dim=dense_config.hidden_dim, num_layers=dense_config.num_layers,
            num_heads=dense_config.num_heads, ffn_dim=dense_config.ffn_dim, history_length=history_length,
        )
        model = TemporalDenseVLA(config)
        model.eval()
        batch = make_batch(config, tokenizer, history_length=history_length, batch_size=2)
        with torch.no_grad():
            output = model(*batch)
        assert output["joint_targets_normalized"].shape == (2, 7)


def test_temporal_position_embedding_has_history_length_slots(temporal_config):
    model = TemporalDenseVLA(temporal_config)
    assert model.temporal_position_embedding.shape == (1, HISTORY_LENGTH, temporal_config.hidden_dim)


def test_action_history_encoder_maps_8d_to_hidden_dim(temporal_config):
    model = TemporalDenseVLA(temporal_config)
    x = torch.randn(5, 8)
    out = model.action_history_encoder(x)
    assert out.shape == (5, temporal_config.hidden_dim)


def test_parameter_count_is_close_to_dense(dense_model, temporal_config):
    temporal_model = convert_dense_to_temporal(dense_model, temporal_config)
    dense_total, _ = count_parameters(dense_model)
    temporal_total, _ = count_parameters(temporal_model)
    # New components (action-history encoder, action_proj, temporal
    # embeddings) should add capacity, but the experiment is about history,
    # not a much bigger model -- assert the increase is modest (<5%).
    assert temporal_total > dense_total
    assert temporal_total < dense_total * 1.05


def test_dense_to_temporal_conversion_copies_shared_weights(dense_model, temporal_config):
    temporal_model = convert_dense_to_temporal(dense_model, temporal_config)

    assert torch.equal(
        dict(dense_model.vision_encoder.named_parameters())["backbone.conv1.weight"],
        dict(temporal_model.vision_encoder.named_parameters())["backbone.conv1.weight"],
    )
    assert torch.equal(dense_model.action_head[0].weight, temporal_model.action_head[0].weight)
    assert torch.equal(dense_model.fusion.vision_proj.weight, temporal_model.vision_proj.weight)
    assert torch.equal(dense_model.fusion.action_query, temporal_model.action_query)
    for layer_index in range(dense_model.config.num_layers):
        dense_layer = dense_model.transformer.layers[layer_index]
        temporal_block = temporal_model.transformer.blocks[layer_index]
        assert torch.equal(dense_layer.linear1.weight, temporal_block.ffn.linear1.weight)


def test_conversion_rejects_mismatched_architecture(dense_model):
    bad_config = TemporalDenseVLAConfig(hidden_dim=999, num_layers=dense_model.config.num_layers)
    with pytest.raises(ValueError):
        convert_dense_to_temporal(dense_model, bad_config)
