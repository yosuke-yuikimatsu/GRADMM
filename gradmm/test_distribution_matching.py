"""CPU sanity checks for the distribution matching extension."""

from types import SimpleNamespace

import torch

from distribution_matching import (
    RandomMLPProjector,
    build_dm_projectors,
    compute_dm_loss,
)


def _args(**overrides):
    defaults = dict(
        dm_projector="mlp",
        dm_feature_dim=7,
        dm_num_projectors=1,
        dm_match="mean",
        dm_by_class=True,
        dm_detach_real_features=True,
        dm_pooling="mean",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_random_mlp_projector_shape():
    projector = RandomMLPProjector(input_dim=5, feature_dim=7)
    inputs = torch.randn(4, 6, 5)
    attention_mask = torch.ones(4, 6, dtype=torch.long)
    outputs = projector(inputs, attention_mask)
    assert outputs.shape == (4, 7)


def test_dm_loss_backward_and_frozen_projector_params():
    args = _args(dm_match="mean_var")
    projectors = build_dm_projectors(args, lm_embed_dim=5, device=torch.device("cpu"))
    real_embeds = torch.randn(4, 6, 5)
    syn_embeds = torch.randn(4, 6, 5, requires_grad=True)
    attention_mask = torch.ones(4, 6, dtype=torch.long)
    labels = torch.tensor([0, 1, 0, 1])

    loss = compute_dm_loss(
        args,
        projectors,
        real_embeds,
        attention_mask,
        labels,
        syn_embeds,
        attention_mask,
        labels,
    )

    assert loss.ndim == 0
    loss.backward()
    assert syn_embeds.grad is not None
    assert syn_embeds.grad.abs().sum() > 0
    for projector in projectors:
        for param in projector.parameters():
            assert not param.requires_grad
            assert param.grad is None


def test_class_aware_mmd_loss_with_two_classes():
    args = _args(dm_match="mmd", dm_by_class=True)
    projectors = build_dm_projectors(args, lm_embed_dim=3, device=torch.device("cpu"))
    real_embeds = torch.randn(4, 5, 3)
    syn_embeds = torch.randn(4, 5, 3, requires_grad=True)
    attention_mask = torch.ones(4, 5, dtype=torch.long)
    labels = torch.tensor([0, 1, 0, 1])

    loss = compute_dm_loss(
        args,
        projectors,
        real_embeds,
        attention_mask,
        labels,
        syn_embeds,
        attention_mask,
        labels,
    )
    assert loss.ndim == 0
    loss.backward()
    assert syn_embeds.grad is not None


if __name__ == "__main__":
    test_random_mlp_projector_shape()
    test_dm_loss_backward_and_frozen_projector_params()
    test_class_aware_mmd_loss_with_two_classes()
    print("Distribution matching sanity checks passed.")
