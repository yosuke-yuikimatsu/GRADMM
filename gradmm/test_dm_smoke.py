"""CPU smoke checks for Distribution Matching and embedding dtype alignment."""

from types import SimpleNamespace

import torch
from torch import nn

from generate import _ids_for_lm_embeddings
from distribution_matching import (
    RandomMLPProjector,
    build_dm_projectors,
    compute_dm_loss,
)
from utilities import (
    align_embeds_to_model,
    compute_grads_lm,
    cos_sim,
    cos_sim_batch,
    grad_dist,
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


class _TinyModel(nn.Module):
    def __init__(self, dtype=torch.float16):
        super().__init__()
        self.embeddings = nn.Embedding(8, 5, dtype=dtype)

    def get_input_embeddings(self):
        return self.embeddings


class _TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=11, embed_dim=6, dtype=torch.float32):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, dtype=dtype)
        self.lm_head = nn.Linear(embed_dim, vocab_size, dtype=dtype)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        del attention_mask
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        return SimpleNamespace(logits=self.lm_head(inputs_embeds))


def _assert_higher_order_grad_clip_backward(gen_grad_clip):
    torch.manual_seed(0)
    model = _TinyCausalLM()
    x_embeds = torch.randn(2, 3, 6, requires_grad=True) * 25
    attention_mask = torch.ones(2, 3, dtype=torch.long)
    y_labels = torch.tensor([1])

    grads = compute_grads_lm(
        model,
        x_embeds,
        attention_mask,
        y_labels,
        create_graph=True,
        gen_grad_clip=gen_grad_clip,
    )
    grad_loss = sum(g.float().square().sum() for g in grads if g is not None)

    grad_loss.backward()

    assert any(param.grad is not None for param in model.parameters())


def test_compute_grads_lm_norm_clip_supports_higher_order_backward():
    _assert_higher_order_grad_clip_backward("norm")


def test_compute_grads_lm_elem_clip_supports_higher_order_backward():
    _assert_higher_order_grad_clip_backward("elem")


def test_cos_sim_zero_vectors_is_finite():
    value = cos_sim(torch.zeros(4), torch.zeros(4))
    assert torch.isfinite(value)


def test_cos_sim_batch_zero_rows_is_finite():
    value = cos_sim_batch(torch.zeros(3, 4), torch.zeros(3, 4))
    assert torch.isfinite(value)


def test_grad_dist_zero_cosine_gradients_is_finite():
    args = SimpleNamespace(loss="cos")
    value = grad_dist([torch.zeros(2, 3)], [torch.zeros(2, 3)], args)
    assert torch.isfinite(value)


def test_grad_dist_all_none_returns_finite_zero_scalar():
    args = SimpleNamespace(loss="cos")
    value = grad_dist([None, None], [None, None], args)
    assert value.ndim == 0
    assert torch.isfinite(value)
    assert value.item() == 0.0


def test_random_mlp_projector_shape():
    projector = RandomMLPProjector(input_dim=5, feature_dim=7)
    inputs = torch.randn(4, 6, 5)
    attention_mask = torch.ones(4, 6, dtype=torch.long)
    outputs = projector(inputs, attention_mask)
    assert outputs.shape == (4, 7)


def test_dm_loss_backward_frozen_params_and_class_aware_labels():
    args = _args(dm_match="mean_var", dm_by_class=True)
    projectors = build_dm_projectors(args, lm_embed_dim=5, device=torch.device("cpu"))
    real_embeds = torch.randn(4, 6, 5)
    syn_embeds = torch.randn(4, 6, 5, requires_grad=True)
    attention_mask = torch.ones(4, 6, dtype=torch.long)
    labels = torch.tensor([0, 0, 1, 1])

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


def test_dtype_alignment_preserves_gradient_flow():
    model = _TinyModel(dtype=torch.float16)
    syn_embeds = torch.randn(2, 3, 5, dtype=torch.float32, requires_grad=True)
    aligned = align_embeds_to_model(syn_embeds, model)

    assert aligned.dtype == torch.float16
    assert aligned.device == model.get_input_embeddings().weight.device
    assert aligned.requires_grad

    loss = aligned.float().square().mean()
    loss.backward()
    assert syn_embeds.grad is not None
    assert syn_embeds.grad.abs().sum() > 0


def test_adam_optimizes_fp32_embeds_with_half_precision_forward():
    torch.manual_seed(0)
    model = _TinyCausalLM(dtype=torch.float16)
    x_embeds = torch.randn(2, 3, 6, dtype=torch.float16)
    x_embeds = x_embeds.detach().clone().float()
    x_embeds.requires_grad_(True)
    optimizer = torch.optim.Adam([x_embeds], lr=1e-5)
    attention_mask = torch.ones(2, 3, dtype=torch.long)

    optimizer.zero_grad()
    aligned = align_embeds_to_model(x_embeds, model)
    assert aligned.dtype == torch.float16
    outputs = model(inputs_embeds=aligned, attention_mask=attention_mask)
    loss = outputs.logits.float().square().mean()
    loss.backward()

    assert x_embeds.grad is not None
    assert torch.isfinite(x_embeds.grad).all()
    optimizer.step()

    assert x_embeds.dtype == torch.float32
    assert torch.isfinite(x_embeds).all()


def test_dm_loss_logging_value_is_always_defined_in_closure_path():
    args = _args(dm_match="mean")
    projectors = build_dm_projectors(args, lm_embed_dim=3, device=torch.device("cpu"))
    real_embeds = torch.randn(4, 2, 3)
    syn_embeds = torch.randn(4, 2, 3, requires_grad=True)
    attention_mask = torch.ones(4, 2, dtype=torch.long)
    labels = torch.tensor([0, 0, 1, 1])
    dm_loss_value = 0.0

    def closure(use_dm=True):
        nonlocal dm_loss_value
        current_dm_loss = syn_embeds.new_zeros(())
        if use_dm:
            current_dm_loss = compute_dm_loss(
                args,
                projectors,
                real_embeds,
                attention_mask,
                labels,
                syn_embeds,
                attention_mask,
                labels,
            )
            dm_loss_value = float(current_dm_loss.detach().cpu())
        total_loss = current_dm_loss + syn_embeds.square().mean()
        total_loss.backward()
        return total_loss

    closure(use_dm=True)
    assert isinstance(dm_loss_value, float)
    assert syn_embeds.grad is not None

    dm_loss_value = 0.0
    syn_embeds.grad = None
    closure(use_dm=False)
    assert dm_loss_value == 0.0
    assert syn_embeds.grad is not None


def test_ids_for_lm_embeddings_preserves_batched_admm_ids_for_all_conversions():
    embeddings = nn.Embedding(13, 5)
    for conversion_method in ("topk", "concat", "proj"):
        for batch_size in (1, 3):
            token_ids = torch.arange(batch_size * 4).view(batch_size, 4)
            z_embeds = torch.empty(batch_size, 4, 5)

            embeds = embeddings(_ids_for_lm_embeddings(token_ids, name="z_ids"))
            z_embeds.copy_(embeds)

            assert z_embeds.shape == (batch_size, 4, 5), conversion_method


def test_ids_for_lm_embeddings_adds_batch_only_for_unbatched_ids():
    embeddings = nn.Embedding(13, 5)
    token_ids = torch.arange(4)
    embeds = embeddings(_ids_for_lm_embeddings(token_ids, name="z_ids"))

    assert embeds.shape == (1, 4, 5)


def test_ids_for_lm_embeddings_rejects_extra_admm_dimension():
    token_ids = torch.zeros(1, 1, 4, dtype=torch.long)

    try:
        _ids_for_lm_embeddings(token_ids, name="z_ids")
    except ValueError as exc:
        assert "Unexpected z_ids shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for rank-3 token ids")


if __name__ == "__main__":
    test_cos_sim_zero_vectors_is_finite()
    test_cos_sim_batch_zero_rows_is_finite()
    test_grad_dist_zero_cosine_gradients_is_finite()
    test_grad_dist_all_none_returns_finite_zero_scalar()
    test_random_mlp_projector_shape()
    test_dm_loss_backward_frozen_params_and_class_aware_labels()
    test_dtype_alignment_preserves_gradient_flow()
    test_dm_loss_logging_value_is_always_defined_in_closure_path()
    test_adam_optimizes_fp32_embeds_with_half_precision_forward()
    test_compute_grads_lm_norm_clip_supports_higher_order_backward()
    test_compute_grads_lm_elem_clip_supports_higher_order_backward()
    test_ids_for_lm_embeddings_preserves_batched_admm_ids_for_all_conversions()
    test_ids_for_lm_embeddings_adds_batch_only_for_unbatched_ids()
    test_ids_for_lm_embeddings_rejects_extra_admm_dimension()
    print("DM smoke checks passed.")
