"""Distribution matching objectives for synthetic embedding generation."""

import torch
from torch import nn
from transformers import BertConfig, BertModel


def _make_attention_mask(inputs_embeds, attention_mask):
    if attention_mask is None:
        return torch.ones(
            inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long
        )
    return attention_mask.to(device=inputs_embeds.device)


def _compatible_heads(hidden_dim, requested_heads=4):
    for nhead in range(min(requested_heads, hidden_dim), 0, -1):
        if hidden_dim % nhead == 0:
            return nhead
    return 1


def masked_pool(features, attention_mask, mode="mean"):
    """Pool token features with an attention mask.

    Args:
        features: Tensor of shape [batch, seq_len, feature_dim].
        attention_mask: Tensor of shape [batch, seq_len]. Non-zero entries are valid.
        mode: ``"mean"`` for masked mean or ``"last"`` for the last valid token.

    Returns:
        Tensor of shape [batch, feature_dim].
    """
    if mode not in ["mean", "last"]:
        raise ValueError(f"Unsupported pooling mode: {mode}")

    attention_mask = _make_attention_mask(features, attention_mask)
    mask = attention_mask.to(dtype=features.dtype).unsqueeze(-1)

    if mode == "mean":
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (features * mask).sum(dim=1) / denom

    lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
    batch_index = torch.arange(features.shape[0], device=features.device)
    return features[batch_index, lengths]


class RandomMLPProjector(nn.Module):
    """Frozen random token-wise MLP projector over LM input embeddings."""

    def __init__(self, input_dim, feature_dim=256, hidden_dim=None, pooling="mean"):
        super().__init__()
        hidden_dim = hidden_dim or min(max(input_dim, feature_dim), 512)
        self.pooling = pooling
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, inputs_embeds, attention_mask=None):
        attention_mask = _make_attention_mask(inputs_embeds, attention_mask)
        token_features = self.net(inputs_embeds)
        return masked_pool(token_features, attention_mask, mode=self.pooling)


class RandomTinyTransformerProjector(nn.Module):
    """Frozen small random Transformer encoder projector."""

    def __init__(self, input_dim, feature_dim=256, hidden_dim=None, pooling="mean"):
        super().__init__()
        hidden_dim = hidden_dim or min(input_dim, 256)
        hidden_dim = max(hidden_dim, 4)
        nhead = _compatible_heads(hidden_dim, requested_heads=4)
        self.pooling = pooling
        self.input_proj = (
            nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

    def forward(self, inputs_embeds, attention_mask=None):
        attention_mask = _make_attention_mask(inputs_embeds, attention_mask)
        hidden = self.input_proj(inputs_embeds)
        padding_mask = attention_mask == 0
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        token_features = self.output_proj(hidden)
        return masked_pool(token_features, attention_mask, mode=self.pooling)


class RandomBertProjector(nn.Module):
    """Frozen randomly initialized BERT projector that consumes inputs_embeds."""

    def __init__(self, input_dim, feature_dim=256, hidden_size=None, pooling="mean"):
        super().__init__()
        hidden_size = hidden_size or min(input_dim, 256)
        hidden_size = max(hidden_size, 4)
        nhead = _compatible_heads(hidden_size, requested_heads=4)
        self.pooling = pooling
        self.adapter = (
            nn.Identity() if input_dim == hidden_size else nn.Linear(input_dim, hidden_size)
        )
        config = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=2,
            num_attention_heads=nhead,
            intermediate_size=4 * hidden_size,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        self.bert = BertModel(config)
        self.output_proj = nn.Linear(hidden_size, feature_dim)

    def forward(self, inputs_embeds, attention_mask=None):
        attention_mask = _make_attention_mask(inputs_embeds, attention_mask)
        hidden = self.adapter(inputs_embeds)
        outputs = self.bert(inputs_embeds=hidden, attention_mask=attention_mask)
        token_features = self.output_proj(outputs.last_hidden_state)
        return masked_pool(token_features, attention_mask, mode=self.pooling)


def build_dm_projectors(args, lm_embed_dim, device):
    """Build frozen random DM projectors while preserving input gradients."""
    projectors = []
    for _ in range(args.dm_num_projectors):
        if args.dm_projector == "mlp":
            projector = RandomMLPProjector(
                lm_embed_dim, feature_dim=args.dm_feature_dim, pooling=args.dm_pooling
            )
        elif args.dm_projector == "tiny_transformer":
            projector = RandomTinyTransformerProjector(
                lm_embed_dim, feature_dim=args.dm_feature_dim, pooling=args.dm_pooling
            )
        elif args.dm_projector == "random_bert":
            projector = RandomBertProjector(
                lm_embed_dim, feature_dim=args.dm_feature_dim, pooling=args.dm_pooling
            )
        else:
            raise ValueError(f"Unsupported DM projector: {args.dm_projector}")
        projector.to(device)
        projector.eval()
        for param in projector.parameters():
            param.requires_grad_(False)
        projectors.append(projector)
    return nn.ModuleList(projectors)


def _mean_loss(real_features, syn_features):
    return (real_features.mean(dim=0) - syn_features.mean(dim=0)).square().sum()


def _mean_var_loss(real_features, syn_features):
    mean_loss = _mean_loss(real_features, syn_features)
    real_var = real_features.var(dim=0, unbiased=False)
    syn_var = syn_features.var(dim=0, unbiased=False)
    return mean_loss + (real_var - syn_var).square().sum()


def _rbf_mmd(real_features, syn_features):
    joint = torch.cat([real_features, syn_features], dim=0)
    with torch.no_grad():
        sq_dists = torch.pdist(joint).square()
        positive = sq_dists[sq_dists > 0]
        if positive.numel() == 0:
            sigma2 = torch.tensor(1.0, device=joint.device, dtype=joint.dtype)
        else:
            sigma2 = positive.median().clamp_min(1e-6)
    gamma = 1.0 / (2.0 * sigma2)

    def kernel(x, y):
        return torch.exp(-gamma * torch.cdist(x, y).square())

    k_xx = kernel(real_features, real_features).mean()
    k_yy = kernel(syn_features, syn_features).mean()
    k_xy = kernel(real_features, syn_features).mean()
    return k_xx + k_yy - 2.0 * k_xy


def _discrepancy(args, real_features, syn_features):
    if args.dm_match == "mean":
        return _mean_loss(real_features, syn_features)
    if args.dm_match == "mean_var":
        return _mean_var_loss(real_features, syn_features)
    if args.dm_match == "mmd":
        return _rbf_mmd(real_features, syn_features)
    raise ValueError(f"Unsupported DM match objective: {args.dm_match}")


def compute_dm_loss(
    args,
    projectors,
    real_embeds,
    real_attention_mask,
    real_labels,
    syn_embeds,
    syn_attention_mask,
    syn_labels,
):
    """Compute distribution matching loss in frozen random feature spaces."""
    if projectors is None or len(projectors) == 0:
        return syn_embeds.new_zeros(())

    real_labels = real_labels.to(device=syn_embeds.device).view(-1)
    syn_labels = syn_labels.to(device=syn_embeds.device).view(-1)
    losses = []
    for projector in projectors:
        real_features = projector(real_embeds, real_attention_mask)
        syn_features = projector(syn_embeds, syn_attention_mask)
        if args.dm_detach_real_features:
            real_features = real_features.detach()

        if args.dm_by_class:
            class_losses = []
            common_labels = torch.unique(syn_labels)
            for label in common_labels:
                real_idx = real_labels == label
                syn_idx = syn_labels == label
                if real_idx.any() and syn_idx.any():
                    class_losses.append(
                        _discrepancy(args, real_features[real_idx], syn_features[syn_idx])
                    )
            if class_losses:
                losses.append(torch.stack(class_losses).mean())
        else:
            losses.append(_discrepancy(args, real_features, syn_features))

    if not losses:
        return syn_embeds.new_zeros(())
    return torch.stack(losses).mean()
