"""Optional learned set pooling modules for player-season mechanics priors.

These classes are imported by Colab training code when a learned prior is used.
They are not invoked by the dependency-free contract baseline, but they provide
the Deep Sets / attention / Set Transformer building blocks required by the
project design.
"""

from __future__ import annotations


def _import_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        return torch, nn
    except ImportError as exc:  # pragma: no cover - Colab dependency path
        raise RuntimeError("Learned set pooling requires PyTorch in Colab.") from exc


def build_deep_sets_pooler(input_dim: int, hidden_dim: int = 128, output_dim: int | None = None):
    """Return a Deep Sets style permutation-invariant pooler."""

    _torch, nn = _import_torch()
    output_dim = output_dim or hidden_dim

    class DeepSetsPooler(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.rho = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, output_dim))

        def forward(self, x, mask=None):  # type: ignore[no-untyped-def]
            values = self.phi(x)
            if mask is not None:
                values = values * mask.unsqueeze(-1).float()
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
                pooled = values.sum(dim=1) / denom
            else:
                pooled = values.mean(dim=1)
            return self.rho(pooled)

    return DeepSetsPooler()


def build_attention_pooler(input_dim: int, hidden_dim: int = 128, output_dim: int | None = None):
    """Return a learned attention pooling module over same batter-season clips."""

    _torch, nn = _import_torch()
    output_dim = output_dim or hidden_dim

    class AttentionPooler(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.score = nn.Linear(hidden_dim, 1)
            self.out = nn.Linear(hidden_dim, output_dim)

        def forward(self, x, mask=None):  # type: ignore[no-untyped-def]
            values = self.value(x)
            logits = self.score(values).squeeze(-1)
            if mask is not None:
                logits = logits.masked_fill(~mask.bool(), -1e9)
            weights = logits.softmax(dim=1)
            pooled = (values * weights.unsqueeze(-1)).sum(dim=1)
            return self.out(pooled)

    return AttentionPooler()


def build_set_transformer_pooler(input_dim: int, hidden_dim: int = 128, num_heads: int = 4, output_dim: int | None = None):
    """Return a small Transformer-encoder set pooler.

    This is a practical Set Transformer-inspired module for variable clip sets:
    no positional encoding is added, so the representation remains permutation
    invariant after masked mean pooling.
    """

    _torch, nn = _import_torch()
    output_dim = output_dim or hidden_dim

    class SetTransformerPooler(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(input_dim, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.out = nn.Linear(hidden_dim, output_dim)

        def forward(self, x, mask=None):  # type: ignore[no-untyped-def]
            values = self.in_proj(x)
            key_padding_mask = None if mask is None else ~mask.bool()
            encoded = self.encoder(values, src_key_padding_mask=key_padding_mask)
            if mask is not None:
                encoded = encoded * mask.unsqueeze(-1).float()
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
                pooled = encoded.sum(dim=1) / denom
            else:
                pooled = encoded.mean(dim=1)
            return self.out(pooled)

    return SetTransformerPooler()
