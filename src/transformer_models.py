"""Compact encoder-only Transformer for sequence-to-one RUL regression.

Data flow
    windows [B, T, F] -> Linear(F, d_model) -> + sinusoidal positional encoding -> dropout
    -> `num_layers` pre-norm Transformer encoder layers (multi-head self-attention and a
       feed-forward block, each with a residual connection) -> final LayerNorm  [B, T, d_model]
    -> pooling over time ('last' or 'mean')  [B, d_model] -> dropout -> Linear(d_model, 1) -> [B]

Encoder layers are `torch.nn.TransformerEncoderLayer` with `batch_first=True`. They are constructed
one by one (not cloned), so every layer has its own initial weights.

Attention-mask policy: NO mask by default. The task is sequence-to-one on a fully observed window:
every one of the T cycles is available when the prediction is made and the output refers to the
last cycle, so attention between any two cycles of the window uses no future information relative
to the prediction time. No cycle outside the window can be attended to, and there is no padding to
mask (all windows have length T). A boolean mask (True = blocked) can be passed to `forward`.

The number of input sensors only affects the input projection, so the same architecture can be
rebuilt for other sensor counts (Session 7).

Training, early stopping, clipping, scheduling, target scaling (RUL / cap), seeding and checkpoint
selection are inherited unchanged from `RecurrentRegressor` (Session 4).
"""
import json
import math
import numbers
from pathlib import Path

import torch
from torch import nn

from .recurrent_models import RecurrentRegressor, seed_everything

POOLINGS = ("last", "mean")
ACTIVATIONS = ("relu", "gelu")
ARCHITECTURE_KEYS = ("d_model", "num_heads", "num_layers", "ff_multiplier", "dropout", "pooling", "activation")


def sinusoidal_encoding(length, d_model):
    """Deterministic sine / cosine positional encoding [length, d_model] (Vaswani et al., 2017)."""
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    table = torch.zeros(length, d_model)
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency[: d_model // 2])
    return table


def require_positive_int(name, value, minimum=1):
    """Integer-valued settings must be true integers (Python or NumPy). Booleans and floats are
    rejected, even 4.0, so that nothing is ever silently truncated."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def validate_architecture(d_model, num_heads, num_layers, ff_multiplier, dropout, pooling, activation="relu",
                          sequence_length=None, n_sensors=None):
    require_positive_int("d_model", d_model, 2)
    require_positive_int("num_heads", num_heads)
    require_positive_int("num_layers", num_layers)
    require_positive_int("ff_multiplier", ff_multiplier)
    if sequence_length is not None:
        require_positive_int("sequence_length", sequence_length)
    if n_sensors is not None:
        require_positive_int("n_sensors", n_sensors)
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    if isinstance(dropout, bool) or not isinstance(dropout, numbers.Real) or not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be a number in [0, 1), got {dropout!r}")
    if pooling not in POOLINGS:
        raise ValueError(f"unsupported pooling '{pooling}', expected one of {POOLINGS}")
    if activation not in ACTIVATIONS:
        raise ValueError(f"unsupported activation '{activation}', expected one of {ACTIVATIONS}")


class TransformerNet(nn.Module):
    def __init__(self, n_sensors, sequence_length, d_model, num_heads, num_layers, ff_multiplier=2,
                 dropout=0.1, pooling="last", activation="relu"):
        super().__init__()
        validate_architecture(d_model, num_heads, num_layers, ff_multiplier, dropout, pooling, activation,
                              sequence_length, n_sensors)
        self.pooling = pooling
        self.input_projection = nn.Linear(n_sensors, d_model)
        self.register_buffer("positions", sinusoidal_encoding(sequence_length, d_model), persistent=False)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=ff_multiplier * d_model,
                                       dropout=dropout, activation=activation, batch_first=True, norm_first=True)
            for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def embed(self, x):                              # [B, T, F] -> [B, T, d_model]
        if x.ndim != 3 or x.shape[1] > len(self.positions):
            raise ValueError(f"expected [batch, time <= {len(self.positions)}, sensors], got {tuple(x.shape)}")
        return self.input_projection(x) + self.positions[: x.shape[1]]

    def encode(self, x, attn_mask=None):             # [B, T, F] -> [B, T, d_model]
        h = self.input_dropout(self.embed(x))
        for layer in self.layers:
            h = layer(h, src_mask=attn_mask)
        return self.final_norm(h)

    def pool(self, h):                               # [B, T, d_model] -> [B, d_model]
        return h[:, -1, :] if self.pooling == "last" else h.mean(dim=1)

    def forward(self, x, attn_mask=None):            # [B, T, F] -> [B]
        return self.head(self.head_dropout(self.pool(self.encode(x, attn_mask)))).squeeze(-1)

    def attention_weights(self, x, attn_mask=None):
        """Per-layer attention weights [B, heads, T, T] (inspection only, evaluation mode)."""
        was_training = self.training
        self.eval()
        weights = []
        with torch.no_grad():
            h = self.embed(x)
            for layer in self.layers:
                q = layer.norm1(h)
                _, w = layer.self_attn(q, q, q, attn_mask=attn_mask, need_weights=True,
                                       average_attn_weights=False)
                weights.append(w)
                h = layer(h, src_mask=attn_mask)
        self.train(was_training)
        return weights


class TransformerRegressor(RecurrentRegressor):
    """Transformer regressor with the benchmark model interface.

    Only the network and its description are defined here; `fit`, `predict`, `n_parameters`,
    `weights_sha256` and the whole training protocol come from `RecurrentRegressor`.
    """

    def __init__(self, name, *, d_model, num_heads, num_layers, ff_multiplier=2, dropout=0.1, pooling="last",
                 activation="relu", sequence_length, n_sensors, training, seed, device="cpu", validation=None,
                 target_scale=125.0):
        validate_architecture(d_model, num_heads, num_layers, ff_multiplier, dropout, pooling, activation,
                              sequence_length, n_sensors)
        self.name, self.cell = name, "transformer"
        self.d_model, self.num_heads, self.num_layers = int(d_model), int(num_heads), int(num_layers)
        self.ff_multiplier, self.dropout = int(ff_multiplier), float(dropout)
        self.pooling, self.activation = pooling, activation
        self.sequence_length, self.n_sensors = int(sequence_length), int(n_sensors)
        self.training, self.seed, self.device = dict(training), int(seed), device
        self.target_scale, self.validation = float(target_scale), validation
        self.net_, self.history_, self.fit_summary_ = None, [], {}

    def _build(self):
        seed_everything(self.seed)                   # deterministic initialisation
        return TransformerNet(self.n_sensors, self.sequence_length, self.d_model, self.num_heads,
                              self.num_layers, self.ff_multiplier, self.dropout, self.pooling,
                              self.activation).to(self.device)

    def architecture(self):
        return {"d_model": self.d_model, "num_heads": self.num_heads, "num_layers": self.num_layers,
                "ff_multiplier": self.ff_multiplier, "ff_dim": self.ff_multiplier * self.d_model,
                "dropout": self.dropout, "pooling": self.pooling, "activation": self.activation,
                "positional_encoding": "sinusoidal", "normalization": "pre-norm layers + final LayerNorm",
                "attention_mask": "none", "sequence_length": self.sequence_length, "n_sensors": self.n_sensors}

    def get_config(self):
        return {"type": "transformer", **self.architecture(), "seed": self.seed, "device": self.device,
                "target_scale": self.target_scale, "training": self.training,
                "n_parameters": self.n_parameters(), **self.fit_summary_}

    def save(self, directory):
        """Best-validation checkpoint and config (the model files); history is written alongside."""
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        ckpt = directory / "transformer.pt"
        state = {k: v.detach().cpu() for k, v in self.net_.state_dict().items()}
        torch.save({"state_dict": state, "config": self.get_config()}, ckpt)
        config = directory / "transformer_config.json"
        config.write_text(json.dumps(self.get_config(), indent=2) + "\n")
        (directory / "training_history.json").write_text(json.dumps(self.history_, indent=2) + "\n")
        return [ckpt, config]

    @classmethod
    def load(cls, directory, name=None, device="cpu"):
        directory = Path(directory)
        blob = torch.load(next(directory.glob("*.pt")), map_location="cpu", weights_only=True)
        c = blob["config"]
        model = cls(name or "transformer", **{k: c[k] for k in ARCHITECTURE_KEYS},
                    sequence_length=c["sequence_length"], n_sensors=c["n_sensors"], training=c["training"],
                    seed=c["seed"], device=device, target_scale=c["target_scale"])
        model.net_ = model._build()
        model.net_.load_state_dict(blob["state_dict"])
        model.fit_summary_ = {k: c[k] for k in ("best_epoch", "best_val_loss", "epochs_run", "stop_reason",
                                                 "stable") if k in c}
        history = directory / "training_history.json"
        model.history_ = json.loads(history.read_text()) if history.exists() else []
        return model


def build_from_definition(definition, *, n_sensors, validation=None, name="transformer", device=None):
    """Rebuild the frozen architecture (artifacts/session6/selected_transformer.json) for a given
    number of input sensors. Only the input projection depends on it; every other hyperparameter and
    the training protocol are taken from the definition unchanged."""
    arch, training = definition["architecture"], dict(definition["training"])
    seed, trained_on = training.pop("seed"), training.pop("device")
    return TransformerRegressor(name, **{k: arch[k] for k in ARCHITECTURE_KEYS},
                                sequence_length=arch["sequence_length"], n_sensors=n_sensors,
                                training=training, seed=seed, device=device or trained_on,
                                validation=validation, target_scale=float(training["target_scale"]))
