"""Temporal convolutional regressors (plain 1D-CNN and TCN) for the Session 2 benchmark engine.

Both models read the Session 1 windows [batch, time, sensors] directly (no tabular features) and
transpose them internally to the [batch, channels, time] layout of torch.nn.Conv1d, so the stored
arrays are never altered.

Plain 1D-CNN (minimal convolutional baseline)
    `depth` x [Conv1d(kernel_size, stride 1, no padding) -> ReLU -> Dropout]. The time axis shrinks
    by (kernel_size - 1) per layer. Readout: the LAST time step of the final feature map ->
    Linear(channels, 1). Receptive field of the readout = 1 + depth * (kernel_size - 1).

TCN
    One residual block per entry of `dilations`. Block = causal Conv1d(dilation) -> ReLU -> Dropout ->
    causal Conv1d(dilation) -> ReLU -> Dropout, plus the block input (through a 1x1 convolution when
    the number of channels changes), then ReLU. "Causal" means the input is padded with zeros on the
    LEFT by (kernel_size - 1) * dilation, so the output at time t never depends on later time
    steps and the time axis keeps its length. Readout: the LAST time step -> Linear(channels, 1).
    Theoretical receptive field = 1 + 2 * (kernel_size - 1) * sum(dilations).

The receptive field can exceed the 30-cycle window. The extra positions fall on the zero padding
to the left of the window: they add no observations, the model still sees at most 30 real cycles.

Training, early stopping, clipping, scheduling, prediction scaling, seeding and checkpoint
selection are inherited unchanged from `RecurrentRegressor` (Session 4), so every protocol
detail is identical to the recurrent family by construction.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .recurrent_models import RecurrentRegressor, seed_everything

KINDS = ("cnn", "tcn")


def cnn_receptive_field(kernel_size, depth):
    return 1 + int(depth) * (int(kernel_size) - 1)


def tcn_receptive_field(kernel_size, dilations):
    """RF = 1 + 2 * (kernel_size - 1) * sum(dilations): two convolutions per residual block."""
    return 1 + 2 * (int(kernel_size) - 1) * int(sum(dilations))


def cnn_output_length(sequence_length, kernel_size, depth):
    return int(sequence_length) - int(depth) * (int(kernel_size) - 1)


class PlainCNNNet(nn.Module):
    def __init__(self, n_sensors, channels, kernel_size, depth, dropout, sequence_length):
        super().__init__()
        if cnn_output_length(sequence_length, kernel_size, depth) < 1:
            raise ValueError(f"{depth} unpadded layers of kernel {kernel_size} do not fit "
                             f"sequence_length={sequence_length}")
        self.convs = nn.ModuleList([nn.Conv1d(n_sensors if i == 0 else channels, channels, kernel_size)
                                    for i in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels, 1)

    def features(self, x):                      # x: [B, F, T] -> [B, C, T - depth * (k - 1)]
        for conv in self.convs:
            x = self.dropout(F.relu(conv(x)))
        return x

    def forward(self, x):                       # x: [B, T, F]
        h = self.features(x.transpose(1, 2))[:, :, -1]      # last time step
        return self.head(self.dropout(h)).squeeze(-1)       # [B]


class CausalConv1d(nn.Module):
    """Conv1d with dilation and LEFT zero padding of (kernel_size - 1) * dilation: output keeps the
    input length and output[t] depends on input[<= t] only."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        # 1x1 projection on the residual path only when the channel count changes
        self.project = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.dropout(F.relu(self.conv1(x)))
        out = self.dropout(F.relu(self.conv2(out)))
        residual = x if self.project is None else self.project(x)
        return F.relu(out + residual)


class TCNNet(nn.Module):
    def __init__(self, n_sensors, channels, kernel_size, dilations, dropout):
        super().__init__()
        if not dilations or any(int(d) < 1 for d in dilations):
            raise ValueError("dilations must be a non-empty list of positive integers")
        self.blocks = nn.ModuleList([TemporalBlock(n_sensors if i == 0 else channels, channels, kernel_size,
                                                   int(d), dropout) for i, d in enumerate(dilations)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels, 1)

    def features(self, x):                      # x: [B, F, T] -> [B, C, T]
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x):                       # x: [B, T, F]
        h = self.features(x.transpose(1, 2))[:, :, -1]
        return self.head(self.dropout(h)).squeeze(-1)


class ConvolutionalRegressor(RecurrentRegressor):
    """CNN / TCN regressor with the benchmark model interface.

    Only the network and its description are defined here; `fit`, `predict`, `n_parameters`,
    `weights_sha256` and the whole training protocol come from `RecurrentRegressor`.
    """

    def __init__(self, name, kind, *, channels, kernel_size, depth=None, dilations=None, dropout,
                 sequence_length, n_sensors, training, seed, device="cpu", validation=None,
                 target_scale=125.0):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if kind == "cnn" and depth is None:
            raise ValueError("a cnn needs `depth`")
        if kind == "tcn" and not dilations:
            raise ValueError("a tcn needs `dilations`")
        self.name, self.cell, self.kind = name, kind, kind
        self.channels, self.kernel_size, self.dropout = int(channels), int(kernel_size), float(dropout)
        self.depth = None if depth is None else int(depth)
        self.dilations = None if dilations is None else [int(d) for d in dilations]
        self.sequence_length, self.n_sensors = int(sequence_length), int(n_sensors)
        self.training, self.seed, self.device = dict(training), int(seed), device
        self.target_scale, self.validation = float(target_scale), validation
        self.net_, self.history_, self.fit_summary_ = None, [], {}
        if kind == "cnn" and cnn_output_length(self.sequence_length, self.kernel_size, self.depth) < 1:
            raise ValueError("the unpadded convolution stack is longer than the sequence")

    @property
    def receptive_field(self):
        return (cnn_receptive_field(self.kernel_size, self.depth) if self.kind == "cnn"
                else tcn_receptive_field(self.kernel_size, self.dilations))

    def _build(self):
        seed_everything(self.seed)              # deterministic initialisation
        if self.kind == "cnn":
            net = PlainCNNNet(self.n_sensors, self.channels, self.kernel_size, self.depth, self.dropout,
                              self.sequence_length)
        else:
            net = TCNNet(self.n_sensors, self.channels, self.kernel_size, self.dilations, self.dropout)
        return net.to(self.device)

    def architecture(self):
        arch = {"channels": self.channels, "kernel_size": self.kernel_size, "dropout": self.dropout,
                "sequence_length": self.sequence_length, "n_sensors": self.n_sensors,
                "receptive_field": self.receptive_field, "readout": "last time step"}
        if self.kind == "cnn":
            arch.update(depth=self.depth, padding="none (valid)",
                        output_length=cnn_output_length(self.sequence_length, self.kernel_size, self.depth))
        else:
            arch.update(dilations=list(self.dilations), n_blocks=len(self.dilations),
                        padding="causal (left zero padding)")
        return arch

    def get_config(self):
        return {"type": self.kind, **self.architecture(), "seed": self.seed, "device": self.device,
                "target_scale": self.target_scale, "training": self.training,
                "n_parameters": self.n_parameters(), **self.fit_summary_}

    def save(self, directory):
        """Best-validation checkpoint and config (the model files); history is written alongside."""
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        ckpt = directory / f"{self.kind}.pt"
        state = {k: v.detach().cpu() for k, v in self.net_.state_dict().items()}
        torch.save({"state_dict": state, "config": self.get_config()}, ckpt)
        config = directory / f"{self.kind}_config.json"
        config.write_text(json.dumps(self.get_config(), indent=2) + "\n")
        (directory / "training_history.json").write_text(json.dumps(self.history_, indent=2) + "\n")
        return [ckpt, config]

    @classmethod
    def load(cls, directory, name=None, device="cpu"):
        directory = Path(directory)
        blob = torch.load(next(directory.glob("*.pt")), map_location="cpu", weights_only=True)
        c = blob["config"]
        model = cls(name or c["type"], c["type"], channels=c["channels"], kernel_size=c["kernel_size"],
                    depth=c.get("depth"), dilations=c.get("dilations"), dropout=c["dropout"],
                    sequence_length=c["sequence_length"], n_sensors=c["n_sensors"], training=c["training"],
                    seed=c["seed"], device=device, target_scale=c["target_scale"])
        model.net_ = model._build()
        model.net_.load_state_dict(blob["state_dict"])
        model.fit_summary_ = {k: c[k] for k in ("best_epoch", "best_val_loss", "epochs_run", "stop_reason",
                                                 "stable") if k in c}
        history = directory / "training_history.json"
        model.history_ = json.loads(history.read_text()) if history.exists() else []
        return model

