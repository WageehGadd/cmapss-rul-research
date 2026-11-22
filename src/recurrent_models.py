"""Recurrent sequence-to-one RUL regressors (LSTM, GRU) for the Session 2 benchmark engine.

The wrapper follows the benchmark model interface (name, fit, predict, get_config, save,
n_parameters) and hides every PyTorch detail. It consumes the Session 1 windows directly
([n, sequence_length, n_sensors]); no tabular features are involved.

Architecture
    input [batch, time, sensors] -> LSTM | GRU (batch_first=True, `num_layers`, inter-layer dropout)
    -> hidden state of the LAST time step of the top layer -> dropout -> Linear(hidden, 1) -> RUL

Training protocol (shared by both cells, see `fit`)
    * targets are divided by `target_scale` (the RUL cap) inside the model and multiplied back in
      `predict`, so the loss is the MSE of RUL / cap; metrics are always computed in cycles;
    * Adam, MSE loss, mini-batches reshuffled every epoch from a seeded generator;
    * gradient-norm clipping at `grad_clip` (the pre-clip norm is recorded every step);
    * ReduceLROnPlateau on the VALIDATION loss;
    * early stopping on the VALIDATION loss only, best weights restored at the end.
Validation arrays are supplied at construction (`validation=(X, y)`); the test set never reaches
this class.

Determinism: seeds for Python, NumPy and PyTorch are set at the start of `fit`, and PyTorch's
deterministic algorithms are requested. CPU runs on one machine are repeatable. MPS and CUDA
kernels are not guaranteed to be bit-reproducible, so the device is part of the recorded config.
"""
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

CELLS = ("lstm", "gru")
DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")


def resolve_device(preference="auto"):
    """Pick a torch device. 'auto' prefers CUDA, then MPS, then CPU; an explicit choice that is
    unavailable raises instead of silently falling back."""
    if preference not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {DEVICE_CHOICES}, got '{preference}'")
    has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        raise ValueError("device 'cuda' requested but CUDA is not available")
    if preference == "mps" and not has_mps:
        raise ValueError("device 'mps' requested but MPS is not available")
    return preference


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch (CPU and, when present, CUDA)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RecurrentNet(nn.Module):
    def __init__(self, cell, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {CELLS}")
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):                       # x: [batch, time, sensors]
        out, _ = self.rnn(x)                    # [batch, time, hidden]
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)    # [batch]


def count_parameters(net):
    return int(sum(p.numel() for p in net.parameters()))


def weights_digest(state_dict):
    """SHA-256 over parameter names, dtypes, shapes and values (independent of file layout)."""
    h = hashlib.sha256()
    for key in sorted(state_dict):
        t = state_dict[key].detach().cpu().contiguous()
        h.update(f"{key}{t.dtype}{tuple(t.shape)}".encode() + t.numpy().tobytes())
    return h.hexdigest()


class RecurrentRegressor:
    """LSTM/GRU sequence-to-one regressor following the benchmark model interface."""

    def __init__(self, name, cell, *, hidden_size, num_layers, dropout, sequence_length, n_sensors,
                 training, seed, device="cpu", validation=None, target_scale=125.0):
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {CELLS}")
        self.name, self.cell = name, cell
        self.hidden_size, self.num_layers, self.dropout = int(hidden_size), int(num_layers), float(dropout)
        self.sequence_length, self.n_sensors = int(sequence_length), int(n_sensors)
        self.training = dict(training)
        self.seed, self.device, self.target_scale = int(seed), device, float(target_scale)
        self.validation = validation
        self.net_ = None
        self.history_ = []
        self.fit_summary_ = {}

    # ---- construction helpers -------------------------------------------------------------
    def _build(self):
        seed_everything(self.seed)              # deterministic initialisation
        return RecurrentNet(self.cell, self.n_sensors, self.hidden_size, self.num_layers,
                            self.dropout).to(self.device)

    def _check_windows(self, X):
        X = np.asarray(X)
        if X.ndim != 3 or X.shape[1:] != (self.sequence_length, self.n_sensors):
            raise ValueError(f"expected windows [n, {self.sequence_length}, {self.n_sensors}], "
                             f"got {X.shape}")
        return X

    def _tensor(self, X):
        return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(self.device)

    # ---- training -------------------------------------------------------------------------
    def fit(self, X, y):
        if self.validation is None:
            raise ValueError("a validation set is required for early stopping and checkpoint selection")
        tr = self.training
        X, Xv = self._check_windows(X), self._check_windows(self.validation[0])
        y = np.asarray(y, dtype=np.float32) / self.target_scale
        yv = np.asarray(self.validation[1], dtype=np.float32) / self.target_scale
        if len(X) != len(y) or len(Xv) != len(yv):
            raise ValueError("windows and targets differ in length")

        self.net_ = net = self._build()
        if self.device != "mps":
            torch.use_deterministic_algorithms(True, warn_only=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=float(tr["learning_rate"]),
                                     weight_decay=float(tr.get("weight_decay", 0.0)))
        sched = tr.get("scheduler") or {}
        scheduler = None
        if sched.get("type", "none") == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=float(sched["factor"]), patience=int(sched["patience"]),
                min_lr=float(sched["min_lr"]))
        elif sched.get("type", "none") != "none":
            raise ValueError("scheduler.type must be 'reduce_on_plateau' or 'none'")

        Xt, yt, Xvt, yvt = self._tensor(X), self._tensor(y), self._tensor(Xv), self._tensor(yv)
        loss_fn = nn.MSELoss()
        generator = torch.Generator().manual_seed(self.seed)
        batch, clip = int(tr["batch_size"]), float(tr["grad_clip"])
        patience, delta = int(tr["early_stopping_patience"]), float(tr["min_delta"])
        best_loss, best_epoch, best_state, stale = math.inf, 0, None, 0
        stop_reason, self.history_ = "max_epochs", []

        for epoch in range(1, int(tr["max_epochs"]) + 1):
            start = time.perf_counter()
            net.train()
            order = torch.randperm(len(Xt), generator=generator).to(self.device)
            total, norms, clipped, finite = 0.0, [], 0, True
            for i in range(0, len(order), batch):
                idx = order[i:i + batch]
                loss = loss_fn(net(Xt[idx]), yt[idx])
                optimizer.zero_grad()
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), clip))
                loss_value = float(loss.detach())
                if not (math.isfinite(loss_value) and math.isfinite(norm)):
                    finite = False
                    break
                norms.append(norm)
                clipped += norm > clip
                optimizer.step()
                total += loss_value * len(idx)
            if not finite:
                stop_reason = "non_finite_loss_or_gradient"
                self.history_.append({"epoch": epoch, "train_loss": float("nan"), "val_loss": float("nan"),
                                      "lr": optimizer.param_groups[0]["lr"], "grad_norm_mean": float("nan"),
                                      "grad_norm_max": float("nan"), "clip_fraction": float("nan"),
                                      "epoch_time_s": time.perf_counter() - start})
                break
            val_loss = self._loss(net, Xvt, yvt, loss_fn)
            self.history_.append({
                "epoch": epoch, "train_loss": total / len(Xt), "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"], "grad_norm_mean": float(np.mean(norms)),
                "grad_norm_max": float(np.max(norms)), "clip_fraction": clipped / len(norms),
                "epoch_time_s": time.perf_counter() - start})
            if scheduler is not None:
                scheduler.step(val_loss)        # validation information only
            if val_loss < best_loss - delta:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    stop_reason = "early_stopping"
                    break

        if best_state is not None:
            net.load_state_dict(best_state)     # restore the best validation checkpoint
        self.fit_summary_ = {"best_epoch": best_epoch, "best_val_loss": best_loss,
                             "epochs_run": len(self.history_), "stop_reason": stop_reason,
                             "stable": stop_reason != "non_finite_loss_or_gradient" and best_state is not None}
        return self

    @staticmethod
    def _loss(net, X, y, loss_fn, chunk=2048):
        net.eval()
        total = 0.0
        with torch.no_grad():
            for i in range(0, len(X), chunk):
                total += float(loss_fn(net(X[i:i + chunk]), y[i:i + chunk])) * len(X[i:i + chunk])
        return total / len(X)

    # ---- inference ------------------------------------------------------------------------
    def predict(self, X, chunk=2048):
        self._check_fitted()
        X = self._check_windows(X)
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), chunk):
                out.append(self.net_(self._tensor(X[i:i + chunk])).cpu().numpy())
        return np.concatenate(out).astype(np.float64) * self.target_scale

    # ---- benchmark interface --------------------------------------------------------------
    def n_parameters(self):
        net = self.net_ if self.net_ is not None else self._build()
        return count_parameters(net)

    def get_config(self):
        return {"type": self.cell, "hidden_size": self.hidden_size, "num_layers": self.num_layers,
                "dropout": self.dropout, "sequence_length": self.sequence_length,
                "n_sensors": self.n_sensors, "seed": self.seed, "device": self.device,
                "target_scale": self.target_scale, "training": self.training,
                "n_parameters": self.n_parameters(), **self.fit_summary_}

    def weights_sha256(self):
        self._check_fitted()
        return weights_digest(self.net_.state_dict())

    def save(self, directory):
        """Write the best-validation checkpoint and its config; returns the files that make up the
        model. The training history is written alongside but is not part of the model size."""
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        ckpt = directory / f"{self.cell}.pt"
        state = {k: v.detach().cpu() for k, v in self.net_.state_dict().items()}
        torch.save({"state_dict": state, "config": self.get_config()}, ckpt)
        config = directory / f"{self.cell}_config.json"
        config.write_text(json.dumps(self.get_config(), indent=2) + "\n")
        (directory / "training_history.json").write_text(json.dumps(self.history_, indent=2) + "\n")
        return [ckpt, config]

    @classmethod
    def load(cls, directory, name=None, device="cpu"):
        """Rebuild a fitted model from a checkpoint written by `save`."""
        directory = Path(directory)
        ckpt_path = next(directory.glob("*.pt"))
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        c = blob["config"]
        model = cls(name or c["type"], c["type"], hidden_size=c["hidden_size"], num_layers=c["num_layers"],
                    dropout=c["dropout"], sequence_length=c["sequence_length"], n_sensors=c["n_sensors"],
                    training=c["training"], seed=c["seed"], device=device, target_scale=c["target_scale"])
        model.net_ = model._build()
        model.net_.load_state_dict(blob["state_dict"])
        model.fit_summary_ = {k: c[k] for k in ("best_epoch", "best_val_loss", "epochs_run", "stop_reason", "stable")
                              if k in c}
        history = directory / "training_history.json"
        model.history_ = json.loads(history.read_text()) if history.exists() else []
        return model

    def _check_fitted(self):
        if self.net_ is None:
            raise RuntimeError(f"{self.name} is not fitted")
