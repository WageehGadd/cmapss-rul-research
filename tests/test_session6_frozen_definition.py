"""The real frozen Transformer definition (artifacts/session6/selected_transformer.json).

This file is version-controlled once the canonical Session 6 run has generated it. Until then the
tests are skipped, so the suite never depends on untracked verification output being present.
Nothing here is hard-coded from the config: every expectation is read from the artifact itself.
"""
import json

import numpy as np
import pytest
import torch

from src.benchmark_config import PROJECT_ROOT
from src.transformer_models import ARCHITECTURE_KEYS, build_from_definition, validate_architecture

DEFINITION = PROJECT_ROOT / "artifacts/session6/selected_transformer.json"
SENSOR_COUNTS = (14, 10, 7, 4, 2)
pytestmark = pytest.mark.skipif(not DEFINITION.exists(), reason="canonical Session 6 definition not generated yet")


@pytest.fixture(scope="module")
def definition():
    return json.loads(DEFINITION.read_text())


def built(definition, k):
    model = build_from_definition(definition, n_sensors=k, device="cpu")
    return model, model._build().eval()


def test_schema(definition):
    assert definition["schema_version"] == 1 and definition["family"] == "transformer"
    arch = definition["architecture"]
    for key in (*ARCHITECTURE_KEYS, "ff_dim", "positional_encoding", "normalization", "attention_mask",
                "sequence_length", "n_sensors"):
        assert key in arch, key
    validate_architecture(**{k: arch[k] for k in ARCHITECTURE_KEYS}, sequence_length=arch["sequence_length"],
                          n_sensors=arch["n_sensors"])
    for key in ("learning_rate", "batch_size", "max_epochs", "min_delta", "seed", "device", "target_scale"):
        assert key in definition["training"], key
    assert definition["selection"]["data"] == "validation only"
    assert definition["n_parameters"] > 0 and definition["checkpoint"]["weights_sha256"]


@pytest.mark.parametrize("k", SENSOR_COUNTS)
def test_rebuild_for_each_sensor_budget_keeps_everything_but_the_projection(definition, k):
    arch = definition["architecture"]
    model, net = built(definition, k)
    with torch.no_grad():
        out = net(torch.from_numpy(np.random.default_rng(0).normal(size=(3, arch["sequence_length"], k)).astype("float32")))
    assert tuple(out.shape) == (3,) and bool(torch.isfinite(out).all())
    rebuilt = model.architecture()
    assert {key: v for key, v in rebuilt.items() if key != "n_sensors"} == \
        {key: v for key, v in arch.items() if key != "n_sensors"}
    assert net.input_projection.in_features == k
    _, full = built(definition, arch["n_sensors"])
    others = lambda n: {name: tuple(p.shape) for name, p in n.named_parameters()
                        if not name.startswith("input_projection")}
    assert others(net) == others(full)
    count = lambda n: sum(p.numel() for p in n.parameters())
    assert count(full) == definition["n_parameters"]
    assert count(full) - count(net) == (arch["n_sensors"] - k) * arch["d_model"]    # projection width only


def test_frozen_weights_load_into_the_rebuilt_14_sensor_model(definition):
    checkpoint = PROJECT_ROOT / definition["checkpoint"]["path"]
    if not checkpoint.exists():
        pytest.skip("checkpoint binaries are not version-controlled")
    _, net = built(definition, definition["architecture"]["n_sensors"])
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
