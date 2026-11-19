"""Writers for the Session 3 artifacts: rankings, comparison, subsets, manifest, report, figures.

The frozen sensor definitions go to a version-controlled directory and contain no timestamps
or timings, so identical inputs reproduce byte-identical files.
"""
import csv
import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = 1


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_frozen_artifacts(frozen_dir, *, ranking_a, ranking_b, comparison, topk, randomk, features,
                           metadata=None):
    """Write the versioned ranking, comparison and subset files; returns {name: path}."""
    frozen_dir = Path(frozen_dir)
    written = {}

    path = frozen_dir / "sensor_ranking_correlation.csv"
    write_csv(path, [{"sensor": r["sensor"], "correlation": repr(r["correlation"]),
                      "abs_correlation": repr(r["abs_correlation"]), "rank": r["rank"]} for r in ranking_a],
              ["sensor", "correlation", "abs_correlation", "rank"])
    written[path.name] = path

    path = frozen_dir / "sensor_ranking_permutation.csv"
    columns = ["sensor", "importance", *[f"importance_{f}" for f in features], "rank"]
    rows = [{"sensor": r["sensor"], "importance": repr(r["importance"]),
             **{f"importance_{f}": repr(r["components"][f]) for f in features}, "rank": r["rank"]}
            for r in ranking_b]
    write_csv(path, rows, columns)
    written[path.name] = path

    for name, content in (("sensor_ranking_comparison.json", comparison),
                          ("sensor_subsets_topk.json", topk),
                          ("sensor_subsets_randomk.json", randomk)):
        path = frozen_dir / name
        write_json(path, content)
        written[name] = path

    return written


def selected_models_document(selected, *, features, seed, metadata):
    """Frozen definition of the already-selected full-sensor models.

    selected: {family: {params, candidate, rule, validation, n_parameters}} (a record of the
    completed validation-only selection; nothing is recomputed here).
    """
    models = {}
    for family, info in selected.items():
        rule = info["rule"]
        models[family] = {
            "estimator": ESTIMATOR_CLASS[family],
            "params": info["params"],
            "seed": int(seed),
            "temporal_features": list(features),
            "selected_candidate": info["candidate"],
            "selection": {"metric": rule["metric"], "tie_tolerance_rel": rule["tie_tolerance_rel"],
                          "tiebreak": rule["tiebreak"],
                          "best_validation_value": rule["best_validation_value"],
                          "n_equivalent": rule["n_equivalent"], "data": "validation only"},
            "validation_metrics": info["validation"],
            "n_parameters": info["n_parameters"],
        }
    return {"schema_version": SCHEMA_VERSION, **metadata, "models": models}


ESTIMATOR_CLASS = {"random_forest": "sklearn.ensemble.RandomForestRegressor",
                   "gradient_boosting": "sklearn.ensemble.HistGradientBoostingRegressor"}


def session1_identity(session1_cfg, *, sensors, sequence_length, rul_cap, target_type, root):
    """Hashes and metadata that pin the exact Session 1 foundation (no data is copied)."""
    root = Path(root)
    processed = (root / session1_cfg["processed_dir"]) if not Path(session1_cfg["processed_dir"]).is_absolute() \
        else Path(session1_cfg["processed_dir"])
    config = root / session1_cfg["config"] if not Path(session1_cfg["config"]).is_absolute() \
        else Path(session1_cfg["config"])
    split = json.loads((processed / "split.json").read_text())
    units = lambda ids: hashlib.sha256(json.dumps(sorted(int(i) for i in ids)).encode()).hexdigest()
    return {
        "config": session1_cfg["config"], "config_sha256": file_sha256(config),
        "processed_dir": session1_cfg["processed_dir"],
        "manifest_sha256": file_sha256(processed / "manifest.json"),
        "split_sha256": file_sha256(processed / "split.json"),
        "split_seed": split["seed"],
        "n_train_engines": len(split["train_units"]), "n_validation_engines": len(split["val_units"]),
        "train_units_sha256": units(split["train_units"]), "validation_units_sha256": units(split["val_units"]),
        "sensors": list(sensors), "sequence_length": int(sequence_length),
        "rul_cap": int(rul_cap), "target_type": target_type,
    }


def write_manifest(frozen_dir, metadata, provenance):
    """manifest.json: SHA-256 of every other file in the frozen directory plus identity fields.

    `provenance` (producing run ID, Git state) is run-specific and kept out of the data files,
    so those stay byte-identical across reruns; only the manifest records it.
    """
    frozen_dir = Path(frozen_dir)
    files = {p.name: file_sha256(p) for p in sorted(frozen_dir.iterdir())
             if p.is_file() and p.name != "manifest.json"}
    manifest = {"schema_version": SCHEMA_VERSION, **metadata, **provenance, "files": files}
    write_json(frozen_dir / "manifest.json", manifest)
    return manifest


def markdown_table(rows, columns, formats=None):
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        cells = [format(row[c], formats[c]) if c in formats else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
