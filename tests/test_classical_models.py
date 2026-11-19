import numpy as np
import pytest

from src.classical_models import build_model
from src.evaluation import evaluate_model
from src.tabular_features import FEATURES
from src.tuning import expand_grid, search, select
from tests.conftest import SENSORS, SEQ_LEN, synthetic_data

TIMING = {"warmup_runs": 1, "timed_runs": 2, "batch_size": 16}
RF = {"n_estimators": 5, "max_depth": 4, "min_samples_leaf": 3, "max_features": 0.5}
GB = {"max_iter": 10, "learning_rate": 0.1, "max_depth": 3}


def make(family, params):
    return build_model(family, params, sensors=SENSORS, features=FEATURES, sequence_length=SEQ_LEN, seed=1)


@pytest.mark.parametrize("family,params", [("random_forest", RF), ("gradient_boosting", GB)])
def test_wrappers_work_with_the_session2_evaluator(family, params, tmp_path):
    data = synthetic_data()
    model = make(family, params).fit(data.X["train"], data.y["train"])
    result = evaluate_model(model, {"validation": (data.X["validation"], data.y["validation"]),
                                    "test": (data.X["test"], data.y["test"])},
                            data.X["validation"], target_type="capped", train_time_s=1.0,
                            timing_config=TIMING, artifact_dir=tmp_path / "m")
    assert result.model_name == family
    assert set(result.metrics["validation"]) == {"rmse", "mae", "r2", "nasa_score"}
    assert result.predictions["test"].shape == (40,)
    assert result.model_size_bytes > 0 and result.n_parameters and result.n_parameters > 0
    assert (tmp_path / "m" / f"{family}.joblib").exists()
    assert model.get_config()["n_features"] == 84


def test_same_seed_gives_identical_predictions():
    data = synthetic_data()
    for family, params in (("random_forest", RF), ("gradient_boosting", {**GB, "max_features": 0.5})):
        a = make(family, params).fit(data.X["train"], data.y["train"]).predict(data.X["test"])
        b = make(family, params).fit(data.X["train"], data.y["train"]).predict(data.X["test"])
        np.testing.assert_array_equal(a, b)


def test_unfitted_model_and_unknown_family_raise():
    with pytest.raises(RuntimeError, match="not fitted"):
        make("random_forest", RF).predict(np.zeros((2, SEQ_LEN, 14), dtype="float32"))
    with pytest.raises(ValueError, match="unknown model family"):
        make("svm", {})


def test_grid_expansion_is_ordered_and_complete():
    grid = expand_grid({"a": [1, 2], "b": [10, 20, 30]})
    assert len(grid) == 6 and grid[0] == {"a": 1, "b": 10} and grid[-1] == {"a": 2, "b": 30}


def _records(rmses, nodes):
    return [{"candidate": i, "params": {}, "validation": {"rmse": r}, "n_parameters": n, "fit_time_s": 0}
            for i, (r, n) in enumerate(zip(rmses, nodes))]


def test_selection_prefers_lowest_error_then_smaller_model_within_tolerance():
    records = _records([10.0, 10.05, 12.0], [900, 100, 50])
    assert select(records, "rmse", 0.0)[0]["candidate"] == 0
    chosen, rule = select(records, "rmse", 0.01)             # 10.05 is within 1% of 10.0
    assert chosen["candidate"] == 1 and rule["n_equivalent"] == 2


def test_search_and_selection_use_only_training_and_validation_data():
    """Corrupting everything that is not train/validation cannot change the outcome."""
    data = synthetic_data()
    from src.tabular_features import WindowFeatureTransformer
    t = WindowFeatureTransformer(SENSORS, FEATURES, SEQ_LEN)
    F_tr, F_va = t.transform(data.X["train"]), t.transform(data.X["validation"])

    def build(params):
        return make("random_forest", params)

    grid = {"n_estimators": [3, 6], "max_depth": [2, 4]}
    first = search(build, grid, F_tr, data.y["train"], F_va, data.y["validation"])
    data.X["test"][:] = np.nan
    data.y["test"][:] = np.nan
    second = search(build, grid, F_tr, data.y["train"], F_va, data.y["validation"])
    assert [r["validation"] for r in first] == [r["validation"] for r in second]
    assert select(first)[0]["candidate"] == select(second)[0]["candidate"]
