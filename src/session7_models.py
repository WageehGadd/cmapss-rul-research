"""Build each frozen architecture for a given sensor subset. Only the number of input sensors
varies; hyperparameters, training protocol and seed come from the frozen definitions."""
from .classical_models import build_model
from .convolutional_models import ConvolutionalRegressor
from .recurrent_models import RecurrentRegressor
from .session7_plan import CLASSICAL
from .transformer_models import build_from_definition


def build_condition_model(model, sources, data, *, execution, device="cpu"):
    """A fresh, unfitted model for `data` (already restricted to the condition's sensors).
    Deep models get the validation split of `data` for early stopping, as in their sessions."""
    n_sensors, length = len(data.sensors), data.sequence_length
    if model in CLASSICAL:
        d = sources["selected_models"]["models"][model]
        return build_model(model, d["params"], sensors=data.sensors, features=d["temporal_features"],
                           sequence_length=length, seed=d["seed"], name=model,
                           n_jobs=execution["classical_n_jobs_train"],
                           predict_n_jobs=execution["classical_n_jobs_predict"])
    definition = sources[model]
    validation = (data.X["validation"], data.y["validation"])
    if model == "transformer":
        return build_from_definition(definition, n_sensors=n_sensors, validation=validation, name="transformer",
                                     device=device)
    arch, training = definition["architecture"], dict(definition["training"])
    seed = training.pop("seed")
    training.pop("device")
    common = dict(dropout=arch["dropout"], sequence_length=arch["sequence_length"], n_sensors=n_sensors,
                  training=training, seed=seed, device=device, validation=validation,
                  target_scale=float(training["target_scale"]))
    if arch["sequence_length"] != length:
        raise ValueError("sequence length differs from the frozen definition")
    if model in ("lstm", "gru"):
        return RecurrentRegressor(model, model, hidden_size=arch["hidden_size"], num_layers=arch["num_layers"],
                                  **common)
    if model == "tcn":
        return ConvolutionalRegressor("tcn", "tcn", channels=arch["channels"], kernel_size=arch["kernel_size"],
                                      dilations=arch["dilations"], **common)
    raise ValueError(f"unknown model '{model}'")
