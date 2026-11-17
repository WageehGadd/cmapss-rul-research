"""Shared timing utilities (wall clock, time.perf_counter).

Training time
-------------
One wall-clock measurement of `model.fit(X_train, y_train)`.

Inference protocol
------------------
`predict_fn(batch)` is called `warmup_runs` times (not recorded) and then `timed_runs` times,
each call timed individually. One call processes one batch of `batch_size` samples.
  * latency per batch  = median of the timed calls, in milliseconds
  * latency per sample = latency per batch / batch_size
  * throughput         = batch_size / (median latency per batch in seconds), samples per second
The 95th percentile and minimum per-batch latency are reported alongside the median.
"""
import time

import numpy as np

LATENCY_UNIT = "ms"


def time_training(model, X_train, y_train):
    """Wall-clock seconds taken by one `model.fit(X_train, y_train)` call."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return time.perf_counter() - start


def measure_inference(predict_fn, batch, warmup_runs, timed_runs):
    """Time `predict_fn(batch)` following the protocol in the module docstring."""
    batch_size = len(batch)
    if batch_size < 1:
        raise ValueError("the timing batch must contain at least one sample")
    if warmup_runs < 0 or timed_runs < 1:
        raise ValueError("warmup_runs must be >= 0 and timed_runs must be >= 1")

    for _ in range(warmup_runs):
        predict_fn(batch)
    seconds = np.empty(timed_runs)
    for i in range(timed_runs):
        start = time.perf_counter()
        predict_fn(batch)
        seconds[i] = time.perf_counter() - start

    median_s = float(np.median(seconds))
    return {
        "latency_ms_per_batch": median_s * 1000.0,
        "latency_ms_per_sample": median_s * 1000.0 / batch_size,
        "throughput_samples_per_s": batch_size / median_s,
        "latency_ms_per_batch_p95": float(np.percentile(seconds, 95)) * 1000.0,
        "latency_ms_per_batch_min": float(seconds.min()) * 1000.0,
        "batch_size": batch_size,
        "warmup_runs": int(warmup_runs),
        "timed_runs": int(timed_runs),
        "aggregate": "median",
        "latency_unit": LATENCY_UNIT,
    }
