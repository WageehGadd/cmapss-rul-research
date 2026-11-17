"""Capture the hardware / software environment of a benchmark run.

Fields that cannot be determined reliably are recorded as "unavailable"; nothing is guessed.
"""
import os
import platform
import subprocess
import sys
from importlib import metadata

UNAVAILABLE = "unavailable"


def _run(cmd, cwd=None):
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _sysctl(key):
    return _run(["sysctl", "-n", key]) if sys.platform == "darwin" else None


def _cpu_brand():
    if sys.platform == "darwin":
        return _sysctl("machdep.cpu.brand_string") or UNAVAILABLE
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return UNAVAILABLE


def _memory_bytes():
    if sys.platform == "darwin":
        value = _sysctl("hw.memsize")
        return int(value) if value and value.isdigit() else UNAVAILABLE
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            pass
    return UNAVAILABLE


def _physical_cores():
    value = _sysctl("hw.physicalcpu")
    return int(value) if value and value.isdigit() else UNAVAILABLE


def git_info(root):
    """Current commit SHA and whether the working tree has uncommitted changes."""
    commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
    if commit is None:
        return {"git_commit": UNAVAILABLE, "git_dirty": UNAVAILABLE}
    status = _run(["git", "status", "--porcelain"], cwd=root)
    return {"git_commit": commit, "git_dirty": bool(status)}


def library_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = UNAVAILABLE
    return versions


def hardware_description(hardware):
    """One-line summary built only from fields that were determined."""
    parts = []
    if hardware["cpu_brand"] != UNAVAILABLE:
        parts.append(hardware["cpu_brand"])
    if hardware["logical_cpu_count"] != UNAVAILABLE:
        cores = hardware["physical_cpu_count"]
        parts.append(f"{cores}/{hardware['logical_cpu_count']} physical/logical cores"
                     if cores != UNAVAILABLE else f"{hardware['logical_cpu_count']} logical cores")
    if hardware["memory_bytes"] != UNAVAILABLE:
        parts.append(f"{hardware['memory_bytes'] / 2**30:.0f} GiB RAM")
    return ", ".join(parts) if parts else UNAVAILABLE


def capture_environment(root, library_names, session_configs, hardware_note="auto"):
    """Environment metadata for one benchmark run.

    session_configs maps a label (e.g. "session1_config") to a reference string.
    hardware_note is "auto" or a free-text description supplied in the config.
    """
    hardware = {
        "cpu_brand": _cpu_brand(),
        "physical_cpu_count": _physical_cores(),
        "logical_cpu_count": os.cpu_count() or UNAVAILABLE,
        "memory_bytes": _memory_bytes(),
        "machine": platform.machine() or UNAVAILABLE,
        "processor": platform.processor() or UNAVAILABLE,
    }
    hardware["description"] = hardware_description(hardware)
    if hardware_note != "auto":
        hardware["config_note"] = hardware_note
    return {
        "os": {
            "system": platform.system() or UNAVAILABLE,
            "release": platform.release() or UNAVAILABLE,
            "version": platform.version() or UNAVAILABLE,
            "platform": platform.platform() or UNAVAILABLE,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "hardware": hardware,
        "libraries": library_versions(library_names),
        "thread_env": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                  "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
                       if k in os.environ},
        "git": git_info(root),
        "configs": dict(session_configs),
    }
