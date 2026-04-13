"""Python wrapper for CPPJoules energy measurement via ctypes."""

import ctypes
import csv
import io
import os
from pathlib import Path

_LIB_DIR = Path(__file__).resolve().parent / "portability_solved_cpp" / "CPPJoules" / "build"
_WRAPPER_SO = _LIB_DIR / "libenergy_wrapper.so"
_JOULES_SO = _LIB_DIR / "libCPP_Joules.so"

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not _WRAPPER_SO.exists():
        raise FileNotFoundError(
            f"libenergy_wrapper.so not found at {_WRAPPER_SO}. "
            "Build CPPJoules first: cd portability_solved_cpp/CPPJoules/build && cmake .. && make"
        )
    # Ensure the library directory is in the loader search path
    lib_dir = str(_LIB_DIR)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + ld_path if ld_path else "")
    # Load the CPPJoules library first so symbols are available
    ctypes.CDLL(str(_JOULES_SO), mode=ctypes.RTLD_GLOBAL)
    _lib = ctypes.CDLL(str(_WRAPPER_SO))

    # Declare function signatures
    _lib.energy_tracker_new.restype = ctypes.c_void_p
    _lib.energy_tracker_new.argtypes = []
    _lib.energy_tracker_delete.restype = None
    _lib.energy_tracker_delete.argtypes = [ctypes.c_void_p]
    _lib.energy_tracker_start.restype = None
    _lib.energy_tracker_start.argtypes = [ctypes.c_void_p]
    _lib.energy_tracker_stop.restype = None
    _lib.energy_tracker_stop.argtypes = [ctypes.c_void_p]
    _lib.energy_tracker_calculate.restype = None
    _lib.energy_tracker_calculate.argtypes = [ctypes.c_void_p]
    _lib.energy_tracker_get_csv.restype = ctypes.c_void_p
    _lib.energy_tracker_get_csv.argtypes = [ctypes.c_void_p]
    _lib.energy_tracker_free_str.restype = None
    _lib.energy_tracker_free_str.argtypes = [ctypes.c_void_p]
    _lib.energy_tracker_save_csv.restype = None
    _lib.energy_tracker_save_csv.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    return _lib


class EnergyTracker:
    """Context-manager wrapper around CPPJoules EnergyTracker.

    Usage::

        tracker = EnergyTracker()
        tracker.start()
        # ... do work ...
        tracker.stop()
        results = tracker.results()   # dict[str, float]

    Or as a context manager::

        with EnergyTracker() as t:
            # ... do work ...
        print(t.results())
    """

    def __init__(self):
        lib = _load_lib()
        self._lib = lib
        self._handle = lib.energy_tracker_new()
        self._results = None

    def start(self):
        """Record starting energy counters."""
        self._lib.energy_tracker_start(self._handle)

    def stop(self):
        """Record ending energy counters and compute deltas."""
        self._lib.energy_tracker_stop(self._handle)
        self._lib.energy_tracker_calculate(self._handle)
        self._results = None  # invalidate cache

    def results(self) -> dict:
        """Return energy results as ``{domain: value}`` dict.

        Keys include ``'Time'`` (seconds) and energy domains like
        ``'package-0'``, ``'core-0'``, ``'dram-0'``, ``'nvidia_gpu_0'``
        (values in micro-joules for RAPL, milli-joules for NVML).
        """
        if self._results is not None:
            return dict(self._results)

        raw_ptr = self._lib.energy_tracker_get_csv(self._handle)
        if raw_ptr is None:
            return {}
        text = ctypes.string_at(raw_ptr).decode("utf-8", errors="replace")
        self._lib.energy_tracker_free_str(raw_ptr)

        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if len(rows) < 2:
            return {}
        headers = [h.strip() for h in rows[0] if h.strip()]
        values = [v.strip() for v in rows[1] if v.strip()]
        res = {}
        for h, v in zip(headers, values):
            try:
                res[h] = float(v)
            except ValueError:
                res[h] = v
        self._results = res
        return dict(res)

    def save_csv(self, path: str):
        """Save energy results to a CSV file."""
        self._lib.energy_tracker_save_csv(self._handle, path.encode("utf-8"))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.energy_tracker_delete(self._handle)
            self._handle = None

    def summary_str(self) -> str:
        """One-line summary of energy measurement results."""
        r = self.results()
        if not r:
            return "No energy data"
        time_s = r.get("Time", 0)
        parts = [f"Time={time_s:.3f}s"]
        for k, v in r.items():
            if k == "Time":
                continue
            if isinstance(v, float):
                # Convert micro-joules to joules for readability
                parts.append(f"{k}={v / 1e6:.4f}J")
            else:
                parts.append(f"{k}={v}")
        return "  ".join(parts)
