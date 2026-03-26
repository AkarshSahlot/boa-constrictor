"""
ATLAS jets round-trip helper

This script:
  1) Downloads the ATLAS HDF5 file from EOS (tries xrdcp, falls back to HTTPS)
  2) Extracts the 'jets' dataset to:
      - atlas.npz (compressed npz)
      - atlas.bin (raw bytes) + atlas.meta.json (shape/dtype metadata)
  3) Reconstructs an HDF5 file from a decompressed atlas.bin (+ meta)
  4) Compares the original and reconstructed HDF5 for equality of 'jets'

Example end-to-end:
  python download.py --all-steps \
    --src root://eospublic.cern.ch//eos/opendata/atlas/datascience/ATLAS-FTAG-2023-05/mc-flavtag-ttbar-large.h5

If you already have a decompressed 'atlas.bin' from an external compressor, you can run:
  python download.py --reconstruct --compare --bin atlas.bin --meta atlas.meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import urllib.parse
import urllib.request
from typing import Iterable, Tuple

import numpy as np

try:
    import h5py  # type: ignore
except Exception as e:  # pragma: no cover - surfaced at runtime
    print("h5py is required. Please pip install h5py.", file=sys.stderr)
    raise


EOS_ROOT = (
    "root://eospublic.cern.ch//eos/opendata/atlas/datascience/ATLAS-FTAG-2023-05/"
    "mc-flavtag-ttbar-small.h5"
)


def root_to_https(url: str) -> str:
    # Convert root://eospublic.cern.ch//path -> https://eospublic.cern.ch/path
    if url.startswith("root://eospublic.cern.ch//"):
        return "https://eospublic.cern.ch/" + url.split("//", 2)[-1]
    return url


def download_atlas_h5(src: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"[download] Already present: {dst}")
        return

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    xrdcp = shutil.which("xrdcp")
    if src.startswith("root://") and xrdcp is not None:
        # Use xrdcp when available
        import subprocess

        print(f"[download] Using xrdcp to fetch {src} -> {dst}")
        subprocess.check_call([xrdcp, "-f", src, dst])
        return

    # Fallback to HTTPS
    https_url = root_to_https(src)
    print(f"[download] Fetching via HTTPS: {https_url} -> {dst}")
    with urllib.request.urlopen(https_url) as r, open(dst, "wb") as f:
        # Stream with a basic progress indicator
        total = int(r.headers.get("Content-Length", 0))
        chunk = 1 << 20
        downloaded = 0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)
            if total:
                pct = 100.0 * downloaded / total
                print(f"\r[download] {downloaded}/{total} bytes ({pct:.1f}%)", end="")
        print()



def save_npz(arr: np.ndarray, out_npz: str) -> None:
    print(f"[npz] Writing {out_npz} (compressed)")
    np.savez_compressed(out_npz, jets=arr)


def save_bin(out_bin: str) -> None:
    print(f"[bin] Writing {out_bin}")
    meta_path = out_bin.replace(".bin", ".meta.json")
    # Stream 'jets' dataset from atlas.h5 to a raw binary file (row-major)
    with h5py.File("atlas.h5", "r") as h5:
        if "jets" not in h5:
            raise KeyError("Dataset 'jets' not found in atlas.h5")
        dset = h5["jets"]
        shape = tuple(dset.shape)
        dtype = dset.dtype
        dtype_descr = dtype.descr  # preserves structured field layout
        # Write raw bytes in chunks to avoid excessive memory usage
        with open(out_bin, "wb") as f:
            chunk_rows = max(1, 1 << 12)  # ~4096 rows per chunk
            n_rows = dset.shape[0] if dset.ndim >= 1 else 1
            for sl in iter_slices(n_rows, chunk_rows):
                chunk = np.ascontiguousarray(dset[sl])
                f.write(chunk.tobytes(order="C"))
    # Save sidecar metadata to reconstruct shape/dtype
    with open(meta_path, "w", encoding="utf-8") as m:
        json.dump({"shape": shape, "dtype": dtype.str, "dtype_descr": dtype_descr, "order": "C"}, m)



def reconstruct_h5_from_bin(bin_path: str, out_h5: str) -> None:
    # Load metadata for dtype/shape
    meta_path = bin_path.replace(".bin", ".meta.json")
    with open(meta_path, "r", encoding="utf-8") as m:
        meta = json.load(m)
    # Prefer structured dtype description when available; coerce to list-of-tuples
    def _coerce_dtype_descr(descr_obj):
        import ast as _ast
        if isinstance(descr_obj, str):
            # Sometimes stored as a string; parse safely
            descr_obj = _ast.literal_eval(descr_obj)
        if isinstance(descr_obj, (list, tuple)):
            coerced = []
            for elem in descr_obj:
                if isinstance(elem, (list, tuple)):
                    coerced.append(tuple(elem))
                elif isinstance(elem, str):
                    coerced.append(tuple(_ast.literal_eval(elem)))
                else:
                    raise TypeError(f"Unsupported dtype_descr element type: {type(elem)}")
            return coerced
        raise TypeError(f"Unsupported dtype_descr type: {type(descr_obj)}")

    if "dtype_descr" in meta:
        dtype = np.dtype(_coerce_dtype_descr(meta["dtype_descr"]))
    else:
        dtype = np.dtype(meta["dtype"])
    shape = tuple(meta["shape"])
    count = int(np.prod(shape))

    # Reconstruct array from raw bytes
    arr = np.fromfile(bin_path, dtype=dtype, count=count).reshape(shape)

    # Write to HDF5 as dataset 'jets'
    print(f"[reconstruct] Writing {out_h5}")
    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("jets", data=arr, chunks=True, compression="gzip", compression_opts=4)


def iter_slices(n_rows: int, chunk_rows: int) -> Iterable[slice]:
    for start in range(0, n_rows, chunk_rows):
        end = min(n_rows, start + chunk_rows)
        yield slice(start, end)


def compare_h5_jets(a_path: str, b_path: str, atol: float = 0.0, rtol: float = 0.0) -> bool:
    """Compare dataset 'jets' between two HDF5 files. Returns True if equal."""
    with h5py.File(a_path, "r") as fa, h5py.File(b_path, "r") as fb:
        if "jets" not in fa or "jets" not in fb:
            raise KeyError("Both files must contain a 'jets' dataset")
        da, db = fa["jets"], fb["jets"]
        if da.shape != db.shape or da.dtype != db.dtype:
            print("[compare] Shape or dtype mismatch:", da.shape, da.dtype, "vs", db.shape, db.dtype)
            return False

        # Chunked comparison along first axis to avoid high memory
        n_rows = da.shape[0] if da.ndim >= 1 else 1
        chunk_rows = max(1, 1 << 12)  # ~4096 rows per chunk
        total_chunks = math.ceil(n_rows / chunk_rows)
        for i, sl in enumerate(iter_slices(n_rows, chunk_rows), 1):
            a = da[sl]
            b = db[sl]
            # Handle structured vs plain numeric arrays
            if a.dtype.fields:  # structured dtype
                for name, (fld_dtype, _) in a.dtype.fields.items():
                    av = a[name]
                    bv = b[name]
                    kind = np.dtype(fld_dtype).kind
                    if kind in ("f",):
                        eq = np.allclose(av, bv, atol=atol, rtol=rtol, equal_nan=True)
                    else:
                        eq = np.array_equal(av, bv)
                    if not eq:
                        print(
                            f"[compare] Field '{name}' mismatch in rows {sl.start}:{sl.stop} (chunk {i}/{total_chunks})"
                        )
                        return False
            else:
                if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
                    print(f"[compare] Mismatch found in rows {sl.start}:{sl.stop} (chunk {i}/{total_chunks})")
                    return False
            if i % 50 == 0 or i == total_chunks:
                print(f"[compare] {i}/{total_chunks} chunks OK")
        print("[compare] All chunks equal")
        return True

def save_200m(out_bin: str) -> None:
    limit_bytes = 200 * 1024 * 1024
    base, ext = os.path.splitext("atlas.bin")
    trunc_path = f"{base}_200m{ext}"
    bufsize = 1024 * 1024
    remaining = limit_bytes
    with open("atlas.bin", "rb") as src, open(out_bin, "wb") as dst:
        while remaining > 0:
            chunk = src.read(min(bufsize, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)


def extract_colmajor(h5_path: str, meta_path: str, out_bin: str, out_200m: str) -> None:
    """Extract column-major binary from ATLAS HDF5 jets dataset.

    Each field's values are written contiguously across all jets, giving a
    column-oriented layout. Also creates a 200 MB training chunk.
    """
    with open(meta_path) as f:
        meta = json.load(f)
    fields = meta["dtype_descr"]
    n_jets = meta["shape"][0]

    print(f"[col-major] ATLAS jets: {n_jets:,} jets, {len(fields)} fields")

    with h5py.File(h5_path, "r") as hf:
        ds = hf["jets"]
        total_bytes = 0
        with open(out_bin, "wb") as f_out:
            for i, (name, dt) in enumerate(fields):
                col = np.array(ds[name])
                raw = col.tobytes()
                f_out.write(raw)
                total_bytes += len(raw)
                print(f"  [{i:2d}] {name:45s}  {dt}  {col.dtype.itemsize}B × {len(col):,} = {len(raw):,} bytes")

    print(f"[col-major] Wrote {out_bin}: {total_bytes:,} bytes ({total_bytes/1e6:.1f} MB)")

    # Create shuffled 200 MB training sample
    rng = np.random.default_rng(42)
    bytes_per_jet = sum(np.dtype(dt).itemsize for _, dt in fields)
    target_jets = 200_000_000 // bytes_per_jet
    indices = rng.permutation(n_jets)[:target_jets]

    with h5py.File(h5_path, "r") as hf:
        ds = hf["jets"]
        with open(out_200m, "wb") as f_out:
            for name, dt in fields:
                col = np.array(ds[name])[indices]
                f_out.write(col.tobytes())

    print(f"[col-major] Training chunk: {out_200m} ({os.path.getsize(out_200m)/1048576:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ATLAS jets round-trip helper")
    p.add_argument("--src", default=EOS_ROOT, help="Source HDF5 (root:// or https:// or local path)")
    p.add_argument("--h5", default="atlas.h5", help="Local HDF5 output filename")
    p.add_argument("--npz", default="atlas.npz", help="NPZ filename to write")
    p.add_argument("--bin", dest="bin_path", default="atlas.bin", help="BIN filename to write/read")
    p.add_argument("--meta", default="atlas.meta.json", help="Metadata JSON filename")
    p.add_argument("--recon-h5", default="atlas_reconstructed.h5", help="Reconstructed HDF5 output filename")
    p.add_argument("--download", action="store_true", help="Download the HDF5 if missing")
    p.add_argument("--extract", action="store_true", help="Extract 'jets' to BIN+META (row-major)")
    p.add_argument("--extract-colmajor", action="store_true", help="Also extract column-major binary + 200 MB chunk")
    p.add_argument("--reconstruct", action="store_true", help="Reconstruct HDF5 from BIN+META")
    p.add_argument("--compare", action="store_true", help="Compare original HDF5 vs reconstructed")
    p.add_argument("--all-steps", action="store_true", help="Run download, extract (row+col), reconstruct, compare")
    args = p.parse_args(argv)

    if args.all_steps:
        args.download = True
        args.extract = True
        args.extract_colmajor = True
        args.reconstruct = True
        args.compare = True

    if args.download:
        src = args.src
        # If src is a local path, just copy
        if os.path.exists(src):
            print(f"[download] Copying local file {src} -> {args.h5}")
            shutil.copy2(src, args.h5)
        else:
            download_atlas_h5(src, args.h5)

    if args.extract:
        # save_npz(arr, args.npz)
        save_bin(args.bin_path)
        save_200m(args.bin_path.replace(".bin", "_200m.bin"))

    if args.extract_colmajor:
        extract_colmajor(
            args.h5, args.meta,
            "atlas_colmajor.bin", "atlas_colmajor_200m.bin",
        )

    if args.reconstruct:
        reconstruct_h5_from_bin(args.bin_path, args.recon_h5)

    ok = True
    if args.compare:
        ok = compare_h5_jets(args.h5, args.recon_h5)

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
