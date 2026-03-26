"""
Download a CMS NANOAOD file via the CERN OpenData index, extract events,
and produce two ~6 GB compressible binary representations plus 200 MB
training chunks:

  1) Row-major:  cms_rowmajor.bin  (~6 GB) + cms_rowmajor_200m.bin  (200 MB)
     Events written in chunks; each event stores all branches sequentially
     (counts + values for jagged, native dtype for scalars).

  2) Col-major:  cms_colmajor.bin  (~6 GB) + cms_colmajor_200m.bin  (200 MB)
     Each branch stored contiguously across all events (counts then values
     for jagged branches).

Additionally creates the original padded float64 row-major matrix
(cms_50k_padded.bin) for backward compatibility.

Usage:
  python download.py --out-dir ./ --extract-formats

Artifacts written to out-dir:
  - cms.root: downloaded ROOT file
  - cms_50k_padded.bin: padded float64 matrix (legacy, 50k events)
  - cms_rowmajor.bin + .meta.json: full row-major binary (~6 GB)
  - cms_rowmajor_200m.bin: 200 MB training chunk
  - cms_colmajor.bin + .meta.json: full col-major binary (~6 GB)
  - cms_colmajor_200m.bin: 200 MB training chunk
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import awkward as ak
import uproot

import time

# ---------------------------
# Configuration and constants
# ---------------------------

DEFAULT_INDEX_URL = (
	"https://opendata.cern.ch/record/30525/files/CMS_Run2016G_JetHT_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_260000_file_index.json_12"
)


@dataclass
class BranchMeta:
	name: str
	is_list: bool
	max_len: int
	dtype: str = "float64"
	col_offset: int = 0  # start column in the flat matrix


@dataclass
class BinMeta:
	n_events: int
	tree_key: str
	branches: List[BranchMeta]
	# For list-like branches, store per-event lengths
	lengths: Dict[str, List[int]]

	def to_json(self) -> str:
		return json.dumps(
			{
				"n_events": self.n_events,
				"tree_key": self.tree_key,
				"branches": [asdict(b) for b in self.branches],
				"lengths": self.lengths,
			},
			indent=2,
		)

	@staticmethod
	def from_json(s: str) -> "BinMeta":
		obj = json.loads(s)
		branches = [BranchMeta(**b) for b in obj["branches"]]
		return BinMeta(
			n_events=int(obj["n_events"]),
			tree_key=str(obj["tree_key"]),
			branches=branches,
			lengths={k: list(v) for k, v in obj["lengths"].items()},
		)


# ---------------------------
# Helpers
# ---------------------------

def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def download_file(url: str, dest_path: str, chunk_size: int = 1 << 20, timeout: float = 60.0) -> None:
	"""Download a file over HTTP(S) to dest_path with a simple chunked reader."""
	import urllib.request
	request = urllib.request.Request(url, headers={"User-Agent": "boa-constrictor/cms-lg"})
	with urllib.request.urlopen(request, timeout=timeout) as resp, open(dest_path, "wb") as out:
		while True:
			chunk = resp.read(chunk_size)
			if not chunk:
				break
			out.write(chunk)


def open_tree(file_url_or_path: str) -> Tuple[uproot.reading.ReadOnlyDirectory, str, uproot.behaviors.TBranch.TTree]:
	"""Open a ROOT file (HTTP/xrootd/local), return (file, tree_key, tree). Prefer 'Events' tree."""
	f = uproot.open(file_url_or_path)
	# Pick Events if present
	tree_key = None
	for k, cls in f.classnames().items():
		if cls == "TTree" and k.split(";")[0].lower() == "events":
			tree_key = k
			break
	if tree_key is None:
		# first TTree
		for k, cls in f.classnames().items():
			if cls == "TTree":
				tree_key = k
				break
	if tree_key is None:
		raise RuntimeError("No TTree found in ROOT file")
	tree = f[tree_key]
	return f, tree_key, tree


def select_numeric_branches(arrs: ak.Array) -> List[str]:
	"""Select branches that are primitives or lists of primitives, convertible to float64."""
	selected: List[str] = []
	for name in arrs.fields:
		a = arrs[name]
		try:
			_ = ak.to_numpy(ak.ravel(a))  # check convertibility
			selected.append(name)
		except Exception:
			# skip nested records/objects
			pass
	return selected


def encode_to_bin(arrs: ak.Array, selected: List[str]) -> Tuple[np.ndarray, BinMeta]:
	"""Pad selected branches and stack into 2D float64 matrix; build metadata with lengths."""
	n_events = len(arrs)
	cols: List[np.ndarray] = []
	branches: List[BranchMeta] = []
	lengths: Dict[str, List[int]] = {}

	col_offset = 0
	for name in selected:
		a = arrs[name]
		# Detect list-like vs scalar
		is_list = isinstance(getattr(ak.type(a), "content", None), ak.types.ListType)
		max_len = 1
		per_lengths: Optional[np.ndarray] = None
		if is_list:
			per_lengths = ak.to_numpy(ak.num(a, axis=-1)).astype(np.int32)
			max_len = int(per_lengths.max()) if per_lengths.size > 0 else 0
		else:
			max_len = 1

		if is_list:
			# pad/clip to max_len, fill missing with 0, shape (N, L)
			a_pad = ak.pad_none(a, max_len, axis=1, clip=True)
			a_fill = ak.fill_none(a_pad, 0)
			np_arr = ak.to_numpy(a_fill).astype(np.float64)
			lengths[name] = per_lengths.astype(int).tolist()
		else:
			np_arr = ak.to_numpy(a).astype(np.float64).reshape(-1, 1)
			lengths[name] = []  # placeholder

		cols.append(np_arr)
		branches.append(BranchMeta(name=name, is_list=is_list, max_len=int(max_len), col_offset=col_offset))
		col_offset += np_arr.shape[1]

	data = np.concatenate(cols, axis=1).astype(np.float64)
	meta = BinMeta(
		n_events=int(n_events),
		tree_key="Events",
		branches=branches,
		lengths=lengths,
	)
	return data, meta


def write_bin_and_meta(bin_path: str, meta_path: str, data: np.ndarray, meta: BinMeta) -> None:
	data.tofile(bin_path)
	with open(meta_path, "w", encoding="utf-8") as f:
		f.write(meta.to_json())


def read_bin_and_meta(bin_path: str, meta_path: str) -> Tuple[np.ndarray, BinMeta]:
	with open(meta_path, "r", encoding="utf-8") as f:
		meta = BinMeta.from_json(f.read())
	total_cols = sum(b.max_len for b in meta.branches)
	data = np.fromfile(bin_path, dtype=np.float64)
	if meta.n_events == 0 or total_cols == 0:
		data = data.reshape(meta.n_events, total_cols)
	else:
		if data.size != meta.n_events * total_cols:
			raise RuntimeError(
				f"Unexpected bin size: got {data.size}, expected {meta.n_events * total_cols} elements"
			)
		data = data.reshape(meta.n_events, total_cols)
	return data, meta


def reconstruct_awkward(data: np.ndarray, meta: BinMeta) -> ak.Array:
	n = meta.n_events
	fields: Dict[str, ak.Array] = {}
	for b in meta.branches:
		cols = slice(b.col_offset, b.col_offset + (b.max_len if b.is_list else 1))
		block = data[:, cols]
		if b.is_list:
			lens = np.array(meta.lengths[b.name], dtype=np.int32)
			clipped = np.minimum(lens, b.max_len)
			# Build Python list-of-lists to avoid depending on internal layouts
			lol: List[List[float]] = []
			for i in range(n):
				Li = int(clipped[i])
				if Li > 0:
					lol.append(block[i, :Li].tolist())
				else:
					lol.append([])
			arr = ak.Array(lol, with_name=None)
		else:
			arr = ak.Array(block[:, 0])
		fields[b.name] = arr
	# Zip only at the top-level (per-event) to avoid broadcasting nested lists across branches
	return ak.zip(fields, depth_limit=1)


def write_root_from_awkward(path: str, tree_name: str, arrs: ak.Array) -> None:
	with uproot.recreate(path) as f:
		# Uproot prefers receiving a dict of branch-name -> array mappings.
		# If `arrs` is an awkward array of records, convert to a dict so fields
		# are written individually. This avoids broadcasting/nesting issues
		# when a record's fields are jagged arrays.
		try:
			if isinstance(arrs, ak.Array) and getattr(arrs, "fields", None):
				data = {name: arrs[name] for name in arrs.fields}
			else:
				data = arrs
		except Exception:
			data = arrs
		f[tree_name] = data


def compare_trees(original: ak.Array, reconstructed: ak.Array, selected: List[str]) -> Tuple[bool, Dict[str, str]]:
	ok = True
	report: Dict[str, str] = {}
	for name in selected:
		a = original[name]
		b = reconstructed[name]
		try:
			# detect list-like
			la = ak.to_numpy(ak.num(a, axis=-1))
			lb = ak.to_numpy(ak.num(b, axis=-1))
			if not np.array_equal(la, lb):
				ok = False
				report[name] = "lengths differ"
				continue
			# compare content up to per-event length
			max_len = int(la.max()) if la.size > 0 else 0
			a_pad = ak.fill_none(ak.pad_none(a, max_len, clip=True), 0)
			b_pad = ak.fill_none(ak.pad_none(b, max_len, clip=True), 0)
			an = ak.to_numpy(a_pad).astype(np.float64)
			bn = ak.to_numpy(b_pad).astype(np.float64)
			if not np.allclose(an, bn, equal_nan=True):
				ok = False
				report[name] = "values differ"
			else:
				report[name] = "ok"
		except Exception:
			# scalar branch
			an = ak.to_numpy(a).astype(np.float64).reshape(-1)
			bn = ak.to_numpy(b).astype(np.float64).reshape(-1)
			if not np.allclose(an, bn, equal_nan=True):
				ok = False
				report[name] = "values differ"
			else:
				report[name] = "ok"
	return ok, report


def extract_rowmajor_binary(tree, nmax, out_bin, out_meta, chunk_size=50_000):
    """Extract ROW-MAJOR binary: events written in chunks, all branches per event.

    For each chunk of events, writes each branch's data: scalar branches as
    native-dtype arrays, jagged branches as int32 counts + flattened native-dtype
    values. This gives a ~6 GB file for a full CMS NANOAOD file.

    Also creates a 200 MB training chunk (first 200 MiB of the output).
    """
    n_entries = min(nmax, tree.num_entries) if nmax > 0 else tree.num_entries

    # Determine branch types from a sample
    sample = tree.arrays(entry_stop=min(1000, n_entries), library="ak")
    branch_names = sample.fields
    schema = []
    for bname in branch_names:
        a = sample[bname]
        try:
            ak.num(a, axis=1)
            is_jagged = True
            flat = ak.to_numpy(ak.ravel(a))
            dtype_str = str(flat.dtype)
        except Exception:
            is_jagged = False
            try:
                flat = ak.to_numpy(a)
                dtype_str = str(flat.dtype)
            except Exception:
                continue
        schema.append({"name": bname, "is_jagged": is_jagged, "dtype": dtype_str})

    print(f"[row-major] {len(schema)} branches, {n_entries} events")
    total_bytes = 0
    with open(out_bin, "wb") as f:
        for start in range(0, n_entries, chunk_size):
            stop = min(start + chunk_size, n_entries)
            arrs = tree.arrays(
                filter_name=[s["name"] for s in schema],
                entry_start=start, entry_stop=stop, library="ak",
            )
            for s in schema:
                a = arrs[s["name"]]
                dtype = np.dtype(s["dtype"])
                if s["is_jagged"]:
                    counts = ak.to_numpy(ak.num(a, axis=1)).astype(np.int32)
                    flat = ak.to_numpy(ak.ravel(a)).astype(dtype)
                    f.write(counts.tobytes())
                    f.write(flat.tobytes())
                    total_bytes += len(counts.tobytes()) + len(flat.tobytes())
                else:
                    arr = ak.to_numpy(a).astype(dtype)
                    f.write(arr.tobytes())
                    total_bytes += len(arr.tobytes())
            pct = stop / n_entries * 100
            print(f"  [{stop}/{n_entries}] {pct:.0f}% — {total_bytes / 1048576:.1f} MB written", end="\r")
    print()

    meta_obj = {"format": "row-major", "n_entries": n_entries, "total_bytes": total_bytes, "schema": schema}
    with open(out_meta, "w") as f:
        json.dump(meta_obj, f, indent=2)

    print(f"[row-major] Output: {out_bin} ({total_bytes:,} bytes, {total_bytes/1048576:.1f} MB)")

    # 200 MB training chunk
    trunc = os.path.splitext(out_bin)[0] + "_200m" + os.path.splitext(out_bin)[1]
    limit = 200 * 1024 * 1024
    with open(out_bin, "rb") as src, open(trunc, "wb") as dst:
        remaining = limit
        while remaining > 0:
            chunk = src.read(min(1 << 20, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    print(f"[row-major] Training chunk: {trunc} ({os.path.getsize(trunc)/1048576:.1f} MB)")
    return total_bytes


def extract_colmajor_binary(tree, nmax, out_bin, out_meta):
    """Extract COLUMN-MAJOR binary: each branch contiguous for ALL events.

    Stores [branch0_all_events][branch1_counts_all][branch1_values_all]...
    This gives a ~6 GB file for a full CMS NANOAOD file.

    Also creates a 200 MB training chunk (first 200 MiB of the output).
    """
    n_entries = min(nmax, tree.num_entries) if nmax > 0 else tree.num_entries

    sample = tree.arrays(entry_stop=min(1000, n_entries), library="ak")
    branch_names = sample.fields
    schema = []
    for bname in branch_names:
        a = sample[bname]
        try:
            ak.num(a, axis=1)
            is_jagged = True
            flat = ak.to_numpy(ak.ravel(a))
            dtype_str = str(flat.dtype)
        except Exception:
            is_jagged = False
            try:
                flat = ak.to_numpy(a)
                dtype_str = str(flat.dtype)
            except Exception:
                continue
        schema.append({"name": bname, "is_jagged": is_jagged, "dtype": dtype_str})

    print(f"[col-major] {len(schema)} branches, {n_entries} events")
    total_bytes = 0
    with open(out_bin, "wb") as f:
        for i, s in enumerate(schema):
            arr = tree.arrays(filter_name=[s["name"]], entry_stop=n_entries, library="ak")[s["name"]]
            dtype = np.dtype(s["dtype"])
            if s["is_jagged"]:
                counts = ak.to_numpy(ak.num(arr, axis=1)).astype(np.int32)
                flat = ak.to_numpy(ak.ravel(arr)).astype(dtype)
                cb = counts.tobytes()
                vb = flat.tobytes()
                f.write(cb)
                f.write(vb)
                s["counts_bytes"] = len(cb)
                s["values_bytes"] = len(vb)
                total_bytes += len(cb) + len(vb)
            else:
                values = ak.to_numpy(arr).astype(dtype)
                buf = values.tobytes()
                f.write(buf)
                s["values_bytes"] = len(buf)
                s["counts_bytes"] = 0
                total_bytes += len(buf)
            s["offset"] = total_bytes - s.get("counts_bytes", 0) - s["values_bytes"]
            pct = (i + 1) / len(schema) * 100
            print(f"  [{i+1}/{len(schema)}] {pct:5.1f}% cumul: {total_bytes/1048576:.1f} MB", end="\r")
    print()

    meta_obj = {"format": "column-major", "n_entries": n_entries, "total_bytes": total_bytes, "n_branches": len(schema), "schema": schema}
    with open(out_meta, "w") as f:
        json.dump(meta_obj, f, indent=2)

    print(f"[col-major] Output: {out_bin} ({total_bytes:,} bytes, {total_bytes/1048576:.1f} MB)")

    # Create representative 200 MB training chunk — proportional prefix of each column
    # (NOT naive first-200MB truncation, which would only cover a few branches)
    trunc = os.path.splitext(out_bin)[0] + "_200m" + os.path.splitext(out_bin)[1]
    target = 200 * 1024 * 1024
    ratio = target / total_bytes
    print(f"[col-major] Creating representative 200 MB chunk (ratio={ratio:.4f}, all {len(schema)} branches)...")
    written = 0
    file_offset = 0
    with open(out_bin, "rb") as src, open(trunc, "wb") as dst:
        for s in schema:
            cb = s.get("counts_bytes", 0)
            vb = s["values_bytes"]
            dtype = np.dtype(s["dtype"])
            elem_size = dtype.itemsize

            if s["is_jagged"]:
                n_events_total = cb // 4
                n_events_take = max(1, int(n_events_total * ratio))
                take_cb = n_events_take * 4
                src.seek(file_offset)
                all_counts = np.frombuffer(src.read(cb), dtype=np.int32)
                taken_counts = all_counts[:n_events_take]
                take_vb = int(taken_counts.sum()) * elem_size
                dst.write(taken_counts.tobytes())
                src.seek(file_offset + cb)
                dst.write(src.read(take_vb))
                written += take_cb + take_vb
                file_offset += cb + vb
            else:
                n_events_total = vb // elem_size
                n_events_take = max(1, int(n_events_total * ratio))
                take_bytes = n_events_take * elem_size
                src.seek(file_offset)
                dst.write(src.read(take_bytes))
                written += take_bytes
                file_offset += vb

    print(f"[col-major] Training chunk: {trunc} ({written/1048576:.1f} MB, all {len(schema)} branches)")
    return total_bytes


def write_rntuple_from_awkward(path, name, arrs, compression=uproot.ZSTD(7), chunk=100_000):
    # arrs: awkward record array with fields as scalars or jagged lists
    with uproot.recreate(path, compression=compression) as f:
        # Build schema directly from Awkward types; jagged becomes "var * T"
        schema = {k: ak.type(arrs[k]) for k in arrs.fields}
        nt = f.mkrntuple(name, schema)   # creates the RNTuple
        n = len(arrs)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            # slice event range and extend
            nt.extend({k: arrs[k][s:e] for k in arrs.fields})
    return os.path.getsize(path)

def main():
	parser = argparse.ArgumentParser(description="CMS ROOT ↔ bin reversible pipeline with two modes")
	parser.add_argument("--url", default=DEFAULT_INDEX_URL, help="OpenData file URL")
	parser.add_argument("--out-dir", default="./data", help="Output directory for artifacts")
	parser.add_argument("--nmax", type=int, default=50000, help="Max number of events to process")

	# Creation path: download + create compressible bin
	parser.add_argument("--create-bin", action="store_true", help="Download, select first N events, and write .bin + .meta.json")
	parser.add_argument("--bin-out", default="cms_50k_padded.bin", help="Output .bin filename (float64 matrix)")
	parser.add_argument("--meta-out", default=None, help="Output metadata JSON filename (default: bin name + .meta.json)")

	# Full-size extraction: row-major and column-major ~6 GB binaries
	parser.add_argument("--extract-formats", action="store_true",
	                    help="Also extract full-size row-major and col-major binaries (~6 GB each) with 200 MB training chunks")

	# Validation path: take an external decompressed bin and fit/verify against original
	parser.add_argument("--validate-bin", action="store_true", help="Validate an external decompressed .bin against the original ROOT")
	parser.add_argument("--decompressed-bin", default=None, help="Path to decompressed .bin produced by another algorithm")
	parser.add_argument("--use-meta", default=None, help="Path to metadata JSON matching the decompressed .bin (default: <bin>.meta.json)")
	parser.add_argument("--fitted-root", default="cms_fitted.root", help="Output ROOT path built from the decompressed .bin + meta")

	args = parser.parse_args()

	# Default behavior: if neither mode specified, run the creation path + extract formats
	if not args.create_bin and not args.validate_bin and not args.extract_formats:
		args.create_bin = True
		args.extract_formats = True

	ensure_dir(args.out_dir)

	# Resolve local ROOT file path: download to <out-dir>/cms.root if missing
	local_root = os.path.join(args.out_dir, "cms.root")
	if os.path.exists(local_root):
		print(f"[prep] Found existing ROOT: {local_root} (skipping download)")
	else:
		print(f"[prep] Resolving ROOT source from: {args.url}")
		root_src = args.url
		print(f"[prep] Downloading ROOT → {local_root}\n        from: {root_src}")
		download_file(root_src, local_root)
		print(f"[prep] Download complete: {local_root}")

	_, tree_key, tree = open_tree(local_root)

	# CREATION PATH
	if args.create_bin:
		n_entries = min(args.nmax, tree.num_entries)
		print(f"[create-bin] Using tree {tree_key} with {n_entries} events (of {tree.num_entries} total)")

		arrs = tree.arrays(entry_stop=n_entries, library="ak")
		selected = select_numeric_branches(arrs)
		print(f"[create-bin] Selected {len(selected)} branches")

		data, meta = encode_to_bin(arrs, selected)
		bin_path = os.path.join(args.out_dir, args.bin_out)
		meta_path = (
			os.path.join(args.out_dir, args.meta_out)
			if args.meta_out
			else os.path.splitext(bin_path)[0] + ".meta.json"
		)
		write_bin_and_meta(bin_path, meta_path, data, meta)
		# Write a 200MB truncated copy of the binary for quick tests/debugging
		limit_bytes = 200 * 1024 * 1024
		base, ext = os.path.splitext(bin_path)
		trunc_path = f"{base}_200m{ext}"
		bufsize = 1024 * 1024
		remaining = limit_bytes
		with open(bin_path, "rb") as src, open(trunc_path, "wb") as dst:
			while remaining > 0:
				chunk = src.read(min(bufsize, remaining))
				if not chunk:
					break
				dst.write(chunk)
				remaining -= len(chunk)
		size_mb = os.path.getsize(bin_path) / 1024 / 1024
		print(f"[create-bin] Wrote bin: {bin_path} ({data.shape[0]}x{data.shape[1]}, {size_mb:.2f} MB)")
		print(f"[create-bin] Wrote meta: {meta_path}")

		# Round-trip from our own bin and compare
		data2, meta2 = read_bin_and_meta(bin_path, meta_path)
		arrs_rec = reconstruct_awkward(data2, meta2)
		roundtrip_root = os.path.join(args.out_dir, "cms_roundtrip.root")
		write_root_from_awkward(roundtrip_root, meta2.tree_key, arrs_rec)
		print(f"[create-bin] Wrote reconstructed ROOT -> {roundtrip_root}")

		# Example usage after you reconstruct:
		# arrs_rec = reconstruct_awkward(data2, meta2)
		t0 = time.perf_counter()
		rntuple_path = os.path.join(args.out_dir, "cms_roundtrip_rntuple.root")
		size_bytes = write_rntuple_from_awkward(rntuple_path, "Events", arrs_rec, compression=None)
		t_rntuple = time.perf_counter() - t0
		print("RNTuple entries, branches:",
			uproot.open(rntuple_path)["Events"].num_entries,
			len(uproot.open(rntuple_path)["Events"].keys()))
		print("size:", size_bytes, "time_s:", t_rntuple)
		# Compare
		f2 = uproot.open(roundtrip_root)
		t2 = f2[meta2.tree_key]
		arrs2 = t2.arrays(library="ak")
		ok, report = compare_trees(arrs, arrs2, selected)
		report_path = os.path.join(args.out_dir, "compare_report.txt")
		with open(report_path, "w", encoding="utf-8") as f:
			f.write("Round-trip comparison report (create-bin)\n")
			f.write(f"Branches compared: {len(selected)}\n")
			for k, v in sorted(report.items()):
				f.write(f"{k}: {v}\n")
			f.write(f"\nOVERALL: {'OK' if ok else 'MISMATCH'}\n")
		print(f"[create-bin] Comparison {'OK' if ok else 'FAILED'} -> {report_path}")

		if not ok:
			# Do not exit immediately; user may also want to run validation in same call
			print("[create-bin] WARNING: Round-trip mismatch detected.")

	# FULL-SIZE FORMAT EXTRACTION PATH
	if args.extract_formats:
		print("\n=== Extracting full-size row-major and column-major binaries ===")
		nmax_full = tree.num_entries  # Use ALL events

		# Row-major
		rm_bin = os.path.join(args.out_dir, "cms_rowmajor.bin")
		rm_meta = os.path.join(args.out_dir, "cms_rowmajor.meta.json")
		if os.path.exists(rm_bin):
			print(f"[extract] Row-major already exists: {rm_bin} ({os.path.getsize(rm_bin)/1073741824:.1f} GB) — skipping")
		else:
			t0 = time.perf_counter()
			extract_rowmajor_binary(tree, nmax_full, rm_bin, rm_meta)
			print(f"  Time: {time.perf_counter()-t0:.1f}s")

		# Column-major
		cm_bin = os.path.join(args.out_dir, "cms_colmajor.bin")
		cm_meta = os.path.join(args.out_dir, "cms_colmajor.meta.json")
		if os.path.exists(cm_bin):
			print(f"[extract] Col-major already exists: {cm_bin} ({os.path.getsize(cm_bin)/1073741824:.1f} GB) — skipping")
		else:
			t0 = time.perf_counter()
			extract_colmajor_binary(tree, nmax_full, cm_bin, cm_meta)
			print(f"  Time: {time.perf_counter()-t0:.1f}s")

	# VALIDATION PATH
	if args.validate_bin:
		if not args.decompressed_bin:
			raise SystemExit("--validate-bin requires --decompressed-bin path")
		dec_bin_path = os.path.join(args.out_dir, args.decompressed_bin) if not os.path.isabs(args.decompressed_bin) else args.decompressed_bin
		use_meta_path = (
			args.use_meta
			if args.use_meta
			else os.path.splitext(dec_bin_path)[0] + ".meta.json"
		)
		if not os.path.exists(dec_bin_path):
			raise SystemExit(f"Decompressed bin not found: {dec_bin_path}")
		if not os.path.exists(use_meta_path):
			raise SystemExit(f"Metadata JSON not found for decompressed bin: {use_meta_path}")

		print(f"[validate-bin] Reading decompressed bin -> {dec_bin_path}")
		data_dec, meta_dec = read_bin_and_meta(dec_bin_path, use_meta_path)

		# Load original arrays for the same branches and number of events
		selected2 = [b.name for b in meta_dec.branches]
		n_entries_val = min(meta_dec.n_events, tree.num_entries)
		print(f"[validate-bin] Using first {n_entries_val} events for validation")
		arrs_orig = tree.arrays(filter_name=selected2, entry_stop=n_entries_val, library="ak")

		# Reconstruct from decompressed bin
		arrs_fit = reconstruct_awkward(data_dec, meta_dec)
		fitted_root = os.path.join(args.out_dir, args.fitted_root)
		write_root_from_awkward(fitted_root, meta_dec.tree_key, arrs_fit)
		print(f"[validate-bin] Wrote fitted ROOT -> {fitted_root}")

		# Compare
		ok2, report2 = compare_trees(arrs_orig, arrs_fit, selected2)
		report_path2 = os.path.join(args.out_dir, "compare_report_validate.txt")
		with open(report_path2, "w", encoding="utf-8") as f:
			f.write("Round-trip comparison report (validate-bin)\n")
			f.write(f"Branches compared: {len(selected2)}\n")
			for k, v in sorted(report2.items()):
				f.write(f"{k}: {v}\n")
			f.write(f"\nOVERALL: {'OK' if ok2 else 'MISMATCH'}\n")
		print(f"[validate-bin] Comparison {'OK' if ok2 else 'FAILED'} -> {report_path2}")

		if not ok2:
			sys.exit(3)


if __name__ == "__main__":
	main()

