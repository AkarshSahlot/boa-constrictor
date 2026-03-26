from pathlib import Path
import struct, zlib, math, hashlib
import os
import numpy as np
import torch

# Online head adaptation (test-time training of prediction head)
try:
    from online_adapt import HeadAdapter
    _HAS_ADAPT = True
except ImportError:
    _HAS_ADAPT = False


def BOA(device, filepath: str, model, adapt_config=None):
    IS_CUDA = device == "cuda" and torch.cuda.is_available()

    if IS_CUDA:
        from codec import compress_GPU as compress, decompress_GPU as decompress
        device = "cuda"
    else:
        from codec import compress_CPU as compress, decompress_CPU as decompress
        device = "cpu"

    def _uvarint_encode(x: int) -> bytes:
        out = bytearray()
        while True:
            b = x & 0x7F; x >>= 7
            out.append(b | (0x80 if x else 0))
            if not x: break
        return bytes(out)

    def _uvarint_decode(buf: memoryview, pos: int):
        x = 0; s = 0
        while True:
            b = buf[pos]; pos += 1
            x |= (b & 0x7F) << s
            if not (b & 0x80): return x, pos
            s += 7

    def _as_bytes(obj) -> bytes:
        if isinstance(obj, (bytes, bytearray)): return bytes(obj)
        if torch.is_tensor(obj):
            t = obj.detach().contiguous().to("cpu")
            if t.dtype != torch.uint8: t = t.view(torch.uint8)
            return t.numpy().tobytes()
        arr = np.asarray(obj)
        if arr.dtype != np.uint8: arr = arr.view(np.uint8)
        return arr.tobytes()

    def _pad4(b: bytes) -> bytes:
        r = len(b) & 3
        return b if r == 0 else (b + b"\x00" * (4 - r))

    class BoaFile:
        MAGIC = b'BOA2'
        IDX   = b'IDX2'
        VERSION = 1

        def __init__(self, filepath: str, model):
            self.filepath = Path(filepath)
            self.model = model
            self.compressed_data = []
            self.first_bytes = []
            self.lengths = []
            self.metadata = {}

        def _split_to_chunks(self, data_bytes: bytes, seq_size: int = 0, chunks_count: int = 0):
            n = len(data_bytes)
            if seq_size and chunks_count:
                chunk_len = int(seq_size); n_chunks = math.ceil(n / chunk_len)
            elif seq_size:
                chunk_len = int(seq_size); n_chunks = math.ceil(n / chunk_len)
            elif chunks_count:
                n_chunks = max(int(chunks_count), 1); chunk_len = math.ceil(n / n_chunks)
            else:
                raise ValueError("Provide either 'seq_size' or 'chunks_count'.")
            chunks = []
            for i in range(n_chunks):
                s = i * chunk_len; e = min(s + chunk_len, n)
                if s >= e: break
                arr = np.frombuffer(memoryview(data_bytes)[s:e], dtype=np.uint8).astype(np.int64)
                chunks.append(arr)
            last_len = len(chunks[-1]) if chunks else 0
            self.metadata = {
                'chunk_len': int(chunk_len),
                'n_chunks': int(len(chunks)),
                'uncompressed_len': int(n),
                'last_chunk_len': int(last_len if last_len else chunk_len),
            }
            return chunks, int(chunk_len)

        def _model_fingerprint(self) -> bytes:
            name = getattr(self.model, "__class__", type(self.model)).__name__
            return hashlib.blake2s(name.encode(), digest_size=16).digest()

        def _write_file(self, compressed_list, first_bytes, uncompressed_len, chunk_len, last_chunk_len):
            n = len(compressed_list); fp = self._model_fingerprint()
            with open(self.filepath, 'wb') as f:
                f.write(self.MAGIC)
                f.write(struct.pack('<I', self.VERSION))
                f.write(struct.pack('<I', 0))  # flags
                f.write(struct.pack('<Q', uncompressed_len))
                f.write(struct.pack('<I', chunk_len))
                f.write(struct.pack('<I', n))
                f.write(struct.pack('<I', last_chunk_len))
                f.write(struct.pack('<B', len(fp))); f.write(fp)

                offsets = []; off = 0
                for c in compressed_list:
                    offsets.append(off); f.write(c); off += len(c)

                idx = bytearray()
                idx += self.IDX
                idx += bytes(first_bytes)  # n bytes
                # IDX2: only lengths (offsets reconstructed as cumulative sum)
                for L in lengths:
                    idx += _uvarint_encode(L)
                crc = zlib.crc32(idx) & 0xFFFFFFFF
                f.write(idx); f.write(struct.pack('<I', crc))

        def _read_file(self):
            data = Path(self.filepath).read_bytes()
            mm = memoryview(data); p = 0
            if bytes(mm[p:p+4]) != self.MAGIC: raise ValueError("Bad file magic")
            p += 4
            version, = struct.unpack_from('<I', mm, p); p += 4
            if version != self.VERSION: raise ValueError(f"Unsupported version {version}")
            _flags, = struct.unpack_from('<I', mm, p); p += 4
            ulen, = struct.unpack_from('<Q', mm, p); p += 8
            chunk_len, = struct.unpack_from('<I', mm, p); p += 4
            n, = struct.unpack_from('<I', mm, p); p += 4
            last_chunk_len, = struct.unpack_from('<I', mm, p); p += 4
            hlen = mm[p]; p += 1
            _fp = bytes(mm[p:p+hlen]); p += hlen

            crc = struct.unpack_from('<I', mm, len(mm)-4)[0]
            # Support both IDX1 and IDX2
            idx_pos_2 = data.rfind(b'IDX2')
            idx_pos_1 = data.rfind(b'IDX1')
            idx_pos = max(idx_pos_2, idx_pos_1)
            if idx_pos < 0: raise ValueError("Index not found")
            idx_ver = 2 if idx_pos == idx_pos_2 else 1
            if (zlib.crc32(data[idx_pos:len(data)-4]) & 0xFFFFFFFF) != crc:
                raise ValueError("Bad index CRC")

            q = idx_pos + 4  # skip IDXn tag
            first_bytes = list(mm[q:q+n]); q += n

            if idx_ver == 1:
                offsets = [0]*n; pos = q; prev = 0
                for i in range(n):
                    d, pos = _uvarint_decode(mm, pos); prev += d; offsets[i] = prev
                comp_lens = [0]*n
                for i in range(n):
                    L, pos = _uvarint_decode(mm, pos); comp_lens[i] = L
            else:
                # IDX2: only lengths, reconstruct offsets
                comp_lens = [0]*n; pos = q
                for i in range(n):
                    L, pos = _uvarint_decode(mm, pos); comp_lens[i] = L
                offsets = [0]*n; off = 0
                for i in range(n):
                    offsets[i] = off; off += comp_lens[i]

            payload = mm[p:idx_pos]
            compressed_list = [bytes(payload[offsets[i]: offsets[i]+comp_lens[i]]) for i in range(n)]
            full_lens = [int(chunk_len)]*(n-1) + [int(last_chunk_len)]

            self.compressed_data = compressed_list          # bytes
            self.first_bytes = first_bytes                  # list[int]
            self.lengths = full_lens                        # uncompressed lens
            self.metadata = {
                'chunk_len': int(chunk_len),
                'n_chunks': int(n),
                'last_chunk_len': int(last_chunk_len),
                'uncompressed_len': int(ulen),
            }

        def compress(self, data_path: str, seq_size: int = 0, chunks_count: int = 0, progress: bool = True):
            # Determine chunking from file size without loading entire file into RAM
            p = Path(data_path)
            total_size = p.stat().st_size
            if total_size <= 0:
                raise ValueError("Input file is empty")

            # Compute chunk_len and number of chunks similar to _split_to_chunks
            if seq_size and chunks_count:
                chunk_len = int(seq_size); n_chunks = math.ceil(total_size / chunk_len)
            elif seq_size:
                chunk_len = int(seq_size); n_chunks = math.ceil(total_size / chunk_len)
            elif chunks_count:
                n_chunks = max(int(chunks_count), 1); chunk_len = math.ceil(total_size / n_chunks)
            else:
                raise ValueError("Provide either 'seq_size' or 'chunks_count'.")

            last_chunk_len = int(total_size - (n_chunks - 1) * chunk_len) if n_chunks > 1 else int(total_size)
            self.metadata = {
                'chunk_len': int(chunk_len),
                'n_chunks': int(n_chunks),
                'uncompressed_len': int(total_size),
                'last_chunk_len': int(last_chunk_len if last_chunk_len else chunk_len),
            }

            # Prepare output file and write header
            fp = self._model_fingerprint()
            with open(self.filepath, 'wb') as f_out:
                f_out.write(self.MAGIC)
                f_out.write(struct.pack('<I', self.VERSION))
                f_out.write(struct.pack('<I', 0))  # flags
                f_out.write(struct.pack('<Q', total_size))
                f_out.write(struct.pack('<I', chunk_len))
                f_out.write(struct.pack('<I', n_chunks))
                f_out.write(struct.pack('<I', last_chunk_len))
                f_out.write(struct.pack('<B', len(fp))); f_out.write(fp)

                payload_start = f_out.tell()
                offsets: list[int] = []
                lengths: list[int] = []
                first_bytes: list[int] = []
                off = 0

                # Memory-map the input file to avoid loading it into RAM
                mm = np.memmap(p, dtype=np.uint8, mode='r')

                # Number of chunks to process per batch (streams). Default 5000; can be overridden for demos via env.
                try:
                    gpu_streams = int(os.getenv("BOA_GPU_STREAMS", "5000"))
                except Exception:
                    gpu_streams = 5000
                gpu_streams = max(1, min(int(gpu_streams), int(n_chunks)))
                if progress:
                    try:
                        if device == "cuda" and torch.cuda.is_available():
                            free_mem, total_mem = torch.cuda.mem_get_info()
                            print(f"[compress] gpu_streams={gpu_streams} (chunk_len={chunk_len}, free={free_mem/2**30:.1f}GiB)")
                        else:
                            print(f"[compress] streams (CPU)={gpu_streams} (chunk_len={chunk_len})")
                    except Exception:
                        print(f"[compress] streams={gpu_streams}")

                # Online head adaptation: adapt the prediction head between batches
                adapter = None
                if _HAS_ADAPT and adapt_config is not None:
                    _lr, _K = adapt_config
                    adapter = HeadAdapter(self.model, lr=_lr, adapt_steps=_K)
                    if progress:
                        print(f"[compress] Online head adaptation enabled (lr={_lr}, K={_K})")

                # Process chunks in GPU batches to reduce H2D overhead
                for batch_start in range(0, n_chunks, gpu_streams):
                    batch_end = min(batch_start + gpu_streams, n_chunks)
                    x_list = []
                    Ls_batch = []
                    for i in range(batch_start, batch_end):
                        s = i * chunk_len
                        e = min(s + chunk_len, total_size)
                        sl = mm[s:e]
                        # Torch tensor on CPU; GPU transfer handled inside compress() call
                        t = torch.from_numpy(np.ascontiguousarray(sl)).unsqueeze(0)
                        x_list.append(t)
                        Ls_batch.append(int(e - s))

                    compressed_list, fb_batch, _Ls = compress(
                        self.model, x_list, device=device, progress=progress
                    )
                    # Stream write compressed payload and record offsets/lengths
                    for j, comp_u32 in enumerate(compressed_list):
                        # Ensure deterministic little-endian serialization of u32 words
                        comp_arr = np.asarray(comp_u32, dtype=np.uint32)
                        if comp_arr.dtype.byteorder == '>':
                            comp_arr = comp_arr.byteswap().newbyteorder('<')
                        comp_bytes = comp_arr.tobytes(order='C')
                        f_out.write(comp_bytes)
                        offsets.append(off)
                        lengths.append(len(comp_bytes))
                        off += len(comp_bytes)
                    first_bytes.extend(int(b) & 0xFF for b in fb_batch)

                    # Adapt head on the batch we just compressed
                    if adapter is not None:
                        adapter.adapt_on_batch(x_list, device=device)

                # Build and write index (IDX2: only lengths, no offsets)
                idx = bytearray()
                idx += self.IDX
                idx += bytes(first_bytes)  # n bytes
                for L in lengths:
                    idx += _uvarint_encode(L)
                crc = zlib.crc32(idx) & 0xFFFFFFFF
                f_out.write(idx)
                f_out.write(struct.pack('<I', crc))

                # Restore head to original state so decompress starts identically
                if adapter is not None:
                    adapter.restore_head()

            # Update object state (do not keep compressed payload in RAM)
            self.compressed_data = []
            self.first_bytes = first_bytes
            self.lengths = [chunk_len] * (n_chunks - 1) + [last_chunk_len]
            print(f"Compression complete: {n_chunks} chunks, chunk_len={chunk_len}, last={last_chunk_len}")

        def read_from_disk(self):
            self._read_file()
            print("File loaded successfully")

        def decompress(self, progress: bool = True) -> bytes:
            self._read_file()
            print(f"Total compressed size from disk: {sum(len(c) for c in self.compressed_data)} bytes")

            # Decompress in batches to limit GPU memory and align with encoder batch semantics
            try:
                gpu_streams = int(os.getenv("BOA_GPU_STREAMS", "5000"))
            except Exception:
                gpu_streams = 5000
            n = len(self.compressed_data)
            gpu_streams = max(1, min(int(gpu_streams), int(n)))
            if progress:
                try:
                    if device == "cuda" and torch.cuda.is_available():
                        free_mem, total_mem = torch.cuda.mem_get_info()
                        print(f"[decompress] gpu_streams={gpu_streams} (free={free_mem/2**30:.1f}GiB)")
                    else:
                        print(f"[decompress] streams (CPU)={gpu_streams}")
                except Exception:
                    print(f"[decompress] streams={gpu_streams}")

            # Online head adaptation for decompression (must mirror encoder)
            adapter = None
            if _HAS_ADAPT and adapt_config is not None:
                _lr, _K = adapt_config
                adapter = HeadAdapter(self.model, lr=_lr, adapt_steps=_K)
                if progress:
                    print(f"[decompress] Online head adaptation enabled (lr={_lr}, K={_K})")

            out_parts: list[bytes] = []
            for batch_start in range(0, n, gpu_streams):
                batch_end = min(batch_start + gpu_streams, n)
                comp_u32_batch = [np.frombuffer(c, dtype='<u4').copy() for c in self.compressed_data[batch_start:batch_end]]
                lens_batch = self.lengths[batch_start:batch_end]
                fb_batch = self.first_bytes[batch_start:batch_end]

                decoded_list = decompress(
                    self.model, comp_u32_batch, lens_batch, fb_batch, device=device, progress=progress
                )
                # Adapt head on the batch we just decompressed (mirrors encoder)
                if adapter is not None:
                    byte_seqs = [torch.from_numpy(d.flatten()).unsqueeze(0) for d in decoded_list]
                    adapter.adapt_on_batch(byte_seqs, device=device)

                for d in decoded_list:
                    out_parts.append(d.tobytes() if hasattr(d, "tobytes") else bytes(d))

            # Restore head to original state (clean up after decompression)
            if adapter is not None:
                adapter.restore_head()

            return b"".join(out_parts)

        def get_metadata(self):
            return dict(self.metadata)
        
    return BoaFile(filepath, model)