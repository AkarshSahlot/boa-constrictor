#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>
#include "boa_gpu.hpp"
#include "gemm_gpu.hpp" 

void gpu_select_tokens(const int* d_chunk_data, int* d_tokens, int t, int chunk_size, int batch_size);
void gpu_store_tokens(const int* d_out_symbols, int* d_chunk_data, int t, int chunk_size, int batch_size);

#include "activations.hpp"

bool g_show_timings = false;
bool g_no_graph = false;

void print_progress(int current, int total, double elapsed_sec, size_t total_bytes, const std::string& prefix = "") {
    float percent = (float)current / total;
    int bar_width = 40;
    int pos = (int)(bar_width * percent);

    std::cout << "\r" << prefix << " [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    
    double mb_s = 0.0;
    if (elapsed_sec > 0) {
        double processed_bytes = (double)current / total * total_bytes;
        mb_s = processed_bytes / (1024.0 * 1024.0) / elapsed_sec;
    }

    std::cout << "] " << (int)(percent * 100) << "% (" << current << "/" << total << ") "
              << std::fixed << std::setprecision(2) << mb_s << " MB/s" << std::flush;
}

// Simple Softmax
std::vector<float> softmax(const float* logits, int size) {
    std::vector<float> probs(size);
    float max_val = logits[0];
    for(int i=1; i<size; ++i) if(logits[i] > max_val) max_val = logits[i];
    
    double sum_exp = 0.0;
    for(int i=0; i<size; ++i) {
        double val = (double)repro_exp(logits[i] - max_val);
        probs[i] = (float)val; 
        sum_exp += val;
    }
    double inv_sum = 1.0 / sum_exp;
    for(int i=0; i<size; ++i) probs[i] *= (float)inv_sum;
    return probs;
}

inline void softmax_into(const float* logits, float* probs, int size) {
    float max_val = logits[0];
    for(int i=1; i<size; ++i) if(logits[i] > max_val) max_val = logits[i];

    double sum_exp = 0.0;
    for(int i=0; i<size; ++i) {
        double val = (double)repro_exp(logits[i] - max_val);
        probs[i] = (float)val;
        sum_exp += val;
    }
    double inv_sum = 1.0 / sum_exp;
    for(int i=0; i<size; ++i) probs[i] *= (float)inv_sum;
}

static size_t g_chunk_size = 4096;

static uint32_t crc32_table[256];
static bool crc32_table_init = false;

static void init_crc32_table() {
    if (crc32_table_init) return;
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) {
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        crc32_table[i] = c;
    }
    crc32_table_init = true;
}

static uint32_t crc32_compute(const uint8_t* data, size_t len) {
    init_crc32_table();
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        c = crc32_table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

void auto_size_params(const MambaConfig& config, int& batch_size, size_t& chunk_size, bool is_compression) {
    size_t free_vram, total_vram;
    get_gpu_vram_info(free_vram, total_vram);
    
    // Safety buffer: use 90% of free VRAM
    size_t target_vram = (size_t)(free_vram * 0.9);
    
    if (chunk_size == 0) chunk_size = 4096;
    
    // Initial guess for batch size if not provided
    if (batch_size <= 0) {
        if (is_compression) batch_size = 64; // Conservative start
        else batch_size = 1024; // Decompression is lighter
    }

    // Adjust batch size based on memory
    size_t mem = BoaPredictorGPU::estimate_memory_static(config, 256, config.n_layers, batch_size, (int)chunk_size, is_compression);
    
    if (mem > target_vram) {
        // Reduce batch size
        while (batch_size > 1 && mem > target_vram) {
            batch_size /= 2;
            mem = BoaPredictorGPU::estimate_memory_static(config, 256, config.n_layers, batch_size, (int)chunk_size, is_compression);
        }
        // If even batch=1 doesn't fit, reduce chunk_size
        while (chunk_size > 256 && mem > target_vram) {
            chunk_size /= 2;
            mem = BoaPredictorGPU::estimate_memory_static(config, 256, config.n_layers, batch_size, (int)chunk_size, is_compression);
        }
    } else {
        // Increase batch size to fill VRAM
        int new_batch = batch_size;
        while (true) {
            int test_batch = new_batch * 2;
            size_t test_mem = BoaPredictorGPU::estimate_memory_static(config, 256, config.n_layers, test_batch, (int)chunk_size, is_compression);
            if (test_mem < target_vram && test_batch <= 8192) {
                new_batch = test_batch;
            } else {
                break;
            }
        }
        batch_size = new_batch;
    }

    mem = BoaPredictorGPU::estimate_memory_static(config, 256, config.n_layers, batch_size, (int)chunk_size, is_compression);
    std::cout << "Auto-sizing: Selected Batch=" << batch_size << ", Chunk=" << chunk_size 
              << " (" << mem / (1024*1024) << " MB VRAM est, " << free_vram / (1024*1024) << " MB free)\n";
}

static void uvarint_encode(std::vector<uint8_t>& out, uint64_t x) {
    while (true) {
        uint8_t b = (uint8_t)(x & 0x7Fu);
        x >>= 7;
        out.push_back((uint8_t)(b | (x ? 0x80u : 0u)));
        if (!x) break;
    }
}

static uint64_t uvarint_decode(const std::vector<uint8_t>& buf, size_t& pos) {
    uint64_t x = 0; int shift = 0;
    while (true) {
        uint8_t b = buf[pos++];
        x |= (uint64_t)(b & 0x7Fu) << shift;
        if (!(b & 0x80u)) break;
        shift += 7;
    }
    return x;
}

struct Boa2Container {
    uint64_t total_size = 0;
    uint32_t chunk_len = 0;
    uint32_t last_chunk_len = 0;
    int num_chunks = 0;
    bool warm_start = false;
    uint32_t interleave_batch = 0;
    std::vector<uint8_t> first_bytes;
    std::vector<std::vector<uint8_t>> streams;
    std::vector<int> lengths;
};

static Boa2Container read_boa2(const std::string& path) {
    Boa2Container out;
    std::ifstream fin(path, std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    fin.close();
    if (data.size() < 32) throw std::runtime_error("Invalid BOA2 file");

    size_t p = 0;
    if (std::memcmp(data.data(), "BOA2", 4) != 0) throw std::runtime_error("Bad BOA2 magic");
    p += 4;
    uint32_t version = 0; std::memcpy(&version, data.data() + p, 4); p += 4;
    if (version != 1) throw std::runtime_error("Unsupported BOA2 version");
    uint32_t flags = 0; std::memcpy(&flags, data.data() + p, 4); p += 4;
    out.warm_start = (flags & 1u) != 0;
    std::memcpy(&out.total_size, data.data() + p, 8); p += 8;
    std::memcpy(&out.chunk_len, data.data() + p, 4); p += 4;
    uint32_t n_chunks_u32 = 0; std::memcpy(&n_chunks_u32, data.data() + p, 4); p += 4;
    out.num_chunks = (int)n_chunks_u32;
    std::memcpy(&out.last_chunk_len, data.data() + p, 4); p += 4;
    uint8_t fp_len = data[p++];
    if (out.warm_start && fp_len >= 4) {
        std::memcpy(&out.interleave_batch, data.data() + p, 4);
        std::cout << "Warm-start file detected. Interleave batch=" << out.interleave_batch << std::endl;
    }
    p += fp_len;
    size_t payload_start = p;

    // find IDX1 or IDX2
    size_t idx_pos = std::string::npos;
    int idx_version = 0;
    for (size_t i = data.size() - 4; i-- > 0;) {
        if (data[i] == 'I' && data[i+1] == 'D' && data[i+2] == 'X') {
            if (data[i+3] == '2') { idx_pos = i; idx_version = 2; break; }
            if (data[i+3] == '1') { idx_pos = i; idx_version = 1; break; }
        }
    }
    if (idx_pos == std::string::npos) throw std::runtime_error("IDX not found");
    if (data.size() < idx_pos + 4 + 4) throw std::runtime_error("Bad BOA2 IDX");

    uint32_t crc_file = 0; std::memcpy(&crc_file, data.data() + data.size() - 4, 4);
    uint32_t crc_calc = crc32_compute(data.data() + idx_pos, data.size() - idx_pos - 4);
    if (crc_file != crc_calc) throw std::runtime_error("Bad BOA2 CRC");

    size_t q = idx_pos + 4;
    out.first_bytes.resize(out.num_chunks);
    for (int i = 0; i < out.num_chunks; ++i) out.first_bytes[i] = data[q++];

    std::vector<uint64_t> offsets(out.num_chunks, 0);
    std::vector<uint64_t> lengths(out.num_chunks, 0);

    if (idx_version == 1) {
        // IDX1: delta-encoded offsets + lengths
        uint64_t prev = 0;
        for (int i = 0; i < out.num_chunks; ++i) {
            uint64_t delta = uvarint_decode(data, q);
            prev += delta;
            offsets[i] = prev;
        }
        for (int i = 0; i < out.num_chunks; ++i) {
            lengths[i] = uvarint_decode(data, q);
        }
    } else {
        // IDX2: only lengths (offsets reconstructed)
        for (int i = 0; i < out.num_chunks; ++i) {
            lengths[i] = uvarint_decode(data, q);
        }
        uint64_t off = 0;
        for (int i = 0; i < out.num_chunks; ++i) {
            offsets[i] = off;
            off += lengths[i];
        }
    }

    std::vector<uint8_t> payload(data.begin() + payload_start, data.begin() + idx_pos);
    out.streams.resize(out.num_chunks);
    for (int i = 0; i < out.num_chunks; ++i) {
        uint64_t off = offsets[i];
        uint64_t len = lengths[i];
        if (off + len > payload.size()) throw std::runtime_error("BOA2 payload out of range");
        out.streams[i].assign(payload.begin() + off, payload.begin() + off + len);
    }

    out.lengths.resize(out.num_chunks);
    for (int i = 0; i < out.num_chunks; ++i) {
        out.lengths[i] = (i == out.num_chunks - 1) ? (int)out.last_chunk_len : (int)out.chunk_len;
    }
    return out;
}

// Compress V2
void compress_v2(const std::string& model_path, const std::string& input_path, const std::string& output_path, MambaConfig config, int max_chunks, int gpu_batch, size_t chunk_size, bool warm_start, int tile_size = 0) {
    std::cout << "Loading Data..." << std::endl;
    std::ifstream fin(input_path, std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    fin.close();
    
    size_t total_size = data.size();
    int num_chunks = (int)((total_size + chunk_size - 1) / chunk_size);
    if (max_chunks > 0) {
        num_chunks = std::min(num_chunks, max_chunks);
    }
    size_t processed_size = std::min(total_size, (size_t)num_chunks * chunk_size);
    
    std::cout << "Compressing " << processed_size << " bytes in " << num_chunks << " chunks..." << std::endl;
    
    size_t last_chunk_len = (num_chunks > 1)
        ? (processed_size - (size_t)(num_chunks - 1) * chunk_size)
        : processed_size;
    
    std::vector<std::vector<uint8_t>> all_compressed_chunks(num_chunks);
    std::vector<uint8_t> all_first_bytes(num_chunks);
    
    int BATCH_SIZE = (gpu_batch > 0) ? gpu_batch : 64;
    if (warm_start && BATCH_SIZE > num_chunks) BATCH_SIZE = num_chunks;

    // Tile size: sub-chunk the forward pass to reduce VRAM, allowing larger batches
    if (tile_size <= 0 || tile_size > (int)chunk_size) tile_size = (int)chunk_size;
    int num_tiles = ((int)chunk_size + tile_size - 1) / tile_size;

    // Warm-start interleaving: S = steps per lane, each lane processes contiguous original chunks
    int S = warm_start ? ((num_chunks + BATCH_SIZE - 1) / BATCH_SIZE) : 1;
    int num_iters = warm_start ? S : ((num_chunks + BATCH_SIZE - 1) / BATCH_SIZE);

    if (warm_start) {
        std::cout << "Using GPU Warm-Start Interleaved Compression. Batch=" << BATCH_SIZE
                  << ", Steps/lane=" << S << " (cold-starts: " << BATCH_SIZE << " instead of " << num_chunks << ")" << std::endl;
    } else {
        std::cout << "Using GPU Parallel Batching. Batch=" << BATCH_SIZE << std::endl;
    }
    if (tile_size < (int)chunk_size) {
        std::cout << "Tiled forward: tile=" << tile_size << ", " << num_tiles << " tiles/chunk" << std::endl;
    }

    gpu_init_exp_lut();
    
    BoaPredictorGPU gpu_model(config, 256, config.n_layers, BATCH_SIZE);
    gpu_model.load_weights(model_path);
    
    RCState* d_rc_states;
    unsigned char* d_out_bufs;
    int pitch_words = (int)((chunk_size * 10 + 4096 + 3) / 4);
    int pitch_bytes = pitch_words * 4;

    malloc_device((float**)&d_rc_states, BATCH_SIZE * sizeof(RCState));
    malloc_device((float**)&d_out_bufs, (size_t)BATCH_SIZE * pitch_bytes);
    
    int* d_batch_data[2] = {nullptr, nullptr};
    int* h_batch_data[2] = {nullptr, nullptr};
    std::vector<int> h_batch_lengths[2];
    for (int i = 0; i < 2; ++i) {
        malloc_device((float**)&d_batch_data[i], (size_t)BATCH_SIZE * chunk_size * sizeof(int));
        checkCudaErrors(cudaHostAlloc((void**)&h_batch_data[i], (size_t)BATCH_SIZE * chunk_size * sizeof(int), cudaHostAllocDefault));
        h_batch_lengths[i].resize(BATCH_SIZE);
    }
    
    int* d_sizes;
    malloc_device((float**)&d_sizes, BATCH_SIZE * sizeof(int));
    
    int* d_tokens;
    int* d_lengths;
    float* d_logits = nullptr;
    int* d_sub_tokens = nullptr;
    malloc_device((float**)&d_tokens, BATCH_SIZE * sizeof(int));
    malloc_device((float**)&d_lengths, BATCH_SIZE * sizeof(int));
    malloc_device(&d_logits, (size_t)BATCH_SIZE * tile_size * 256 * sizeof(float));
    if (tile_size < (int)chunk_size) {
        malloc_device((float**)&d_sub_tokens, (size_t)BATCH_SIZE * tile_size * sizeof(int));
    }
    
    unsigned char* h_out_bufs[2] = {nullptr, nullptr};
    int* h_sizes[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i) {
        checkCudaErrors(cudaHostAlloc((void**)&h_out_bufs[i], (size_t)BATCH_SIZE * pitch_bytes, cudaHostAllocDefault));
        checkCudaErrors(cudaHostAlloc((void**)&h_sizes[i], (size_t)BATCH_SIZE * sizeof(int), cudaHostAllocDefault));
    }

    gpu_model.allocate_chunk(tile_size);

    double t_h2d = 0.0;
    double t_compute = 0.0;
    double t_d2h = 0.0;
    double t_fwd = 0.0;
    double t_encode = 0.0;

    cudaEvent_t ev_start, ev_mid, ev_end;
    checkCudaErrors(cudaEventCreate(&ev_start));
    checkCudaErrors(cudaEventCreate(&ev_mid));
    checkCudaErrors(cudaEventCreate(&ev_end));

    auto start_time = std::chrono::high_resolution_clock::now();
    int chunks_done = 0;

    for (int iter = 0; iter < num_iters; ++iter) {
        int buf = iter & 1;

        // Compute current_batch and abs_idx mapping
        int current_batch;
        if (warm_start) {
            // Interleaved: lane j -> original chunk (j * S + iter)
            // Lanes with j*S+iter >= num_chunks are inactive (always trailing)
            current_batch = 0;
            for (int b = 0; b < BATCH_SIZE; ++b) {
                if ((int64_t)b * S + iter < num_chunks) current_batch = b + 1;
                else break;
            }
        } else {
            int chunk_idx = iter * BATCH_SIZE;
            current_batch = std::min(BATCH_SIZE, num_chunks - chunk_idx);
        }
        
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        print_progress(chunks_done, num_chunks, elapsed, processed_size, "Compressing");

        memset(h_batch_data[buf], 0, (size_t)BATCH_SIZE * chunk_size * sizeof(int));
        for (int b = 0; b < current_batch; ++b) {
            int abs_idx = warm_start ? (b * S + iter) : (iter * BATCH_SIZE + b);
            if (abs_idx >= num_chunks) { h_batch_lengths[buf][b] = 0; continue; }
            size_t start_off = (size_t)abs_idx * chunk_size;
            size_t len = std::min(chunk_size, total_size - start_off);
            for(size_t k=0; k<len; ++k) {
                h_batch_data[buf][(size_t)b * chunk_size + k] = data[start_off + k];
            }
            all_first_bytes[abs_idx] = (len > 0) ? data[start_off] : 0;
            h_batch_lengths[buf][b] = (int)len;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        checkCudaErrors(cudaMemcpy(d_batch_data[buf], h_batch_data[buf], (size_t)BATCH_SIZE * chunk_size * sizeof(int), cudaMemcpyHostToDevice));
        auto t1 = std::chrono::high_resolution_clock::now();
        t_h2d += std::chrono::duration<double>(t1 - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        gpu_rc_init(d_rc_states, current_batch);
        // Warm-start: only reset model cache on first iteration
        if (!warm_start || iter == 0) {
            gpu_model.reset_cache();
        }

        checkCudaErrors(cudaMemcpy(d_lengths, h_batch_lengths[buf].data(), current_batch * sizeof(int), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaEventRecord(ev_start));

        // Tiled forward + encode: process chunk in tiles of tile_size
        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            int sub_start = tile_idx * tile_size;
            int sub_end = std::min(sub_start + tile_size, (int)chunk_size);
            int sub_len = sub_end - sub_start;

            // Gather sub-chunk tokens from strided [B, chunk_size] to compact [B, sub_len]
            int* fwd_tokens;
            if (d_sub_tokens) {
                checkCudaErrors(cudaMemcpy2D(d_sub_tokens, sub_len * sizeof(int),
                                              d_batch_data[buf] + sub_start, chunk_size * sizeof(int),
                                              sub_len * sizeof(int), current_batch,
                                              cudaMemcpyDeviceToDevice));
                fwd_tokens = d_sub_tokens;
            } else {
                fwd_tokens = d_batch_data[buf]; // no tiling, use directly
            }

            // Forward model (state carries forward between tiles)
            gpu_model.forward_chunk(fwd_tokens, d_logits, sub_len);

            // Encode this tile's token range
            // For non-final tiles: max_len = sub_end + 1 so the loop encodes up to sub_end
            // For final tile: max_len = sub_end, loop stops at len-1 (natural chunk end)
            bool is_last_tile = (sub_end >= (int)chunk_size);
            int enc_max_len = is_last_tile ? sub_end : (sub_end + 1);
            gpu_rc_encode_chunk_warp(d_logits, d_batch_data[buf], d_lengths, d_rc_states, (unsigned int*)d_out_bufs,
                                      pitch_words, 256, current_batch, (int)chunk_size, enc_max_len, sub_start, sub_len * 256);
        }

        checkCudaErrors(cudaEventRecord(ev_end));
        checkCudaErrors(cudaEventSynchronize(ev_end));
        float ms_fwd = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&ms_fwd, ev_start, ev_end));
        t_fwd += ms_fwd / 1000.0;

        gpu_rc_finish_batch(d_rc_states, (unsigned int*)d_out_bufs, pitch_words, d_sizes, current_batch);
        checkCudaErrors(cudaDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        t_compute += std::chrono::duration<double>(t1 - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        checkCudaErrors(cudaMemcpy(h_out_bufs[buf], d_out_bufs, (size_t)current_batch * pitch_bytes, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_sizes[buf], d_sizes, (size_t)current_batch * sizeof(int), cudaMemcpyDeviceToHost));
        t1 = std::chrono::high_resolution_clock::now();
        t_d2h += std::chrono::duration<double>(t1 - t0).count();
        for (int b = 0; b < current_batch; ++b) {
            int abs_idx = warm_start ? (b * S + iter) : (iter * BATCH_SIZE + b);
            if (abs_idx >= num_chunks) continue;
            int size_words = h_sizes[buf][b];
            all_compressed_chunks[abs_idx].resize(size_words * 4);
            memcpy(all_compressed_chunks[abs_idx].data(), h_out_bufs[buf] + (size_t)b * pitch_bytes, size_words * 4);
        }
        chunks_done += current_batch;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    print_progress(num_chunks, num_chunks, duration, processed_size, "Compressing");
    
    double mb_s = (double)processed_size / (1024.0 * 1024.0) / duration;
    std::cout << "\nDone. Time: " << std::fixed << std::setprecision(3) << duration << "s, Avg Throughput: " << mb_s << " MB/s\n";
    
    if (g_show_timings) {
        std::cout << "Timing (s): H2D=" << t_h2d << " Compute=" << t_compute << " D2H=" << t_d2h
              << " Forward=" << t_fwd << " Encode=" << t_encode << "\n";
    }
    
    checkCudaErrors(cudaEventDestroy(ev_start));
    checkCudaErrors(cudaEventDestroy(ev_mid));
    checkCudaErrors(cudaEventDestroy(ev_end));

    free_device((float*)d_rc_states); free_device((float*)d_out_bufs);
    free_device((float*)d_tokens); free_device((float*)d_lengths); free_device(d_logits);
    if (d_sub_tokens) free_device((float*)d_sub_tokens);
    for (int i = 0; i < 2; ++i) {
        free_device((float*)d_batch_data[i]);
        checkCudaErrors(cudaFreeHost(h_batch_data[i]));
        checkCudaErrors(cudaFreeHost(h_out_bufs[i]));
        checkCudaErrors(cudaFreeHost(h_sizes[i]));
    }
    
    std::cout << " Done.\n";

    // Write BOA2 container
    std::ofstream fout(output_path, std::ios::binary);
    const char magic[4] = {'B','O','A','2'};
    const uint32_t version = 1;
    const uint32_t flags_val = warm_start ? 1u : 0u;
    const uint32_t chunk_len_u32 = (uint32_t)chunk_size;
    const uint32_t n_chunks_u32 = (uint32_t)num_chunks;
    const uint32_t last_chunk_u32 = (uint32_t)last_chunk_len;
    const uint8_t fp_len = warm_start ? 4 : 0; // 4 bytes for interleave_batch when warm-start

    fout.write(magic, 4);
    fout.write(reinterpret_cast<const char*>(&version), 4);
    fout.write(reinterpret_cast<const char*>(&flags_val), 4);
    fout.write(reinterpret_cast<const char*>(&processed_size), 8);
    fout.write(reinterpret_cast<const char*>(&chunk_len_u32), 4);
    fout.write(reinterpret_cast<const char*>(&n_chunks_u32), 4);
    fout.write(reinterpret_cast<const char*>(&last_chunk_u32), 4);
    fout.write(reinterpret_cast<const char*>(&fp_len), 1);
    if (warm_start) {
        uint32_t interleave_batch = (uint32_t)BATCH_SIZE;
        fout.write(reinterpret_cast<const char*>(&interleave_batch), 4);
    }

    std::vector<uint64_t> offsets(num_chunks, 0);
    std::vector<uint64_t> lengths(num_chunks, 0);
    uint64_t off = 0;
    for (int i = 0; i < num_chunks; ++i) {
        offsets[i] = off;
        lengths[i] = all_compressed_chunks[i].size();
        fout.write(reinterpret_cast<const char*>(all_compressed_chunks[i].data()), all_compressed_chunks[i].size());
        off += lengths[i];
    }

    std::vector<uint8_t> idx;
    idx.insert(idx.end(), {'I','D','X','2'});
    idx.insert(idx.end(), all_first_bytes.begin(), all_first_bytes.end());
    // IDX2: only lengths (offsets are reconstructed as cumulative sum)
    for (int i = 0; i < num_chunks; ++i) {
        uvarint_encode(idx, lengths[i]);
    }
    uint32_t crc = crc32_compute(idx.data(), idx.size());
    fout.write(reinterpret_cast<const char*>(idx.data()), idx.size());
    fout.write(reinterpret_cast<const char*>(&crc), 4);
    fout.close();
}

void decompress_v2(const std::string& model_path, const std::string& input_path, const std::string& output_path, MambaConfig config, int gpu_batch, size_t chunk_size) {
    int BATCH_SIZE = (gpu_batch > 0) ? gpu_batch : 64;

    gpu_init_exp_lut();
    
    std::ofstream fout(output_path, std::ios::binary);
    Boa2Container container = read_boa2(input_path);
    size_t total_size = container.total_size;
    chunk_size = container.chunk_len;
    int num_chunks = container.num_chunks;
    bool warm_start = container.warm_start;
    std::vector<std::vector<uint8_t>> all_streams = std::move(container.streams);
    std::vector<uint8_t> all_first_bytes = std::move(container.first_bytes);
    std::vector<int> chunk_lengths = std::move(container.lengths);
    
    if (warm_start) {
        BATCH_SIZE = (int)container.interleave_batch;
        std::cout << "Warm-start decompression. Batch=" << BATCH_SIZE << " (from file)" << std::endl;
    } else {
        std::cout << "Using GPU Batched Decompression. Batch=" << BATCH_SIZE << std::endl;
    }

    BoaPredictorGPU gpu_model(config, 256, config.n_layers, BATCH_SIZE);
    gpu_model.load_weights(model_path);
    
    std::cout << "Decompressing " << total_size << " bytes in " << num_chunks << " chunks..." << std::endl;
    
    // Warm-start interleaving parameters
    int S = warm_start ? ((num_chunks + BATCH_SIZE - 1) / BATCH_SIZE) : 1;
    int num_iters = warm_start ? S : ((num_chunks + BATCH_SIZE - 1) / BATCH_SIZE);

    // Allocate GPU buffers
    int pitch_words = (int)((chunk_size * 10 + 4096 + 3) / 4);
    int pitch_bytes = pitch_words * 4;
    
    RCDecState* d_rc_states;
    unsigned char* d_in_bufs;
    int* d_stream_lengths;
    malloc_device((float**)&d_rc_states, BATCH_SIZE * sizeof(RCDecState));
    malloc_device((float**)&d_in_bufs, (size_t)BATCH_SIZE * pitch_bytes);
    malloc_device((float**)&d_stream_lengths, BATCH_SIZE * sizeof(int));
    
    int* d_tokens;
    int* d_lengths;
    int* d_out_symbols;
    float* d_logits;
    malloc_device((float**)&d_tokens, BATCH_SIZE * sizeof(int));
    malloc_device((float**)&d_lengths, BATCH_SIZE * sizeof(int));
    malloc_device((float**)&d_out_symbols, BATCH_SIZE * sizeof(int));
    malloc_device(&d_logits, (size_t)BATCH_SIZE * 256 * sizeof(float));
    
    std::vector<int> h_tokens(BATCH_SIZE);
    std::vector<int> h_stream_lengths(BATCH_SIZE);
    std::vector<int> h_chunk_lengths(BATCH_SIZE);
    std::vector<unsigned char> h_in_bufs((size_t)BATCH_SIZE * pitch_bytes, 0);
    
    // Output buffer
    std::vector<std::vector<uint8_t>> all_outputs(num_chunks);
    
    int* d_batch_output;
    malloc_device((float**)&d_batch_output, (size_t)BATCH_SIZE * chunk_size * sizeof(int));
    std::vector<int> h_batch_output(BATCH_SIZE * chunk_size);

    auto start_time = std::chrono::high_resolution_clock::now();

    // --- CUDA Graph setup (capture once, replay for all full batches) ---
    int* d_t_counter;
    malloc_device((float**)&d_t_counter, sizeof(int));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaStream_t work_stream;
    checkCudaErrors(cudaStreamCreate(&work_stream));
    g_stream = work_stream;

    int graph_batch_size = -1; // batch size the graph was captured for
    int max_len = (int)chunk_size;
    int chunks_done = 0;

    for (int iter = 0; iter < num_iters; ++iter) {
        // Compute current_batch and abs_idx mapping (same as compress)
        int current_batch;
        if (warm_start) {
            current_batch = 0;
            for (int b = 0; b < BATCH_SIZE; ++b) {
                if ((int64_t)b * S + iter < num_chunks) current_batch = b + 1;
                else break;
            }
        } else {
            int chunk_idx = iter * BATCH_SIZE;
            current_batch = std::min(BATCH_SIZE, num_chunks - chunk_idx);
        }
        
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        print_progress(chunks_done, num_chunks, elapsed, total_size, "Decompressing");

        // Reset model state: warm-start resets only on first iter
        if (!warm_start || iter == 0) {
            gpu_model.reset_cache();
        }
        
        // Prepare input buffers
        memset(h_in_bufs.data(), 0, h_in_bufs.size());
        for (int b = 0; b < current_batch; ++b) {
            int abs_idx = warm_start ? (b * S + iter) : (iter * BATCH_SIZE + b);
            if (abs_idx >= num_chunks) continue;
            size_t sz = all_streams[abs_idx].size();
            h_stream_lengths[b] = (int)(sz / 4);
            if (sz > (size_t)pitch_bytes) { std::cerr << "Stream too large!\n"; return; }
            memcpy(h_in_bufs.data() + (size_t)b * pitch_bytes, all_streams[abs_idx].data(), sz);
            h_tokens[b] = all_first_bytes[abs_idx];
            all_outputs[abs_idx].push_back(all_first_bytes[abs_idx]);
            h_chunk_lengths[b] = chunk_lengths[abs_idx];
        }
        
        checkCudaErrors(cudaMemcpyAsync(d_in_bufs, h_in_bufs.data(), (size_t)current_batch * pitch_bytes, cudaMemcpyHostToDevice, work_stream));
        checkCudaErrors(cudaMemcpyAsync(d_stream_lengths, h_stream_lengths.data(), current_batch * sizeof(int), cudaMemcpyHostToDevice, work_stream));
        checkCudaErrors(cudaMemcpyAsync(d_tokens, h_tokens.data(), current_batch * sizeof(int), cudaMemcpyHostToDevice, work_stream));
        checkCudaErrors(cudaMemcpyAsync(d_lengths, h_chunk_lengths.data(), current_batch * sizeof(int), cudaMemcpyHostToDevice, work_stream));
        checkCudaErrors(cudaStreamSynchronize(work_stream));
        
        // Store initial tokens to output buffer (timestep 0)
        gpu_store_tokens(d_tokens, d_batch_output, 0, (int)chunk_size, current_batch);
        
        // Initialize decoder states
        gpu_rc_init_decoder(d_rc_states, (unsigned int*)d_in_bufs, d_stream_lengths, pitch_words, current_batch);
        
        // Reset t counter to 1
        int t_init = 1;
        checkCudaErrors(cudaMemcpyAsync(d_t_counter, &t_init, sizeof(int), cudaMemcpyHostToDevice, work_stream));

        if (g_no_graph) {
            // --- No-graph fallback: standard per-step loop ---
            for (int t = 1; t < max_len; ++t) {
                gpu_model.step_batch(d_tokens, d_logits, false);
                gpu_rc_decode_fused_step_batch(d_logits, gpu_model.head_b2, d_tokens, d_lengths, t, d_batch_output, (int)chunk_size,
                                               d_rc_states, (unsigned int*)d_in_bufs, d_stream_lengths, pitch_words, 256, current_batch);
            }
            checkCudaErrors(cudaStreamSynchronize(work_stream));
        } else {
            // --- CUDA Graph path ---
            // Capture graph on first full batch, or if batch size changed (last batch)
            if (graph_batch_size != current_batch) {
                if (graph_exec) { checkCudaErrors(cudaGraphExecDestroy(graph_exec)); graph_exec = nullptr; }
                if (graph) { checkCudaErrors(cudaGraphDestroy(graph)); graph = nullptr; }

                checkCudaErrors(cudaStreamSynchronize(work_stream));
                checkCudaErrors(cudaStreamBeginCapture(work_stream, cudaStreamCaptureModeGlobal));

                gpu_model.step_batch(d_tokens, d_logits, false);
                gpu_rc_decode_fused_step_batch_graph(d_logits, gpu_model.head_b2, d_tokens, d_lengths, d_t_counter, d_batch_output, (int)chunk_size,
                                                      d_rc_states, (unsigned int*)d_in_bufs, d_stream_lengths, pitch_words, 256, current_batch);
                gpu_increment_counter(d_t_counter);

                checkCudaErrors(cudaStreamEndCapture(work_stream, &graph));
                checkCudaErrors(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
                graph_batch_size = current_batch;
            }

            // Replay the graph for each timestep
            for (int t = 1; t < max_len; ++t) {
                checkCudaErrors(cudaGraphLaunch(graph_exec, work_stream));
            }
            checkCudaErrors(cudaStreamSynchronize(work_stream));
        }
        
        // Copy all outputs back to CPU
        checkCudaErrors(cudaMemcpy(h_batch_output.data(), d_batch_output, (size_t)BATCH_SIZE * chunk_size * sizeof(int), cudaMemcpyDeviceToHost));
        
        for (int b = 0; b < current_batch; ++b) {
            int abs_idx = warm_start ? (b * S + iter) : (iter * BATCH_SIZE + b);
            if (abs_idx >= num_chunks) continue;
            size_t expected = (size_t)chunk_lengths[abs_idx];
            for(int t=1; t<max_len; ++t) {
                if (t < (int)expected) {
                    all_outputs[abs_idx].push_back((uint8_t)h_batch_output[b * chunk_size + t]);
                }
            }
        }
        chunks_done += current_batch;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    print_progress(num_chunks, num_chunks, duration, total_size, "Decompressing");
    
    double mb_s = (double)total_size / (1024.0 * 1024.0) / duration;
    std::cout << "\nDone. Time: " << duration << "s, Avg Throughput: " << mb_s << " MB/s\n";

    // Write all outputs
    for(int i=0; i<num_chunks; ++i) {
        fout.write((char*)all_outputs[i].data(), all_outputs[i].size());
    }
    
    free_device((float*)d_rc_states); 
    free_device((float*)d_in_bufs);
    free_device((float*)d_stream_lengths);
    free_device((float*)d_tokens); 
    free_device((float*)d_lengths);
    free_device((float*)d_out_symbols);
    free_device(d_logits);
    free_device((float*)d_batch_output);
    free_device((float*)d_t_counter);
    if (graph_exec) checkCudaErrors(cudaGraphExecDestroy(graph_exec));
    if (graph) checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaStreamDestroy(work_stream));
    g_stream = 0;
    
    fout.close();
    std::cout << " Done.\n";
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: boa <mode> <model> <input> <output> [d_model] [n_layers] [--backbone TYPE] [--gpu-batch B] [--max-chunks C] [--chunk-size S] [--tile T] [--warm-start] [--temperature T]\n";
        std::cerr << "  Backbones: mamba (default), lstm, gru, mingru\n";
        std::cerr << "  --tile T: Sub-chunk forward pass into tiles of T tokens (reduces VRAM, allows larger batches)\n";
        std::cerr << "  --warm-start: Interleaved chunk order for model state carry-forward (compress only)\n";
        std::cerr << "  --temperature T: Scale logits by 1/T before softmax (default: 1.0, <1 = sharper, >1 = softer)\n";
        return 1;
    }

    std::string mode = argv[1];
    std::vector<std::string> args;
    int gpu_batch = 0;
    int max_chunks = -1;
    size_t chunk_size = g_chunk_size;
    std::string backbone_str = "mamba";
    bool warm_start = false;
    int tile_size = 0;
    float temperature = 1.0f;

    for(int i=0; i<argc; ++i) {
        std::string s = argv[i];
        if (s == "--gpu-batch" && i + 1 < argc) {
            gpu_batch = std::stoi(argv[i+1]);
            ++i;
            continue;
        }
        if (s == "--max-chunks" && i + 1 < argc) {
            max_chunks = std::stoi(argv[i+1]);
            ++i;
            continue;
        }
        if (s == "--chunk-size" && i + 1 < argc) {
            chunk_size = (size_t)std::stoull(argv[i+1]);
            ++i;
            continue;
        }
        if (s == "--backbone" && i + 1 < argc) {
            backbone_str = argv[i+1];
            ++i;
            continue;
        }
        if (s == "--show-timings") {
            g_show_timings = true;
            continue;
        }
        if (s == "--no-graph") {
            g_no_graph = true;
            continue;
        }
        if (s == "--warm-start") {
            warm_start = true;
            continue;
        }
        if (s == "--tile" && i + 1 < argc) {
            tile_size = std::stoi(argv[i+1]);
            ++i;
            continue;
        }
        if (s == "--temperature" && i + 1 < argc) {
            temperature = std::stof(argv[i+1]);
            ++i;
            continue;
        }
        if (s == "--gpu") continue; // Ignore legacy flag
        if (s.rfind("--", 0) == 0) {
            std::cerr << "Warning: Unknown flag '" << s << "' ignored.\n";
            continue;
        }
        args.push_back(s);
    }

    if (args.size() < 5) return 1;

    std::string model_path = args[2];
    std::string input_path = args[3];
    std::string output_path = args[4];

    MambaConfig config;
    config.d_model = 256;
    config.n_layers = 1;
    config.backbone = backbone_from_string(backbone_str.c_str());

    if (args.size() > 5) config.d_model = std::stoi(args[5]);
    if (args.size() > 6) config.n_layers = std::stoi(args[6]);
    config.update();
    config.use_rmsnorm = false;

    std::cout << "Backbone: " << backbone_name(config.backbone) << std::endl;
    if (temperature != 1.0f) {
        std::cout << "Temperature: " << temperature << std::endl;
        gpu_set_temperature(temperature);
    }

    if (chunk_size == 0) chunk_size = g_chunk_size;

    if (mode == "compress") {
        if (gpu_batch == 0) auto_size_params(config, gpu_batch, chunk_size, true);
        compress_v2(model_path, input_path, output_path, config, max_chunks, gpu_batch, chunk_size, warm_start, tile_size);
    } else if (mode == "decompress") {
        if (gpu_batch == 0) auto_size_params(config, gpu_batch, chunk_size, false);
        decompress_v2(model_path, input_path, output_path, config, gpu_batch, chunk_size);
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }
    
    return 0;
}
