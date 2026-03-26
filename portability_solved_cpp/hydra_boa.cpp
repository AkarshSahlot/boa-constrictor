/*
 * hydra_boa.cpp — C++/CUDA HydraBOA compressor/decompressor.
 *
 * Standalone binary (does NOT modify boa.cpp).  Reuses the same GPU
 * kernels (gemm_gpu, boa_gpu, range_coder_kernels, …) and adds:
 *
 *   • Tiled compression  (--tile T)  — reduces VRAM, warp-encodes
 *     full tiles at once with double-precision CDF (matching decoder).
 *   • CUDA-graph decompression (--no-graph to disable) — captures
 *     one backbone step (K chained head decodes) and replays it.
 *   • All inner-loop operations stay on-device — no host↔device
 *     copies during backbone / decode loops.
 *
 * Architecture:
 *   K=4 bytes per backbone step.  Chunked Mamba backbone with K
 *   chained prediction heads (head k receives backbone context
 *   concatenated with embeddings of the k preceding bytes).
 *
 * Compilation:
 *   nvcc -o hydra_boa hydra_boa.cpp hydra_kernels.cu gemm_kernels.cu \
 *        inference_kernels.cu mamba_kernels.cu rnn_kernels.cu      \
 *        range_coder_kernels.cu utility_kernels.cu                  \
 *        -O3 -std=c++17 --fmad=false                               \
 *        -DENABLE_GPU -DGPU_DEBUG_LOGITS=0 -DGPU_FAST_EXP=0
 *
 * Usage:
 *   ./hydra_boa compress   <model.bin> <in> <out.boa> [--gpu-batch B] [--tile T] [--chunk-size S]
 *   ./hydra_boa decompress <model.bin> <in.boa> <out>  [--gpu-batch B] [--no-graph]
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>
#include "boa_gpu.hpp"
#include "gemm_gpu.hpp"
#include "activations.hpp"

// Hydra-specific GPU kernels (hydra_kernels.cu)
void hydra_store_decoded(const int* d_decoded, int* d_batch_out,
                         const int* d_step, int K, int chunk_size, int batch);
void hydra_store_decoded_host(const int* d_decoded, int* d_batch_out,
                              int step, int K, int chunk_size, int batch);
void hydra_build_head_input(const float* H_t, const float* byte_embeds,
                            float* head_inp, int D, int k, int K,
                            int out_stride, int batch);
void hydra_broadcast_vector(const float* src, float* dst, int D, int batch);
void hydra_fill_bos_strided(const float* src, float* dst,
                            int D, int tile_T, int batch);
void hydra_scatter_logits(const float* src, float* dst,
                          int tile_T, int K, int k, int V, int batch);

bool g_show_timings = false;
bool g_no_graph     = false;

// ════════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════════
static void print_progress(int cur, int tot, double elapsed,
                           size_t total_bytes, const std::string& prefix = "") {
    float pct = (float)cur / tot;
    int bw = 40, pos = (int)(bw * pct);
    std::cout << "\r" << prefix << " [";
    for (int i = 0; i < bw; ++i) std::cout << (i < pos ? '=' : (i == pos ? '>' : ' '));
    double mb_s = elapsed > 0 ? ((double)cur / tot * total_bytes) / (1024.0*1024.0) / elapsed : 0;
    std::cout << "] " << (int)(pct*100) << "% (" << cur << "/" << tot << ") "
              << std::fixed << std::setprecision(2) << mb_s << " MB/s" << std::flush;
}

// CRC32
static uint32_t crc32_table[256];
static bool crc32_init = false;
static void init_crc32() {
    if (crc32_init) return;
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i; for (int k = 0; k < 8; ++k) c = (c&1) ? (0xEDB88320u ^ (c>>1)) : (c>>1);
        crc32_table[i] = c;
    }
    crc32_init = true;
}
static uint32_t crc32_compute(const uint8_t* d, size_t n) {
    init_crc32(); uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = crc32_table[(c^d[i])&0xFF] ^ (c>>8);
    return c ^ 0xFFFFFFFFu;
}
static void uvarint_encode(std::vector<uint8_t>& out, uint64_t x) {
    while (true) { uint8_t b = x & 0x7F; x >>= 7; out.push_back(b | (x ? 0x80 : 0)); if (!x) break; }
}
static uint64_t uvarint_decode(const std::vector<uint8_t>& buf, size_t& p) {
    uint64_t x = 0; int s = 0;
    while (true) { uint8_t b = buf[p++]; x |= (uint64_t)(b&0x7F)<<s; if (!(b&0x80)) break; s += 7; }
    return x;
}

// BOA2 container
struct Boa2Container {
    uint64_t total_size=0; uint32_t chunk_len=0; uint32_t last_chunk_len=0;
    int num_chunks=0; bool warm_start=false; uint32_t interleave_batch=0;
    std::vector<uint8_t> first_bytes;
    std::vector<std::vector<uint8_t>> streams;
    std::vector<int> lengths;
};
static Boa2Container read_boa2(const std::string& path) {
    Boa2Container out;
    std::ifstream fin(path, std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(fin)), {});
    fin.close();
    if (data.size() < 32) throw std::runtime_error("Invalid BOA2");
    size_t p = 0;
    if (memcmp(data.data(), "BOA2", 4)) throw std::runtime_error("Bad magic");
    p += 4;
    uint32_t ver; memcpy(&ver, data.data()+p, 4); p += 4;
    uint32_t fl; memcpy(&fl, data.data()+p, 4); p += 4;
    out.warm_start = fl & 1;
    memcpy(&out.total_size, data.data()+p, 8); p += 8;
    memcpy(&out.chunk_len, data.data()+p, 4); p += 4;
    uint32_t nc; memcpy(&nc, data.data()+p, 4); p += 4; out.num_chunks = nc;
    memcpy(&out.last_chunk_len, data.data()+p, 4); p += 4;
    uint8_t fp = data[p++];
    if (out.warm_start && fp >= 4) memcpy(&out.interleave_batch, data.data()+p, 4);
    p += fp;
    size_t payload_start = p;
    size_t idx_pos = std::string::npos; int idx_ver = 0;
    for (size_t i = data.size()-4; i-- > 0;) {
        if (data[i]=='I'&&data[i+1]=='D'&&data[i+2]=='X') {
            if (data[i+3]=='2') { idx_pos=i; idx_ver=2; break; }
            if (data[i+3]=='1') { idx_pos=i; idx_ver=1; break; }
        }
    }
    if (idx_pos == std::string::npos) throw std::runtime_error("IDX not found");
    uint32_t crc_f; memcpy(&crc_f, data.data()+data.size()-4, 4);
    if (crc_f != crc32_compute(data.data()+idx_pos, data.size()-idx_pos-4))
        throw std::runtime_error("Bad CRC");
    size_t q = idx_pos + 4;
    out.first_bytes.resize(out.num_chunks);
    for (int i = 0; i < out.num_chunks; ++i) out.first_bytes[i] = data[q++];
    std::vector<uint64_t> offs(out.num_chunks), lens(out.num_chunks);
    if (idx_ver == 1) {
        uint64_t pv = 0;
        for (int i = 0; i < out.num_chunks; ++i) { uint64_t d=uvarint_decode(data,q); pv+=d; offs[i]=pv; }
        for (int i = 0; i < out.num_chunks; ++i) lens[i]=uvarint_decode(data,q);
    } else {
        for (int i = 0; i < out.num_chunks; ++i) lens[i]=uvarint_decode(data,q);
        uint64_t o=0; for (int i = 0; i < out.num_chunks; ++i) { offs[i]=o; o+=lens[i]; }
    }
    std::vector<uint8_t> pay(data.begin()+payload_start, data.begin()+idx_pos);
    out.streams.resize(out.num_chunks);
    for (int i = 0; i < out.num_chunks; ++i) {
        if (offs[i]+lens[i] > pay.size()) throw std::runtime_error("OOB");
        out.streams[i].assign(pay.begin()+offs[i], pay.begin()+offs[i]+lens[i]);
    }
    out.lengths.resize(out.num_chunks);
    for (int i = 0; i < out.num_chunks; ++i) {
        if (i == out.num_chunks - 1)
            out.lengths[i] = (int)(out.total_size - (uint64_t)(out.num_chunks-1) * out.chunk_len);
        else
            out.lengths[i] = (int)out.chunk_len;
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
//  HydraBoaGPU — model on device
// ════════════════════════════════════════════════════════════════════
struct HydraHeadGPU {
    float *w1, *b1, *w2, *b2, *buf;
    int in_dim, d_model, vocab_size, batch_size;

    void allocate(int k, int d, int V, int batch) {
        in_dim = (1+k)*d; d_model = d; vocab_size = V; batch_size = batch;
        malloc_device(&w1, (size_t)d*in_dim*sizeof(float));
        malloc_device(&b1, d*sizeof(float));
        malloc_device(&w2, (size_t)V*d*sizeof(float));
        malloc_device(&b2, V*sizeof(float));
        malloc_device(&buf, (size_t)batch*d*sizeof(float));
    }
    void free_mem() { free_device(w1); free_device(b1); free_device(w2); free_device(b2); free_device(buf); }
    void load_weights(std::ifstream& f) {
        load_vec(f, w1, d_model*in_dim, true, d_model, in_dim);
        load_vec(f, b1, d_model);
        load_vec(f, w2, vocab_size*d_model, true, vocab_size, d_model);
        load_vec(f, b2, vocab_size);
    }
    void forward(float* inp, float* logits, int cb) {
        gemm_gpu_batch_bias_act(inp, w1, b1, buf, cb, d_model, in_dim, GEMM_ACT_RELU);
        gemm_gpu_batch_bias(buf, w2, b2, logits, cb, vocab_size, d_model);
    }
    // Chunk-mode forward using external intermediate buffer (avoids buf size limit)
    void forward_ext(float* inp, float* logits, float* ext_buf, int cb) {
        gemm_gpu_batch_bias_act(inp, w1, b1, ext_buf, cb, d_model, in_dim, GEMM_ACT_RELU);
        gemm_gpu_batch_bias(ext_buf, w2, b2, logits, cb, vocab_size, d_model);
    }
};

struct HydraBoaGPU {
    int d_model, n_layers, K, vocab_size, batch_size;

    float *embedding, *bos_embed;
    float *chunk_proj_w, *chunk_proj_b;
    BoaBlockGPU* blocks;
    float *backbone_norm_w, *backbone_norm_b;
    HydraHeadGPU* heads;

    // Persistent buffers
    float *buf_x, *buf_res, *buf_chunk_emb, *buf_logits;
    float *buf_head_inp;      // [batch, K*D]  max head input
    float *buf_byte_embeds;   // [batch, K, D]
    float *buf_single_embed;  // [batch, D]  scratch for one embedding

    void allocate(int d, int layers, int k, int V, int batch) {
        d_model = d; n_layers = layers; K = k; vocab_size = V; batch_size = batch;

        malloc_device(&embedding, (size_t)V*d*sizeof(float));
        malloc_device(&bos_embed, d*sizeof(float));
        malloc_device(&chunk_proj_w, (size_t)d*(k*d)*sizeof(float));
        malloc_device(&chunk_proj_b, d*sizeof(float));

        MambaConfig conf;
        conf.d_model = d; conf.n_layers = layers;
        conf.backbone = BACKBONE_MAMBA; conf.use_rmsnorm = false; conf.update();

        blocks = new BoaBlockGPU[layers];
        for (int i = 0; i < layers; ++i) blocks[i].allocate(conf, batch);

        malloc_device(&backbone_norm_w, d*sizeof(float));
        malloc_device(&backbone_norm_b, d*sizeof(float));

        heads = new HydraHeadGPU[K];
        for (int i = 0; i < K; ++i) heads[i].allocate(i, d, V, batch);

        malloc_device(&buf_x,            (size_t)batch*d*sizeof(float));
        malloc_device(&buf_res,          (size_t)batch*d*sizeof(float));
        malloc_device(&buf_chunk_emb,    (size_t)batch*K*d*sizeof(float));
        malloc_device(&buf_logits,       (size_t)batch*V*sizeof(float));
        malloc_device(&buf_head_inp,     (size_t)batch*K*d*sizeof(float));
        malloc_device(&buf_byte_embeds,  (size_t)batch*K*d*sizeof(float));
        malloc_device(&buf_single_embed, (size_t)batch*d*sizeof(float));
    }

    void load_weights(const std::string& path) {
        std::cout << "Loading HydraBOA weights..." << std::endl;
        std::ifstream f(path, std::ios::binary);
        char magic[4]; f.read(magic, 4);
        if (magic[0]!='H'||magic[1]!='Y'||magic[2]!='D')
            throw std::runtime_error("Bad Hydra magic");
        g_model_fp16 = (magic[3] == 0x01);
        uint32_t hd, hl, hk;
        f.read((char*)&hd,4); f.read((char*)&hl,4); f.read((char*)&hk,4);
        std::cout << "  d=" << hd << " L=" << hl << " K=" << hk
                  << (g_model_fp16?" fp16":" fp32") << std::endl;
        if ((int)hd!=d_model||(int)hl!=n_layers||(int)hk!=K)
            throw std::runtime_error("Shape mismatch");

        { int sz = vocab_size*d_model; std::vector<float> tmp(sz);
          if (g_model_fp16) read_fp16_to_fp32(f,tmp,sz); else f.read((char*)tmp.data(),sz*4);
          copy_to_device(embedding,tmp.data(),sz*4); }
        load_vec(f, bos_embed, d_model);
        load_vec(f, chunk_proj_w, d_model*K*d_model, true, d_model, K*d_model);
        load_vec(f, chunk_proj_b, d_model);
        for (int i = 0; i < n_layers; ++i) blocks[i].load_weights(f);
        load_vec(f, backbone_norm_w, d_model);
        load_vec(f, backbone_norm_b, d_model);
        for (int i = 0; i < K; ++i) heads[i].load_weights(f);
        std::cout << "  Done." << std::endl;
    }

    void reset_cache() { for (int i = 0; i < n_layers; ++i) blocks[i].reset_cache(); }

    // Backbone step.  d_prev_chunk: [cb, K] or NULL (BOS).
    // Writes H_t into buf_x [cb, D].
    void step_backbone(int* d_prev_chunk, int cb) {
        if (d_prev_chunk == nullptr) {
            hydra_broadcast_vector(bos_embed, buf_x, d_model, cb);
        } else {
            gpu_embedding_lookup_batch(d_prev_chunk, embedding, buf_chunk_emb, d_model, (size_t)cb*K);
            gemm_gpu_batch_bias(buf_chunk_emb, chunk_proj_w, chunk_proj_b, buf_x, cb, d_model, K*d_model);
        }
        for (int i = 0; i < n_layers; ++i) blocks[i].step_batch(buf_x, buf_x, buf_res);
        gpu_layernorm_batch(buf_x, backbone_norm_w, backbone_norm_b, d_model, cb);
    }

    // Head k forward.  For k>0, builds input on device.
    // Writes logits to buf_logits [cb, V].
    void predict_head(int k, int cb) {
        if (k == 0) {
            heads[0].forward(buf_x, buf_logits, cb);
        } else {
            hydra_build_head_input(buf_x, buf_byte_embeds, buf_head_inp,
                                   d_model, k, K, (1+k)*d_model, cb);
            heads[k].forward(buf_head_inp, buf_logits, cb);
        }
    }

    // After decoding byte k, embed it and scatter into buf_byte_embeds.
    // d_out_sym: [cb] decoded symbols.
    void embed_and_scatter(int* d_out_sym, int k, int cb) {
        gpu_embedding_lookup_batch(d_out_sym, embedding, buf_single_embed, d_model, cb);
        gpu_scatter_timestep(buf_single_embed, buf_byte_embeds, cb, K, k, d_model);
    }

    void free_mem() {
        free_device(embedding); free_device(bos_embed);
        free_device(chunk_proj_w); free_device(chunk_proj_b);
        for (int i = 0; i < n_layers; ++i) blocks[i].free();
        delete[] blocks;
        free_device(backbone_norm_w); free_device(backbone_norm_b);
        for (int i = 0; i < K; ++i) heads[i].free_mem();
        delete[] heads;
        free_device(buf_x); free_device(buf_res); free_device(buf_chunk_emb);
        free_device(buf_logits); free_device(buf_head_inp);
        free_device(buf_byte_embeds); free_device(buf_single_embed);
    }

    // ── Chunk-parallel compression infrastructure ──
    int max_tile_T = 0;
    float* chunk_backbone_in  = nullptr;   // [batch, tile_T, D]
    float* chunk_backbone_res = nullptr;   // [batch, tile_T, D]
    float* chunk_head_buf     = nullptr;   // [batch*tile_T, D] (intermediate for head GEMM)
    float* chunk_head_logits  = nullptr;   // [batch*tile_T, V] (contiguous head output)
    float* chunk_all_byte_embeds = nullptr; // [batch*tile_T, K, D] (byte embeds for heads)
    float* chunk_head_inp_buf = nullptr;   // [batch*tile_T, K*D] (head k input — max size)
    int*   chunk_backbone_tokens = nullptr; // [batch*tile_T, K] (int tokens for embed+proj)

    void allocate_chunk(int tile_T) {
        if (max_tile_T >= tile_T) return;
        if (max_tile_T > 0) free_chunk();
        max_tile_T = tile_T;

        // Allocate block chunk buffers (Mamba SSM, conv, etc.)
        for (int i = 0; i < n_layers; ++i) blocks[i].allocate_chunk(tile_T);

        size_t eff = (size_t)batch_size * tile_T;
        malloc_device(&chunk_backbone_in,   eff * d_model * sizeof(float));
        malloc_device(&chunk_backbone_res,  eff * d_model * sizeof(float));
        malloc_device(&chunk_head_buf,      eff * d_model * sizeof(float));
        malloc_device(&chunk_head_logits,   eff * vocab_size * sizeof(float));
        malloc_device(&chunk_all_byte_embeds, eff * K * d_model * sizeof(float));
        malloc_device(&chunk_head_inp_buf,  eff * K * d_model * sizeof(float));
        malloc_device((float**)&chunk_backbone_tokens, eff * K * sizeof(int));
    }

    void free_chunk() {
        if (max_tile_T == 0) return;
        for (int i = 0; i < n_layers; ++i) blocks[i].free_chunk();
        free_device(chunk_backbone_in);   free_device(chunk_backbone_res);
        free_device(chunk_head_buf);      free_device(chunk_head_logits);
        free_device(chunk_all_byte_embeds); free_device(chunk_head_inp_buf);
        free_device((float*)chunk_backbone_tokens);
        max_tile_T = 0;
    }

    // Chunk-parallel backbone + heads for a tile during compression.
    // All bytes are teacher-forced (known).
    //
    //   d_chunk_data: [cb, chunk_data_stride] int — raw bytes with dummy prefix
    //   t_start_bb:   first backbone step index in the chunk
    //   this_tile_T:  number of backbone steps in this tile
    //   d_tile_logits: [cb, this_tile_T*K, V] — OUTPUT interleaved logits
    //   cb:           current batch size
    //   chunk_data_stride: row stride of d_chunk_data
    void forward_chunk_compress(int* d_chunk_data, int t_start_bb,
                                int this_tile_T, float* d_tile_logits,
                                int cb, int chunk_data_stride) {
        int D = d_model;
        int V = vocab_size;
        size_t eff = (size_t)cb * this_tile_T;  // total rows

        // ── 1. Build backbone inputs [cb, this_tile_T, D] ──
        // Zero backbone tokens for valid batch slots
        checkCudaErrors(cudaMemsetAsync(chunk_backbone_tokens, 0,
            (size_t)cb * this_tile_T * K * sizeof(int), g_stream));

        // Gather backbone tokens for cb valid batch entries
        if (t_start_bb == 0) {
            if (this_tile_T > 1) {
                // Steps 1..this_tile_T-1: prev bytes start at chunk_data offset 1
                checkCudaErrors(cudaMemcpy2DAsync(
                    chunk_backbone_tokens + K,                        // skip step-0 slot
                    (size_t)this_tile_T * K * sizeof(int),            // dst pitch
                    d_chunk_data + 1,                                 // src: byte 0
                    (size_t)chunk_data_stride * sizeof(int),          // src pitch
                    (size_t)(this_tile_T - 1) * K * sizeof(int),      // width
                    cb,                                               // height
                    cudaMemcpyDeviceToDevice, g_stream));
            }
        } else {
            int byte_offset = 1 + (t_start_bb - 1) * K;
            checkCudaErrors(cudaMemcpy2DAsync(
                chunk_backbone_tokens,
                (size_t)this_tile_T * K * sizeof(int),
                d_chunk_data + byte_offset,
                (size_t)chunk_data_stride * sizeof(int),
                (size_t)this_tile_T * K * sizeof(int),
                cb,
                cudaMemcpyDeviceToDevice, g_stream));
        }

        // Embed tokens: [cb*this_tile_T*K, D] into temp buffer
        gpu_embedding_lookup_batch(chunk_backbone_tokens, embedding,
            chunk_head_inp_buf, D, (int)(eff * K));

        // Chunk proj GEMM: [cb*this_tile_T, K*D] → [cb*this_tile_T, D]
        gemm_gpu_batch_bias(chunk_head_inp_buf, chunk_proj_w, chunk_proj_b,
            chunk_backbone_in, (int)eff, D, K * D);

        // Overwrite BOS positions if needed
        if (t_start_bb == 0) {
            hydra_fill_bos_strided(bos_embed, chunk_backbone_in, D, this_tile_T, cb);
        }

        // ── 2. Forward through blocks (chunk-parallel Mamba, fused SSM) ──
        for (int i = 0; i < n_layers; ++i) {
            blocks[i].forward_chunk_fused(chunk_backbone_in, chunk_backbone_in,
                                    chunk_backbone_res, this_tile_T);
        }

        // ── 3. Backbone norm ──
        gpu_layernorm_batch(chunk_backbone_in, backbone_norm_w, backbone_norm_b,
                           D, (int)eff);

        // Now chunk_backbone_in = H [cb, this_tile_T, D] = [cb*this_tile_T, D]

        // ── 4. Pre-compute all byte embeddings for head inputs ──
        // Bytes being predicted at step t are bytes[t*K+j] for j=0..K-1
        // = chunk_data[b*stride + 1 + t_start_bb*K + t*K + j]
        {
            int pred_byte_offset = 1 + t_start_bb * K;
            checkCudaErrors(cudaMemcpy2DAsync(
                chunk_backbone_tokens, (size_t)this_tile_T * K * sizeof(int),
                d_chunk_data + pred_byte_offset, (size_t)chunk_data_stride * sizeof(int),
                (size_t)this_tile_T * K * sizeof(int), cb,
                cudaMemcpyDeviceToDevice, g_stream));

            // Embed: [cb*this_tile_T*K] → [cb*this_tile_T*K, D]
            gpu_embedding_lookup_batch(chunk_backbone_tokens, embedding,
                chunk_all_byte_embeds, D, (int)(eff * K));
            // Layout: [cb*this_tile_T, K, D] — K embeddings per (batch,step)
        }

        // ── 5. Compute K head logits and scatter ──
        for (int k = 0; k < K; ++k) {
            if (k == 0) {
                // Head 0: input = H_t [eff, D]
                heads[0].forward_ext(chunk_backbone_in, chunk_head_logits,
                                     chunk_head_buf, (int)eff);
            } else {
                // Head k: input = [H_t, embed(byte_0)..embed(byte_{k-1})]
                // Build [eff, (1+k)*D] from H_t and chunk_all_byte_embeds
                hydra_build_head_input(chunk_backbone_in, chunk_all_byte_embeds,
                    chunk_head_inp_buf, D, k, K, (1+k)*D, (int)eff);
                heads[k].forward_ext(chunk_head_inp_buf, chunk_head_logits,
                                     chunk_head_buf, (int)eff);
            }
            // Scatter [eff, V] → d_tile_logits[cb, this_tile_T*K, V] at slice k
            hydra_scatter_logits(chunk_head_logits, d_tile_logits,
                                this_tile_T, K, k, V, cb);
        }
    }
};

// ════════════════════════════════════════════════════════════════════
//  Compress  (tiled, warp-encoded)
// ════════════════════════════════════════════════════════════════════
//
//  • Each tile spans tile_T backbone steps = tile_T*K bytes.
//  • All K heads per backbone step are teacher-forced (bytes known).
//  • Logits are accumulated in [batch, tile_T*K, V] then warp-encoded
//    in one call — double-precision CDF matches the warp decoder.
//  • Layout trick: we prepend a dummy byte-0 to the chunk_data so
//    that warp encoder token indexing (t+1) aligns naturally.
//
void hydra_compress(const std::string& model_path, const std::string& input_path,
                    const std::string& output_path, int d_model, int n_layers,
                    int K, int gpu_batch, size_t chunk_size, int tile_size) {
    std::cout << "Loading data..." << std::endl;
    std::ifstream fin(input_path, std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(fin)), {});
    fin.close();
    size_t total_size = data.size();

    if (chunk_size == 0) chunk_size = 4096;
    chunk_size = (chunk_size / K) * K;
    if (chunk_size == 0) chunk_size = K;

    int num_chunks = (int)((total_size + chunk_size - 1) / chunk_size);
    size_t processed_size = std::min(total_size, (size_t)num_chunks * chunk_size);
    size_t last_chunk_len = (num_chunks > 1) ? (processed_size - (size_t)(num_chunks-1)*chunk_size) : processed_size;
    if (last_chunk_len % K != 0) {
        size_t pad = K - (last_chunk_len % K);
        data.resize(data.size() + pad, 0);
        last_chunk_len += pad; processed_size += pad;
    }

    int BATCH = (gpu_batch > 0) ? gpu_batch : 64;
    int T_chunk = (int)(chunk_size / K);         // backbone steps per chunk

    // Tile: number of backbone steps per tile
    if (tile_size <= 0 || tile_size > (int)chunk_size) tile_size = (int)chunk_size;
    tile_size = (tile_size / K) * K;
    if (tile_size == 0) tile_size = (int)chunk_size;
    int tile_T   = tile_size / K;                // backbone steps per tile

    // Adjust tile_T to evenly divide T_chunk (avoids remainder-tile non-determinism)
    if (T_chunk % tile_T != 0) {
        // Find nearest divisor of T_chunk to requested tile_T
        int below = tile_T;
        while (below > 0 && T_chunk % below != 0) below--;
        int above = tile_T;
        while (above <= T_chunk && T_chunk % above != 0) above++;
        if (above <= T_chunk && (above - tile_T) < (tile_T - below))
            tile_T = above;
        else
            tile_T = below;
        tile_size = tile_T * K;
    }
    int num_tiles = T_chunk / tile_T;

    std::cout << "Hydra compress: " << processed_size << " B, " << num_chunks
              << " chunks (T=" << T_chunk << "), K=" << K << ", batch=" << BATCH
              << ", tile=" << tile_size << " (" << num_tiles << " tiles/chunk)" << std::endl;

    gpu_init_exp_lut();

    HydraBoaGPU model;
    model.allocate(d_model, n_layers, K, 256, BATCH);
    model.load_weights(model_path);

    // Allocate chunk-parallel buffers for the tile
    model.allocate_chunk(tile_T);

    // ── Create two CUDA streams for pipeline ──
    cudaStream_t stream_fwd, stream_enc;
    checkCudaErrors(cudaStreamCreate(&stream_fwd));
    checkCudaErrors(cudaStreamCreate(&stream_enc));
    // CUDA events for synchronization between streams
    cudaEvent_t evt_fwd_done, evt_enc_done;
    checkCudaErrors(cudaEventCreate(&evt_fwd_done));
    checkCudaErrors(cudaEventCreate(&evt_enc_done));

    // ── device buffers ──
    // chunk_data with dummy prefix: [BATCH, chunk_size + 1]
    int chunk_data_stride = (int)chunk_size + 1;
    // Double-buffered chunk data for upload overlap
    int* d_chunk_data[2];
    malloc_device((float**)&d_chunk_data[0], (size_t)BATCH * chunk_data_stride * sizeof(int));
    malloc_device((float**)&d_chunk_data[1], (size_t)BATCH * chunk_data_stride * sizeof(int));

    // Double-buffered tile logits: [BATCH, tile_T*K, 256]
    float* d_tile_logits[2];
    malloc_device(&d_tile_logits[0], (size_t)BATCH * tile_T * K * 256 * sizeof(float));
    malloc_device(&d_tile_logits[1], (size_t)BATCH * tile_T * K * 256 * sizeof(float));

    // Lengths: [BATCH] = chunk_data_stride for all
    int* d_lengths;
    malloc_device((float**)&d_lengths, BATCH * sizeof(int));

    // RC state + output
    int pitch_words = (int)((chunk_size * 10 + 4096 + 3) / 4);
    int pitch_bytes = pitch_words * 4;
    RCState* d_rc;
    malloc_device((float**)&d_rc, BATCH * sizeof(RCState));
    unsigned char* d_out_bufs;
    malloc_device((float**)&d_out_bufs, (size_t)BATCH * pitch_bytes);
    int* d_sizes;
    malloc_device((float**)&d_sizes, BATCH * sizeof(int));

    // Double-buffered host staging for upload overlap
    int* h_chunk_data[2];
    checkCudaErrors(cudaHostAlloc((void**)&h_chunk_data[0], (size_t)BATCH * chunk_data_stride * sizeof(int), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc((void**)&h_chunk_data[1], (size_t)BATCH * chunk_data_stride * sizeof(int), cudaHostAllocDefault));
    unsigned char* h_out_buf;
    checkCudaErrors(cudaHostAlloc((void**)&h_out_buf, (size_t)BATCH * pitch_bytes, cudaHostAllocDefault));
    int* h_sizes;
    checkCudaErrors(cudaHostAlloc((void**)&h_sizes, BATCH * sizeof(int), cudaHostAllocDefault));

    std::vector<std::vector<uint8_t>> all_compressed(num_chunks);
    std::vector<uint8_t> all_first_bytes(num_chunks, 0);
    int num_iters = (num_chunks + BATCH - 1) / BATCH;
    int chunks_done = 0;
    uint64_t running_logit_hash = 0;  // running FNV-1a hash of logits

    // Timing accumulators
    double t_host_fill = 0;

    // Set lengths once (doesn't change between iterations)
    { std::vector<int> lens(BATCH, chunk_data_stride);
      checkCudaErrors(cudaMemcpy(d_lengths, lens.data(), BATCH*sizeof(int), cudaMemcpyHostToDevice)); }

    // Pre-fill first iteration's host data
    int buf_idx = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_iters; ++iter) {
        int ci = iter * BATCH;
        int cb = std::min(BATCH, num_chunks - ci);

        { auto now = std::chrono::high_resolution_clock::now();
          print_progress(chunks_done, num_chunks,
              std::chrono::duration<double>(now-t_start).count(), processed_size, "Compressing"); }

        // ── fill host chunk data (with dummy prefix) ──
        int cur_buf = buf_idx;
        buf_idx ^= 1;
        auto tp0 = std::chrono::high_resolution_clock::now();
        memset(h_chunk_data[cur_buf], 0, (size_t)BATCH * chunk_data_stride * sizeof(int));
        for (int b = 0; b < cb; ++b) {
            int abs = ci + b;
            size_t off = (size_t)abs * chunk_size;
            size_t len = std::min(chunk_size, data.size() - off);
            all_first_bytes[abs] = (len > 0) ? data[off] : 0;
            h_chunk_data[cur_buf][b * chunk_data_stride] = 0;  // dummy
            for (size_t j = 0; j < len; ++j)
                h_chunk_data[cur_buf][b * chunk_data_stride + 1 + j] = data[off + j];
        }
        auto tp1 = std::chrono::high_resolution_clock::now();
        t_host_fill += std::chrono::duration<double>(tp1 - tp0).count();

        // Upload on stream_fwd (will sync within this stream)
        g_stream = stream_fwd;
        checkCudaErrors(cudaMemcpyAsync(d_chunk_data[cur_buf], h_chunk_data[cur_buf],
                                   (size_t)cb * chunk_data_stride * sizeof(int),
                                   cudaMemcpyHostToDevice, stream_fwd));

        gpu_rc_init(d_rc, cb);
        model.reset_cache();

        // ── tiled chunk-parallel forward + pipelined encode ──
        int logit_buf = 0;
        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            int t_start_bb = tile_idx * tile_T;
            int t_end_bb   = std::min(t_start_bb + tile_T, T_chunk);
            int this_tile_T = t_end_bb - t_start_bb;

            // Forward on stream_fwd
            // No need to wait for encode — encode only reads d_tile_logits[prev_buf]
            // while forward writes d_tile_logits[logit_buf] (different buffer)
            g_stream = stream_fwd;
            model.forward_chunk_compress(d_chunk_data[cur_buf], t_start_bb, this_tile_T,
                                         d_tile_logits[logit_buf], cb, chunk_data_stride);

            // Signal forward done
            checkCudaErrors(cudaEventRecord(evt_fwd_done, stream_fwd));

            // Encode on stream_enc — wait for forward to produce logits
            checkCudaErrors(cudaStreamWaitEvent(stream_enc, evt_fwd_done, 0));

            int byte_start = t_start_bb * K;
            int byte_end   = t_end_bb * K;
            bool is_last = (tile_idx == num_tiles - 1);
            int enc_max_len = is_last ? chunk_data_stride : (byte_end + 1);

            g_stream = stream_enc;
            gpu_rc_encode_chunk_warp(
                d_tile_logits[logit_buf], d_chunk_data[cur_buf], d_lengths,
                d_rc, (unsigned int*)d_out_bufs,
                pitch_words, 256, cb,
                chunk_data_stride,
                enc_max_len,
                byte_start,
                this_tile_T * K * 256
            );

            // Signal encode done
            checkCudaErrors(cudaEventRecord(evt_enc_done, stream_enc));

            // Alternate logit buffer
            logit_buf ^= 1;
        }

        // Wait for final encode to finish
        g_stream = stream_fwd;
        checkCudaErrors(cudaStreamWaitEvent(stream_fwd, evt_enc_done, 0));
        gpu_rc_finish_batch(d_rc, (unsigned int*)d_out_bufs, pitch_words, d_sizes, cb);
        checkCudaErrors(cudaStreamSynchronize(stream_fwd));

        checkCudaErrors(cudaMemcpy(h_out_buf, d_out_bufs, (size_t)cb*pitch_bytes, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_sizes, d_sizes, cb*sizeof(int), cudaMemcpyDeviceToHost));
        for (int b = 0; b < cb; ++b) {
            int abs = ci + b;
            int sw = h_sizes[b];
            all_compressed[abs].resize(sw*4);
            memcpy(all_compressed[abs].data(), h_out_buf + (size_t)b*pitch_bytes, sw*4);
        }
        chunks_done += cb;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration<double>(t_end - t_start).count();
    print_progress(num_chunks, num_chunks, dur, processed_size, "Compressing");
    std::cout << "\nDone. " << std::fixed << std::setprecision(3) << dur << "s  "
              << (double)processed_size/(1024.0*1024.0)/dur << " MB/s\n";
    std::cout << "  host_fill=" << std::setprecision(3) << t_host_fill << "s\n";
#ifdef LOGIT_CHECKSUM
    fprintf(stderr, "LOGIT_HASH=%016lx\n", running_logit_hash);
#endif

    // ── Write BOA2 ──
    { std::ofstream fout(output_path, std::ios::binary);
      const char m[4]={'B','O','A','2'}; uint32_t ver=1, fl=0;
      uint32_t cl=(uint32_t)chunk_size, nc=(uint32_t)num_chunks, lc=(uint32_t)last_chunk_len;
      uint8_t fp=0;
      fout.write(m,4); fout.write((char*)&ver,4); fout.write((char*)&fl,4);
      fout.write((char*)&total_size,8); fout.write((char*)&cl,4);
      fout.write((char*)&nc,4); fout.write((char*)&lc,4); fout.write((char*)&fp,1);
      std::vector<uint64_t> lens(num_chunks);
      for (int i=0;i<num_chunks;++i) { lens[i]=all_compressed[i].size(); fout.write((char*)all_compressed[i].data(),all_compressed[i].size()); }
      std::vector<uint8_t> idx; idx.insert(idx.end(),{'I','D','X','2'});
      idx.insert(idx.end(), all_first_bytes.begin(), all_first_bytes.end());
      for (int i=0;i<num_chunks;++i) uvarint_encode(idx, lens[i]);
      uint32_t crc = crc32_compute(idx.data(), idx.size());
      fout.write((char*)idx.data(), idx.size());
      fout.write((char*)&crc, 4); fout.close(); }

    size_t comp_total = 0;
    for (auto& c : all_compressed) comp_total += c.size();
    std::cout << "Compressed: " << comp_total << " B (" << std::setprecision(2)
              << (double)processed_size/comp_total << "x)\n";

    // Cleanup
    free_device((float*)d_chunk_data[0]); free_device((float*)d_chunk_data[1]);
    free_device(d_tile_logits[0]); free_device(d_tile_logits[1]);
    free_device((float*)d_lengths);
    free_device((float*)d_rc); free_device((float*)d_out_bufs); free_device((float*)d_sizes);
    checkCudaErrors(cudaFreeHost(h_chunk_data[0])); checkCudaErrors(cudaFreeHost(h_chunk_data[1]));
    checkCudaErrors(cudaFreeHost(h_out_buf));
    checkCudaErrors(cudaFreeHost(h_sizes));
    checkCudaErrors(cudaStreamDestroy(stream_fwd)); checkCudaErrors(cudaStreamDestroy(stream_enc));
    checkCudaErrors(cudaEventDestroy(evt_fwd_done)); checkCudaErrors(cudaEventDestroy(evt_enc_done));
    model.free_chunk();
    model.free_mem();
}

// ════════════════════════════════════════════════════════════════════
//  Decompress  (CUDA-graph accelerated)
// ════════════════════════════════════════════════════════════════════
//
//  • All decoded bytes stay on-device in d_batch_output [batch, chunk_size].
//  • No host↔device copies in the inner loop.
//  • CUDA graph captures one backbone step (with K chained head decodes)
//    and replays it for t=1..T-1.  t=0 (BOS) runs outside the graph.
//
void hydra_decompress(const std::string& model_path, const std::string& input_path,
                      const std::string& output_path, int d_model, int n_layers,
                      int K, int gpu_batch) {
    Boa2Container container = read_boa2(input_path);
    size_t total_size = container.total_size;
    size_t chunk_size = container.chunk_len;
    int num_chunks = container.num_chunks;
    int BATCH = (gpu_batch > 0) ? gpu_batch : 64;
    int T_chunk = (int)(chunk_size / K);

    std::cout << "Hydra decompress: " << total_size << " B, " << num_chunks
              << " chunks (T=" << T_chunk << "), K=" << K << ", batch=" << BATCH
              << (g_no_graph ? "  --no-graph" : "  CUDA-graph") << std::endl;

    gpu_init_exp_lut();
    HydraBoaGPU model;
    model.allocate(d_model, n_layers, K, 256, BATCH);
    model.load_weights(model_path);

    // ── device buffers ──
    int pitch_words = (int)((chunk_size * 10 + 4096 + 3) / 4);
    int pitch_bytes = pitch_words * 4;

    RCDecState* d_rc;       malloc_device((float**)&d_rc, BATCH*sizeof(RCDecState));
    unsigned char* d_in_bufs; malloc_device((float**)&d_in_bufs, (size_t)BATCH*pitch_bytes);
    int* d_stream_lengths;  malloc_device((float**)&d_stream_lengths, BATCH*sizeof(int));

    int* d_prev_chunk;      malloc_device((float**)&d_prev_chunk, (size_t)BATCH*K*sizeof(int));
    int* d_decoded;         malloc_device((float**)&d_decoded, (size_t)BATCH*K*sizeof(int));
    int* d_out_sym;         malloc_device((float**)&d_out_sym, BATCH*sizeof(int));
    int* d_batch_output;    malloc_device((float**)&d_batch_output, (size_t)BATCH*chunk_size*sizeof(int));

    // Device-side step counter for CUDA graph
    int* d_step_counter;    malloc_device((float**)&d_step_counter, sizeof(int));

    // Host staging for compressed-stream upload (pinned)
    std::vector<int> h_stream_lengths(BATCH, 0);
    unsigned char* h_in_bufs;
    checkCudaErrors(cudaHostAlloc((void**)&h_in_bufs, (size_t)BATCH*pitch_bytes, cudaHostAllocDefault));

    // Output copy buffer (pinned)
    int* h_batch_output;
    checkCudaErrors(cudaHostAlloc((void**)&h_batch_output, (size_t)BATCH*chunk_size*sizeof(int), cudaHostAllocDefault));

    // ── CUDA graph plumbing ──
    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaStream_t    work_stream;
    checkCudaErrors(cudaStreamCreate(&work_stream));
    g_stream = work_stream;
    int graph_cb = -1;  // batch size graph was captured for

    std::vector<std::vector<uint8_t>> all_outputs(num_chunks);
    int num_iters = (num_chunks + BATCH - 1) / BATCH;
    int chunks_done = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_iters; ++iter) {
        int ci = iter * BATCH;
        int cb = std::min(BATCH, num_chunks - ci);

        { auto now = std::chrono::high_resolution_clock::now();
          print_progress(chunks_done, num_chunks,
              std::chrono::duration<double>(now-t_start).count(), total_size, "Decompressing"); }

        model.reset_cache();

        // Upload compressed streams
        memset(h_in_bufs, 0, (size_t)BATCH*pitch_bytes);
        for (int b = 0; b < cb; ++b) {
            int abs = ci + b;
            size_t sz = container.streams[abs].size();
            h_stream_lengths[b] = (int)(sz/4);
            memcpy(h_in_bufs + (size_t)b*pitch_bytes, container.streams[abs].data(), sz);
        }
        checkCudaErrors(cudaMemcpyAsync(d_in_bufs, h_in_bufs,
            (size_t)cb*pitch_bytes, cudaMemcpyHostToDevice, work_stream));
        checkCudaErrors(cudaMemcpyAsync(d_stream_lengths, h_stream_lengths.data(),
            cb*sizeof(int), cudaMemcpyHostToDevice, work_stream));
        checkCudaErrors(cudaStreamSynchronize(work_stream));

        gpu_rc_init_decoder(d_rc, (unsigned int*)d_in_bufs, d_stream_lengths, pitch_words, cb);

        // ── t=0: BOS backbone step (outside graph) ──
        model.step_backbone(nullptr, cb);
        for (int k = 0; k < K; ++k) {
            model.predict_head(k, cb);
            // Decode — pass nullptr for lengths to skip masking (all positions valid)
            gpu_rc_decode_step_batch(model.buf_logits, nullptr, nullptr, 0, d_out_sym,
                                     d_rc, (unsigned int*)d_in_bufs, d_stream_lengths,
                                     pitch_words, 256, cb);
            gpu_store_tokens(d_out_sym, d_decoded, k, K, cb);
            if (k < K-1) model.embed_and_scatter(d_out_sym, k, cb);
        }
        // Store decoded K bytes at step 0
        { int zero = 0;
          checkCudaErrors(cudaMemcpyAsync(d_step_counter, &zero, sizeof(int), cudaMemcpyHostToDevice, work_stream)); }
        hydra_store_decoded_host(d_decoded, d_batch_output, 0, K, (int)chunk_size, cb);
        // Copy decoded to prev_chunk for step 1
        checkCudaErrors(cudaMemcpyAsync(d_prev_chunk, d_decoded, cb*K*sizeof(int),
            cudaMemcpyDeviceToDevice, work_stream));

        if (T_chunk <= 1) goto copy_out;

        // Init device step counter to 1 (next step)
        { int one = 1;
          checkCudaErrors(cudaMemcpyAsync(d_step_counter, &one, sizeof(int), cudaMemcpyHostToDevice, work_stream));
          checkCudaErrors(cudaStreamSynchronize(work_stream)); }

        if (g_no_graph) {
            // ── no-graph fallback ──
            for (int t = 1; t < T_chunk; ++t) {
                model.step_backbone(d_prev_chunk, cb);
                for (int k = 0; k < K; ++k) {
                    model.predict_head(k, cb);
                    gpu_rc_decode_step_batch(model.buf_logits, nullptr, nullptr, 0, d_out_sym,
                                             d_rc, (unsigned int*)d_in_bufs, d_stream_lengths,
                                             pitch_words, 256, cb);
                    gpu_store_tokens(d_out_sym, d_decoded, k, K, cb);
                    if (k < K-1) model.embed_and_scatter(d_out_sym, k, cb);
                }
                hydra_store_decoded_host(d_decoded, d_batch_output, t, K, (int)chunk_size, cb);
                checkCudaErrors(cudaMemcpyAsync(d_prev_chunk, d_decoded, cb*K*sizeof(int),
                    cudaMemcpyDeviceToDevice, work_stream));
            }
            checkCudaErrors(cudaStreamSynchronize(work_stream));
        } else {
            // ── CUDA graph path ──
            if (graph_cb != cb) {
                if (graph_exec) { checkCudaErrors(cudaGraphExecDestroy(graph_exec)); graph_exec = nullptr; }
                if (graph)      { checkCudaErrors(cudaGraphDestroy(graph)); graph = nullptr; }

                checkCudaErrors(cudaStreamSynchronize(work_stream));
                checkCudaErrors(cudaStreamBeginCapture(work_stream, cudaStreamCaptureModeGlobal));

                // One backbone step (t>0)
                model.step_backbone(d_prev_chunk, cb);
                for (int k = 0; k < K; ++k) {
                    model.predict_head(k, cb);
                    gpu_rc_decode_step_batch(model.buf_logits, nullptr, nullptr, 0, d_out_sym,
                                             d_rc, (unsigned int*)d_in_bufs, d_stream_lengths,
                                             pitch_words, 256, cb);
                    gpu_store_tokens(d_out_sym, d_decoded, k, K, cb);
                    if (k < K-1) model.embed_and_scatter(d_out_sym, k, cb);
                }
                hydra_store_decoded(d_decoded, d_batch_output, d_step_counter, K, (int)chunk_size, cb);
                checkCudaErrors(cudaMemcpyAsync(d_prev_chunk, d_decoded, cb*K*sizeof(int),
                    cudaMemcpyDeviceToDevice, work_stream));
                gpu_increment_counter(d_step_counter);

                checkCudaErrors(cudaStreamEndCapture(work_stream, &graph));
                checkCudaErrors(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
                graph_cb = cb;
            }

            // Replay for t=1..T_chunk-1
            for (int t = 1; t < T_chunk; ++t)
                checkCudaErrors(cudaGraphLaunch(graph_exec, work_stream));
            checkCudaErrors(cudaStreamSynchronize(work_stream));
        }

    copy_out:
        // Copy batch output to host
        checkCudaErrors(cudaMemcpy(h_batch_output, d_batch_output,
            (size_t)cb*chunk_size*sizeof(int), cudaMemcpyDeviceToHost));
        for (int b = 0; b < cb; ++b) {
            int abs = ci + b;
            int len = container.lengths[abs];
            all_outputs[abs].resize(len);
            for (int j = 0; j < len; ++j)
                all_outputs[abs][j] = (uint8_t)h_batch_output[(size_t)b*chunk_size + j];
        }
        chunks_done += cb;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration<double>(t_end - t_start).count();
    print_progress(num_chunks, num_chunks, dur, total_size, "Decompressing");
    std::cout << "\nDone. " << std::fixed << std::setprecision(3) << dur << "s  "
              << (double)total_size/(1024.0*1024.0)/dur << " MB/s\n";

    // Write output
    std::ofstream fout(output_path, std::ios::binary);
    for (int i = 0; i < num_chunks; ++i)
        fout.write((char*)all_outputs[i].data(), all_outputs[i].size());
    fout.close();

    // Cleanup
    free_device((float*)d_rc); free_device((float*)d_in_bufs);
    free_device((float*)d_stream_lengths); free_device((float*)d_prev_chunk);
    free_device((float*)d_decoded); free_device((float*)d_out_sym);
    free_device((float*)d_batch_output); free_device((float*)d_step_counter);
    checkCudaErrors(cudaFreeHost(h_in_bufs));
    checkCudaErrors(cudaFreeHost(h_batch_output));
    if (graph_exec) checkCudaErrors(cudaGraphExecDestroy(graph_exec));
    if (graph) checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaStreamDestroy(work_stream));
    g_stream = 0;
    model.free_mem();
}

// ════════════════════════════════════════════════════════════════════
//  main
// ════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: hydra_boa <compress|decompress> <model.bin> <input> <output>\n"
                  << "  [d_model] [n_layers] [K]\n"
                  << "  [--gpu-batch B] [--tile T] [--chunk-size S]\n"
                  << "  [--no-graph] [--show-timings]\n";
        return 1;
    }

    std::string mode = argv[1];
    int gpu_batch = 0, tile_size = 0;
    size_t chunk_size = 4096;
    std::vector<std::string> pos;

    for (int i = 0; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--gpu-batch"  && i+1<argc) { gpu_batch  = std::stoi(argv[++i]); continue; }
        if (s == "--tile"       && i+1<argc) { tile_size  = std::stoi(argv[++i]);  continue; }
        if (s == "--chunk-size" && i+1<argc) { chunk_size = std::stoull(argv[++i]); continue; }
        if (s == "--no-graph")   { g_no_graph = true;     continue; }
        if (s == "--show-timings") { g_show_timings = true; continue; }
        if (s.rfind("--",0)==0) continue;
        pos.push_back(s);
    }
    if (pos.size() < 5) { std::cerr << "Not enough arguments.\n"; return 1; }

    std::string model_path = pos[2], input_path = pos[3], output_path = pos[4];
    int d_model=64, n_layers=1, K=4;
    if (pos.size()>5) d_model  = std::stoi(pos[5]);
    if (pos.size()>6) n_layers = std::stoi(pos[6]);
    if (pos.size()>7) K        = std::stoi(pos[7]);

    // Read dims from model header
    { std::ifstream mf(model_path, std::ios::binary);
      char magic[4]; mf.read(magic,4);
      if (magic[0]=='H'&&magic[1]=='Y'&&magic[2]=='D') {
          uint32_t hd,hl,hk;
          mf.read((char*)&hd,4); mf.read((char*)&hl,4); mf.read((char*)&hk,4);
          d_model=(int)hd; n_layers=(int)hl; K=(int)hk;
      } mf.close(); }

    std::cout << "HydraBOA C++ | d=" << d_model << " L=" << n_layers << " K=" << K << std::endl;

    if (mode == "compress")
        hydra_compress(model_path, input_path, output_path, d_model, n_layers, K, gpu_batch, chunk_size, tile_size);
    else if (mode == "decompress")
        hydra_decompress(model_path, input_path, output_path, d_model, n_layers, K, gpu_batch);
    else { std::cerr << "Unknown mode: " << mode << "\n"; return 1; }

    return 0;
}
