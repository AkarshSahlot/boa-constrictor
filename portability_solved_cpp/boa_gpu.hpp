#pragma once

#include "gemm_gpu.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstring>

// GPU Structures

// FP16 support: global flag set by header detection in load_weights
inline bool g_model_fp16 = false;

// Helper: read fp16 values from file and convert to fp32
inline void read_fp16_to_fp32(std::ifstream& f, std::vector<float>& dst, int count) {
    std::vector<uint16_t> tmp16(count);
    f.read(reinterpret_cast<char*>(tmp16.data()), count * sizeof(uint16_t));
    for (int i = 0; i < count; ++i) {
        // IEEE 754 half-precision to single-precision conversion
        uint16_t h = tmp16[i];
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f32;
        if (exp == 0) {
            if (mant == 0) {
                f32 = sign; // +/- zero
            } else {
                // Denormalized: convert to normalized fp32
                exp = 1;
                while (!(mant & 0x400)) { mant <<= 1; exp--; }
                mant &= 0x3FF;
                f32 = sign | ((127 - 15 + exp) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            f32 = sign | 0x7F800000 | (mant << 13); // Inf/NaN
        } else {
            f32 = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        std::memcpy(&dst[i], &f32, 4);
    }
}

// Helper for transposing weights (Out, In) -> (In, Out) for RowMajor GEMM
std::vector<float> transpose_weights(const std::vector<float>& w, int rows, int cols) {
    std::vector<float> t(w.size());
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            t[j*rows + i] = w[i*cols + j];
        }
    }
    return t;
}

struct LayerNormGPU {
    float* weight;
    float* bias;
    
    LayerNormGPU() = default;
    
    void allocate(int size) {
        malloc_device(&weight, size * sizeof(float));
        malloc_device(&bias, size * sizeof(float));
    }
    
    void free() {
        free_device(weight); free_device(bias);
    }
    
    void load_weights(std::ifstream& f) {
        std::vector<float> tmp(256); // Assuming d_model known or read?
        // Actually load_weights needs size.
        // We assume caller knows context or we read generic.
        // For simplicity: hardcoded based on boa_model.hpp usage calling convention?
        // No, caller logic in BoaBlockGPU calls load_weights with implicit size in mind.
        // Let's rely on MambaConfig?
    }
    
    // Better: load_vec helper in parent
};




// Helper to read vector and copy (alloc not needed) to existing
inline void load_vec(std::ifstream& f, float* d_ptr, int size, bool transpose=false, int rows=0, int cols=0) {
    std::vector<float> tmp(size);
    if (g_model_fp16) {
        read_fp16_to_fp32(f, tmp, size);
    } else {
        f.read(reinterpret_cast<char*>(tmp.data()), size * sizeof(float));
    }
    if (transpose) {
        auto t = transpose_weights(tmp, rows, cols); // utils.hpp
        copy_to_device(d_ptr, t.data(), size * sizeof(float));
    } else {
        copy_to_device(d_ptr, tmp.data(), size * sizeof(float));
    }
}

struct MambaBlockGPU {
    MambaConfig config;
    int batch_size;
    
    // Weights (Shared)
    float* in_proj_w; 
    float* in_proj_b;
    float* conv1d_w;
    float* conv1d_b;
    float* x_proj_w;
    float* dt_proj_w;
    float* dt_proj_b;
    float* A_log;
    float* D;
    float* out_proj_w;
    float* out_proj_b;
    float* norm_w; // For RMSNorm
    
    // State (Per Batch)
    float* conv_state; // [Batch, d_inner, d_conv]
    float* ssm_state;  // [Batch, d_inner, d_state]
    
    // Buffers (Per Batch)
    float* buf_xz;     // [Batch, 2*d_inner]
    float* buf_x;      // Alias
    float* buf_z;      // Alias
    float* buf_conv;   // [Batch, d_inner]
    float* buf_delta;  // [Batch, dt_rank + 2*d_state] (Row Major)
    float* buf_dt_in;  // [Batch, dt_rank] (Extracted)
    float* buf_dt;     // [Batch, d_inner]
    float* buf_y;      // [Batch, d_inner]
    
    MambaBlockGPU() = default;

    void allocate(MambaConfig c, int batch) {
        config = c;
        batch_size = batch;
        
        // Weights (One copy)
        malloc_device(&in_proj_w, 2 * config.d_inner * config.d_model * sizeof(float));
        malloc_device(&in_proj_b, 2 * config.d_inner * sizeof(float));
        malloc_device(&conv1d_w, config.d_inner * config.d_conv * sizeof(float));
        malloc_device(&conv1d_b, config.d_inner * sizeof(float));
        malloc_device(&x_proj_w, (config.dt_rank + 2 * config.d_state) * config.d_inner * sizeof(float));
        malloc_device(&dt_proj_w, config.d_inner * config.dt_rank * sizeof(float));
        malloc_device(&dt_proj_b, config.d_inner * sizeof(float));
        malloc_device(&A_log, config.d_inner * config.d_state * sizeof(float));
        malloc_device(&D, config.d_inner * sizeof(float));
        malloc_device(&out_proj_w, config.d_model * config.d_inner * sizeof(float));
        malloc_device(&out_proj_b, config.d_model * sizeof(float));
        if(config.use_rmsnorm) malloc_device(&norm_w, config.d_model * sizeof(float));
        else malloc_device(&norm_w, 1);
        
        // State (Batch * Size)
        malloc_device(&conv_state, (size_t)batch_size * config.d_inner * config.d_conv * sizeof(float));
        malloc_device(&ssm_state, (size_t)batch_size * config.d_inner * config.d_state * sizeof(float));
        
        // Buffers
        malloc_device(&buf_xz, (size_t)batch_size * 2 * config.d_inner * sizeof(float));
        buf_x = buf_xz; 
        // buf_z = buf_xz + d_inner (Per batch slice).
        // Wait, buf_xz is [Batch, 2*d_inner]. 
        // Row 0: [x0, z0]. Row 1: [x1, z1].
        // Split pointers? No. 
        // buf_x points to start.
        // buf_z needs to point to... wait.
        // If we split via striding, buf_z is not contiguous block.
        // SOLUTION: Split projection into `in_proj_x` and `in_proj_z`?
        // Or separate buffers `buf_x` and `buf_z`.
        // Let's allocate separate.
        // in_proj normally produces one [2*d] output.
        // If I keep it one [Batch, 2d], accessing x part is strided!
        // `gemm_gpu_batch` produces [Batch, 2d].
        // `conv1d` reads `x`.
        // I need Contiguous `x`.
        // I will copy `buf_xz` (strided) to `buf_x` and `buf_z` (contiguous)?
        // Or implement `split_kernel`?
        // Using `split_kernel` is best.
        
        malloc_device(&buf_x, (size_t)batch_size * config.d_inner * sizeof(float));
        malloc_device(&buf_z, (size_t)batch_size * config.d_inner * sizeof(float));
        
        malloc_device(&buf_conv, (size_t)batch_size * config.d_inner * sizeof(float));
        
        int stride_delta = config.dt_rank + 2 * config.d_state;
        malloc_device(&buf_delta, (size_t)batch_size * stride_delta * sizeof(float));
        malloc_device(&buf_dt_in, (size_t)batch_size * config.dt_rank * sizeof(float));
        
        malloc_device(&buf_dt, (size_t)batch_size * config.d_inner * sizeof(float));
        malloc_device(&buf_y, (size_t)batch_size * config.d_inner * sizeof(float));
        
        reset_cache();
    }
    
    void free() {
        free_device(in_proj_w); free_device(in_proj_b);
        free_device(conv1d_w); free_device(conv1d_b);
        free_device(x_proj_w); free_device(dt_proj_w); free_device(dt_proj_b);
        free_device(A_log); free_device(D);
        free_device(out_proj_w); free_device(out_proj_b);
        free_device(norm_w);
        free_device(conv_state); free_device(ssm_state);
        free_device(buf_xz); free_device(buf_x); free_device(buf_z);
        free_device(buf_conv); free_device(buf_delta); free_device(buf_dt_in); free_device(buf_dt); free_device(buf_y);
    }
    

    
     // Helper to read vector and copy (alloc not needed) to existing

    void load_weights(std::ifstream& f) {
        load_vec(f, in_proj_w, 2*config.d_inner * config.d_model, true, 2*config.d_inner, config.d_model);
        load_vec(f, in_proj_b, 2*config.d_inner); 
        load_vec(f, conv1d_w, config.d_inner * config.d_conv); 
        load_vec(f, conv1d_b, config.d_inner);
        load_vec(f, x_proj_w, (config.dt_rank + 2*config.d_state) * config.d_inner, true, (config.dt_rank + 2*config.d_state), config.d_inner);
        load_vec(f, dt_proj_w, config.d_inner * config.dt_rank, true, config.d_inner, config.dt_rank);
        load_vec(f, dt_proj_b, config.d_inner);
        load_vec(f, A_log, config.d_inner * config.d_state);
        load_vec(f, D, config.d_inner);
        load_vec(f, out_proj_w, config.d_model * config.d_inner, true, config.d_model, config.d_inner);
        load_vec(f, out_proj_b, config.d_model);
        if(config.use_rmsnorm) load_vec(f, norm_w, config.d_model);
    }
    
    void step_batch(float* x_in, float* x_out) {
        // 1. In Proj (fused GEMM+bias)
        gemm_gpu_batch_bias(x_in, in_proj_w, in_proj_b, buf_xz, batch_size, 2*config.d_inner, config.d_model);
        
        // 2. Optimized Conv1D (Direct from buf_xz)
        gpu_mamba_conv1d_batch(buf_xz, conv_state, conv1d_w, conv1d_b, buf_conv, config.d_inner, config.d_conv, 2*config.d_inner, batch_size);
        
        // 3. x_proj
        int stride_delta = config.dt_rank + 2*config.d_state;
        gemm_gpu_batch(buf_conv, x_proj_w, buf_delta, batch_size, stride_delta, config.d_inner);
        
        // 4. dt_proj (Strided access from buf_delta, skip copy)
        gemm_gpu_batch_strided(buf_delta, stride_delta, dt_proj_w, config.d_inner, buf_dt, config.d_inner, batch_size, config.d_inner, config.dt_rank);
        
        // 5. Fused SSM Tail: Softplus(logits) + SSM(x) + Gating(z)
        // Passes buf_conv for x, buf_xz directly for z extraction.
        gpu_mamba_ssm_fused_batch(buf_conv, buf_xz, buf_dt, dt_proj_b, A_log, D, buf_delta, ssm_state, buf_y, config.d_inner, config.d_state, config.dt_rank, batch_size);
        
        // 6. Out Proj (fused GEMM+bias)
        gemm_gpu_batch_bias(buf_y, out_proj_w, out_proj_b, x_out, batch_size, config.d_model, config.d_inner);
    }

    void reset_cache() {
        checkCudaErrors(cudaMemsetAsync(conv_state, 0, (size_t)batch_size * config.d_inner * config.d_conv * sizeof(float), g_stream));
        checkCudaErrors(cudaMemsetAsync(ssm_state, 0, (size_t)batch_size * config.d_inner * config.d_state * sizeof(float), g_stream));
    }

    // Chunk Buffers
    float* chunk_buf_xz;     
    // float* chunk_buf_x; sent to void
    // float* chunk_buf_z; sent to void
    float* chunk_buf_conv;   
    float* chunk_buf_delta;  
    float* chunk_buf_dt_in;  
    float* chunk_buf_dt;     
    float* chunk_buf_y;
    float* chunk_buf_B;
    float* chunk_buf_C;
    
    int max_chunk_len_alloc = 0;

    void allocate_chunk(int length) {
        if (max_chunk_len_alloc >= length) return;
        if (max_chunk_len_alloc > 0) free_chunk();
        
        max_chunk_len_alloc = length;
        size_t total_tokens = (size_t)batch_size * length;
        
        malloc_device(&chunk_buf_xz, total_tokens * 2 * config.d_inner * sizeof(float));
        // malloc_device(&chunk_buf_x, total_tokens * config.d_inner * sizeof(float));
        // malloc_device(&chunk_buf_z, total_tokens * config.d_inner * sizeof(float));
        malloc_device(&chunk_buf_conv, total_tokens * config.d_inner * sizeof(float));
        
        int stride_delta = config.dt_rank + 2 * config.d_state;
        malloc_device(&chunk_buf_delta, total_tokens * stride_delta * sizeof(float));
        malloc_device(&chunk_buf_dt_in, total_tokens * config.dt_rank * sizeof(float));
        malloc_device(&chunk_buf_dt, total_tokens * config.d_inner * sizeof(float));
        malloc_device(&chunk_buf_y, total_tokens * config.d_inner * sizeof(float));
        malloc_device(&chunk_buf_B, total_tokens * config.d_state * sizeof(float));
        malloc_device(&chunk_buf_C, total_tokens * config.d_state * sizeof(float));
    }
    
    void free_chunk() {
        if (max_chunk_len_alloc == 0) return;
        free_device(chunk_buf_xz); // free_device(chunk_buf_x); free_device(chunk_buf_z);
        free_device(chunk_buf_conv); free_device(chunk_buf_delta); 
        free_device(chunk_buf_dt_in); free_device(chunk_buf_dt); free_device(chunk_buf_y);
        free_device(chunk_buf_B); free_device(chunk_buf_C);
        max_chunk_len_alloc = 0;
    }

    void forward_chunk(float* x_in, float* x_out, int length) {
        int batch_eff = batch_size * length;
        
        // 1. In Proj (fused GEMM+bias)
        gemm_gpu_batch_bias(x_in, in_proj_w, in_proj_b, chunk_buf_xz, batch_eff, 2*config.d_inner, config.d_model);
        
        // 2. Conv1d (Strided x input)
        gpu_mamba_conv1d_chunk(chunk_buf_xz, conv_state, conv1d_w, conv1d_b, chunk_buf_conv, batch_size, length, config.d_inner, config.d_conv, 2*config.d_inner);
        
        // 3. Delta Proj
        int stride_delta = config.dt_rank + 2*config.d_state;
        gemm_gpu_batch(chunk_buf_conv, x_proj_w, chunk_buf_delta, batch_eff, stride_delta, config.d_inner);
        
        // Split Delta
        gpu_copy_strided(chunk_buf_delta, chunk_buf_dt_in, stride_delta, config.dt_rank, batch_eff);
        gpu_copy_strided(chunk_buf_delta + config.dt_rank, chunk_buf_B, stride_delta, config.d_state, batch_eff);
        gpu_copy_strided(chunk_buf_delta + config.dt_rank + config.d_state, chunk_buf_C, stride_delta, config.d_state, batch_eff);
        
        // 4. dt Proj
        gemm_gpu_batch(chunk_buf_dt_in, dt_proj_w, chunk_buf_dt, batch_eff, config.d_inner, config.dt_rank);
        gpu_add_bias_softplus_batch(chunk_buf_dt, dt_proj_b, config.d_inner, batch_eff);
        
        // 5. SSM
        gpu_mamba_ssm_chunk(chunk_buf_conv, chunk_buf_dt, A_log, D,
                            chunk_buf_B, chunk_buf_C, ssm_state, chunk_buf_y,
                            batch_size, length, config.d_inner, config.d_state);
        
        // 6. Gate (Strided z)
        gpu_gate_strided(chunk_buf_y, chunk_buf_xz + config.d_inner, config.d_inner, 2*config.d_inner, batch_eff * config.d_inner);
        
        // 7. Out Proj (fused GEMM+bias)
        gemm_gpu_batch_bias(chunk_buf_y, out_proj_w, out_proj_b, x_out, batch_eff, config.d_model, config.d_inner);
    }

    // Fused variant for HydraBOA: inlines dt_proj GEMV + softplus + SSM
    // Eliminates 5 kernel launches but may differ in float rounding from GEMM path
    void forward_chunk_fused(float* x_in, float* x_out, int length) {
        int batch_eff = batch_size * length;
        
        // 1. In Proj (fused GEMM+bias)
        gemm_gpu_batch_bias(x_in, in_proj_w, in_proj_b, chunk_buf_xz, batch_eff, 2*config.d_inner, config.d_model);
        
        // 2. Conv1d (Strided x input)
        gpu_mamba_conv1d_chunk(chunk_buf_xz, conv_state, conv1d_w, conv1d_b, chunk_buf_conv, batch_size, length, config.d_inner, config.d_conv, 2*config.d_inner);
        
        // 3. Delta Proj
        int stride_delta = config.dt_rank + 2*config.d_state;
        gemm_gpu_batch(chunk_buf_conv, x_proj_w, chunk_buf_delta, batch_eff, stride_delta, config.d_inner);
        
        // 4+5. Fused: dt_proj(GEMV) + softplus + SSM with inline B/C from delta
        gpu_mamba_ssm_chunk_fused(chunk_buf_conv, chunk_buf_delta,
                                  dt_proj_w, dt_proj_b, A_log, D, 
                                  ssm_state, chunk_buf_y, 
                                  batch_size, length, config.d_inner, config.d_state,
                                  config.dt_rank, stride_delta);
        
        // 6. Gate (Strided z)
        gpu_gate_strided(chunk_buf_y, chunk_buf_xz + config.d_inner, config.d_inner, 2*config.d_inner, batch_eff * config.d_inner);
        
        // 7. Out Proj (fused GEMM+bias)
        gemm_gpu_batch_bias(chunk_buf_y, out_proj_w, out_proj_b, x_out, batch_eff, config.d_model, config.d_inner);
    }
};

// ==========================================
// LSTM Block GPU
// ==========================================
struct LSTMBlockGPU {
    int d_model;
    int batch_size;

    // Weights: W_ih [4*d, d], b_ih [4*d], W_hh [4*d, d], b_hh [4*d]
    float* w_ih; float* b_ih;
    float* w_hh; float* b_hh;

    // State: h [B, d], c [B, d]
    float* h_state;
    float* c_state;

    // Buffers: [B, 4*d]
    float* buf_ih;
    float* buf_hh;

    LSTMBlockGPU() = default;

    void allocate(MambaConfig c, int batch) {
        d_model = c.d_model;
        batch_size = batch;
        int d4 = 4 * d_model;

        malloc_device(&w_ih, d4 * d_model * sizeof(float));
        malloc_device(&b_ih, d4 * sizeof(float));
        malloc_device(&w_hh, d4 * d_model * sizeof(float));
        malloc_device(&b_hh, d4 * sizeof(float));

        malloc_device(&h_state, (size_t)batch * d_model * sizeof(float));
        malloc_device(&c_state, (size_t)batch * d_model * sizeof(float));

        malloc_device(&buf_ih, (size_t)batch * d4 * sizeof(float));
        malloc_device(&buf_hh, (size_t)batch * d4 * sizeof(float));

        reset_cache();
    }

    void free() {
        free_device(w_ih); free_device(b_ih);
        free_device(w_hh); free_device(b_hh);
        free_device(h_state); free_device(c_state);
        free_device(buf_ih); free_device(buf_hh);
    }

    void load_weights(std::ifstream& f) {
        int d4 = 4 * d_model;
        load_vec(f, w_ih, d4 * d_model, true, d4, d_model);
        load_vec(f, b_ih, d4);
        load_vec(f, w_hh, d4 * d_model, true, d4, d_model);
        load_vec(f, b_hh, d4);
    }

    void step_batch(float* x_in, float* x_out) {
        int d4 = 4 * d_model;
        // ih = x @ W_ih^T + b_ih (fused GEMM+bias)
        gemm_gpu_batch_bias(x_in, w_ih, b_ih, buf_ih, batch_size, d4, d_model);
        // hh = h @ W_hh^T + b_hh (fused GEMM+bias)
        gemm_gpu_batch_bias(h_state, w_hh, b_hh, buf_hh, batch_size, d4, d_model);
        // Fused gates → updates h_state, c_state in-place
        gpu_lstm_fused_batch(buf_ih, buf_hh, h_state, c_state, d_model, batch_size);
        // Output is h_state
        checkCudaErrors(cudaMemcpyAsync(x_out, h_state, (size_t)batch_size * d_model * sizeof(float), cudaMemcpyDeviceToDevice, g_stream));
    }

    void reset_cache() {
        checkCudaErrors(cudaMemsetAsync(h_state, 0, (size_t)batch_size * d_model * sizeof(float), g_stream));
        checkCudaErrors(cudaMemsetAsync(c_state, 0, (size_t)batch_size * d_model * sizeof(float), g_stream));
    }

    // Chunk buffers
    float* chunk_buf_ih = nullptr;
    float* chunk_buf_out = nullptr;
    int max_chunk_len_alloc = 0;

    void allocate_chunk(int length) {
        if (max_chunk_len_alloc >= length) return;
        if (max_chunk_len_alloc > 0) free_chunk();
        max_chunk_len_alloc = length;
        size_t total = (size_t)batch_size * length;
        int d4 = 4 * d_model;
        // Pre-compute all ih projections in parallel
        malloc_device(&chunk_buf_ih, total * d4 * sizeof(float));
        malloc_device(&chunk_buf_out, total * d_model * sizeof(float));
    }

    void free_chunk() {
        if (max_chunk_len_alloc == 0) return;
        free_device(chunk_buf_ih); free_device(chunk_buf_out);
        max_chunk_len_alloc = 0;
    }

    void forward_chunk(float* x_in, float* x_out, int length) {
        int batch_eff = batch_size * length;
        int d4 = 4 * d_model;

        // 1. Pre-compute ih for all timesteps: [B*L, 4d] (fused GEMM+bias)
        gemm_gpu_batch_bias(x_in, w_ih, b_ih, chunk_buf_ih, batch_eff, d4, d_model);

        // 2. Sequential loop over timesteps
        for (int t = 0; t < length; ++t) {
            // hh = h @ W_hh^T + b_hh for current timestep (fused GEMM+bias)
            gemm_gpu_batch_bias(h_state, w_hh, b_hh, buf_hh, batch_size, d4, d_model);

            // Extract ih for timestep t: pointer into pre-computed buffer
            // Layout: [b0_t0, b0_t1, ..., b0_tL-1, b1_t0, ...]
            // We need ih[b*L+t] for each b. This is strided access.
            // Use a gather kernel or loop with per-batch offsets.
            // Simpler: compute ih per-step too (less memory but more compute)
            // Actually, with layout [B*L, 4d] where B*L is contiguous batches,
            // timestep t for batch b is at index (b*length + t).
            // We need a contiguous [B, 4d] slice. Use scatter_timestep in reverse.
            // For simplicity, extract to buf_ih using a strided copy.

            // Gather ih[b*L+t, :] for all b into buf_ih[b, :]
            for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
                size_t src_off = ((size_t)b_idx * length + t) * d4;
                size_t dst_off = (size_t)b_idx * d4;
                checkCudaErrors(cudaMemcpyAsync(
                    (char*)buf_ih + dst_off * sizeof(float),
                    (char*)chunk_buf_ih + src_off * sizeof(float),
                    d4 * sizeof(float), cudaMemcpyDeviceToDevice, g_stream));
            }

            // Fused gates
            gpu_lstm_fused_batch(buf_ih, buf_hh, h_state, c_state, d_model, batch_size);

            // Scatter h_state to output: out[b*L+t, :] = h[b, :]
            gpu_scatter_timestep(h_state, x_out, batch_size, length, t, d_model);
        }
    }
};

// ==========================================
// GRU Block GPU
// ==========================================
struct GRUBlockGPU {
    int d_model;
    int batch_size;

    // Weights: W_ih [3*d, d], b_ih [3*d], W_hh [3*d, d], b_hh [3*d]
    float* w_ih; float* b_ih;
    float* w_hh; float* b_hh;

    // State: h [B, d]
    float* h_state;

    // Buffers: [B, 3*d]
    float* buf_ih;
    float* buf_hh;

    GRUBlockGPU() = default;

    void allocate(MambaConfig c, int batch) {
        d_model = c.d_model;
        batch_size = batch;
        int d3 = 3 * d_model;

        malloc_device(&w_ih, d3 * d_model * sizeof(float));
        malloc_device(&b_ih, d3 * sizeof(float));
        malloc_device(&w_hh, d3 * d_model * sizeof(float));
        malloc_device(&b_hh, d3 * sizeof(float));

        malloc_device(&h_state, (size_t)batch * d_model * sizeof(float));
        malloc_device(&buf_ih, (size_t)batch * d3 * sizeof(float));
        malloc_device(&buf_hh, (size_t)batch * d3 * sizeof(float));

        reset_cache();
    }

    void free() {
        free_device(w_ih); free_device(b_ih);
        free_device(w_hh); free_device(b_hh);
        free_device(h_state);
        free_device(buf_ih); free_device(buf_hh);
    }

    void load_weights(std::ifstream& f) {
        int d3 = 3 * d_model;
        load_vec(f, w_ih, d3 * d_model, true, d3, d_model);
        load_vec(f, b_ih, d3);
        load_vec(f, w_hh, d3 * d_model, true, d3, d_model);
        load_vec(f, b_hh, d3);
    }

    void step_batch(float* x_in, float* x_out) {
        int d3 = 3 * d_model;
        gemm_gpu_batch_bias(x_in, w_ih, b_ih, buf_ih, batch_size, d3, d_model);
        gemm_gpu_batch_bias(h_state, w_hh, b_hh, buf_hh, batch_size, d3, d_model);
        gpu_gru_fused_batch(buf_ih, buf_hh, h_state, d_model, batch_size);
        checkCudaErrors(cudaMemcpyAsync(x_out, h_state, (size_t)batch_size * d_model * sizeof(float), cudaMemcpyDeviceToDevice, g_stream));
    }

    void reset_cache() {
        checkCudaErrors(cudaMemsetAsync(h_state, 0, (size_t)batch_size * d_model * sizeof(float), g_stream));
    }

    // Chunk buffers
    float* chunk_buf_ih = nullptr;
    float* chunk_buf_out = nullptr;
    int max_chunk_len_alloc = 0;

    void allocate_chunk(int length) {
        if (max_chunk_len_alloc >= length) return;
        if (max_chunk_len_alloc > 0) free_chunk();
        max_chunk_len_alloc = length;
        size_t total = (size_t)batch_size * length;
        int d3 = 3 * d_model;
        malloc_device(&chunk_buf_ih, total * d3 * sizeof(float));
        malloc_device(&chunk_buf_out, total * d_model * sizeof(float));
    }

    void free_chunk() {
        if (max_chunk_len_alloc == 0) return;
        free_device(chunk_buf_ih); free_device(chunk_buf_out);
        max_chunk_len_alloc = 0;
    }

    void forward_chunk(float* x_in, float* x_out, int length) {
        int batch_eff = batch_size * length;
        int d3 = 3 * d_model;

        // Pre-compute ih for all timesteps in one batched GEMM (fused GEMM+bias)
        gemm_gpu_batch_bias(x_in, w_ih, b_ih, chunk_buf_ih, batch_eff, d3, d_model);

        for (int t = 0; t < length; ++t) {
            // h_state @ W_hh — unavoidable sequential dependency (fused GEMM+bias)
            gemm_gpu_batch_bias(h_state, w_hh, b_hh, buf_hh, batch_size, d3, d_model);

            // Gather ih for timestep t — single kernel instead of per-batch memcpy
            gpu_gather_timestep(chunk_buf_ih, buf_ih, batch_size, length, t, d3);

            gpu_gru_fused_batch(buf_ih, buf_hh, h_state, d_model, batch_size);
            gpu_scatter_timestep(h_state, x_out, batch_size, length, t, d_model);
        }
    }
};

// ==========================================
// minGRU Block GPU
// ==========================================
struct MinGRUBlockGPU {
    int d_model;
    int batch_size;

    // Weights: W_z [d, d], b_z [d], W_h [d, d], b_h [d]
    float* w_z; float* b_z;
    float* w_h; float* b_h;

    // State: h [B, d]
    float* h_state;

    // Buffers: [B, d]
    float* buf_z;
    float* buf_h;

    MinGRUBlockGPU() = default;

    void allocate(MambaConfig c, int batch) {
        d_model = c.d_model;
        batch_size = batch;

        malloc_device(&w_z, d_model * d_model * sizeof(float));
        malloc_device(&b_z, d_model * sizeof(float));
        malloc_device(&w_h, d_model * d_model * sizeof(float));
        malloc_device(&b_h, d_model * sizeof(float));

        malloc_device(&h_state, (size_t)batch * d_model * sizeof(float));
        malloc_device(&buf_z, (size_t)batch * d_model * sizeof(float));
        malloc_device(&buf_h, (size_t)batch * d_model * sizeof(float));

        reset_cache();
    }

    void free() {
        free_device(w_z); free_device(b_z);
        free_device(w_h); free_device(b_h);
        free_device(h_state);
        free_device(buf_z); free_device(buf_h);
    }

    void load_weights(std::ifstream& f) {
        load_vec(f, w_z, d_model * d_model, true, d_model, d_model);
        load_vec(f, b_z, d_model);
        load_vec(f, w_h, d_model * d_model, true, d_model, d_model);
        load_vec(f, b_h, d_model);
    }

    void step_batch(float* x_in, float* x_out) {
        // z_logits = x @ W_z^T + b_z (fused GEMM+bias)
        gemm_gpu_batch_bias(x_in, w_z, b_z, buf_z, batch_size, d_model, d_model);
        // h_tilde = x @ W_h^T + b_h (fused GEMM+bias)
        gemm_gpu_batch_bias(x_in, w_h, b_h, buf_h, batch_size, d_model, d_model);
        // Fused: z = sigmoid(z), h = (1-z)*h + z*h_tilde
        gpu_mingru_fused_batch(buf_z, buf_h, h_state, d_model, batch_size);
        checkCudaErrors(cudaMemcpyAsync(x_out, h_state, (size_t)batch_size * d_model * sizeof(float), cudaMemcpyDeviceToDevice, g_stream));
    }

    void reset_cache() {
        checkCudaErrors(cudaMemsetAsync(h_state, 0, (size_t)batch_size * d_model * sizeof(float), g_stream));
    }

    // Chunk buffers
    float* chunk_buf_z = nullptr;
    float* chunk_buf_h = nullptr;
    int max_chunk_len_alloc = 0;

    void allocate_chunk(int length) {
        if (max_chunk_len_alloc >= length) return;
        if (max_chunk_len_alloc > 0) free_chunk();
        max_chunk_len_alloc = length;
        size_t total = (size_t)batch_size * length;
        malloc_device(&chunk_buf_z, total * d_model * sizeof(float));
        malloc_device(&chunk_buf_h, total * d_model * sizeof(float));
    }

    void free_chunk() {
        if (max_chunk_len_alloc == 0) return;
        free_device(chunk_buf_z); free_device(chunk_buf_h);
        max_chunk_len_alloc = 0;
    }

    void forward_chunk(float* x_in, float* x_out, int length) {
        int batch_eff = batch_size * length;
        // Pre-compute z and h_tilde for all timesteps (fused GEMM+bias)
        gemm_gpu_batch_bias(x_in, w_z, b_z, chunk_buf_z, batch_eff, d_model, d_model);
        gemm_gpu_batch_bias(x_in, w_h, b_h, chunk_buf_h, batch_eff, d_model, d_model);
        // Sequential scan over time (parallel over batch × dim)
        gpu_mingru_chunk(chunk_buf_z, chunk_buf_h, h_state, x_out, batch_size, length, d_model);
    }
};

// ==========================================
// Feed-Forward GPU
// ==========================================
struct FeedForwardGPU {
    float* w1; float* b1;
    float* w2; float* b2;
    float* buf; // [Batch, 4*d]
    int d_model; int d_hidden; int batch_size;

    void allocate(int d, int batch) {
        d_model = d; d_hidden = 4*d; batch_size = batch;
        malloc_device(&w1, d_hidden * d_model * sizeof(float));
        malloc_device(&b1, d_hidden * sizeof(float));
        malloc_device(&w2, d_model * d_hidden * sizeof(float));
        malloc_device(&b2, d_model * sizeof(float));
        malloc_device(&buf, (size_t)batch * d_hidden * sizeof(float));
    }
    void free() {
        free_device(w1); free_device(b1); free_device(w2); free_device(b2); free_device(buf);
    }
    void load_weights(std::ifstream& f) {
        load_vec(f, w1, d_hidden * d_model, true, d_hidden, d_model);
        load_vec(f, b1, d_hidden);
        load_vec(f, w2, d_model * d_hidden, true, d_model, d_hidden);
        load_vec(f, b2, d_model);
    }
    
    // Chunk Logic
    float* chunk_buf; 
    int max_alloc = 0;
    
    void allocate_chunk(int length) {
        if (max_alloc >= length) return;
        if (max_alloc > 0) free_device(chunk_buf);
        max_alloc = length;
        size_t total = (size_t)batch_size * length;
        malloc_device(&chunk_buf, total * d_hidden * sizeof(float));
    }
    void free_chunk() {
        if (max_alloc > 0) free_device(chunk_buf);
        max_alloc = 0;
    }
    
    void forward_chunk(float* x, float* out, int length, const float* residual = nullptr) {
        int batch_eff = batch_size * length;
        // Fused GEMM + bias + GELU (single kernel)
        gemm_gpu_batch_bias_act(x, w1, b1, chunk_buf, batch_eff, d_hidden, d_model, GEMM_ACT_GELU);
        
        if (residual) {
            // Fused GEMM + bias + residual add (single kernel)
            gemm_gpu_batch_bias_res(chunk_buf, w2, b2, residual, out, batch_eff, d_model, d_hidden);
        } else {
            gemm_gpu_batch_bias(chunk_buf, w2, b2, out, batch_eff, d_model, d_hidden);
        }
    }

    void step_batch(float* x, float* out, const float* residual = nullptr) {
        // x: [batch, d_model], w1: [d_model, d_hidden], buf: [batch, d_hidden]
        // Fused GEMM + bias + GELU (single kernel)
        gemm_gpu_batch_bias_act(x, w1, b1, buf, batch_size, d_hidden, d_model, GEMM_ACT_GELU);
        
        // buf: [batch, d_hidden], w2: [d_hidden, d_model], out: [batch, d_model]
        if (residual) {
            // Fused GEMM + bias + residual add (single kernel)
            gemm_gpu_batch_bias_res(buf, w2, b2, residual, out, batch_size, d_model, d_hidden);
        } else {
            gemm_gpu_batch_bias(buf, w2, b2, out, batch_size, d_model, d_hidden);
        }
    }
};

struct BoaBlockGPU {
    BackboneType backbone_type;
    MambaBlockGPU mamba;
    LSTMBlockGPU lstm;
    GRUBlockGPU gru;
    MinGRUBlockGPU mingru;
    LayerNormGPU ln1;
    LayerNormGPU ln2;
    FeedForwardGPU ff;
    int d_model;
    int batch_size;

    BoaBlockGPU() = default;

    void allocate(MambaConfig conf, int batch) {
        d_model = conf.d_model;
        batch_size = batch;
        backbone_type = conf.backbone;

        switch(backbone_type) {
            case BACKBONE_MAMBA:  mamba.allocate(conf, batch); break;
            case BACKBONE_LSTM:   lstm.allocate(conf, batch); break;
            case BACKBONE_GRU:    gru.allocate(conf, batch); break;
            case BACKBONE_MINGRU: mingru.allocate(conf, batch); break;
        }

        ln1.allocate(d_model);
        ln2.allocate(d_model);
        ff.allocate(d_model, batch);
    }

    void free() {
        switch(backbone_type) {
            case BACKBONE_MAMBA:  mamba.free(); break;
            case BACKBONE_LSTM:   lstm.free(); break;
            case BACKBONE_GRU:    gru.free(); break;
            case BACKBONE_MINGRU: mingru.free(); break;
        }
        ln1.free(); ln2.free(); ff.free();
    }

    void reset_cache() {
        switch(backbone_type) {
            case BACKBONE_MAMBA:  mamba.reset_cache(); break;
            case BACKBONE_LSTM:   lstm.reset_cache(); break;
            case BACKBONE_GRU:    gru.reset_cache(); break;
            case BACKBONE_MINGRU: mingru.reset_cache(); break;
        }
    }

    void load_weights(std::ifstream& f) {
        load_vec(f, ln1.weight, d_model);
        load_vec(f, ln1.bias, d_model);

        switch(backbone_type) {
            case BACKBONE_MAMBA:  mamba.load_weights(f); break;
            case BACKBONE_LSTM:   lstm.load_weights(f); break;
            case BACKBONE_GRU:    gru.load_weights(f); break;
            case BACKBONE_MINGRU: mingru.load_weights(f); break;
        }

        load_vec(f, ln2.weight, d_model);
        load_vec(f, ln2.bias, d_model);

        ff.load_weights(f);
    }

    void step_batch(float* x_in, float* x_out, float* buf_res) {
        if (backbone_type == BACKBONE_MAMBA) {
            // Mamba V1: single residual — x + ff(ln2(mamba(ln1(x))))
            // Fused: save residual + layernorm in one kernel (eliminates D2D memcpy)
            gpu_layernorm_save_batch(x_in, buf_res, ln1.weight, ln1.bias, d_model, batch_size);

            mamba.step_batch(x_in, x_in);
            gpu_layernorm_batch(x_in, ln2.weight, ln2.bias, d_model, batch_size);
            // Fused: FF + residual add (residual folded into w2 GEMM store)
            ff.step_batch(x_in, x_in, buf_res);
        } else {
            // RNN backbones: dual residual — x = x + backbone(ln1(x)); x = x + ff(ln2(x))
            // Fused: save residual + layernorm in one kernel
            gpu_layernorm_save_batch(x_in, buf_res, ln1.weight, ln1.bias, d_model, batch_size);

            switch(backbone_type) {
                case BACKBONE_LSTM:   lstm.step_batch(x_in, x_in); break;
                case BACKBONE_GRU:    gru.step_batch(x_in, x_in); break;
                case BACKBONE_MINGRU: mingru.step_batch(x_in, x_in); break;
                default: break;
            }

            // First residual: x = backbone_out + original_x
            gpu_add_batch(x_in, buf_res, d_model, batch_size);

            // Second residual: x = x + ff(ln2(x))
            // Fused: save residual + layernorm in one kernel
            gpu_layernorm_save_batch(x_in, buf_res, ln2.weight, ln2.bias, d_model, batch_size);
            // Fused: FF + residual add (residual folded into w2 GEMM store)
            ff.step_batch(x_in, x_in, buf_res);
        }

        if (x_in != x_out) {
            checkCudaErrors(cudaMemcpyAsync(x_out, x_in, (size_t)batch_size * d_model * sizeof(float), cudaMemcpyDeviceToDevice, g_stream));
        }
    }

    // Chunk Logic
    void allocate_chunk(int length) {
        switch(backbone_type) {
            case BACKBONE_MAMBA:  mamba.allocate_chunk(length); break;
            case BACKBONE_LSTM:   lstm.allocate_chunk(length); break;
            case BACKBONE_GRU:    gru.allocate_chunk(length); break;
            case BACKBONE_MINGRU: mingru.allocate_chunk(length); break;
        }
        ff.allocate_chunk(length);
    }
    void free_chunk() {
        switch(backbone_type) {
            case BACKBONE_MAMBA:  mamba.free_chunk(); break;
            case BACKBONE_LSTM:   lstm.free_chunk(); break;
            case BACKBONE_GRU:    gru.free_chunk(); break;
            case BACKBONE_MINGRU: mingru.free_chunk(); break;
        }
        ff.free_chunk();
    }

    // x_in and buf_res are flattened [Batch*Length, d_model]
    void forward_chunk(float* x_in, float* x_out, float* buf_res, int length) {
        int batch_eff = batch_size * length;

        if (backbone_type == BACKBONE_MAMBA) {
            // Mamba V1: single residual
            // Fused: save residual + layernorm in one kernel
            gpu_layernorm_save_batch(x_in, buf_res, ln1.weight, ln1.bias, d_model, batch_eff);
            mamba.forward_chunk(x_in, x_in, length);
            gpu_layernorm_batch(x_in, ln2.weight, ln2.bias, d_model, batch_eff);
            // Fused: FF + residual add
            ff.forward_chunk(x_in, x_in, length, buf_res);
        } else {
            // RNN backbones: dual residual
            // Fused: save residual + layernorm in one kernel
            gpu_layernorm_save_batch(x_in, buf_res, ln1.weight, ln1.bias, d_model, batch_eff);

            switch(backbone_type) {
                case BACKBONE_LSTM:   lstm.forward_chunk(x_in, x_in, length); break;
                case BACKBONE_GRU:    gru.forward_chunk(x_in, x_in, length); break;
                case BACKBONE_MINGRU: mingru.forward_chunk(x_in, x_in, length); break;
                default: break;
            }

            // First residual
            gpu_add_batch(x_in, buf_res, d_model, batch_eff);

            // Second residual: x = x + ff(ln2(x))
            // Fused: save residual + layernorm in one kernel
            gpu_layernorm_save_batch(x_in, buf_res, ln2.weight, ln2.bias, d_model, batch_eff);
            // Fused: FF + residual add
            ff.forward_chunk(x_in, x_in, length, buf_res);
        }

        if (x_in != x_out) {
            checkCudaErrors(cudaMemcpy(x_out, x_in, (size_t)batch_eff * d_model * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }

    // Fused variant: uses mamba.forward_chunk_fused (inlined dt_proj GEMV)
    // Only valid for BACKBONE_MAMBA. Used by HydraBOA for speed.
    void forward_chunk_fused(float* x_in, float* x_out, float* buf_res, int length) {
        int batch_eff = batch_size * length;
        gpu_layernorm_save_batch(x_in, buf_res, ln1.weight, ln1.bias, d_model, batch_eff);
        mamba.forward_chunk_fused(x_in, x_in, length);
        gpu_layernorm_batch(x_in, ln2.weight, ln2.bias, d_model, batch_eff);
        ff.forward_chunk(x_in, x_in, length, buf_res);
        if (x_in != x_out) {
            checkCudaErrors(cudaMemcpy(x_out, x_in, (size_t)batch_eff * d_model * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
};

struct BoaPredictorGPU {
    BoaPredictorGPU(MambaConfig conf, int v_size, int n_l, int batch) : config(conf), vocab_size(v_size), n_layers(n_l), batch_size(batch) {
        embedding_size = vocab_size * config.d_model;
        malloc_device(&embedding, embedding_size * sizeof(float));
        
        
        blocks = new BoaBlockGPU[n_layers];
        for(int i=0; i<n_layers; ++i) { 
            blocks[i].allocate(config, batch);
        }
        
        malloc_device(&head_w1, config.d_model * config.d_model * sizeof(float));
        malloc_device(&head_b1, config.d_model * sizeof(float));
        malloc_device(&head_w2, vocab_size * config.d_model * sizeof(float));
        malloc_device(&head_b2, vocab_size * sizeof(float));
        
        // Final RMSNorm for non-mamba backbones
        has_final_norm = (config.backbone != BACKBONE_MAMBA);
        if (has_final_norm) {
            malloc_device(&final_norm_w, config.d_model * sizeof(float));
        }

        malloc_device(&buf_x, (size_t)batch * config.d_model * sizeof(float));
        malloc_device(&buf_res, (size_t)batch * config.d_model * sizeof(float));
        malloc_device(&buf_head, (size_t)batch * config.d_model * sizeof(float));
    }
    
    // ... (Members)
    MambaConfig config;
    int vocab_size;
    int n_layers;
    int batch_size;
    long long embedding_size;
    
    float* embedding; 
    BoaBlockGPU* blocks; // Raw pointer array
    float* head_w1; float* head_b1;
    float* head_w2; float* head_b2;
    
    float* final_norm_w;  // RMSNorm weight for non-mamba backbones
    bool has_final_norm;   // true for non-mamba backbones

    float* buf_x;
    float* buf_res;
    float* buf_head;
    
    void load_weights(std::string path) {
         std::cout << "Loading weights to GPU..." << std::endl;
         std::ifstream f(path, std::ios::binary);
         
         // Detect file format: check for "BOA\x00" (fp32) or "BOA\x01" (fp16) header
         char magic[4];
         f.read(magic, 4);
         if (magic[0] == 'B' && magic[1] == 'O' && magic[2] == 'A') {
             g_model_fp16 = (magic[3] == 0x01);
             std::cout << "  Model format: " << (g_model_fp16 ? "fp16" : "fp32") << " (with header)" << std::endl;
             // Header consumed, data follows
         } else {
             // Legacy format — no header, rewind
             g_model_fp16 = false;
             f.seekg(0, std::ios::beg);
             std::cout << "  Model format: fp32 (legacy, no header)" << std::endl;
         }
         
         // Embedding
         std::vector<float> h_emb(embedding_size);
         if (g_model_fp16) {
             read_fp16_to_fp32(f, h_emb, (int)embedding_size);
         } else {
             f.read(reinterpret_cast<char*>(h_emb.data()), embedding_size * sizeof(float));
         }
         copy_to_device(embedding, h_emb.data(), embedding_size * sizeof(float));
         
         for(int i=0; i<n_layers; ++i) blocks[i].load_weights(f);
         
         // Final RMSNorm (non-mamba backbones have final_norm.weight before head)
         if (has_final_norm) {
             load_vec(f, final_norm_w, config.d_model);
         }

         // Head weights — use load_vec which handles fp16 automatically
         load_vec(f, head_w1, config.d_model * config.d_model, true, config.d_model, config.d_model);
         load_vec(f, head_b1, config.d_model);
         load_vec(f, head_w2, vocab_size * config.d_model, true, vocab_size, config.d_model);
         load_vec(f, head_b2, vocab_size);
    }
    

    // Input: Tokens [Batch]. (Device Pointer)
    // Output: Logits [Batch, Vocab]. (Device Pointer)
    void step_batch(int* d_tokens, float* d_logits_out, bool add_bias = true) {
        // 1. Embedding lookup
        gpu_embedding_lookup_batch(d_tokens, embedding, buf_x, config.d_model, batch_size);
        
        // 2. Blocks
        for(int i=0; i<n_layers; ++i) {
            blocks[i].step_batch(buf_x, buf_x, buf_res);
        }
        
        // 3. Final norm (RMSNorm for non-mamba backbones)
        if (has_final_norm) {
            gpu_rmsnorm_batch(buf_x, final_norm_w, config.d_model, batch_size);
        }

        // 4. Head: Linear -> Activation -> Linear (fused GEMM+bias+activation)
        int head_act = has_final_norm ? GEMM_ACT_SILU : GEMM_ACT_RELU;
        gemm_gpu_batch_bias_act(buf_x, head_w1, head_b1, buf_head, batch_size, config.d_model, config.d_model, head_act);
        
        gemm_gpu_batch_bias(buf_head, head_w2, add_bias ? head_b2 : nullptr, d_logits_out, batch_size, vocab_size, config.d_model);
    }

    // Chunk Logic
    float* chunk_buf_main;
    float* chunk_buf_res;
    float* chunk_buf_head;
    int max_chunk = 0;
    
    void allocate_chunk(int length) {
         if (max_chunk >= length) return;
         if (max_chunk > 0) free_chunk();
         max_chunk = length;
         // Propagate
         for(int i=0; i<n_layers; ++i) blocks[i].allocate_chunk(length);
         
         size_t total = (size_t)batch_size * length;
         malloc_device(&chunk_buf_main, total * config.d_model * sizeof(float));
         malloc_device(&chunk_buf_res, total * config.d_model * sizeof(float));
         malloc_device(&chunk_buf_head, total * config.d_model * sizeof(float)); 
    }
    
    void free_chunk() {
         if (max_chunk == 0) return;
         for(int i=0; i<n_layers; ++i) blocks[i].free_chunk();
         free_device(chunk_buf_main); free_device(chunk_buf_res); free_device(chunk_buf_head);
         max_chunk = 0;
    }
    
    void forward_chunk(const int* d_tokens, float* d_logits, int length) {
        // d_tokens: [Batch, Length] -> Flattened [Batch*Length]
        int batch_eff = batch_size * length;
        
        gpu_embedding_lookup_batch(d_tokens, embedding, chunk_buf_main, config.d_model, batch_eff);
        
        for(int i=0; i<n_layers; ++i) {
            blocks[i].forward_chunk(chunk_buf_main, chunk_buf_main, chunk_buf_res, length);
        }
        
        // Final norm (RMSNorm for non-mamba backbones)
        if (has_final_norm) {
            gpu_rmsnorm_batch(chunk_buf_main, final_norm_w, config.d_model, batch_eff);
        }

        // Head (fused GEMM+bias+activation)
        int head_act = has_final_norm ? GEMM_ACT_SILU : GEMM_ACT_RELU;
        gemm_gpu_batch_bias_act(chunk_buf_main, head_w1, head_b1, chunk_buf_head, batch_eff, config.d_model, config.d_model, head_act);
        
        // Final Logits (fused GEMM+bias)
        gemm_gpu_batch_bias(chunk_buf_head, head_w2, head_b2, d_logits, batch_eff, vocab_size, config.d_model);
    }
    static size_t estimate_memory_static(MambaConfig config, int vocab_size, int n_layers, int batch_size, int chunk_len, bool is_compression) {
        size_t dmod = config.d_model;
        size_t w_ff = (4*dmod*dmod + 4*dmod + dmod*4*dmod + dmod) * 4;
        size_t w_ln = 2 * (2 * dmod) * 4; // ln1 + ln2
        size_t w_head = (dmod*dmod + dmod + vocab_size*dmod + vocab_size) * 4;
        size_t w_emb = (size_t)vocab_size * dmod * 4;
        size_t b_ff = (4*dmod) * 4;
        size_t b_pred = (dmod + dmod + dmod) * 4;

        size_t w_backbone = 0, b_backbone = 0, c_backbone = 0;

        switch(config.backbone) {
            case BACKBONE_MAMBA: {
                size_t din = config.d_inner;
                size_t dstate = config.d_state;
                size_t dconv = config.d_conv;
                size_t dtrank = config.dt_rank;
                size_t stride_delta = dtrank + 2 * dstate;
                w_backbone = (2*din*dmod + 2*din + din*dconv + din + stride_delta*din + din*dtrank + din + din*dstate + din + dmod*din + dmod) * 4;
                b_backbone = (din*dconv + din*dstate + 2*din + din + din + stride_delta + dtrank + din + din) * 4;
                if (is_compression) c_backbone = (2*din + din + stride_delta + dtrank + din + din + 2*dstate) * 4;
                break;
            }
            case BACKBONE_LSTM:
                w_backbone = (4*dmod*dmod + 4*dmod + 4*dmod*dmod + 4*dmod) * 4; // w_ih, b_ih, w_hh, b_hh
                b_backbone = (dmod + dmod + 4*dmod + 4*dmod) * 4; // h, c, buf_ih, buf_hh
                if (is_compression) c_backbone = (4*dmod + dmod) * 4; // chunk_buf_ih, chunk_buf_out
                break;
            case BACKBONE_GRU:
                w_backbone = (3*dmod*dmod + 3*dmod + 3*dmod*dmod + 3*dmod) * 4;
                b_backbone = (dmod + 3*dmod + 3*dmod) * 4; // h, buf_ih, buf_hh
                if (is_compression) c_backbone = (3*dmod + dmod) * 4;
                break;
            case BACKBONE_MINGRU:
                w_backbone = (dmod*dmod + dmod + dmod*dmod + dmod) * 4; // w_z, b_z, w_h, b_h
                b_backbone = (dmod + dmod + dmod) * 4; // h, buf_z, buf_h
                if (is_compression) c_backbone = (dmod + dmod) * 4; // chunk_buf_z, chunk_buf_h
                break;
        }

        size_t fixed = (w_backbone + w_ff + w_ln) * n_layers + w_head + w_emb;
        size_t per_batch = (size_t)batch_size * ((b_backbone + b_ff) * n_layers + b_pred);

        size_t per_chunk = 0;
        if (is_compression) {
            size_t c_ff = (4*dmod) * 4;
            size_t c_pred = (dmod + dmod + dmod) * 4;
            size_t c_logits = (size_t)vocab_size * 4;
            per_chunk = (size_t)batch_size * (size_t)chunk_len * ((c_backbone + c_ff) * n_layers + c_pred + c_logits);
        } else {
            per_chunk = (size_t)batch_size * (size_t)chunk_len * sizeof(int);
        }

        return fixed + per_batch + per_chunk;
    }

    void reset_cache() {
        for(int i=0; i<n_layers; ++i) blocks[i].reset_cache();
    }
    
    void free() {
        free_device(embedding);
        for(int i=0; i<n_layers; ++i) blocks[i].free();
        delete[] blocks;
        
        free_device(head_w1); free_device(head_b1);
        free_device(head_w2); free_device(head_b2);
        if (has_final_norm) free_device(final_norm_w);
        
        free_device(buf_x); free_device(buf_res); free_device(buf_head);
    }
};

