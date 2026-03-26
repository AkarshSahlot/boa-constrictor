// Inference Kernels - GELU, ReLU, Softmax, LayerNorm, Bias operations
#include "gemm_gpu_common.cuh"

// ==========================================
// GELU Activation
// ==========================================

__global__ void ker_gelu(float* x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        x[idx] = repro_gelu(x[idx]);
    }
}

void gpu_gelu(float* x, int N) {
    int block = 256;
    int grid = (N + block - 1) / block;
    ker_gelu<<<grid, block>>>(x, N);
}

// ==========================================
// ReLU Activation
// ==========================================

__global__ void ker_relu(float* x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

void gpu_relu(float* x, int N) {
    int block = 256;
    int grid = (N + block - 1) / block;
    ker_relu<<<grid, block, 0, g_stream>>>(x, N);
}

// ==========================================
// SiLU (Swish) Activation — x * sigmoid(x)
// ==========================================

__global__ void ker_silu_batch(float* x, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        #if GPU_REPRO_MATH
        x[i] = repro_silu(x[i]);
        #else
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
        #endif
    }
}

void gpu_silu_batch(float* x, size_t total) {
    int grid = (int)((total + 255) / 256);
    ker_silu_batch<<<grid, 256, 0, g_stream>>>(x, total);
}

// ==========================================
// Batched RMSNorm (weight-only, no bias)
// ==========================================

__global__ void ker_rmsnorm_batch(float* x, const float* w, int N, size_t batch) {
    size_t row_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= batch) return;

    float* row = x + row_idx * N;

    float sq = 0.0f;
    int i = 0;
    int limit = N - (N % 4);
    for (; i < limit; i += 4) {
        float4 v = reinterpret_cast<float4*>(row)[i/4];
        sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (; i < N; ++i) {
        float val = row[i];
        sq += val * val;
    }

    float inv_rms = rsqrtf(sq / N + 1e-6f);

    i = 0;
    for (; i < limit; i += 4) {
        float4 v = reinterpret_cast<float4*>(row)[i/4];
        float4 wc = reinterpret_cast<const float4*>(w)[i/4];
        v.x = v.x * inv_rms * wc.x;
        v.y = v.y * inv_rms * wc.y;
        v.z = v.z * inv_rms * wc.z;
        v.w = v.w * inv_rms * wc.w;
        reinterpret_cast<float4*>(row)[i/4] = v;
    }
    for (; i < N; ++i) {
        row[i] = row[i] * inv_rms * w[i];
    }
}

void gpu_rmsnorm_batch(float* x, const float* w, int N, size_t batch) {
    int block = 256;
    int grid = (int)((batch + block - 1) / block);
    ker_rmsnorm_batch<<<grid, block, 0, g_stream>>>(x, w, N, batch);
}

// ==========================================
// Add Bias
// ==========================================

__global__ void ker_add_bias(float* x, const float* b, int N, size_t total_elements) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        int col = i % N;
        x[i] += b[col];
    }
}

void gpu_add_bias(float* x, const float* b, int M, int N) {
    size_t total = (size_t)M * N;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    ker_add_bias<<<grid, block, 0, g_stream>>>(x, b, N, total);
}

// ==========================================
// Softmax (Row-wise)
// ==========================================

__global__ void ker_softmax_row(float* x, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // We launch 1 thread per row to ensure sequential reduction matching CPU
    if (row < M) { 
        float max_val = -1e30f;
        for(int j=0; j<N; ++j) {
            float v = x[row*N + j];
            if(v > max_val) max_val = v;
        }
        
        float sum_exp = 0.0f;
        for(int j=0; j<N; ++j) {
            float v = repro_exp(x[row*N + j] - max_val);
            x[row*N + j] = v;
            sum_exp += v;
        }
        
        float inv_sum = 1.0f / sum_exp;
        for(int j=0; j<N; ++j) {
            x[row*N + j] *= inv_sum;
        }
    }
}

void gpu_softmax(float* x, int M, int N) {
    int block = 1; 
    int grid = M;
    ker_softmax_row<<<grid, block>>>(x, M, N);
    GPU_DEVICE_SYNC();
}

// ==========================================
// LayerNorm (Single Row)
// ==========================================

__global__ void ker_layernorm_repro(float* x, const float* w, const float* b, int N, float eps) {
    int tx = threadIdx.x;
    
    __shared__ float s_mem[1024];
    
    // Use single thread for deterministic reduction matching CPU
    if (tx == 0) {
        float sum = 0.0f;
        for(int i=0; i<N; ++i) sum += x[i];
        float mean = sum / N;
        
        float sum_sq = 0.0f;
        for(int i=0; i<N; ++i) {
            float diff = x[i] - mean;
            sum_sq += diff * diff;
        }
        float var = sum_sq / N;
        float inv_std = rsqrtf(var + eps);
        
        s_mem[0] = mean;
        s_mem[1] = inv_std;
    }
    __syncthreads();
    
    float mean = s_mem[0];
    float inv_std = s_mem[1];
    
    if(tx < N) {
        x[tx] = (x[tx] - mean) * inv_std * w[tx] + b[tx];
    }
}

void gpu_layernorm(float* x, const float* w, const float* b, int N) {
    // Single block, N threads (assuming N <= 1024)
    if (N > 1024) { printf("Error: LayerNorm N too large for naive kernel\n"); return; }
    ker_layernorm_repro<<<1, N>>>(x, w, b, N, 1e-5f);
    GPU_DEVICE_SYNC();
}

// ==========================================
// Batched GELU
// ==========================================

__global__ void ker_gelu_batch(float* x, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        #if GPU_REPRO_MATH
        x[i] = repro_gelu(x[i]);
        #else
        float val = x[i];
        float cdf = 0.5f * (1.0f + erff(val * 0.70710678f));
        x[i] = val * cdf;
        #endif
    }
}

void gpu_gelu_batch(float* x, int N, size_t batch) {
    size_t total = (size_t)N * batch;
    int grid = (int)((total+255)/256);
    ker_gelu_batch<<<grid, 256, 0, g_stream>>>(x, total);
}

// ==========================================
// Batched LayerNorm (Vectorized)
// ==========================================

__global__ void ker_layernorm_batch_vec4(float* x, const float* w, const float* b, int N, size_t batch) {
    size_t row_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= batch) return;
    
    float* row = x + row_idx * N;
    
    float sum = 0.0f;
    float sq = 0.0f;
    
    int i = 0;
    // Vectorized loop
    int limit = N - (N % 4);
    for (; i < limit; i += 4) {
        float4 v = reinterpret_cast<float4*>(row)[i/4];
        // Order: x, y, z, w
        sum += v.x; sq += v.x * v.x;
        sum += v.y; sq += v.y * v.y;
        sum += v.z; sq += v.z * v.z;
        sum += v.w; sq += v.w * v.w;
    }
    // Tail
    for (; i < N; ++i) {
        float val = row[i];
        sum += val;
        sq += val * val;
    }
    
    float mean = sum / N;
    float var = sq / N - mean * mean;
    float inv_std = rsqrtf(var + 1e-5f);
    
    // Second pass for normalize
    i = 0;
    for (; i < limit; i += 4) {
        float4 v = reinterpret_cast<float4*>(row)[i/4];
        float4 wc = reinterpret_cast<const float4*>(w)[i/4];
        float4 bc = reinterpret_cast<const float4*>(b)[i/4];
        
        v.x = (v.x - mean) * inv_std * wc.x + bc.x;
        v.y = (v.y - mean) * inv_std * wc.y + bc.y;
        v.z = (v.z - mean) * inv_std * wc.z + bc.z;
        v.w = (v.w - mean) * inv_std * wc.w + bc.w;
        
        reinterpret_cast<float4*>(row)[i/4] = v;
    }
    for (; i < N; ++i) {
        row[i] = (row[i] - mean) * inv_std * w[i] + b[i];
    }
}

void gpu_layernorm_batch(float* x, const float* w, const float* b, int N, size_t batch) {
    // Launch 1 thread per row.
    int block = 256;
    int grid = (int)((batch + block - 1) / block);
    ker_layernorm_batch_vec4<<<grid, block, 0, g_stream>>>(x, w, b, N, batch);
}

// ==========================================
// Fused LayerNorm + Residual Save (Warp-Cooperative)
// ==========================================
// One warp (32 threads) per row. Saves pre-norm x to 'save' buffer,
// then normalizes x in-place. Eliminates separate cudaMemcpyAsync.
// Designed for d_model up to 512 (each thread handles up to 16 elements).

__global__ void ker_layernorm_save_warp(float* __restrict__ x, float* __restrict__ save,
                                        const float* __restrict__ w, const float* __restrict__ b,
                                        int N, size_t batch) {
    // 1 warp per row, 8 warps per block (256 threads)
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    size_t row_idx = (size_t)blockIdx.x * 8 + warp_id;
    if (row_idx >= batch) return;

    float* row = x + row_idx * N;
    float* sav = save + row_idx * N;

    // Each thread handles N/32 elements (e.g. 4 for N=128)
    int elems_per_thread = (N + 31) / 32;

    // Pass 1: copy to save buffer + compute sum and sum-of-squares
    float local_sum = 0.0f;
    float local_sq = 0.0f;
    for (int e = 0; e < elems_per_thread; ++e) {
        int idx = lane + e * 32;
        if (idx < N) {
            float val = row[idx];
            sav[idx] = val;
            local_sum += val;
            local_sq += val * val;
        }
    }

    // Warp reduction for sum and sq
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, offset);
    }
    // Broadcast from lane 0
    float mean = __shfl_sync(0xFFFFFFFF, local_sum, 0) / (float)N;
    float var = __shfl_sync(0xFFFFFFFF, local_sq, 0) / (float)N - mean * mean;
    float inv_std = rsqrtf(var + 1e-5f);

    // Pass 2: normalize in-place
    for (int e = 0; e < elems_per_thread; ++e) {
        int idx = lane + e * 32;
        if (idx < N) {
            float val = row[idx]; // still original value (not yet overwritten)
            row[idx] = (val - mean) * inv_std * w[idx] + b[idx];
        }
    }
}

void gpu_layernorm_save_batch(float* x, float* save, const float* w, const float* b, int N, size_t batch) {
    // 8 warps per block (256 threads), 1 warp per row
    int block = 256;
    int warps_per_block = 8;
    int grid = (int)((batch + warps_per_block - 1) / warps_per_block);
    ker_layernorm_save_warp<<<grid, block, 0, g_stream>>>(x, save, w, b, N, batch);
}

// ==========================================
// Batched Add
// ==========================================

__global__ void ker_add_batch(float* x, const float* y, int N, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        x[i] = __fadd_rn(x[i], y[i]);
    }
}

void gpu_add_batch(float* x, const float* y, int N, size_t batch) {
    size_t total = (size_t)N * batch;
    int grid = (int)((total+255)/256);
    ker_add_batch<<<grid, 256, 0, g_stream>>>(x, y, N, total);
    GPU_DEVICE_SYNC();
}

// ==========================================
// Batched Bias + Softplus
// ==========================================

__global__ void ker_add_bias_softplus_batch(float* x, const float* b, int N, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int col = i % N;
        float val = __fadd_rn(x[i], b[col]);
        #if GPU_REPRO_MATH
        float e = repro_exp(val);
        #else
        float e = expf(val);
        #endif
        x[i] = logf(__fadd_rn(1.0f, e));
    }
}

void gpu_add_bias_softplus_batch(float* x, const float* b, int N, size_t batch) {
    size_t total = (size_t)batch * N;
    int grid = (int)((total+255)/256);
    ker_add_bias_softplus_batch<<<grid, 256, 0, g_stream>>>(x, b, N, total);
}
