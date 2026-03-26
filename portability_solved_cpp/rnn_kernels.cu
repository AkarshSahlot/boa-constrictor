// RNN Kernels - LSTM, GRU, minGRU operations for multi-backbone support
#include "gemm_gpu_common.cuh"

// ==========================================
// LSTM Fused Gate Kernel
// ==========================================
// After GEMM: ih = x @ W_ih^T + b_ih, hh = h @ W_hh^T + b_hh
// Gate layout (PyTorch convention): [i, f, g, o] each of size d_model
// i = sigmoid(ih_i + hh_i)   — input gate
// f = sigmoid(ih_f + hh_f)   — forget gate
// g = tanh(ih_g + hh_g)      — cell candidate
// o = sigmoid(ih_o + hh_o)   — output gate
// c_new = f * c_old + i * g
// h_new = o * tanh(c_new)

__global__ void ker_lstm_fused_batch(const float* ih, const float* hh,
                                      float* h_state, float* c_state,
                                      int d_model, size_t total) {
    // Process 4 elements per thread when d_model is aligned
    size_t base = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < total && (d_model % 4 == 0)) {
        size_t b = base / d_model;
        int d = base % d_model;
        int d4 = 4 * d_model;
        size_t off = b * d4;

        // Load all 4 gates for 4 elements via float4
        float4 ih_i = reinterpret_cast<const float4*>(&ih[off + d])[0];
        float4 hh_i = reinterpret_cast<const float4*>(&hh[off + d])[0];
        float4 ih_f = reinterpret_cast<const float4*>(&ih[off + d_model + d])[0];
        float4 hh_f = reinterpret_cast<const float4*>(&hh[off + d_model + d])[0];
        float4 ih_g = reinterpret_cast<const float4*>(&ih[off + 2*d_model + d])[0];
        float4 hh_g = reinterpret_cast<const float4*>(&hh[off + 2*d_model + d])[0];
        float4 ih_o = reinterpret_cast<const float4*>(&ih[off + 3*d_model + d])[0];
        float4 hh_o = reinterpret_cast<const float4*>(&hh[off + 3*d_model + d])[0];
        float4 c_old = reinterpret_cast<float4*>(&c_state[base])[0];

        float vals_i[4] = {ih_i.x+hh_i.x, ih_i.y+hh_i.y, ih_i.z+hh_i.z, ih_i.w+hh_i.w};
        float vals_f[4] = {ih_f.x+hh_f.x, ih_f.y+hh_f.y, ih_f.z+hh_f.z, ih_f.w+hh_f.w};
        float vals_g[4] = {ih_g.x+hh_g.x, ih_g.y+hh_g.y, ih_g.z+hh_g.z, ih_g.w+hh_g.w};
        float vals_o[4] = {ih_o.x+hh_o.x, ih_o.y+hh_o.y, ih_o.z+hh_o.z, ih_o.w+hh_o.w};
        float c_arr[4] = {c_old.x, c_old.y, c_old.z, c_old.w};

        float h_arr[4], c_new[4];
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            #if GPU_REPRO_MATH
            float iv = repro_sigmoid(vals_i[n]);
            float fv = repro_sigmoid(vals_f[n]);
            float gv = repro_tanh(vals_g[n]);
            float ov = repro_sigmoid(vals_o[n]);
            #else
            float iv = 1.0f / (1.0f + expf(-vals_i[n]));
            float fv = 1.0f / (1.0f + expf(-vals_f[n]));
            float gv = tanhf(vals_g[n]);
            float ov = 1.0f / (1.0f + expf(-vals_o[n]));
            #endif
            float c = __fadd_rn(__fmul_rn(fv, c_arr[n]), __fmul_rn(iv, gv));
            #if GPU_REPRO_MATH
            h_arr[n] = __fmul_rn(ov, repro_tanh(c));
            #else
            h_arr[n] = __fmul_rn(ov, tanhf(c));
            #endif
            c_new[n] = c;
        }

        reinterpret_cast<float4*>(&c_state[base])[0] = make_float4(c_new[0], c_new[1], c_new[2], c_new[3]);
        reinterpret_cast<float4*>(&h_state[base])[0] = make_float4(h_arr[0], h_arr[1], h_arr[2], h_arr[3]);
    } else {
        // Scalar fallback
        for (size_t idx = base; idx < total && idx < base + 4; ++idx) {
            size_t b = idx / d_model;
            int d = idx % d_model;
            int d4 = 4 * d_model;

            float i_val = ih[b * d4 + d]              + hh[b * d4 + d];
            float f_val = ih[b * d4 + d_model + d]    + hh[b * d4 + d_model + d];
            float g_val = ih[b * d4 + 2*d_model + d]  + hh[b * d4 + 2*d_model + d];
            float o_val = ih[b * d4 + 3*d_model + d]  + hh[b * d4 + 3*d_model + d];

            #if GPU_REPRO_MATH
            i_val = repro_sigmoid(i_val);
            f_val = repro_sigmoid(f_val);
            g_val = repro_tanh(g_val);
            o_val = repro_sigmoid(o_val);
            #else
            i_val = 1.0f / (1.0f + expf(-i_val));
            f_val = 1.0f / (1.0f + expf(-f_val));
            g_val = tanhf(g_val);
            o_val = 1.0f / (1.0f + expf(-o_val));
            #endif

            float c = __fadd_rn(__fmul_rn(f_val, c_state[idx]), __fmul_rn(i_val, g_val));
            #if GPU_REPRO_MATH
            float h = __fmul_rn(o_val, repro_tanh(c));
            #else
            float h = __fmul_rn(o_val, tanhf(c));
            #endif

            c_state[idx] = c;
            h_state[idx] = h;
        }
    }
}

void gpu_lstm_fused_batch(const float* ih, const float* hh,
                           float* h_state, float* c_state,
                           int d_model, size_t batch) {
    size_t total = batch * d_model;
    size_t elements_vec4 = (total + 3) / 4;
    int grid = (int)((elements_vec4 + 255) / 256);
    ker_lstm_fused_batch<<<grid, 256, 0, g_stream>>>(ih, hh, h_state, c_state, d_model, total);
}

// ==========================================
// GRU Fused Gate Kernel
// ==========================================
// Gate layout (PyTorch convention): [r, z, n] each of size d_model
// r = sigmoid(ih_r + hh_r)   — reset gate
// z = sigmoid(ih_z + hh_z)   — update gate
// n = tanh(ih_n + r * hh_n)  — new gate
// h_new = (1 - z) * n + z * h_old

__global__ void ker_gru_fused_batch(const float* ih, const float* hh,
                                     float* h_state,
                                     int d_model, size_t total) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t b = idx / d_model;
    int d = idx % d_model;
    int d3 = 3 * d_model;

    float r_i = ih[b * d3 + d];
    float z_i = ih[b * d3 + d_model + d];
    float n_i = ih[b * d3 + 2*d_model + d];

    float r_h = hh[b * d3 + d];
    float z_h = hh[b * d3 + d_model + d];
    float n_h = hh[b * d3 + 2*d_model + d];

    #if GPU_REPRO_MATH
    float r = repro_sigmoid(__fadd_rn(r_i, r_h));
    float z = repro_sigmoid(__fadd_rn(z_i, z_h));
    float n = repro_tanh(__fadd_rn(n_i, __fmul_rn(r, n_h)));
    #else
    float r = 1.0f / (1.0f + expf(-(r_i + r_h)));
    float z = 1.0f / (1.0f + expf(-(z_i + z_h)));
    float n = tanhf(n_i + r * n_h);
    #endif

    h_state[idx] = __fadd_rn(__fmul_rn(1.0f - z, n), __fmul_rn(z, h_state[idx]));
}

void gpu_gru_fused_batch(const float* ih, const float* hh,
                          float* h_state,
                          int d_model, size_t batch) {
    size_t total = batch * d_model;
    int grid = (int)((total + 255) / 256);
    ker_gru_fused_batch<<<grid, 256, 0, g_stream>>>(ih, hh, h_state, d_model, total);
}

// ==========================================
// minGRU activation g(x) = x >= 0 ? x + 0.5 : sigmoid(x)
// (Appendix B.3, Feng et al. 2024)
// ==========================================
__device__ __forceinline__ float mingru_g(float x) {
    return x >= 0.0f ? (x + 0.5f) : (1.0f / (1.0f + expf(-x)));
}

// ==========================================
// minGRU Fused Step Kernel
// ==========================================
// z = sigmoid(z_logits)
// h_tilde = g(h_raw)    — positive activation (log-space compatible)
// h_new = (1 - z) * h_prev + z * h_tilde

__global__ void ker_mingru_fused_batch(const float* z_logits, const float* h_raw,
                                        float* h_state, size_t total) {
    size_t base = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < total) {
        float4 zv = reinterpret_cast<const float4*>(&z_logits[base])[0];
        float4 hv = reinterpret_cast<const float4*>(&h_raw[base])[0];
        float4 hs = reinterpret_cast<float4*>(&h_state[base])[0];

        float z_arr[4] = {zv.x, zv.y, zv.z, zv.w};
        float h_arr[4] = {hv.x, hv.y, hv.z, hv.w};
        float s_arr[4] = {hs.x, hs.y, hs.z, hs.w};

        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            #if GPU_REPRO_MATH
            float z = repro_sigmoid(z_arr[n]);
            #else
            float z = 1.0f / (1.0f + expf(-z_arr[n]));
            #endif
            float ht = mingru_g(h_arr[n]);
            s_arr[n] = __fadd_rn(__fmul_rn(1.0f - z, s_arr[n]), __fmul_rn(z, ht));
        }

        reinterpret_cast<float4*>(&h_state[base])[0] = make_float4(s_arr[0], s_arr[1], s_arr[2], s_arr[3]);
    } else {
        for (size_t idx = base; idx < total && idx < base + 4; ++idx) {
            #if GPU_REPRO_MATH
            float z = repro_sigmoid(z_logits[idx]);
            #else
            float z = 1.0f / (1.0f + expf(-z_logits[idx]));
            #endif
            float ht = mingru_g(h_raw[idx]);
            h_state[idx] = __fadd_rn(__fmul_rn(1.0f - z, h_state[idx]), __fmul_rn(z, ht));
        }
    }
}

void gpu_mingru_fused_batch(const float* z_logits, const float* h_tilde,
                             float* h_state, int d_model, size_t batch) {
    size_t total = batch * d_model;
    size_t elements_vec4 = (total + 3) / 4;
    int grid = (int)((elements_vec4 + 255) / 256);
    ker_mingru_fused_batch<<<grid, 256, 0, g_stream>>>(z_logits, h_tilde, h_state, total);
}

// ==========================================
// minGRU Chunk Kernel (sequential over time)
// ==========================================
// Pre-computed: z_all [B, L, d], h_all [B, L, d]
// Each thread handles one (batch, dim) pair and loops over L timesteps.
// Layout: [b*L + t] * d_model + d  (row-major [B, L, D])

__global__ void ker_mingru_chunk(const float* z_all, const float* h_all,
                                  float* h_state, float* out,
                                  int batch, int length, int d_model,
                                  size_t total_channels) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_channels) return;

    int b = idx / d_model;
    int d = idx % d_model;

    float h = h_state[idx];

    for (int t = 0; t < length; ++t) {
        size_t flat = (size_t)(b * length + t) * d_model + d;
        #if GPU_REPRO_MATH
        float z = repro_sigmoid(z_all[flat]);
        #else
        float z = 1.0f / (1.0f + expf(-z_all[flat]));
        #endif
        float ht = mingru_g(h_all[flat]);
        h = __fadd_rn(__fmul_rn(1.0f - z, h), __fmul_rn(z, ht));
        out[flat] = h;
    }

    h_state[idx] = h;
}

void gpu_mingru_chunk(const float* z_all, const float* h_all,
                       float* h_state, float* out,
                       int batch, int length, int d_model) {
    size_t total = (size_t)batch * d_model;
    int grid = (int)((total + 255) / 256);
    ker_mingru_chunk<<<grid, 256, 0, g_stream>>>(z_all, h_all, h_state, out, batch, length, d_model, total);
}

// ==========================================
// Scatter Timestep: copy h[b,:] → out[b*L+t, :]
// ==========================================

__global__ void ker_scatter_timestep(const float* h, float* out,
                                      int batch, int length, int t, int d_model,
                                      size_t total) {
    size_t base = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < total && (d_model % 4 == 0)) {
        int b = base / d_model;
        int d = base % d_model;
        float4 v = reinterpret_cast<const float4*>(&h[base])[0];
        reinterpret_cast<float4*>(&out[(size_t)(b * length + t) * d_model + d])[0] = v;
    } else {
        for (size_t idx = base; idx < total && idx < base + 4; ++idx) {
            int b = idx / d_model;
            int d = idx % d_model;
            out[(size_t)(b * length + t) * d_model + d] = h[idx];
        }
    }
}

void gpu_scatter_timestep(const float* h, float* out,
                           int batch, int length, int t, int d_model) {
    size_t total = (size_t)batch * d_model;
    size_t elements_vec4 = (total + 3) / 4;
    int grid = (int)((elements_vec4 + 255) / 256);
    ker_scatter_timestep<<<grid, 256, 0, g_stream>>>(h, out, batch, length, t, d_model, total);
}

// ==========================================
// Gather Timestep: copy src[b*L+t, :] → dst[b, :]
// Replaces per-batch cudaMemcpyAsync with a single kernel
// ==========================================

__global__ void ker_gather_timestep(const float* src, float* dst,
                                     int batch, int length, int t, int width,
                                     size_t total) {
    size_t base = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < total && (width % 4 == 0)) {
        int b = base / width;
        int d = base % width;
        float4 v = reinterpret_cast<const float4*>(&src[((size_t)b * length + t) * width + d])[0];
        reinterpret_cast<float4*>(&dst[(size_t)b * width + d])[0] = v;
    } else {
        for (size_t idx = base; idx < total && idx < base + 4; ++idx) {
            int b = idx / width;
            int d = idx % width;
            dst[(size_t)b * width + d] = src[((size_t)b * length + t) * width + d];
        }
    }
}

void gpu_gather_timestep(const float* src, float* dst,
                          int batch, int length, int t, int width) {
    size_t total = (size_t)batch * width;
    size_t elements_vec4 = (total + 3) / 4;
    int grid = (int)((elements_vec4 + 255) / 256);
    ker_gather_timestep<<<grid, 256, 0, g_stream>>>(src, dst, batch, length, t, width, total);
}
