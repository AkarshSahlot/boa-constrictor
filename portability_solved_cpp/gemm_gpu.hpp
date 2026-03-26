#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>

// ==========================================
// Backbone Type
// ==========================================
enum BackboneType {
    BACKBONE_MAMBA = 0,
    BACKBONE_LSTM = 1,
    BACKBONE_GRU = 2,
    BACKBONE_MINGRU = 3,
};

inline const char* backbone_name(BackboneType t) {
    switch(t) {
        case BACKBONE_MAMBA:  return "mamba";
        case BACKBONE_LSTM:   return "lstm";
        case BACKBONE_GRU:    return "gru";
        case BACKBONE_MINGRU: return "mingru";
        default:              return "unknown";
    }
}

inline BackboneType backbone_from_string(const char* s) {
    if (strcmp(s, "mamba") == 0 || strcmp(s, "mambav1") == 0) return BACKBONE_MAMBA;
    if (strcmp(s, "lstm") == 0)   return BACKBONE_LSTM;
    if (strcmp(s, "gru") == 0)    return BACKBONE_GRU;
    if (strcmp(s, "mingru") == 0) return BACKBONE_MINGRU;
    return BACKBONE_MAMBA; // default
}

// ==========================================
// Model Configuration
// ==========================================
struct MambaConfig {
    int d_model;
    int n_layers;
    BackboneType backbone = BACKBONE_MAMBA;

    // Mamba-specific
    int d_state = 16;
    int expand_factor = 2;
    int d_conv = 4;
    int dt_rank = 0; // if 0, set to ceil(d_model / 16)

    // Calculated
    int d_inner;
    bool use_rmsnorm = true;

    MambaConfig() : d_model(256), n_layers(4) {
        update();
    }

    MambaConfig(int d_model, int n_layers) : d_model(d_model), n_layers(n_layers) {
        if (dt_rank == 0) dt_rank = (d_model + 15) / 16;
        d_inner = expand_factor * d_model;
    }

    void update() {
        dt_rank = (d_model + 15) / 16;
        d_inner = expand_factor * d_model;
    }
};

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

#ifndef checkCudaErrors
#define checkCudaErrors(call) CHECK_CUDA(call)
#endif

#ifndef GPU_SYNC
#define GPU_SYNC 0
#endif

#ifndef GPU_FAST_EXP
#define GPU_FAST_EXP 0
#endif

#ifndef GPU_SOFTMAX_LUT
#define GPU_SOFTMAX_LUT 0
#endif

#ifndef GPU_REPRO_MATH
#define GPU_REPRO_MATH 0
#endif

#ifndef GPU_DEBUG_LOGITS
#define GPU_DEBUG_LOGITS 0
#endif


#if GPU_SYNC
#define GPU_DEVICE_SYNC() checkCudaErrors(cudaDeviceSynchronize())
#else
#define GPU_DEVICE_SYNC() do {} while(0)
#endif

// Global stream for CUDA Graph support.
// Default is 0 (legacy default stream). Set to a capture stream
// before cudaStreamBeginCapture so all kernels are captured.
extern cudaStream_t g_stream;

// Structs
// Structs
struct RCState {
    unsigned long long lower;
    unsigned long long range;
    int inverted_num;
    unsigned int first_inv_lower_word;
    int write_idx_words;
};

// GPU Kernel Declarations (Sequential/Legacy)
void gemm_gpu_naive(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_gpu_tiled(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_gpu_optimized(const float* A, const float* B, float* C, int M, int N, int K);

void gpu_gelu(float* x, int N);
void gpu_relu(float* x, int N);
void gpu_silu_batch(float* x, size_t total);
void gpu_rmsnorm_batch(float* x, const float* w, int N, size_t batch);
void gpu_add_bias(float* x, const float* b, int M, int N); // M rows, N cols, bias [N]
void gpu_softmax(float* x, int M, int N);
void gpu_layernorm(float* x, const float* w, const float* b, int N);

void gpu_mamba_conv1d(const float* x_in, float* state, const float* weight, const float* bias, float* out, int d_inner, int d_conv);
void gpu_mamba_conv1d_chunk(const float* x, float* state, const float* weight, const float* bias, float* out, int Batch, int Length, int Dim, int Kernel, int x_stride);

void gpu_mamba_ssm(const float* x, const float* dt, const float* A_log, const float* D, 
                   const float* B, const float* C, float* state, float* out, 
                   int d_inner, int d_state);
void gpu_mamba_ssm_chunk(const float* u, const float* dt, const float* A_log, const float* D,
                         const float* B, const float* C, float* state, float* out,
                         size_t Batch, int Length, int Dim, int State);
void gpu_mamba_ssm_chunk_fused(const float* u, const float* delta,
                               const float* dt_proj_w, const float* dt_proj_b,
                               const float* A_log, const float* D,
                               float* state, float* out,
                               size_t Batch, int Length, int Dim, int State,
                               int dt_rank, int stride_delta);

void gpu_add_bias_softplus(float* x, const float* b, int N);
void gpu_gate(float* y, const float* z, int N);
void gpu_gate_strided(float* y, const float* z, int d_inner, int z_stride, size_t total);

void gemv_gpu_repro(const float* A, const float* B, float* C, int N, int K);

// Activation codes for fused GEMM epilogue
#define GEMM_ACT_NONE 0
#define GEMM_ACT_GELU 1
#define GEMM_ACT_RELU 2
#define GEMM_ACT_SILU 3

// Batched Kernels
void gemm_gpu_batch(const float* A, const float* B, float* C, size_t batch, int K, int N); 
void gemm_gpu_batch_strided(const float* A, int lda, const float* B, int ldb, float* C, int ldc, size_t batch, int K, int N);
void gemm_gpu_batch_bias(const float* A, const float* B, const float* bias, float* C, size_t batch, int K, int N); // Fused GEMM + bias add
void gemm_gpu_batch_bias_act(const float* A, const float* B, const float* bias, float* C, size_t batch, int K, int N, int activation); // Fused GEMM + bias + activation
void gemm_gpu_batch_bias_res(const float* A, const float* B, const float* bias, const float* residual, float* C, size_t batch, int K, int N); // Fused GEMM + bias + residual add
void gpu_copy_strided(const float* src, float* dst, int src_stride, int width, size_t batch);

void gpu_embedding_lookup_batch(const int* tokens, const float* embedding, float* out, int d_model, size_t batch);

void gpu_layernorm_batch(float* x, const float* w, const float* b, int row_size, size_t batch);
void gpu_layernorm_save_batch(float* x, float* save, const float* w, const float* b, int row_size, size_t batch); // Fused: save x to save, then layernorm x in-place
void gpu_add_batch(float* x, const float* y, int row_size, size_t batch);

void gpu_mamba_conv1d_batch(const float* x_in, float* state, const float* weight, const float* bias, float* out, int d_inner, int d_conv, int x_stride, size_t batch);
void gpu_mamba_ssm_batch(const float* x, const float* dt, const float* A_log, const float* D, 
                         const float* B, const float* C, float* state, float* out, 
                         int d_inner, int d_state, int dt_rank, size_t batch);

void gpu_add_bias_softplus_batch(float* x, const float* b, int N, size_t batch);
void gpu_gelu_batch(float* x, int N, size_t batch);

void gpu_mamba_ssm_fused_batch(const float* x_conv, const float* x_z, const float* dt_logits, const float* dt_bias, 
                               const float* A_log, const float* D, 
                               const float* B_base_ptr, float* state, float* out, 
                               int d_inner, int d_state, int dt_rank, size_t batch);



// Range Coder Encoder Kernels
void gpu_rc_init(RCState* states, int batch_size);
void gpu_rc_encode_step_batch(const float* logits, const int* targets, RCState* states, unsigned int* out_bufs, int pitch_words, int vocab_size, int batch_size);
void gpu_rc_encode_step_batch_strided(const float* logits, const int* targets, RCState* states, unsigned int* out_bufs, int pitch_words, int vocab_size, int batch_size, int logit_stride);
void gpu_rc_encode_step_batch_masked(const float* logits, const int* targets, const int* lengths, int t, RCState* states, unsigned int* out_bufs, int pitch_words, int vocab_size, int batch_size);
void gpu_rc_encode_chunk_batch_strided(const float* logits, const int* chunk_data, const int* lengths, RCState* states, unsigned int* out_bufs,
                                       int pitch_words, int vocab_size, int batch_size, int len, int max_len, int sub_start, int logit_stride);
// Warp Parallel Version
void gpu_rc_encode_chunk_warp(const float* logits, const int* chunk_data, const int* lengths,
                              RCState* states, unsigned int* out_bufs,
                              int pitch_words, int vocab_size, int batch_size,
                              int chunk_len, int max_len, int start_t, int logit_stride);

void gpu_init_exp_lut();
void gpu_set_temperature(float temperature);

void gpu_rc_finish_batch(RCState* states, unsigned int* out_bufs, int pitch_words, int* sizes_words, int batch_size);

// Range Coder Decoder State
struct RCDecState {
    unsigned long long lower;
    unsigned long long range;
    unsigned long long point;
    int read_idx_words;
};

// Range Coder Decoder Kernels
void gpu_rc_init_decoder(RCDecState* states, const unsigned int* in_bufs, const int* sizes_words, int pitch_words, int batch_size);
void gpu_rc_decode_step_batch(const float* logits, const int* in_tokens, const int* lengths, int t, int* out_symbols, RCDecState* states, const unsigned int* in_bufs, const int* sizes_words, int pitch_words, int vocab_size, int batch_size);
void gpu_rc_decode_fused_step_batch(const float* logits, const float* head_b2, int* d_tokens, const int* lengths, int t, int* d_batch_output, int chunk_size, RCDecState* states, const unsigned int* in_bufs, const int* sizes_words, int pitch_words, int vocab_size, int batch_size);
void gpu_rc_decode_fused_step_batch_graph(const float* logits, const float* head_b2, int* d_tokens, const int* lengths, const int* d_t, int* d_batch_output, int chunk_size, RCDecState* states, const unsigned int* in_bufs, const int* sizes_words, int pitch_words, int vocab_size, int batch_size);
void gpu_increment_counter(int* d_counter);

// Optimizations
void gpu_select_tokens(const int* d_chunk_data, int* d_tokens, int t, int chunk_size, int batch_size);
void gpu_store_tokens(const int* d_out_symbols, int* d_chunk_data, int t, int chunk_size, int batch_size);

// ==========================================
// RNN Kernels (LSTM / GRU / minGRU)
// ==========================================

// LSTM: fused gate computation after GEMM projections
// ih, hh: [B, 4*d_model] pre-computed projections (with bias already added)
// h_state: [B, d_model], c_state: [B, d_model] — updated in-place
void gpu_lstm_fused_batch(const float* ih, const float* hh, float* h_state, float* c_state, int d_model, size_t batch);

// GRU: fused gate computation after GEMM projections
// ih, hh: [B, 3*d_model] pre-computed projections (with bias already added)
// h_state: [B, d_model] — updated in-place
void gpu_gru_fused_batch(const float* ih, const float* hh, float* h_state, int d_model, size_t batch);

// minGRU: fused step — z=sigmoid(z_logits), h=(1-z)*h_prev + z*h_tilde
// z_logits, h_tilde: [B, d_model], h_state: [B, d_model] — updated in-place
void gpu_mingru_fused_batch(const float* z_logits, const float* h_tilde, float* h_state, int d_model, size_t batch);

// minGRU: chunk processing — sequential over time, parallel over (batch, dim)
// z_all, h_all: [B*L, d_model] pre-computed, h_state: [B, d_model], out: [B*L, d_model]
void gpu_mingru_chunk(const float* z_all, const float* h_all, float* h_state, float* out,
                      int batch, int length, int d_model);

// Copy hidden state h into output buffer at timestep t: out[b*L+t, :] = h[b, :]
void gpu_scatter_timestep(const float* h, float* out, int batch, int length, int t, int d_model);

// Gather timestep t from [B, L, width] layout into [B, width] buffer
void gpu_gather_timestep(const float* src, float* dst, int batch, int length, int t, int width);

// Helpers
void malloc_device(float** ptr, size_t bytes);
void free_device(float* ptr);
void copy_to_device(float* d_ptr, const float* h_ptr, size_t bytes);
void copy_to_host(float* h_ptr, const float* d_ptr, size_t bytes);

void get_gpu_vram_info(size_t& free_bytes, size_t& total_bytes);
