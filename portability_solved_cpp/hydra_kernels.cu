/*
 * hydra_kernels.cu — Small helper CUDA kernels for HydraBOA.
 *
 * These kernels are all graph-captureable (launch on g_stream, no host sync).
 */

#include <cuda_runtime.h>
#include "gemm_gpu.hpp"   // g_stream, checkCudaErrors

// ─────────────────────────────────────────────────────────────────────
// hydra_store_decoded — store K decoded bytes using a device-side step
//   counter so this works inside a CUDA graph.
//
//   d_decoded    [batch, K]           source (stride K)
//   d_batch_out  [batch, chunk_size]  destination (stride chunk_size)
//   d_step       [1]                  current backbone step (on device)
//
//   Writes: d_batch_out[b * chunk_size + (*d_step)*K + k] = d_decoded[b*K+k]
// ─────────────────────────────────────────────────────────────────────
__global__ void ker_hydra_store_decoded(
    const int* __restrict__ d_decoded,
    int*       __restrict__ d_batch_out,
    const int* __restrict__ d_step,
    int K, int chunk_size, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * K) return;
    int b = idx / K;
    int k = idx % K;
    int step = *d_step;
    int pos = step * K + k;
    if (pos < chunk_size)
        d_batch_out[b * chunk_size + pos] = d_decoded[b * K + k];
}

void hydra_store_decoded(const int* d_decoded, int* d_batch_out,
                         const int* d_step, int K, int chunk_size, int batch)
{
    int total = batch * K;
    ker_hydra_store_decoded<<<(total+255)/256, 256, 0, g_stream>>>(
        d_decoded, d_batch_out, d_step, K, chunk_size, batch);
    checkCudaErrors(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────
// hydra_build_head_input — concatenate [H_t, byte_embeds_0..k-1]
//
//   H_t         [batch, D]        backbone hidden
//   byte_embeds [batch, K, D]     (stride K*D per batch element)
//   head_inp    [batch, out_stride] output  (only first (1+k)*D filled)
//   k           head index (1..K-1)  — k=0 is handled directly
//   out_stride  leading dimension of head_inp (>= (1+k)*D)
// ─────────────────────────────────────────────────────────────────────
__global__ void ker_hydra_build_head_input(
    const float* __restrict__ H_t,
    const float* __restrict__ byte_embeds,
    float*       __restrict__ head_inp,
    int D, int k, int KD,   // KD = K*D (byte_embeds row stride)
    int out_stride, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_len = (1 + k) * D;
    int total = batch * row_len;
    if (idx >= total) return;

    int b   = idx / row_len;
    int col = idx % row_len;

    float val;
    if (col < D)
        val = H_t[b * D + col];
    else
        val = byte_embeds[b * KD + (col - D)];   // col-D is in [0, k*D)

    head_inp[b * out_stride + col] = val;
}

void hydra_build_head_input(const float* H_t, const float* byte_embeds,
                            float* head_inp, int D, int k, int K,
                            int out_stride, int batch)
{
    int total = batch * (1 + k) * D;
    ker_hydra_build_head_input<<<(total+255)/256, 256, 0, g_stream>>>(
        H_t, byte_embeds, head_inp, D, k, K*D, out_stride, batch);
    checkCudaErrors(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────
// hydra_store_decoded_host — same but step counter is a host int
//   (used in the --no-graph fallback path)
// ─────────────────────────────────────────────────────────────────────
__global__ void ker_hydra_store_decoded_host(
    const int* __restrict__ d_decoded,
    int*       __restrict__ d_batch_out,
    int step, int K, int chunk_size, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * K) return;
    int b = idx / K;
    int k = idx % K;
    int pos = step * K + k;
    if (pos < chunk_size)
        d_batch_out[b * chunk_size + pos] = d_decoded[b * K + k];
}

void hydra_store_decoded_host(const int* d_decoded, int* d_batch_out,
                              int step, int K, int chunk_size, int batch)
{
    int total = batch * K;
    ker_hydra_store_decoded_host<<<(total+255)/256, 256, 0, g_stream>>>(
        d_decoded, d_batch_out, step, K, chunk_size, batch);
    checkCudaErrors(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────
// hydra_broadcast_vector — replicate a D-vector to [batch, D]
//   Useful for BOS embedding broadcast.
// ─────────────────────────────────────────────────────────────────────
__global__ void ker_broadcast_vector(const float* __restrict__ src,
                                     float* __restrict__ dst,
                                     int D, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * D) return;
    int d = idx % D;
    dst[idx] = src[d];
}

void hydra_broadcast_vector(const float* src, float* dst, int D, int batch)
{
    int total = batch * D;
    ker_broadcast_vector<<<(total+255)/256, 256, 0, g_stream>>>(src, dst, D, batch);
    checkCudaErrors(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────
// hydra_fill_bos_strided — broadcast BOS embed into position 0 of each
//   batch element's tile, with tile_T stride.
//
//   dst[b * tile_T * D + d] = src[d]   for b=0..batch-1, d=0..D-1
// ─────────────────────────────────────────────────────────────────────
__global__ void ker_fill_bos_strided(const float* __restrict__ src,
                                     float* __restrict__ dst,
                                     int D, int stride_D, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * D) return;
    int b = idx / D;
    int d = idx % D;
    dst[b * stride_D + d] = src[d];
}

void hydra_fill_bos_strided(const float* src, float* dst,
                            int D, int tile_T, int batch)
{
    int total = batch * D;
    int stride_D = tile_T * D;
    ker_fill_bos_strided<<<(total+255)/256, 256, 0, g_stream>>>(
        src, dst, D, stride_D, batch);
    checkCudaErrors(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────
// hydra_scatter_logits — scatter contiguous head logits into the
//   interleaved tile-logits buffer.
//
//   src  [batch * tile_T, V]               contiguous head output
//   dst  [batch, tile_T * K, V]            interleaved tile logits
//
//   dst[b * tile_T*K*V + (t*K+k)*V + v] = src[(b*tile_T+t)*V + v]
// ─────────────────────────────────────────────────────────────────────
__global__ void ker_scatter_logits(const float* __restrict__ src,
                                   float* __restrict__ dst,
                                   int tile_T, int K, int k, int V,
                                   int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * tile_T * V;
    if (idx >= total) return;

    int v  = idx % V;
    int bt = idx / V;
    int b  = bt / tile_T;
    int t  = bt % tile_T;

    dst[(size_t)b * tile_T * K * V + (size_t)(t * K + k) * V + v] =
        src[(size_t)bt * V + v];
}

void hydra_scatter_logits(const float* src, float* dst,
                          int tile_T, int K, int k, int V, int batch)
{
    int total = batch * tile_T * V;
    ker_scatter_logits<<<(total+255)/256, 256, 0, g_stream>>>(
        src, dst, tile_T, K, k, V, batch);
    checkCudaErrors(cudaGetLastError());
}
