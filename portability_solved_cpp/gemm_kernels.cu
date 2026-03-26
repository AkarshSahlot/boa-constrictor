// GEMM Kernels - Basic matrix multiplication implementations
#include "gemm_gpu_common.cuh"

// ==========================================
// Naive GEMM Kernel
// ==========================================

__global__ void gemm_kernel_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void gemm_gpu_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    gemm_kernel_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    GPU_DEVICE_SYNC();

    CHECK_CUDA(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ==========================================
// Tiled GEMM Kernel
// ==========================================

#define TILE_SIZE 32

__global__ void gemm_kernel_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A tile
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            s_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial sum for this tile - strictly sequential K within the tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        // Safe 2D
        C[(size_t)row * N + col] = sum;
    }
}

void gemm_gpu_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_kernel_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    GPU_DEVICE_SYNC();

    CHECK_CUDA(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ==========================================
// Unrolled Tiled GEMM Kernel
// ==========================================

__global__ void gemm_kernel_unrolled(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_col = t * TILE_SIZE + threadIdx.x;
        int tiled_row = t * TILE_SIZE + threadIdx.y;

        // Load A
        if (row < M && tiled_col < K)
             s_A[threadIdx.y][threadIdx.x] = A[(size_t)row * K + tiled_col];
        else s_A[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B
        if (col < N && tiled_row < K)
             s_B[threadIdx.y][threadIdx.x] = B[(size_t)tiled_row * N + col];
        else s_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute - Unroll
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[(size_t)row * N + col] = sum;
    }
}

// ==========================================
// Register Tiled GEMM (64x64 output tile, 16x16 threads)
// Deterministic (Sequential Summation)
// Double-buffered: overlap loading tile k+1 with computing tile k
// ==========================================

// Activation codes for fused GEMM epilogue
// 0=none, 1=gelu, 2=relu, 3=silu
#define GEMM_ACT_NONE 0
#define GEMM_ACT_GELU 1
#define GEMM_ACT_RELU 2
#define GEMM_ACT_SILU 3

__global__ void ker_gemm_register_tiled_64x64_strided(const float* A, int lda, const float* B, int ldb, float* C, int ldc, int M, int N, int K, const float* __restrict__ bias = nullptr, int activation = 0, const float* __restrict__ residual = nullptr) {
    // 256 threads. Each processes 4x4 elements of C.
    // Block handles 64x64.
    __shared__ float s_A[2][64][16];
    __shared__ float s_B[2][16][64];

    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;
    int tid = threadIdx.x;

    int row_start = blockIdx.y * 64 + ty * 4;
    int col_start = blockIdx.x * 64 + tx * 4;

    int r_A = tid / 4;
    int c_A = (tid % 4) * 4;
    int r_B = tid / 16;
    int c_B = (tid % 16) * 4;

    bool a_aligned = (lda % 4 == 0);
    bool b_aligned = (ldb % 4 == 0);

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[i][j] = 0.0f;

    int num_tiles = (K + 15) / 16;

    // Load first tile into buffer 0
    {
        int abs_r_A = blockIdx.y * 64 + r_A;
        int abs_c_A = c_A;
        float4 valA = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (abs_r_A < M && abs_c_A < K) {
             if (a_aligned && (abs_c_A + 3 < K || (K%4==0)))
                 valA = reinterpret_cast<const float4*>(&A[(size_t)abs_r_A * (size_t)lda + abs_c_A])[0];
             else {
                 valA.x = A[(size_t)abs_r_A * (size_t)lda + abs_c_A];
                 if (abs_c_A+1 < K) valA.y = A[(size_t)abs_r_A * (size_t)lda + abs_c_A+1];
                 if (abs_c_A+2 < K) valA.z = A[(size_t)abs_r_A * (size_t)lda + abs_c_A+2];
                 if (abs_c_A+3 < K) valA.w = A[(size_t)abs_r_A * (size_t)lda + abs_c_A+3];
             }
        }
        s_A[0][r_A][c_A+0] = valA.x; s_A[0][r_A][c_A+1] = valA.y;
        s_A[0][r_A][c_A+2] = valA.z; s_A[0][r_A][c_A+3] = valA.w;

        int abs_r_B = r_B;
        int abs_c_B = blockIdx.x * 64 + c_B;
        float4 valB = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (abs_r_B < K && abs_c_B < N) {
             if (b_aligned && (abs_c_B + 3 < N || (N%4==0)))
                 valB = reinterpret_cast<const float4*>(&B[(size_t)abs_r_B * (size_t)ldb + abs_c_B])[0];
             else {
                 valB.x = B[(size_t)abs_r_B * (size_t)ldb + abs_c_B];
                 if (abs_c_B+1 < N) valB.y = B[(size_t)abs_r_B * (size_t)ldb + abs_c_B+1];
                 if (abs_c_B+2 < N) valB.z = B[(size_t)abs_r_B * (size_t)ldb + abs_c_B+2];
                 if (abs_c_B+3 < N) valB.w = B[(size_t)abs_r_B * (size_t)ldb + abs_c_B+3];
             }
        }
        s_B[0][r_B][c_B+0] = valB.x; s_B[0][r_B][c_B+1] = valB.y;
        s_B[0][r_B][c_B+2] = valB.z; s_B[0][r_B][c_B+3] = valB.w;
    }
    __syncthreads();

    for (int t = 0; t < num_tiles; ++t) {
        int buf = t & 1;
        int next_buf = 1 - buf;

        // Prefetch next tile into next_buf
        if (t + 1 < num_tiles) {
            int k_next = (t + 1) * 16;
            int abs_r_A = blockIdx.y * 64 + r_A;
            int abs_c_A = k_next + c_A;
            float4 valA = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (abs_r_A < M && abs_c_A < K) {
                 if (a_aligned && (abs_c_A + 3 < K || (K%4==0)))
                     valA = reinterpret_cast<const float4*>(&A[(size_t)abs_r_A * (size_t)lda + abs_c_A])[0];
                 else {
                     valA.x = A[(size_t)abs_r_A * (size_t)lda + abs_c_A];
                     if (abs_c_A+1 < K) valA.y = A[(size_t)abs_r_A * (size_t)lda + abs_c_A+1];
                     if (abs_c_A+2 < K) valA.z = A[(size_t)abs_r_A * (size_t)lda + abs_c_A+2];
                     if (abs_c_A+3 < K) valA.w = A[(size_t)abs_r_A * (size_t)lda + abs_c_A+3];
                 }
            }
            s_A[next_buf][r_A][c_A+0] = valA.x; s_A[next_buf][r_A][c_A+1] = valA.y;
            s_A[next_buf][r_A][c_A+2] = valA.z; s_A[next_buf][r_A][c_A+3] = valA.w;

            int abs_r_B = k_next + r_B;
            int abs_c_B = blockIdx.x * 64 + c_B;
            float4 valB = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (abs_r_B < K && abs_c_B < N) {
                 if (b_aligned && (abs_c_B + 3 < N || (N%4==0)))
                     valB = reinterpret_cast<const float4*>(&B[(size_t)abs_r_B * (size_t)ldb + abs_c_B])[0];
                 else {
                     valB.x = B[(size_t)abs_r_B * (size_t)ldb + abs_c_B];
                     if (abs_c_B+1 < N) valB.y = B[(size_t)abs_r_B * (size_t)ldb + abs_c_B+1];
                     if (abs_c_B+2 < N) valB.z = B[(size_t)abs_r_B * (size_t)ldb + abs_c_B+2];
                     if (abs_c_B+3 < N) valB.w = B[(size_t)abs_r_B * (size_t)ldb + abs_c_B+3];
                 }
            }
            s_B[next_buf][r_B][c_B+0] = valB.x; s_B[next_buf][r_B][c_B+1] = valB.y;
            s_B[next_buf][r_B][c_B+2] = valB.z; s_B[next_buf][r_B][c_B+3] = valB.w;
        }

        // Compute from current buffer — same accumulation order as original
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            float a[4], b[4];
            a[0] = s_A[buf][ty*4+0][k]; a[1] = s_A[buf][ty*4+1][k];
            a[2] = s_A[buf][ty*4+2][k]; a[3] = s_A[buf][ty*4+3][k];
            b[0] = s_B[buf][k][tx*4+0]; b[1] = s_B[buf][k][tx*4+1];
            b[2] = s_B[buf][k][tx*4+2]; b[3] = s_B[buf][k][tx*4+3];
            #pragma unroll
            for (int i=0; i<4; ++i)
                #pragma unroll
                for (int j=0; j<4; ++j)
                    acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Store C (with optional fused bias + activation + residual add)
    #pragma unroll
    for (int i=0; i<4; ++i)
        for (int j=0; j<4; ++j) {
            int r = row_start + i;
            int col = col_start + j;
            if (r < M && col < N) {
                float val = acc[i][j] + (bias ? bias[col] : 0.0f);
                if (activation == GEMM_ACT_GELU) val = repro_gelu(val);
                else if (activation == GEMM_ACT_RELU) val = fmaxf(0.0f, val);
                else if (activation == GEMM_ACT_SILU) val = repro_silu(val);
                if (residual) val += residual[(size_t)r * (size_t)ldc + col];
                C[(size_t)r * (size_t)ldc + col] = val;
            }
        }
}

void gemm_gpu_optimized(const float* A, const float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    int grid_y = (M + 63) / 64; 
    int grid_x = (N + 63) / 64;
    dim3 grid(grid_x, grid_y);

    ker_gemm_register_tiled_64x64_strided<<<grid, block>>>(d_A, K, d_B, N, d_C, N, M, N, K);
    GPU_DEVICE_SYNC();

    CHECK_CUDA(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ==========================================
// GEMV Optimized Kernel (M=1)
// ==========================================
// Specialized for Vector (1xK) * Matrix (KxN) = Vector (1xN).
// Maintains strictly sequential summation over K to match CPU loop order.

__global__ void gemv_kernel_repro(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N, int K) {
    // One thread per output column j
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < N) {
        float sum = 0.0f;
        // Strictly sequential accumulation loop matching CPU
        for (int k = 0; k < K; ++k) {
            // A is 1xK (row vector)
            // B is KxN (row major)
            // B[k, j] is at index k*N + j
            sum += A[k] * B[k * N + j];
        }
        C[j] = sum;
    }
}

void gemv_gpu_repro(const float* A, const float* B, float* C, int N, int K) {
    int block = 256;
    int grid = (N + block - 1) / block;
    gemv_kernel_repro<<<grid, block>>>(A, B, C, N, K);
    GPU_DEVICE_SYNC();
}

// ==========================================
// Batched GEMM
// ==========================================

void gemm_gpu_batch(const float* A, const float* B, float* C, size_t batch, int K, int N) {
    gemm_gpu_batch_strided(A, N, B, K, C, K, batch, K, N);
}

void gemm_gpu_batch_strided(const float* A, int lda, const float* B, int ldb, float* C, int ldc, size_t batch, int K, int N) {
    int M = (int)batch;
    dim3 block(256); // Flat block for tiled kernel logic
    int grid_y = (M + 63) / 64; 
    int grid_x = (K + 63) / 64;
    dim3 grid(grid_x, grid_y);
    
    ker_gemm_register_tiled_64x64_strided<<<grid, block, 0, g_stream>>>(A, lda, B, ldb, C, ldc, M, K, N);
}

// ==========================================
// Fused GEMM + Bias (single kernel launch)
// ==========================================

void gemm_gpu_batch_bias(const float* A, const float* B, const float* bias, float* C, size_t batch, int K, int N) {
    int M = (int)batch;
    dim3 block(256);
    int grid_y = (M + 63) / 64; 
    int grid_x = (K + 63) / 64;
    dim3 grid(grid_x, grid_y);
    
    ker_gemm_register_tiled_64x64_strided<<<grid, block, 0, g_stream>>>(A, N, B, K, C, K, M, K, N, bias);
}

// Fused GEMM + Bias + Activation (single kernel launch)
void gemm_gpu_batch_bias_act(const float* A, const float* B, const float* bias, float* C, size_t batch, int K, int N, int activation) {
    int M = (int)batch;
    dim3 block(256);
    int grid_y = (M + 63) / 64; 
    int grid_x = (K + 63) / 64;
    dim3 grid(grid_x, grid_y);
    
    ker_gemm_register_tiled_64x64_strided<<<grid, block, 0, g_stream>>>(A, N, B, K, C, K, M, K, N, bias, activation);
}

// Fused GEMM + Bias + Residual Add (single kernel launch)
void gemm_gpu_batch_bias_res(const float* A, const float* B, const float* bias, const float* residual, float* C, size_t batch, int K, int N) {
    int M = (int)batch;
    dim3 block(256);
    int grid_y = (M + 63) / 64; 
    int grid_x = (K + 63) / 64;
    dim3 grid(grid_x, grid_y);
    
    ker_gemm_register_tiled_64x64_strided<<<grid, block, 0, g_stream>>>(A, N, B, K, C, K, M, K, N, bias, 0, residual);
}
