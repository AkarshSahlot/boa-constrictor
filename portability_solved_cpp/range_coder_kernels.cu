// Range Coder Kernels - GPU-accelerated arithmetic coding
#include "gemm_gpu_common.cuh"

#define PRECISION 24

// Temperature scaling: inv_temperature = 1.0/T applied to logits before softmax
__constant__ float d_inv_temperature = 1.0f;

void gpu_set_temperature(float temperature) {
    float inv_t = 1.0f / temperature;
    checkCudaErrors(cudaMemcpyToSymbol(d_inv_temperature, &inv_t, sizeof(float)));
}

// ==========================================
// CDF Building Helper
// ==========================================

__device__ void build_cdf_fast(const float* probs_row, int K, unsigned int* cdf) {
    const unsigned int TOTAL = 1u << PRECISION;
    if (K <= 0) { cdf[0] = 0; cdf[1] = TOTAL; return; }
    const unsigned int free_weight = TOTAL - (unsigned int)K;
    double norm = 0.0;
    for (int i = 0; i < K; ++i) norm += (double)probs_row[i];
    if (!(norm > 0.0)) {
        cdf[0] = 0;
        unsigned int acc = 0;
        for (int i=0;i<K;i++) { cdf[i] = acc; acc += (free_weight / (unsigned int)K) + 1u; }
        cdf[K] = TOTAL; return;
    }
    double scale = (double)free_weight / norm;
    double cumulative_float = 0.0;
    unsigned int accumulated_slack = 0;
    for (int i=0;i<K;i++) {
        unsigned int left = (unsigned int)(cumulative_float * scale) + accumulated_slack;
        cdf[i] = left;
        cumulative_float += (double)probs_row[i];
        accumulated_slack += 1u;
    }
    cdf[K] = TOTAL;
}

// ==========================================
// Encoder Initialization
// ==========================================

__global__ void ker_rc_init(RCState* states, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        states[i].lower = 0ull;
        states[i].range = 0xFFFFFFFFFFFFFFFFull;
        states[i].inverted_num = 0;
        states[i].first_inv_lower_word = 0u;
        states[i].write_idx_words = 0;
    }
}

void gpu_rc_init(RCState* states, int batch_size) {
    ker_rc_init<<<(batch_size+255)/256, 256, 0, g_stream>>>(states, batch_size);
    GPU_DEVICE_SYNC();
}

// ==========================================
// Symbol Encoding Helper
// ==========================================

__device__ inline void rc_encode_symbol_probs(const float* probs, int vocab_size, int target, RCState& st, unsigned int* out) {
    unsigned int cdf[257];
    build_cdf_fast(probs, vocab_size, cdf);

    unsigned int left = cdf[target];
    unsigned int prob = cdf[target+1] - cdf[target];

    unsigned long long scale = st.range >> PRECISION;
    unsigned long long old_lower = st.lower;
    st.range = scale * (unsigned long long)prob;
    st.lower = st.lower + scale * (unsigned long long)left;

    if (st.inverted_num > 0) {
        unsigned long long sum = st.lower + st.range;
        if (sum > st.lower) {
            unsigned int first_word, subsequent;
            if (st.lower < old_lower) { first_word = st.first_inv_lower_word + 1u; subsequent = 0u; }
            else { first_word = st.first_inv_lower_word; subsequent = 0xFFFFFFFFu; }
            int widx = st.write_idx_words;
            out[(size_t)widx++] = first_word;
            for (int k=1; k<st.inverted_num; k++) out[(size_t)widx++] = subsequent;
            st.write_idx_words = widx;
            st.inverted_num = 0;
        }
    }

    while (st.range < (1ull << (64-32))) {
        unsigned int lower_word = (unsigned int)(st.lower >> (64-32));
        st.lower <<= 32;
        st.range <<= 32;
        if (st.inverted_num > 0) {
            if (st.inverted_num < 0x7FFFFFFF) st.inverted_num += 1;
        } else {
            unsigned long long sum = st.lower + st.range;
            if (sum > st.lower) {
                int widx = st.write_idx_words;
                out[(size_t)widx++] = lower_word;
                st.write_idx_words = widx;
            } else {
                st.inverted_num = 1;
                st.first_inv_lower_word = lower_word;
            }
        }
    }
}

// ==========================================
// Encode Step Kernels
// ==========================================

__global__ void ker_rc_encode_step(const float* logits, const int* targets, RCState* states, unsigned int* out_words, int pitch_words, int vocab_size, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch) {
        int target = targets[i];
        const float* l = logits + i * vocab_size;
        
        float max_l = l[0];
        for(int j=1; j<vocab_size; ++j) if(l[j]>max_l) max_l = l[j];
        
        float probs[256];
        float sum = 0.0f;
        for(int j=0; j<vocab_size; ++j) {
            float val = __expf((l[j] - max_l) * d_inv_temperature);
            probs[j] = val;
            sum += val;
        }
        float inv_sum = 1.0f / sum;
        for(int j=0; j<vocab_size; ++j) probs[j] *= inv_sum;

        RCState &st = states[i];
        unsigned int* out = out_words + (size_t)i * (size_t)pitch_words;
        rc_encode_symbol_probs(probs, vocab_size, target, st, out);
    }
}

__global__ void ker_rc_encode_step_strided(const float* logits, const int* targets, RCState* states, unsigned int* out_bufs, int pitch_words, int vocab_size, int batch_size, int logit_stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        const float* l = logits + (size_t)i * (size_t)logit_stride;
        int target = targets[i];
        
        RCState &st = states[i];
        unsigned int* out = out_bufs + (size_t)i * (size_t)pitch_words;
        
        float max_l = l[0];
        for(int j=1; j<vocab_size; ++j) if(l[j]>max_l) max_l = l[j];
        
        float probs[256];
        float sum = 0.0f;
        for(int j=0; j<vocab_size; ++j) {
            float val = __expf((l[j] - max_l) * d_inv_temperature);
            probs[j] = val;
            sum += val;
        }
        float inv_sum = 1.0f / sum;
        for(int j=0; j<vocab_size; ++j) probs[j] *= inv_sum;
        
        rc_encode_symbol_probs(probs, vocab_size, target, st, out);
    }
}

__global__ void ker_rc_encode_step_masked(const float* logits, const int* targets, const int* lengths, int t, RCState* states, unsigned int* out_bufs, int pitch_words, int vocab_size, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        if (lengths && t >= lengths[i]) return;
        int target = targets[i];
        const float* l = logits + (size_t)i * (size_t)vocab_size;

        float max_l = l[0];
        for(int j=1; j<vocab_size; ++j) if(l[j]>max_l) max_l = l[j];

        float probs[256];
        float sum = 0.0f;
        for(int j=0; j<vocab_size; ++j) {
            float val = __expf((l[j] - max_l) * d_inv_temperature);
            probs[j] = val;
            sum += val;
        }
        float inv_sum = 1.0f / sum;
        for(int j=0; j<vocab_size; ++j) probs[j] *= inv_sum;

        RCState &st = states[i];
        unsigned int* out = out_bufs + (size_t)i * (size_t)pitch_words;
        rc_encode_symbol_probs(probs, vocab_size, target, st, out);
    }
}

__global__ void ker_rc_encode_chunk_strided(const float* logits, const int* chunk_data, const int* lengths, RCState* states, unsigned int* out_bufs,
                                            int pitch_words, int vocab_size, int batch_size, int len, int max_len, int sub_start, int logit_stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        int seq_len = lengths ? lengths[i] : max_len;
        if (seq_len <= 1) return;
        RCState &st = states[i];
        unsigned int* out = out_bufs + (size_t)i * (size_t)pitch_words;
        const int* row = chunk_data + (size_t)i * (size_t)max_len;

        float probs[256];

        int max_k = seq_len - 1 - sub_start;
        if (max_k > len) max_k = len;
        if (max_k <= 0) return;

        for (int k = 0; k < max_k; ++k) {
            int target = row[sub_start + k + 1];
            const float* l = logits + (size_t)i * (size_t)logit_stride + (size_t)k * (size_t)vocab_size;

            float max_l = l[0];
            for(int j=1; j<vocab_size; ++j) if(l[j]>max_l) max_l = l[j];

            float sum = 0.0f;
            for(int j=0; j<vocab_size; ++j) {
                float val = __expf((l[j] - max_l) * d_inv_temperature);
                probs[j] = val;
                sum += val;
            }
            float inv_sum = 1.0f / sum;
            for(int j=0; j<vocab_size; ++j) probs[j] *= inv_sum;

            rc_encode_symbol_probs(probs, vocab_size, target, st, out);
        }
    }
}

// ==========================================
// Host-Side Encode Functions
// ==========================================

void gpu_rc_encode_step_batch_strided(const float* logits, const int* targets, RCState* states, unsigned int* out_bufs, int pitch_words, int vocab_size, int batch_size, int logit_stride) {
    static bool stack_set = false;
    if (!stack_set) {
        checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
        stack_set = true;
    }
    ker_rc_encode_step_strided<<<(batch_size+255)/256, 256>>>(logits, targets, states, out_bufs, pitch_words, vocab_size, batch_size, logit_stride);
    checkCudaErrors(cudaGetLastError());
}

void gpu_rc_encode_step_batch_masked(const float* logits, const int* targets, const int* lengths, int t, RCState* states, unsigned int* out_bufs, int pitch_words, int vocab_size, int batch_size) {
    static bool stack_set = false;
    if (!stack_set) {
        checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
        stack_set = true;
    }
    ker_rc_encode_step_masked<<<(batch_size+255)/256, 256>>>(logits, targets, lengths, t, states, out_bufs, pitch_words, vocab_size, batch_size);
    checkCudaErrors(cudaGetLastError());
}

void gpu_rc_encode_chunk_batch_strided(const float* logits, const int* chunk_data, const int* lengths, RCState* states, unsigned int* out_bufs,
                                       int pitch_words, int vocab_size, int batch_size, int len, int max_len, int sub_start, int logit_stride) {
    static bool stack_set = false;
    if (!stack_set) {
        checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
        stack_set = true;
    }
    ker_rc_encode_chunk_strided<<<(batch_size+255)/256, 256>>>(logits, chunk_data, lengths, states, out_bufs, pitch_words, vocab_size, batch_size, len, max_len, sub_start, logit_stride);
    checkCudaErrors(cudaGetLastError());
}

void gpu_rc_encode_step_batch(const float* logits, const int* targets, RCState* states, unsigned int* out_bufs, int pitch_words, int vocab_size, int batch_size) {
    static bool stack_set = false;
    if (!stack_set) {
        checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
        stack_set = true;
    }
    ker_rc_encode_step<<<(batch_size+255)/256, 256>>>(logits, targets, states, out_bufs, pitch_words, vocab_size, batch_size);
    checkCudaErrors(cudaGetLastError());
}

// ==========================================
// Finalize Encoding
// ==========================================

__global__ void ker_rc_finish(RCState* states, unsigned int* out_words, int pitch_words, int* sizes_words, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch) {
        RCState &st = states[i];
        unsigned int* out = out_words + (size_t)i * (size_t)pitch_words;
        
        if (st.range == 0xFFFFFFFFFFFFFFFFull) { sizes_words[i] = 0; return; }
        
        unsigned long long point = st.lower + ((1ull << (64-32)) - 1ull);
        if (st.inverted_num > 0) {
            unsigned int first_word, subsequent;
            if (point < st.lower) { first_word = st.first_inv_lower_word + 1u; subsequent = 0u; }
            else { first_word = st.first_inv_lower_word; subsequent = 0xFFFFFFFFu; }
            int widx = st.write_idx_words;
            out[(size_t)widx++] = first_word;
            for (int k=1; k<st.inverted_num; k++) out[(size_t)widx++] = subsequent;
            st.write_idx_words = widx;
            st.inverted_num = 0;
        }
        unsigned int point_word = (unsigned int)(point >> (64-32));
        int widx = st.write_idx_words;
        out[(size_t)widx++] = point_word;
        unsigned long long upper = st.lower + st.range;
        unsigned int upper_word = (unsigned int)(upper >> (64-32));
        if (upper_word == point_word) out[(size_t)widx++] = 0u;
        
        st.write_idx_words = widx;
        sizes_words[i] = widx;
    }
}

void gpu_rc_finish_batch(RCState* states, unsigned int* out_bufs, int pitch_words, int* sizes_words, int batch_size) {
    ker_rc_finish<<<(batch_size+255)/256, 256, 0, g_stream>>>(states, out_bufs, pitch_words, sizes_words, batch_size);
    checkCudaErrors(cudaGetLastError());
}

// ==========================================
// Decoder Initialization
// ==========================================

__global__ void ker_rc_init_decoder(RCDecState* states, const unsigned int* in_words, const int* sizes_words, int pitch_words, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const unsigned int* in = in_words + (size_t)i * (size_t)pitch_words;
        RCDecState &st = states[i];
        st.lower = 0ull; st.range = 0xFFFFFFFFFFFFFFFFull; st.point = 0ull; st.read_idx_words = 0;
        for (int k=0; k<2; k++) {
            unsigned int w = (st.read_idx_words < sizes_words[i]) ? in[st.read_idx_words++] : 0u;
            st.point = (st.point << 32) | (unsigned long long)w;
        }
    }
}

void gpu_rc_init_decoder(RCDecState* states, const unsigned int* in_bufs, const int* sizes_words, int pitch_words, int batch_size) {
    ker_rc_init_decoder<<<(batch_size+255)/256, 256, 0, g_stream>>>(states, in_bufs, sizes_words, pitch_words, batch_size);
    checkCudaErrors(cudaGetLastError());
}

// ==========================================
// Warp-Parallel Decoder Step
// ==========================================

__global__ void ker_rc_decode_step_warp(const float* logits, const int* in_tokens, const int* lengths, int t, int* out_symbols, RCDecState* states, const unsigned int* in_words, const int* sizes_words, int pitch_words, int vocab_size, int batch) {
    int b = blockIdx.x;
    if (b >= batch) return;
    
    int lane = threadIdx.x;

    if (lengths && t >= lengths[b]) {
        if (lane == 0) out_symbols[b] = in_tokens ? in_tokens[b] : 0;
        return;
    }

    const float* l_ptr = logits + b * vocab_size;

    __shared__ volatile double s_cdf[260];

    // 1. Load & Max Reduction
    float my_vals[8];
    float my_max = -1e30f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = lane * 8 + i;
        float val = l_ptr[idx];
        my_vals[i] = val; 
        my_max = fmaxf(my_max, val);
    }
    float batch_max = warpReduceMax(my_max);
    batch_max = __shfl_sync(0xFFFFFFFF, batch_max, 0); 

    // 2. Exp & Sum (Double)
    double my_sum_d = 0.0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val = __expf((my_vals[i] - batch_max) * d_inv_temperature);
        my_vals[i] = val; 
        my_sum_d += (double)val;
    }
    double batch_sum = warpReduceSumDouble(my_sum_d);
    batch_sum = __shfl_sync(0xFFFFFFFF, batch_sum, 0); 
    
    double inv_sum = 1.0 / batch_sum;

    // 3. Normalize & Local CDF
    double thread_sum = my_sum_d * inv_sum;
    double warp_prefix = warpScanSumDouble(thread_sum); 
    double warp_start = warp_prefix - thread_sum; 

    double running = warp_start;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = lane * 8 + i;
        s_cdf[idx] = running;
        running += (double)my_vals[i] * inv_sum; 
    }
    if (lane == 31) s_cdf[256] = 1.0; 

    __syncwarp();

    // Enforce CDF monotonicity at warp lane boundaries (parallel fixup)
    if (lane > 0) {
        double prev_end = s_cdf[lane * 8 - 1];
        double my_start = s_cdf[lane * 8];
        if (my_start < prev_end) {
            double delta = prev_end - my_start;
            #pragma unroll
            for (int i = 0; i < 8; i++) s_cdf[lane * 8 + i] += delta;
        }
    }
    __syncwarp();

    // 4. Decode — Warp-parallel CDF search (all 32 lanes participate)
    {
        RCDecState st_local;
        unsigned int target;
        double scale_double;
        unsigned long long scale_rc;
        const unsigned int PRECISION_BITS = 24;
        const unsigned int TOTAL = (1u<<PRECISION_BITS);

        if (lane == 0) {
            st_local = states[b];
            scale_rc = st_local.range >> PRECISION_BITS;
            unsigned long long q = (st_local.point - st_local.lower) / scale_rc;
            if (q >= (1ull<<PRECISION_BITS)) q = (1ull<<PRECISION_BITS)-1ull;
            target = (unsigned int)q;
            scale_double = (double)(TOTAL - 256);
        }
        target = __shfl_sync(0xFFFFFFFF, target, 0);
        // Broadcast scale_double via its two 32-bit halves
        {
            unsigned long long sd_bits;
            if (lane == 0) sd_bits = __double_as_longlong(scale_double);
            unsigned int sd_lo = __shfl_sync(0xFFFFFFFF, (unsigned int)(sd_bits), 0);
            unsigned int sd_hi = __shfl_sync(0xFFFFFFFF, (unsigned int)(sd_bits >> 32), 0);
            sd_bits = ((unsigned long long)sd_hi << 32) | sd_lo;
            scale_double = __longlong_as_double(sd_bits);
        }

        int s = 0;
        unsigned int low_s = 0;
        unsigned int high_s = 0;

        for (int round = 0; round < 8; ++round) {
            int k = round * 32 + lane;
            double p_low = s_cdf[k];
            double p_high = (k == 255) ? 1.0 : s_cdf[k + 1];
            unsigned int l_int = (unsigned int)(p_low * scale_double) + k;
            unsigned int h_int = (unsigned int)(p_high * scale_double) + (k + 1);
            if (k == 255) h_int = TOTAL;
            bool match = (target >= l_int && target < h_int);
            unsigned int ballot = __ballot_sync(0xFFFFFFFF, match);
            if (ballot != 0) {
                int winner = __ffs(ballot) - 1;
                s = __shfl_sync(0xFFFFFFFF, k, winner);
                low_s = __shfl_sync(0xFFFFFFFF, l_int, winner);
                high_s = __shfl_sync(0xFFFFFFFF, h_int, winner);
                break;
            }
        }

        if (lane == 0) {
            out_symbols[b] = s;
            unsigned int prob = high_s - low_s;
            if (prob == 0) prob = 1;
            st_local.lower = st_local.lower + scale_rc * (unsigned long long)low_s;
            st_local.range = scale_rc * (unsigned long long)prob;
            const unsigned int* in = in_words + (size_t)b * (size_t)pitch_words;
            while (st_local.range < (1ull << (64-32))) {
                st_local.lower <<= 32;
                st_local.range <<= 32;
                unsigned int w = (st_local.read_idx_words < sizes_words[b]) ? in[st_local.read_idx_words++] : 0u;
                st_local.point = (st_local.point << 32) | (unsigned long long)w;
            }
            states[b] = st_local;
        }
    }
}

void gpu_rc_decode_step_batch(const float* logits, const int* in_tokens, const int* lengths, int t, int* out_symbols, RCDecState* states, const unsigned int* in_bufs, const int* sizes_words, int pitch_words, int vocab_size, int batch_size) {
    if (vocab_size != 256) {
        printf("Error: Warp Decode requires Vocab=256\n"); return;
    }
    ker_rc_decode_step_warp<<<batch_size, 32, 0, g_stream>>>(logits, in_tokens, lengths, t, out_symbols, states, in_bufs, sizes_words, pitch_words, vocab_size, batch_size);
    checkCudaErrors(cudaGetLastError());
}

// ==========================================
// Fused Decode Step
// ==========================================

__global__ void ker_rc_decode_fused_step_warp(const float* logits, const float* head_b2, int* d_tokens, 
                                             const int* lengths, int t, int* d_batch_output, int chunk_size, 
                                             RCDecState* states, const unsigned int* in_words, 
                                             const int* sizes_words, int pitch_words, int vocab_size, int batch_size) {
    int b = blockIdx.x;
    if (b >= batch_size) return;
    
    int lane = threadIdx.x;

    if (lengths && t >= lengths[b]) {
        return;
    }

    const float* l_ptr = logits + (size_t)b * vocab_size;

    __shared__ volatile double s_cdf[260];

    // 1. Load, Add Bias & Max Reduction
    float my_vals[8];
    float my_max = -1e30f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int col = lane * 8 + i;
        float val = l_ptr[col] + head_b2[col];
        my_vals[i] = val; 
        my_max = fmaxf(my_max, val);
    }
    float batch_max = warpReduceMax(my_max);
    batch_max = __shfl_sync(0xFFFFFFFF, batch_max, 0); 

    // 2. Exp & Sum (Double)
    double my_sum_d = 0.0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val = __expf((my_vals[i] - batch_max) * d_inv_temperature);
        my_vals[i] = val; 
        my_sum_d += (double)val;
    }
    double batch_sum = warpReduceSumDouble(my_sum_d);
    batch_sum = __shfl_sync(0xFFFFFFFF, batch_sum, 0); 
    
    double inv_sum = 1.0 / batch_sum;

    // 3. Normalize & Local CDF
    double thread_sum = my_sum_d * inv_sum;
    double warp_prefix = warpScanSumDouble(thread_sum); 
    double warp_start = warp_prefix - thread_sum; 

    double running = warp_start;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = lane * 8 + i;
        s_cdf[idx] = running;
        running += (double)my_vals[i] * inv_sum; 
    }
    if (lane == 31) s_cdf[256] = 1.0; 

    __syncwarp();

    // Enforce CDF monotonicity at warp lane boundaries (parallel fixup)
    if (lane > 0) {
        double prev_end = s_cdf[lane * 8 - 1];
        double my_start = s_cdf[lane * 8];
        if (my_start < prev_end) {
            double delta = prev_end - my_start;
            #pragma unroll
            for (int i = 0; i < 8; i++) s_cdf[lane * 8 + i] += delta;
        }
    }
    __syncwarp();

    // 4. Decode — Warp-parallel CDF search
    {
        RCDecState st_local;
        unsigned int target;
        double scale_double;
        unsigned long long scale_rc;
        const unsigned int PRECISION_BITS = 24;
        const unsigned int TOTAL = (1u<<PRECISION_BITS);

        if (lane == 0) {
            st_local = states[b];
            scale_rc = st_local.range >> PRECISION_BITS;
            unsigned long long q = (st_local.point - st_local.lower) / scale_rc;
            if (q >= (1ull<<PRECISION_BITS)) q = (1ull<<PRECISION_BITS)-1ull;
            target = (unsigned int)q;
            scale_double = (double)(TOTAL - 256);
        }
        target = __shfl_sync(0xFFFFFFFF, target, 0);
        {
            unsigned long long sd_bits;
            if (lane == 0) sd_bits = __double_as_longlong(scale_double);
            unsigned int sd_lo = __shfl_sync(0xFFFFFFFF, (unsigned int)(sd_bits), 0);
            unsigned int sd_hi = __shfl_sync(0xFFFFFFFF, (unsigned int)(sd_bits >> 32), 0);
            sd_bits = ((unsigned long long)sd_hi << 32) | sd_lo;
            scale_double = __longlong_as_double(sd_bits);
        }

        int s = 0;
        unsigned int low_s = 0;
        unsigned int high_s = 0;

        for (int round = 0; round < 8; ++round) {
            int k = round * 32 + lane;
            double p_low = s_cdf[k];
            double p_high = (k == 255) ? 1.0 : s_cdf[k + 1];
            unsigned int l_int = (unsigned int)(p_low * scale_double) + k;
            unsigned int h_int = (unsigned int)(p_high * scale_double) + (k + 1);
            if (k == 255) h_int = TOTAL;
            bool match = (target >= l_int && target < h_int);
            unsigned int ballot = __ballot_sync(0xFFFFFFFF, match);
            if (ballot != 0) {
                int winner = __ffs(ballot) - 1;
                s = __shfl_sync(0xFFFFFFFF, k, winner);
                low_s = __shfl_sync(0xFFFFFFFF, l_int, winner);
                high_s = __shfl_sync(0xFFFFFFFF, h_int, winner);
                break;
            }
        }

        if (lane == 0) {
            d_tokens[b] = s;
            d_batch_output[b * chunk_size + t] = s;
            unsigned int prob = high_s - low_s;
            if (prob == 0) prob = 1;
            st_local.lower = st_local.lower + scale_rc * (unsigned long long)low_s;
            st_local.range = scale_rc * (unsigned long long)prob;
            const unsigned int* in = in_words + (size_t)b * (size_t)pitch_words;
            while (st_local.range < (1ull << (64-32))) {
                st_local.lower <<= 32;
                st_local.range <<= 32;
                unsigned int w = (st_local.read_idx_words < sizes_words[b]) ? in[st_local.read_idx_words++] : 0u;
                st_local.point = (st_local.point << 32) | (unsigned long long)w;
            }
            states[b] = st_local;
        }
    }
}

void gpu_rc_decode_fused_step_batch(const float* logits, const float* head_b2, int* d_tokens, const int* lengths, int t, int* d_batch_output, int chunk_size, RCDecState* states, const unsigned int* in_bufs, const int* sizes_words, int pitch_words, int vocab_size, int batch_size) {
    if (vocab_size != 256) {
        printf("Error: Warp Decode requires Vocab=256\n"); return;
    }
    ker_rc_decode_fused_step_warp<<<batch_size, 32>>>(logits, head_b2, d_tokens, lengths, t, d_batch_output, chunk_size, states, in_bufs, sizes_words, pitch_words, vocab_size, batch_size);
    checkCudaErrors(cudaGetLastError());
}

// ==========================================
// Device-pointer t variant for CUDA Graph
// ==========================================

__global__ void ker_rc_decode_fused_step_warp_dptr(const float* logits, const float* head_b2, int* d_tokens, 
                                             const int* lengths, const int* d_t, int* d_batch_output, int chunk_size, 
                                             RCDecState* states, const unsigned int* in_words, 
                                             const int* sizes_words, int pitch_words, int vocab_size, int batch_size) {
    int b = blockIdx.x;
    if (b >= batch_size) return;
    
    int lane = threadIdx.x;
    int t = *d_t;

    if (lengths && t >= lengths[b]) {
        return;
    }

    const float* l_ptr = logits + (size_t)b * vocab_size;

    __shared__ volatile double s_cdf[260];

    // 1. Load, Add Bias & Max Reduction
    float my_vals[8];
    float my_max = -1e30f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int col = lane * 8 + i;
        float val = l_ptr[col] + head_b2[col];
        my_vals[i] = val; 
        my_max = fmaxf(my_max, val);
    }
    float batch_max = warpReduceMax(my_max);
    batch_max = __shfl_sync(0xFFFFFFFF, batch_max, 0); 

    // 2. Exp & Sum (Double)
    double my_sum_d = 0.0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val = __expf((my_vals[i] - batch_max) * d_inv_temperature);
        my_vals[i] = val; 
        my_sum_d += (double)val;
    }
    double batch_sum = warpReduceSumDouble(my_sum_d);
    batch_sum = __shfl_sync(0xFFFFFFFF, batch_sum, 0); 
    
    double inv_sum = 1.0 / batch_sum;

    // 3. Normalize & Local CDF
    double thread_sum = my_sum_d * inv_sum;
    double warp_prefix = warpScanSumDouble(thread_sum); 
    double warp_start = warp_prefix - thread_sum; 

    double running = warp_start;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = lane * 8 + i;
        s_cdf[idx] = running;
        running += (double)my_vals[i] * inv_sum; 
    }
    if (lane == 31) s_cdf[256] = 1.0; 

    __syncwarp();

    // Enforce CDF monotonicity at warp lane boundaries (parallel fixup)
    if (lane > 0) {
        double prev_end = s_cdf[lane * 8 - 1];
        double my_start = s_cdf[lane * 8];
        if (my_start < prev_end) {
            double delta = prev_end - my_start;
            #pragma unroll
            for (int i = 0; i < 8; i++) s_cdf[lane * 8 + i] += delta;
        }
    }
    __syncwarp();

    // 4. Decode — Warp-parallel CDF search
    {
        RCDecState st_local;
        unsigned int target;
        double scale_double;
        unsigned long long scale_rc;
        const unsigned int PRECISION_BITS = 24;
        const unsigned int TOTAL = (1u<<PRECISION_BITS);

        if (lane == 0) {
            st_local = states[b];
            scale_rc = st_local.range >> PRECISION_BITS;
            unsigned long long q = (st_local.point - st_local.lower) / scale_rc;
            if (q >= (1ull<<PRECISION_BITS)) q = (1ull<<PRECISION_BITS)-1ull;
            target = (unsigned int)q;
            scale_double = (double)(TOTAL - 256);
        }
        target = __shfl_sync(0xFFFFFFFF, target, 0);
        {
            unsigned long long sd_bits;
            if (lane == 0) sd_bits = __double_as_longlong(scale_double);
            unsigned int sd_lo = __shfl_sync(0xFFFFFFFF, (unsigned int)(sd_bits), 0);
            unsigned int sd_hi = __shfl_sync(0xFFFFFFFF, (unsigned int)(sd_bits >> 32), 0);
            sd_bits = ((unsigned long long)sd_hi << 32) | sd_lo;
            scale_double = __longlong_as_double(sd_bits);
        }

        int s = 0;
        unsigned int low_s = 0;
        unsigned int high_s = 0;

        for (int round = 0; round < 8; ++round) {
            int k = round * 32 + lane;
            double p_low = s_cdf[k];
            double p_high = (k == 255) ? 1.0 : s_cdf[k + 1];
            unsigned int l_int = (unsigned int)(p_low * scale_double) + k;
            unsigned int h_int = (unsigned int)(p_high * scale_double) + (k + 1);
            if (k == 255) h_int = TOTAL;
            bool match = (target >= l_int && target < h_int);
            unsigned int ballot = __ballot_sync(0xFFFFFFFF, match);
            if (ballot != 0) {
                int winner = __ffs(ballot) - 1;
                s = __shfl_sync(0xFFFFFFFF, k, winner);
                low_s = __shfl_sync(0xFFFFFFFF, l_int, winner);
                high_s = __shfl_sync(0xFFFFFFFF, h_int, winner);
                break;
            }
        }

        if (lane == 0) {
            d_tokens[b] = s;
            d_batch_output[b * chunk_size + t] = s;
            unsigned int prob = high_s - low_s;
            if (prob == 0) prob = 1;
            st_local.lower = st_local.lower + scale_rc * (unsigned long long)low_s;
            st_local.range = scale_rc * (unsigned long long)prob;
            const unsigned int* in = in_words + (size_t)b * (size_t)pitch_words;
            while (st_local.range < (1ull << (64-32))) {
                st_local.lower <<= 32;
                st_local.range <<= 32;
                unsigned int w = (st_local.read_idx_words < sizes_words[b]) ? in[st_local.read_idx_words++] : 0u;
                st_local.point = (st_local.point << 32) | (unsigned long long)w;
            }
            states[b] = st_local;
        }
    }
}

void gpu_rc_decode_fused_step_batch_graph(const float* logits, const float* head_b2, int* d_tokens, const int* lengths, const int* d_t, int* d_batch_output, int chunk_size, RCDecState* states, const unsigned int* in_bufs, const int* sizes_words, int pitch_words, int vocab_size, int batch_size) {
    if (vocab_size != 256) {
        printf("Error: Warp Decode requires Vocab=256\n"); return;
    }
    ker_rc_decode_fused_step_warp_dptr<<<batch_size, 32, 0, g_stream>>>(logits, head_b2, d_tokens, lengths, d_t, d_batch_output, chunk_size, states, in_bufs, sizes_words, pitch_words, vocab_size, batch_size);
}

// Tiny kernel to increment a device-side counter
__global__ void ker_increment_counter(int* counter) {
    *counter += 1;
}

void gpu_increment_counter(int* d_counter) {
    ker_increment_counter<<<1, 1, 0, g_stream>>>(d_counter);
}

// ==========================================
// Token Selection Helpers
// ==========================================

__global__ void ker_select_tokens(const int* d_chunk_data, int* d_tokens, int t, int chunk_size, int batch_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        d_tokens[b] = d_chunk_data[b * chunk_size + t];
    }
}

__global__ void ker_store_tokens(const int* d_out_symbols, int* d_chunk_data, int t, int chunk_size, int batch_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        d_chunk_data[b * chunk_size + t] = d_out_symbols[b];
    }
}

void gpu_select_tokens(const int* d_chunk_data, int* d_tokens, int t, int chunk_size, int batch_size) {
    ker_select_tokens<<<(batch_size + 255)/256, 256, 0, g_stream>>>(d_chunk_data, d_tokens, t, chunk_size, batch_size);
    checkCudaErrors(cudaGetLastError());
}

void gpu_store_tokens(const int* d_out_symbols, int* d_chunk_data, int t, int chunk_size, int batch_size) {
    ker_store_tokens<<<(batch_size + 255)/256, 256, 0, g_stream>>>(d_out_symbols, d_chunk_data, t, chunk_size, batch_size);
    checkCudaErrors(cudaGetLastError());
}

// ==========================================
// Warp-Parallel Encoding
// ==========================================

__global__ void ker_rc_encode_chunk_parallel_warp(const float* logits_base, const int* tokens_base, const int* lengths, 
                                                  RCState* states, unsigned int* out_bufs, 
                                                  int pitch_words, int vocab_size, int batch_size, 
                                                  int chunk_len, int max_len, int start_t, int logit_stride) {
    int b = blockIdx.x;
    if (b >= batch_size) return;

    int lane = threadIdx.x;

    RCState st;
    if (lane == 0) st = states[b];

    int len = lengths[b];
    int valid_len = (len > max_len) ? max_len : len;

    const float* my_logits = logits_base + (size_t)b * logit_stride;
    const int* my_tokens = tokens_base + (size_t)b * chunk_len;
    
    unsigned int* my_out = out_bufs + (size_t)b * pitch_words;

    __shared__ volatile double s_cdf[260]; 

    int loop_len = valid_len - 1; 

    for (int t = start_t; t < loop_len; ++t) {
        int token = my_tokens[t+1];
        const float* l_ptr = my_logits + (size_t)(t - start_t) * vocab_size;

        // 1. Load & Max Reduction
        float my_vals[8];
        float my_max = -1e30f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = lane * 8 + i;
            float val = l_ptr[idx];
            my_vals[i] = val;
            my_max = fmaxf(my_max, val);
        }
        float batch_max = warpReduceMax(my_max);
        batch_max = __shfl_sync(0xFFFFFFFF, batch_max, 0);

        // 2. Exp & Sum (Double)
        double my_sum_d = 0.0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float val = __expf((my_vals[i] - batch_max) * d_inv_temperature); 
            my_vals[i] = val; 
            my_sum_d += (double)val;
        }
        double batch_sum = warpReduceSumDouble(my_sum_d);
        batch_sum = __shfl_sync(0xFFFFFFFF, batch_sum, 0); 
        
        double inv_sum = 1.0 / batch_sum;

        // 3. Normalize & Local CDF (Double)
        double thread_sum = my_sum_d * inv_sum;
        double warp_prefix = warpScanSumDouble(thread_sum); 
        double warp_start = warp_prefix - thread_sum; 

        double running = warp_start;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = lane * 8 + i;
            s_cdf[idx] = running;
            running += (double)my_vals[i] * inv_sum; 
        }
        if (lane == 31) s_cdf[256] = 1.0; 

        __syncwarp();

        // Enforce CDF monotonicity at warp lane boundaries (parallel fixup)
        if (lane > 0) {
            double prev_end = s_cdf[lane * 8 - 1];
            double my_start = s_cdf[lane * 8];
            if (my_start < prev_end) {
                double delta = prev_end - my_start;
                #pragma unroll
                for (int i = 0; i < 8; i++) s_cdf[lane * 8 + i] += delta;
            }
        }
        __syncwarp();

        // 4. Encode (Lane 0 only)
        if (lane == 0) {
            double prob_low = s_cdf[token];
            double prob_high = (token == 255) ? 1.0 : s_cdf[token+1];

            const unsigned int TOTAL = (1u<<24); 
            double scale_double = (double)(TOTAL - 256);
            
            unsigned int low_int = (unsigned int)(prob_low * scale_double) + token;
            unsigned int high_int = (unsigned int)(prob_high * scale_double) + (token + 1);
            if (token == 255) high_int = TOTAL;
            
            unsigned int left = low_int;
            unsigned int prob = high_int - low_int;
            
            if (prob == 0) {
                 prob = 1;
            }
            
            unsigned long long scale_rc = st.range >> 24; 
            unsigned long long old_lower = st.lower;
            
            st.range = scale_rc * (unsigned long long)prob;
            st.lower = st.lower + scale_rc * (unsigned long long)left;
            
            if (st.inverted_num > 0) {
                 unsigned long long sum = st.lower + st.range;
                 if (sum > st.lower) {
                     unsigned int first_word = (st.lower < old_lower) ? st.first_inv_lower_word + 1u : st.first_inv_lower_word;
                     unsigned int subsequent = (st.lower < old_lower) ? 0u : 0xFFFFFFFFu;
                     my_out[st.write_idx_words++] = first_word;
                     for (int k=1; k<st.inverted_num; k++) my_out[st.write_idx_words++] = subsequent;
                     st.inverted_num = 0;
                 }
            }
            while (st.range < (1ull << (64-32))) {
                unsigned int lower_word = (unsigned int)(st.lower >> (64-32));
                st.lower <<= 32; st.range <<= 32;
                if (st.inverted_num > 0) {
                     st.inverted_num++;
                } else {
                     unsigned long long sum = st.lower + st.range;
                     if (sum > st.lower) {
                         my_out[st.write_idx_words++] = lower_word;
                     } else {
                         st.inverted_num = 1;
                         st.first_inv_lower_word = lower_word;
                     }
                }
            }
        }
    }
    
    if (lane == 0) states[b] = st;
}

void gpu_rc_encode_chunk_warp(const float* logits, const int* chunk_data, const int* lengths, 
                             RCState* states, unsigned int* out_bufs,
                             int pitch_words, int vocab_size, int batch_size, 
                             int chunk_len, int max_len, int start_t, int logit_stride) {
    if (vocab_size != 256) {
        printf("Error: Warp Encode requires Vocab=256\n"); return;
    }
    ker_rc_encode_chunk_parallel_warp<<<batch_size, 32, 0, g_stream>>>(logits, chunk_data, lengths, states, out_bufs, pitch_words, vocab_size, batch_size, chunk_len, max_len, start_t, logit_stride);
    checkCudaErrors(cudaGetLastError());
}
