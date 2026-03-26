#pragma once

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE 
#endif

#include <cmath>

// =================================================================================
// REPRODUCIBLE MATH FUNCTIONS
// =================================================================================
// We use software-defined approximations composed of basic arithmetic operations
// to guarantee bit-exact results across CPU (x86/ARM) and GPU (CUDA).
// Standard std::exp or expf use hardware instructions which differ.

// Constants
#define R_LOG2E 1.44269504088896340736f
#define R_LN2   0.69314718055994530942f

HOST_DEVICE inline float repro_abs(float x) {
    return x < 0.0f ? -x : x;
}

// A simple polynomial approximation for exp(x) on [-87, 87]
// Using a 7th deg Taylor/Horner series centered at 0 is good for small x,
// but for larger x we need range reduction.
// e^x = 2^(x * log2(e)) = 2^k * e^r
// Implementation adapted for strictly standard C arithmetic.
// GPU version uses __fadd_rn/__fmul_rn intrinsics to guarantee bit-exact
// results regardless of which kernel this is inlined into.
HOST_DEVICE inline float repro_exp(float x) {
    #if GPU_FAST_EXP && defined(__CUDA_ARCH__)
    return __expf(x);
    #elif defined(__CUDA_ARCH__)
    // GPU path: use intrinsics for bit-exact determinism across kernels
    if (x <= -88.0f) return 0.0f;
    if (x >= 88.0f) x = 88.0f;

    // Range reduction: k = round(x / ln(2))
    float k_float = floorf(__fadd_rn(__fmul_rn(x, R_LOG2E), 0.5f));
    int k = (int)k_float;
    float r = __fadd_rn(x, -__fmul_rn(k_float, R_LN2));

    // Horner evaluation of Taylor series for e^r using strict intrinsics
    // e^r ≈ 1 + r(1 + r(0.5 + r(1/6 + r(1/24 + r(1/120 + r/720)))))
    float t = __fadd_rn(0.008333333333f, __fmul_rn(r, 0.001388888889f));
    t = __fadd_rn(0.041666666667f, __fmul_rn(r, t));
    t = __fadd_rn(0.166666666667f, __fmul_rn(r, t));
    t = __fadd_rn(0.5f, __fmul_rn(r, t));
    t = __fadd_rn(1.0f, __fmul_rn(r, t));
    t = __fadd_rn(1.0f, __fmul_rn(r, t));

    // Reconstruct 2^k * t via IEEE754 bit manipulation
    union { float f; int i; } u;
    u.i = (k + 127) << 23;
    return __fmul_rn(t, u.f);
    #else
    // CPU path: standard arithmetic
    if (x <= -88.0f) return 0.0f;
    if (x >= 88.0f) x = 88.0f;

    float k_float = floorf(x * R_LOG2E + 0.5f);
    int k = (int)k_float;
    float r = x - k_float * R_LN2;

    float t = 1.0f + r * (
        1.0f + r * (
            0.5f + r * (
                0.166666666667f + r * (
                    0.041666666667f + r * (
                        0.008333333333f + r * 0.001388888889f
                    )
                )
            )
        )
    );

    union { float f; int i; } u;
    u.i = (k + 127) << 23; 
    return t * u.f;
    #endif
}

// Tanh
HOST_DEVICE inline float repro_tanh(float x) {
    float abs_x = repro_abs(x);
    if (abs_x > 10.0f) return (x > 0.0f) ? 1.0f : -1.0f;

    #ifdef __CUDA_ARCH__
    float e2x = repro_exp(__fmul_rn(2.0f, x));
    return __fadd_rn(e2x, -1.0f) / __fadd_rn(e2x, 1.0f);
    #else
    float e2x = repro_exp(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
    #endif
}

// GeLU
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#define SQRT_2_OVER_PI 0.79788456080286535588f
#define GELU_COEFF     0.044715f

HOST_DEVICE inline float repro_gelu(float x) {
    #ifdef __CUDA_ARCH__
    float cube = __fmul_rn(__fmul_rn(x, x), x);
    float inner = __fmul_rn(SQRT_2_OVER_PI, __fadd_rn(x, __fmul_rn(GELU_COEFF, cube)));
    float tanh_inner = repro_tanh(inner);
    return __fmul_rn(0.5f, __fmul_rn(x, __fadd_rn(1.0f, tanh_inner)));
    #else
    float cube = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * cube);
    float tanh_inner = repro_tanh(inner);
    return 0.5f * x * (1.0f + tanh_inner);
    #endif
}

// Sigmoid
HOST_DEVICE inline float repro_sigmoid(float x) {
    float ex = repro_exp(-x);
    #ifdef __CUDA_ARCH__
    return 1.0f / __fadd_rn(1.0f, ex);
    #else
    return 1.0f / (1.0f + ex);
    #endif
}

// SiLU (Swish)
HOST_DEVICE inline float repro_silu(float x) {
    #ifdef __CUDA_ARCH__
    return __fmul_rn(x, repro_sigmoid(x));
    #else
    return x * repro_sigmoid(x);
    #endif
}
