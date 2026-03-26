# Boa Constrictor C++/CUDA Implementation

## Compilation

### Ubuntu / Linux
```bash
nvcc -o boa_constrictor boa.cpp gemm_kernels.cu inference_kernels.cu mamba_kernels.cu rnn_kernels.cu range_coder_kernels.cu utility_kernels.cu -I src -O3 -std=c++17 --fmad=false -DENABLE_GPU -DGPU_DEBUG_LOGITS=0 -DGPU_FAST_EXP=0
```

### Windows
```powershell
nvcc -o boa_constrictor.exe boa.cpp gemm_kernels.cu inference_kernels.cu mamba_kernels.cu rnn_kernels.cu range_coder_kernels.cu utility_kernels.cu -I src -O3 -std=c++17 --fmad=false -DENABLE_GPU -DGPU_DEBUG_LOGITS=0 -DGPU_FAST_EXP=0
```

## Model Weights Conversion

Before running the C++ implementation, you must convert the PyTorch model weights (`.pt`) to the binary format (`.bin`) expected by the C++ loader.

```bash
python convert_boa_weights.py --model path/to/model.pt --output model.bin
```

The backbone type is auto-detected from the model's state dict keys. You can also specify it explicitly:
```bash
python convert_boa_weights.py --model path/to/lstm_model.pt --output model.bin --backbone lstm
```

Supported backbones: `mamba`, `mambav1`, `lstm`, `gru`, `mingru`

## How to Run

Basic usage format:
```
./boa_constrictor <mode> <model> <input> <output> [d_model] [n_layers] [--backbone TYPE] [--gpu-batch B] [--max-chunks C] [--chunk-size S]
```

### Arguments
- `mode`: `compress` or `decompress`
- `model`: Path to model weights
- `input`: Input file path
- `output`: Output file path
- `d_model`: Model dimension (optional, default: 256)
- `n_layers`: Number of layers (optional, default: 1)
- `--backbone`: Backbone architecture: `mamba` (default), `lstm`, `gru`, `mingru`
- `--gpu-batch`: Batch size for GPU processing
- `--chunk-size`: Size of chunks for processing
- `--max-chunks`: Maximum number of chunks to process (for testing)

### Examples

**Compression:**
```bash
./boa_constrictor compress cms_model.bin CMS_DATA_float32.bin cmstest.boa --gpu-batch 375 --chunk-size 4096
```

**Decompression:**
```bash
./boa_constrictor decompress cms_model.bin cmstest.boa test.bin --gpu-batch 4096 --chunk-size 256
```

## Optimizations

This implementation includes several key optimizations to achieve high throughput while maintaining reproducibility:

1.  **Custom Reproducible GEMM Kernels**: 1:1 consistent matrix multiplication ensuring deterministic results across platforms.
2.  **Fused Kernels**: Combining multiple operations (e.g., activation functions, scaling) to reduce global memory accesses.
3.  **GPU Batched Processing**: Parallel processing of multiple chunks to maximize GPU utilization (`--gpu-batch` support).
4.  **GPU Range Coder**: Specialized kernels for entropy coding directly on the GPU, avoiding CPU bottlenecks.
5.  **Memory Optimization**: Efficient memory access patterns and reuse to minimize latency.
6.  **Warp-Level Primitives**: Utilization of warp shuffles and other low-level primitives for high-performance reductions and computations.

## Performance Results

Tested on **NVIDIA RTX 5090** with `cms_experiment`:
- **Compression Speed**: ~7 MB/s
- **Decompression Speed**: ~5 MB/s

Portability confirmed on another CUDA-enabled GPU (RTX 3060 Laptop Edition)
