# Boa Constrictor C++/CUDA Implementation

## Compilation

Both Linux and Windows use a single `nvcc` command. The only difference is the output filename and, for energy measurement, one extra linker flag on Linux.

### Standard Build (no energy measurement)

**Linux:**
```bash
nvcc -o boa_constrictor boa.cpp gemm_kernels.cu inference_kernels.cu mamba_kernels.cu rnn_kernels.cu range_coder_kernels.cu utility_kernels.cu -O3 -std=c++17 --fmad=false -DENABLE_GPU -DGPU_DEBUG_LOGITS=0 -DGPU_FAST_EXP=0
```

**Windows (from a Developer Command Prompt with CUDA on PATH):**
```powershell
nvcc -o boa_constrictor.exe boa.cpp gemm_kernels.cu inference_kernels.cu mamba_kernels.cu rnn_kernels.cu range_coder_kernels.cu utility_kernels.cu -O3 -std=c++17 --fmad=false -DENABLE_GPU -DGPU_DEBUG_LOGITS=0 -DGPU_FAST_EXP=0
```

### Build with Energy Measurement (`-DENABLE_ENERGY`)

Adding `-DENABLE_ENERGY` compiles in [CPPJoules](https://github.com/rishalab/CPPJoules) support for CPU (Intel RAPL) and GPU (NVML) energy measurement. The CPPJoules sources are vendored under `CPPJoules/`.

**Linux:**
```bash
nvcc -o boa_constrictor boa.cpp \
  CPPJoules/src/cppJoules.cpp CPPJoules/src/energy_state.cpp \
  CPPJoules/src/nvidia_devices.cpp CPPJoules/src/rapl_devices.cpp \
  gemm_kernels.cu inference_kernels.cu mamba_kernels.cu rnn_kernels.cu \
  range_coder_kernels.cu utility_kernels.cu \
  -O3 -std=c++17 --fmad=false -DENABLE_GPU -DGPU_DEBUG_LOGITS=0 -DGPU_FAST_EXP=0 \
  -DENABLE_ENERGY -ICPPJoules/include -ICPPJoules/src -ldl
```

**Windows:**
```powershell
nvcc -o boa_constrictor.exe boa.cpp ^
  CPPJoules/src/cppJoules.cpp CPPJoules/src/energy_state.cpp ^
  CPPJoules/src/nvidia_devices.cpp CPPJoules/src/rapl_devices.cpp ^
  gemm_kernels.cu inference_kernels.cu mamba_kernels.cu rnn_kernels.cu ^
  range_coder_kernels.cu utility_kernels.cu ^
  -O3 -std=c++17 --fmad=false -DENABLE_GPU -DGPU_DEBUG_LOGITS=0 -DGPU_FAST_EXP=0 ^
  -DENABLE_ENERGY -ICPPJoules/include -ICPPJoules/src
```

> **Note:** Linux requires `-ldl` for dynamic loading of NVML. Windows does not need it.

#### Energy Measurement Prerequisites

| Platform | CPU (RAPL) | GPU (NVML) |
|----------|-----------|------------|
| **Linux** | Requires `/sys/class/powercap/intel-rapl/` (most bare-metal installs; not available in WSL2) | Requires NVIDIA drivers with NVML |
| **Windows** | Requires [Intel Power Gadget](https://www.intel.com/content/www/us/en/developer/articles/tool/power-gadget.html) (`EnergyLib64.dll`) | Requires NVIDIA drivers with NVML |

On Windows, the `EnergyLib64.dll` is searched in: the `ENERGYLIB64_PATH` environment variable, `System32`, `Program Files\Intel\Power Gadget 3.6\`, `Program Files\Intel\Power Gadget\`, and the current directory. If RAPL is unavailable, only GPU energy is reported.

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
./boa_constrictor <mode> <model> <input> <output> [d_model] [n_layers] [--backbone TYPE] [--gpu-batch B] [--max-chunks C] [--chunk-size S] [--tile T] [--warm-start] [--temperature T] [--measure-energy]
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
- `--tile`: Sub-chunk forward pass into tiles of T tokens (reduces VRAM, allows larger batches)
- `--warm-start`: Interleaved chunk order for model state carry-forward (compress only)
- `--temperature`: Scale logits by 1/T before softmax (default: 1.0)
- `--measure-energy`: Measure CPU/GPU energy via CPPJoules (requires `-DENABLE_ENERGY` build)

### Examples

**Compression:**
```bash
./boa_constrictor compress cms_model.bin CMS_DATA_float32.bin cmstest.boa --gpu-batch 375 --chunk-size 4096
```

**Decompression:**
```bash
./boa_constrictor decompress cms_model.bin cmstest.boa test.bin --gpu-batch 4096 --chunk-size 256
```

**With energy measurement** (requires `-DENABLE_ENERGY` build):
```bash
./boa_constrictor compress model.bin input.bin output.boa --gpu-batch 1400 --measure-energy
```
Energy results are printed to stdout and saved as `<output>.compress_energy.csv` / `<output>.decompress_energy.csv`.

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
