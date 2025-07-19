# GGUF Model Integration for PrefEval

This integration allows you to run PrefEval benchmarks with local GGUF models using llama-cpp-python instead of relying on Amazon Bedrock.

## Setup

### 1. Install Dependencies
```bash
pip install llama-cpp-python
```

### 2. Download GGUF Models
Download GGUF models from sources like HuggingFace. Popular options:
- [Llama 3.2 3B Instruct](https://huggingface.co/hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF)
- [Mistral 7B Instruct](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)

### 3. Configure Models
Update `config.yaml` with your model paths:

```yaml
gguf_models:
  my-llama-model:
    path: "/path/to/llama-3.2-3b-instruct.Q4_K_M.gguf"
    config:
      n_ctx: 4096        # Context window size
      n_threads: 8       # CPU threads to use
      n_gpu_layers: 0    # GPU layers (set > 0 for GPU acceleration)
      temperature: 0.0   # Generation temperature
      verbose: false     # Enable verbose output
```

## Usage

Use GGUF models with the same commands as other models:

```bash
# Explicit preference benchmark
python generation_task/benchmark_generation.py \
    --model=my-llama-model \
    --task=zero-shot \
    --topic=travel_restaurant \
    --inter_turns=2 \
    --pref_form=explicit

# Implicit preference with reminder
python generation_task/benchmark_generation.py \
    --model=my-mistral-model \
    --task=remind \
    --topic=lifestyle_dietary \
    --inter_turns=3 \
    --pref_form=implicit \
    --pref_type=persona
```

## Performance Optimization

### CPU Optimization
- Increase `n_threads` to match your CPU cores
- Use quantized models (Q4_K_M, Q5_K_M) for better performance
- Set `use_mmap: true` and `use_mlock: false` for memory efficiency

### GPU Acceleration
- Install llama-cpp-python with GPU support:
  ```bash
  # For CUDA
  CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
  
  # For Metal (macOS)
  CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
  ```
- Set `n_gpu_layers` > 0 (try 20-40 layers to start)
- Monitor GPU memory usage and adjust accordingly

## Model Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_ctx` | Context window size | 4096 |
| `n_threads` | CPU threads | Auto-detect |
| `n_gpu_layers` | GPU layers to offload | 0 |
| `temperature` | Generation randomness | 0.0 |
| `verbose` | Enable debug output | false |
| `seed` | Random seed | -1 (random) |
| `f16_kv` | Use 16-bit for key/value cache | true |
| `use_mmap` | Memory mapping | true |
| `use_mlock` | Lock memory pages | false |

## Example Script

Run the provided example script:
```bash
cd example_scripts
./run_gguf_example.sh
```

## Benefits of GGUF Integration

- **Cost-effective**: No cloud API costs
- **Privacy**: Models run entirely locally
- **Offline capability**: No internet required after setup
- **Custom models**: Use any GGUF-compatible model
- **Performance control**: Full control over hardware utilization

## Troubleshooting

### Memory Issues
- Use smaller quantized models (Q4_K_M instead of Q8_0)
- Reduce `n_ctx` (context window size)
- Enable memory mapping with `use_mmap: true`

### Slow Performance
- Increase `n_threads` for CPU inference
- Enable GPU acceleration with `n_gpu_layers`
- Use faster quantization formats (Q4_K_M is good balance)

### Model Loading Errors
- Verify GGUF file path is correct
- Check file permissions
- Ensure enough system memory
- Try a different quantization level