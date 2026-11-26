# Throughput summary

6 model(s): Llama-3.2-1B, Llama-3.2-3B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen3.5-2B-Base, Qwen3.5-4B-Base. Per-row detail is in throughput_{model}.md.

## 1. Inference, FitLN vs. FullFT

| Model | FitLN (samples/s) | FullFT (samples/s) | ratio |
|---|---|---|---|
| Llama-3.2-1B | 586.7 | 585.4 | 1.002x |
| Llama-3.2-3B | 231.5 | 231.2 | 1.001x |
| Qwen2.5-1.5B | 422.3 | 421.9 | 1.001x |
| Qwen2.5-3B | 241.9 | 241.8 | 1.001x |
| Qwen3.5-2B-Base | 250.9 | 249.9 | 1.004x |
| Qwen3.5-4B-Base | 148.6 | 148.7 | 0.999x |

## 2. LoRA inference, unmerged and merged

| Model | LoRA-r1 unmerged | LoRA-r1 merged | overhead removed | merged vs FitLN | LoRA-r8 unmerged | LoRA-r8 merged | overhead removed | merged vs FitLN |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 471.5 | 584.8 | +24.0% | 0.997x | 434.2 | 585.0 | +34.7% | 0.997x |
| Llama-3.2-3B | 188.4 | 233.4 | +23.9% | 1.008x | 174.3 | 234.2 | +34.3% | 1.011x |
| Qwen2.5-1.5B | 334.6 | 422.3 | +26.2% | 1.000x | 294.7 | 423.2 | +43.6% | 1.002x |
| Qwen2.5-3B | 200.3 | 241.9 | +20.7% | 1.000x | 189.4 | 241.9 | +27.7% | 1.000x |
| Qwen3.5-2B-Base | 199.4 | 252.4 | +26.6% | 1.006x | 195.1 | 225.5 | +15.6% | 0.899x |
| Qwen3.5-4B-Base | 116.4 | 148.8 | +27.9% | 1.001x | 110.7 | 148.7 | +34.3% | 1.001x |

## 3. Training step (forward, backward, optimizer step)

| Model | FitLN (samples/s) | FullFT (samples/s) | FitLN speedup | LoRA-r1 (samples/s) | LoRA-r8 (samples/s) | FitLN / LoRA-r1 | FitLN / LoRA-r8 | LoRA-r1 / FullFT | LoRA-r8 / FullFT |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 229.0 | 124.7 | 1.84x | 181.0 | 165.7 | 1.27x | 1.38x | 1.45x | 1.33x |
| Llama-3.2-3B | 100.9 | 52.6 | 1.92x | 78.0 | 71.7 | 1.29x | 1.41x | 1.48x | 1.36x |
| Qwen2.5-1.5B | 163.3 | 92.3 | 1.77x | 128.0 | 115.8 | 1.28x | 1.41x | 1.39x | 1.25x |
| Qwen2.5-3B | 101.2 | 52.9 | 1.91x | 80.6 | 75.1 | 1.26x | 1.35x | 1.52x | 1.42x |
| Qwen3.5-2B-Base | 99.6 | 61.0 | 1.63x | 79.8 | 72.1 | 1.25x | 1.38x | 1.31x | 1.18x |
| Qwen3.5-4B-Base | 57.0 | 33.0 | 1.72x | 44.8 | 42.3 | 1.27x | 1.35x | 1.36x | 1.28x |

Range across models (min-max):

| Comparison | Range |
|---|---|
| FitLN vs. FullFT | 1.632-1.919x |
| FitLN vs. LoRA-r8 | 1.347-1.410x |
| FitLN vs. LoRA-r1 | 1.247-1.294x |
| LoRA-r1 vs. FullFT | 1.309-1.523x |
| LoRA-r8 vs. FullFT | 1.182-1.419x |

Setting: one GPU, single process, batch size 8, sequence length 256, fp16 for every method. Training-step throughput, not end-to-end training time.

