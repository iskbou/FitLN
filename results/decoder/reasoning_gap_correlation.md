## Reasoning-gap correlation

| Model | FFT GSM8K | GSM8K retention | SAMSum retention | Alpaca retention |
| --- | --- | --- | --- | --- |
| Llama-3.2-1B | 0.2370 | 0.2932 | 0.8866 | 0.8929 |
| Llama-3.2-3B | 0.4210 | 0.7126 | 0.9396 | 0.9455 |
| Qwen2.5-1.5B | 0.5757 | 0.9776 | 0.8742 | 0.9321 |
| Qwen2.5-3B | 0.6790 | 0.9982 | 0.8745 | 0.9420 |
| Qwen3.5-2B-Base | 0.6315 | 0.9061 | 0.9692 | 0.9723 |
| Qwen3.5-4B-Base | 0.7940 | 0.9921 | 0.9955 | 0.9846 |

| Task | Spearman ρ (n=6) | p-value (informal -- see script docstring) |
| --- | --- | --- |
| gsm8k | 0.886 | 0.019 |
| samsum | 0.371 | 0.468 |
| alpaca_mini | 0.714 | 0.111 |

### Sensitivity check: leave-Llama-3.2-1B-out (n=5)

Llama-3.2-1B is the weakest backbone by FFT GSM8K score, so it sits at the low extreme of every panel simultaneously -- removing it checks whether it alone is driving the n=6 correlations.

| Task | Spearman ρ (n=5, leave-one-out) |
| --- | --- |
| gsm8k | 0.800 |
| samsum | 0.500 |
| alpaca_mini | 0.500 |
