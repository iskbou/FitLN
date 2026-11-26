## Peak GPU training memory by method and model

Source: `global_summary.md`'s "Model / method characteristics" table (seed set as selected when summarize_results.py was run; default 1, 2, 42).

| Model | FitLN (MB) | LoRA (1) (MB) | LoRA (8) (MB) | FFT (MB) | FFT / FitLN | LoRA(8) / FitLN |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B | 4601 ± 1047 | 4550 ± 1045 | 4567 ± 1045 | 10518 ± 4 | 2.29× | 0.99× |
| Llama-3.2-3B | 8523 ± 1121 | 8473 ± 1117 | 8520 ± 1116 | 26104 ± 13 | 3.06× | 1.00× |
| Qwen2.5-1.5B | 5631 ± 1229 | 5581 ± 1229 | 5603 ± 1229 | 12767 ± 16 | 2.27× | 0.99× |
| Qwen2.5-3B | 8766 ± 1275 | 8719 ± 1270 | 8756 ± 1270 | 25156 ± 48 | 2.87× | 1.00× |
| Qwen3.5-2B-Base | 12565 ± 2023 | 12522 ± 2041 | 12559 ± 2041 | 16636 ± 258 | 1.32× | 1.00× |
| Qwen3.5-4B-Base | 21872 ± 2130 | 21820 ± 2123 | 21892 ± 2123 | 34891 ± 51 | 1.60× | 1.00× |
