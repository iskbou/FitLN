## sst2 -- qwen3_5_2b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_2b_base | fitln | 0.0013 | 3 | 51200 | 0.00272 | 0.9545 ± 0.0054 | 208.9 | 9894.3 ± 0.3 |
| qwen3_5_2b_base | fullft | 1e-05 | 2 | 1881825088 | 100.00000 | 0.9501 ± 0.0008 | N/A (cluster only) | 16377.8 ± 5.4 |
| qwen3_5_2b_base | lora_r1_attnonly | 0.0002 | 3 | 461376 | 0.02451 | 0.9495 ± 0.0041 | 1838.3 | 9783.3 ± 28.1 |
| qwen3_5_2b_base | lora_r8_attnonly | 0.0002 | 3 | 3691008 | 0.19576 | 0.9469 ± 0.0029 | 14454.4 | 9820.2 ± 28.1 |

## rte -- qwen3_5_2b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_2b_base | fitln | 0.0012 | 3 | 51200 | 0.00272 | 0.8664 ± 0.0201 | 208.9 | 14402.8 ± 0.0 |
| qwen3_5_2b_base | fullft | 1e-05 | 3 | 1881825088 | 100.00000 | 0.8833 ± 0.0171 | N/A (cluster only) | 16886.5 ± 0.9 |
| qwen3_5_2b_base | lora_r1_attnonly | 5e-05 | 3 | 461376 | 0.02451 | 0.8773 ± 0.0072 | 1838.3 | 14391.9 ± 0.0 |
| qwen3_5_2b_base | lora_r8_attnonly | 0.0001 | 3 | 3691008 | 0.19576 | 0.8700 ± 0.0188 | 14454.4 | 14428.8 ± 0.0 |

## mrpc -- qwen3_5_2b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_2b_base | fitln | 0.0012 | 3 | 51200 | 0.00272 | 0.8693 ± 0.0037 | 208.9 | 11698.1 ± 0.0 |
| qwen3_5_2b_base | fullft | 1e-05 | 3 | 1881825088 | 100.00000 | 0.8971 ± 0.0042 | N/A (cluster only) | 16397.5 ± 1.4 |
| qwen3_5_2b_base | lora_r1_attnonly | 0.0002 | 3 | 461376 | 0.02451 | 0.8758 ± 0.0079 | 1838.3 | 11694.1 ± 0.0 |
| qwen3_5_2b_base | lora_r8_attnonly | 5e-05 | 3 | 3691008 | 0.19576 | 0.8848 ± 0.0209 | 14454.4 | 11731.0 ± 0.0 |

## cola -- qwen3_5_2b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_2b_base | fitln | 0.0013 | 3 | 51200 | 0.00272 | 0.8376 ± 0.0047 | 208.9 | 9675.0 ± 0.2 |
| qwen3_5_2b_base | fullft | 2e-05 | 2 | 1881825088 | 100.00000 | 0.8063 ± 0.0163 | N/A (cluster only) | 16370.9 ± 2.7 |
| qwen3_5_2b_base | lora_r1_attnonly | 0.0002 | 3 | 461376 | 0.02451 | 0.8338 ± 0.0093 | 1838.3 | 9628.6 ± 0.4 |
| qwen3_5_2b_base | lora_r8_attnonly | 0.0001 | 3 | 3691008 | 0.19576 | 0.8354 ± 0.0056 | 14454.4 | 9665.6 ± 0.4 |

## alpaca_mini -- qwen3_5_2b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | ROUGE-L | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_2b_base | fitln | 0.001 | 3 | 51200 | 0.00272 | 3.854 ± 0.070 | 0.4004 ± 0.0079 | 208.9 | 14402.8 ± 0.0 |
| qwen3_5_2b_base | fullft | 1e-05 | 3 | 1881825088 | 100.00000 | 3.771 ± 0.095 | 0.4118 ± 0.0068 | N/A (cluster only) | 16886.1 ± 1.5 |
| qwen3_5_2b_base | lora_r1_attnonly | 0.0002 | 3 | 461376 | 0.02451 | 3.737 ± 0.076 | 0.4072 ± 0.0096 | 1838.3 | 14392.1 ± 0.4 |
| qwen3_5_2b_base | lora_r8_attnonly | 5e-05 | 3 | 3691008 | 0.19576 | 3.714 ± 0.073 | 0.4098 ± 0.0093 | 14454.4 | 14429.0 ± 0.4 |

## gsm8k -- qwen3_5_2b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | Exact Match | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_2b_base | fitln | 0.0013 | 3 | 51200 | 0.00272 | 1.532 ± 0.030 | 0.5722 ± 0.0052 | 208.9 | 14450.6 ± 0.0 |
| qwen3_5_2b_base | fullft | 1e-05 | 3 | 1881825088 | 100.00000 | 1.581 ± 0.041 | 0.6315 ± 0.0217 | N/A (cluster only) | 16885.3 ± 0.3 |
| qwen3_5_2b_base | lora_r1_attnonly | 0.0002 | 3 | 461376 | 0.02451 | 1.491 ± 0.032 | 0.6007 ± 0.0019 | 1838.3 | 14391.9 ± 0.0 |
| qwen3_5_2b_base | lora_r8_attnonly | 0.0002 | 3 | 3691008 | 0.19576 | 1.480 ± 0.032 | 0.6176 ± 0.0077 | 14454.4 | 14428.8 ± 0.0 |

## samsum -- qwen3_5_2b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | ROUGE-L | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_2b_base | fitln | 0.0013 | 3 | 51200 | 0.00272 | 4.813 ± 0.063 | 0.3938 ± 0.0012 | 208.9 | 13429.1 ± 43.9 |
| qwen3_5_2b_base | fullft | 1e-05 | 2 | 1881825088 | 100.00000 | 4.732 ± 0.056 | 0.4063 ± 0.0003 | N/A (cluster only) | 16390.6 ± 0.4 |
| qwen3_5_2b_base | lora_r1_attnonly | 0.0002 | 3 | 461376 | 0.02451 | 4.550 ± 0.059 | 0.4041 ± 0.0021 | 1838.3 | 13373.0 ± 43.8 |
| qwen3_5_2b_base | lora_r8_attnonly | 5e-05 | 3 | 3691008 | 0.19576 | 4.513 ± 0.063 | 0.4031 ± 0.0014 | 14454.4 | 13410.0 ± 43.8 |

