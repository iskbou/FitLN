## sst2 -- qwen3_5_4b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_4b_base | fitln | 0.0013 | 3 | 84480 | 0.00201 | 0.9587 ± 0.0061 | 341.3 | 19023.9 ± 50.8 |
| qwen3_5_4b_base | fullft | 1e-05 | 3 | 4205751296 | 100.00000 | 0.9507 ± 0.0072 | N/A (cluster only) | 34923.4 ± 20.1 |
| qwen3_5_4b_base | lora_r1_attnonly | 0.0001 | 3 | 898560 | 0.02136 | 0.9580 ± 0.0018 | 3556.2 | 18980.6 ± 50.5 |
| qwen3_5_4b_base | lora_r8_attnonly | 0.0002 | 3 | 7188480 | 0.17063 | 0.9557 ± 0.0053 | 28126.7 | 19052.6 ± 50.5 |

## rte -- qwen3_5_4b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_4b_base | fitln | 0.0013 | 3 | 84480 | 0.00201 | 0.9110 ± 0.0075 | 341.3 | 23823.4 ± 0.0 |
| qwen3_5_4b_base | fullft | 1e-05 | 3 | 4205751296 | 100.00000 | 0.9338 ± 0.0110 | N/A (cluster only) | 34836.4 ± 86.9 |
| qwen3_5_4b_base | lora_r1_attnonly | 0.0002 | 3 | 898560 | 0.02136 | 0.9182 ± 0.0055 | 3556.2 | 23765.5 ± 0.1 |
| qwen3_5_4b_base | lora_r8_attnonly | 0.0001 | 3 | 7188480 | 0.17063 | 0.9326 ± 0.0075 | 28126.7 | 23837.5 ± 0.1 |

## mrpc -- qwen3_5_4b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_4b_base | fitln | 0.0013 | 3 | 84480 | 0.00201 | 0.8766 ± 0.0116 | 341.3 | 21005.1 ± 0.2 |
| qwen3_5_4b_base | fullft | 1e-05 | 3 | 4205751296 | 100.00000 | 0.9101 ± 0.0051 | N/A (cluster only) | 34852.8 ± 37.4 |
| qwen3_5_4b_base | lora_r1_attnonly | 0.0002 | 3 | 898560 | 0.02136 | 0.8922 ± 0.0130 | 3556.2 | 20956.0 ± 0.9 |
| qwen3_5_4b_base | lora_r8_attnonly | 5e-05 | 3 | 7188480 | 0.17063 | 0.8807 ± 0.0111 | 28126.7 | 21027.9 ± 0.9 |

## cola -- qwen3_5_4b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_4b_base | fitln | 0.0011 | 3 | 84480 | 0.00201 | 0.8559 ± 0.0040 | 341.3 | 18845.8 ± 0.5 |
| qwen3_5_4b_base | fullft | 1e-05 | 3 | 4205751296 | 100.00000 | 0.8642 ± 0.0055 | N/A (cluster only) | 34933.2 ± 10.8 |
| qwen3_5_4b_base | lora_r1_attnonly | 0.0002 | 3 | 898560 | 0.02136 | 0.8594 ± 0.0045 | 3556.2 | 18803.7 ± 0.4 |
| qwen3_5_4b_base | lora_r8_attnonly | 0.0001 | 3 | 7188480 | 0.17063 | 0.8664 ± 0.0058 | 28126.7 | 18875.6 ± 0.4 |

## alpaca_mini -- qwen3_5_4b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | ROUGE-L | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_4b_base | fitln | 0.0009 | 3 | 84480 | 0.00201 | 3.445 ± 0.067 | 0.4206 ± 0.0118 | 341.3 | 23823.4 ± 0.0 |
| qwen3_5_4b_base | fullft | 1e-05 | 3 | 4205751296 | 100.00000 | 3.494 ± 0.098 | 0.4272 ± 0.0055 | N/A (cluster only) | 34910.4 ± 23.6 |
| qwen3_5_4b_base | lora_r1_attnonly | 5e-05 | 3 | 898560 | 0.02136 | 3.383 ± 0.068 | 0.4290 ± 0.0091 | 3556.2 | 23765.8 ± 0.5 |
| qwen3_5_4b_base | lora_r8_attnonly | 5e-05 | 3 | 7188480 | 0.17063 | 3.367 ± 0.073 | 0.4287 ± 0.0075 | 28126.7 | 23837.7 ± 0.5 |

## gsm8k -- qwen3_5_4b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | Exact Match | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_4b_base | fitln | 0.0013 | 3 | 84480 | 0.00201 | 1.462 ± 0.024 | 0.7877 ± 0.0046 | 341.3 | 23823.4 ± 0.0 |
| qwen3_5_4b_base | fullft | 1e-05 | 3 | 4205751296 | 100.00000 | 1.543 ± 0.029 | 0.7940 ± 0.0096 | N/A (cluster only) | 34909.0 ± 3.5 |
| qwen3_5_4b_base | lora_r1_attnonly | 0.0002 | 3 | 898560 | 0.02136 | 1.433 ± 0.023 | 0.8067 ± 0.0154 | 3556.2 | 23765.5 ± 0.0 |
| qwen3_5_4b_base | lora_r8_attnonly | 0.0002 | 3 | 7188480 | 0.17063 | 1.425 ± 0.026 | 0.7930 ± 0.0055 | 28126.7 | 23837.4 ± 0.0 |

## samsum -- qwen3_5_4b_base

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | ROUGE-L | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_4b_base | fitln | 0.0013 | 3 | 84480 | 0.00201 | 4.159 ± 0.068 | 0.4164 ± 0.0009 | 341.3 | 22758.1 ± 45.2 |
| qwen3_5_4b_base | fullft | 1e-05 | 3 | 4205751296 | 100.00000 | 4.364 ± 0.061 | 0.4183 ± 0.0026 | N/A (cluster only) | 34872.0 ± 56.1 |
| qwen3_5_4b_base | lora_r1_attnonly | 0.0001 | 3 | 898560 | 0.02136 | 4.041 ± 0.030 | 0.4228 ± 0.0041 | 3556.2 | 22703.8 ± 45.0 |
| qwen3_5_4b_base | lora_r8_attnonly | 5e-05 | 3 | 7188480 | 0.17063 | 4.026 ± 0.033 | 0.4211 ± 0.0029 | 28126.7 | 22775.7 ± 45.0 |

