## sst2 -- qwen2_5_1_5b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_1_5b | fitln | 0.0013 | 3 | 44544 | 0.00289 | 0.9396 ± 0.0043 | 97.1 | 4000.8 ± 9.3 |
| qwen2_5_1_5b | fullft | 2e-05 | 3 | 1543714304 | 100.00000 | 0.9415 ± 0.0128 | N/A (cluster only) | 12748.5 ± 4.2 |
| qwen2_5_1_5b | lora_r1_attnonly | 0.0002 | 3 | 272384 | 0.01764 | 0.9564 ± 0.0079 | 1098.4 | 3944.9 ± 8.0 |
| qwen2_5_1_5b | lora_r8_attnonly | 0.0001 | 3 | 2179072 | 0.14096 | 0.9622 ± 0.0057 | 8546.8 | 3966.8 ± 8.0 |

## rte -- qwen2_5_1_5b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_1_5b | fitln | 0.0013 | 3 | 44544 | 0.00289 | 0.8183 ± 0.0021 | 97.1 | 6760.1 ± 0.0 |
| qwen2_5_1_5b | fullft | 1e-05 | 3 | 1543714304 | 100.00000 | 0.8255 ± 0.0146 | N/A (cluster only) | 12774.4 ± 5.3 |
| qwen2_5_1_5b | lora_r1_attnonly | 0.0001 | 3 | 272384 | 0.01764 | 0.8387 ± 0.0055 | 1098.4 | 6709.5 ± 0.1 |
| qwen2_5_1_5b | lora_r8_attnonly | 0.0001 | 3 | 2179072 | 0.14096 | 0.8387 ± 0.0146 | 8546.8 | 6731.3 ± 0.1 |

## mrpc -- qwen2_5_1_5b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_1_5b | fitln | 0.0013 | 3 | 44544 | 0.00289 | 0.8007 ± 0.0037 | 97.1 | 5106.2 ± 1.0 |
| qwen2_5_1_5b | fullft | 2e-05 | 3 | 1543714304 | 100.00000 | 0.8668 ± 0.0186 | N/A (cluster only) | 12756.5 ± 7.2 |
| qwen2_5_1_5b | lora_r1_attnonly | 0.0002 | 3 | 272384 | 0.01764 | 0.8750 ± 0.0042 | 1098.4 | 5058.0 ± 0.7 |
| qwen2_5_1_5b | lora_r8_attnonly | 5e-05 | 3 | 2179072 | 0.14096 | 0.8725 ± 0.0049 | 8546.8 | 5079.8 ± 0.7 |

## cola -- qwen2_5_1_5b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_1_5b | fitln | 0.0013 | 3 | 44544 | 0.00289 | 0.7868 ± 0.0034 | 97.1 | 3885.0 ± 0.0 |
| qwen2_5_1_5b | fullft | 2e-05 | 3 | 1543714304 | 100.00000 | 0.8469 ± 0.0140 | N/A (cluster only) | 12744.5 ± 2.6 |
| qwen2_5_1_5b | lora_r1_attnonly | 0.0002 | 3 | 272384 | 0.01764 | 0.8469 ± 0.0162 | 1098.4 | 3839.6 ± 0.0 |
| qwen2_5_1_5b | lora_r8_attnonly | 0.0002 | 3 | 2179072 | 0.14096 | 0.8514 ± 0.0029 | 8546.8 | 3861.4 ± 0.0 |

## alpaca_mini -- qwen2_5_1_5b

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | ROUGE-L | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_1_5b | fitln | 0.0013 | 3 | 44544 | 0.00289 | 3.911 ± 0.044 | 0.3775 ± 0.0099 | 97.1 | 6760.1 ± 0.0 |
| qwen2_5_1_5b | fullft | 2e-05 | 3 | 1543714304 | 100.00000 | 3.616 ± 0.074 | 0.4050 ± 0.0062 | N/A (cluster only) | 12782.3 ± 3.3 |
| qwen2_5_1_5b | lora_r1_attnonly | 0.0002 | 3 | 272384 | 0.01764 | 3.704 ± 0.055 | 0.3965 ± 0.0076 | 1098.4 | 6709.4 ± 0.0 |
| qwen2_5_1_5b | lora_r8_attnonly | 0.0001 | 3 | 2179072 | 0.14096 | 3.681 ± 0.063 | 0.3997 ± 0.0093 | 8546.8 | 6731.3 ± 0.0 |

## gsm8k -- qwen2_5_1_5b

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | Exact Match | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_1_5b | fitln | 0.0013 | 3 | 44544 | 0.00289 | 1.488 ± 0.022 | 0.5628 ± 0.0024 | 97.1 | 6760.1 ± 0.0 |
| qwen2_5_1_5b | fullft | 2e-05 | 3 | 1543714304 | 100.00000 | 1.402 ± 0.024 | 0.5757 ± 0.0145 | N/A (cluster only) | 12778.8 ± 3.2 |
| qwen2_5_1_5b | lora_r1_attnonly | 0.0002 | 3 | 272384 | 0.01764 | 1.407 ± 0.026 | 0.5797 ± 0.0064 | 1098.4 | 6709.4 ± 0.0 |
| qwen2_5_1_5b | lora_r8_attnonly | 0.0001 | 3 | 2179072 | 0.14096 | 1.405 ± 0.025 | 0.5825 ± 0.0072 | 8546.8 | 6731.3 ± 0.0 |

## samsum -- qwen2_5_1_5b

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | ROUGE-L | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_1_5b | fitln | 0.0013 | 3 | 44544 | 0.00289 | 5.603 ± 0.064 | 0.3502 ± 0.0021 | 97.1 | 6147.4 ± 16.5 |
| qwen2_5_1_5b | fullft | 2e-05 | 3 | 1543714304 | 100.00000 | 4.282 ± 0.054 | 0.4006 ± 0.0007 | N/A (cluster only) | 12784.4 ± 4.1 |
| qwen2_5_1_5b | lora_r1_attnonly | 0.0002 | 3 | 272384 | 0.01764 | 4.382 ± 0.069 | 0.3986 ± 0.0065 | 1098.4 | 6097.5 ± 16.6 |
| qwen2_5_1_5b | lora_r8_attnonly | 0.0001 | 3 | 2179072 | 0.14096 | 4.343 ± 0.085 | 0.4005 ± 0.0023 | 8546.8 | 6119.3 ± 16.6 |

