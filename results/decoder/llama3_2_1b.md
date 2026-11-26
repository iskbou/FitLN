## sst2 -- llama3_2_1b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_1b | fitln | 0.0013 | 3 | 34816 | 0.00282 | 0.9446 ± 0.0048 | 74.5 | 3257.3 ± 14.6 |
| llama3_2_1b | fullft | 1e-05 | 3 | 1235814400 | 100.00000 | 0.9469 ± 0.0126 | N/A (cluster only) | 10515.8 ± 1.4 |
| llama3_2_1b | lora_r1_attnonly | 0.0001 | 3 | 212992 | 0.01723 | 0.9465 ± 0.0076 | 854.3 | 3205.7 ± 7.3 |
| llama3_2_1b | lora_r8_attnonly | 0.0001 | 3 | 1703936 | 0.13769 | 0.9545 ± 0.0048 | 6678.5 | 3222.8 ± 7.3 |

## rte -- llama3_2_1b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_1b | fitln | 0.0013 | 3 | 34816 | 0.00282 | 0.7004 ± 0.0181 | 74.5 | 5580.9 ± 0.0 |
| llama3_2_1b | fullft | 1e-05 | 3 | 1235814400 | 100.00000 | 0.7774 ± 0.0311 | N/A (cluster only) | 10524.0 ± 3.3 |
| llama3_2_1b | lora_r1_attnonly | 0.0002 | 3 | 212992 | 0.01723 | 0.7966 ± 0.0110 | 854.3 | 5527.5 ± 0.0 |
| llama3_2_1b | lora_r8_attnonly | 0.0002 | 3 | 1703936 | 0.13769 | 0.7894 ± 0.0290 | 6678.5 | 5544.6 ± 0.0 |

## mrpc -- llama3_2_1b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_1b | fitln | 0.0013 | 3 | 34816 | 0.00282 | 0.7443 ± 0.0099 | 74.5 | 4002.0 ± 20.4 |
| llama3_2_1b | fullft | 1e-05 | 3 | 1235814400 | 100.00000 | 0.8186 ± 0.0425 | N/A (cluster only) | 10520.3 ± 1.9 |
| llama3_2_1b | lora_r1_attnonly | 0.0002 | 3 | 212992 | 0.01723 | 0.8366 ± 0.0099 | 854.3 | 3952.8 ± 20.6 |
| llama3_2_1b | lora_r8_attnonly | 0.0001 | 3 | 1703936 | 0.13769 | 0.8276 ± 0.0167 | 6678.5 | 3969.9 ± 20.6 |

## cola -- llama3_2_1b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_1b | fitln | 0.0013 | 3 | 34816 | 0.00282 | 0.8156 ± 0.0048 | 74.5 | 3162.6 ± 0.3 |
| llama3_2_1b | fullft | 2e-05 | 2 | 1235814400 | 100.00000 | 0.8207 ± 0.0068 | N/A (cluster only) | 10515.1 ± 0.4 |
| llama3_2_1b | lora_r1_attnonly | 0.0002 | 3 | 212992 | 0.01723 | 0.8498 ± 0.0078 | 854.3 | 3115.3 ± 0.1 |
| llama3_2_1b | lora_r8_attnonly | 0.0002 | 3 | 1703936 | 0.13769 | 0.8376 ± 0.0082 | 6678.5 | 3132.3 ± 0.1 |

## alpaca_mini -- llama3_2_1b

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | ROUGE-L | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_1b | fitln | 0.0013 | 3 | 34816 | 0.00282 | 4.369 ± 0.068 | 0.3375 ± 0.0035 | 74.5 | 5580.9 ± 0.0 |
| llama3_2_1b | fullft | 1e-05 | 3 | 1235814400 | 100.00000 | 3.903 ± 0.064 | 0.3780 ± 0.0083 | N/A (cluster only) | 10519.9 ± 4.6 |
| llama3_2_1b | lora_r1_attnonly | 0.0001 | 3 | 212992 | 0.01723 | 4.038 ± 0.071 | 0.3678 ± 0.0106 | 854.3 | 5527.5 ± 0.0 |
| llama3_2_1b | lora_r8_attnonly | 5e-05 | 3 | 1703936 | 0.13769 | 3.990 ± 0.080 | 0.3733 ± 0.0083 | 6678.5 | 5544.6 ± 0.0 |

## gsm8k -- llama3_2_1b

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | Exact Match | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_1b | fitln | 0.0013 | 3 | 34816 | 0.00282 | 2.156 ± 0.037 | 0.0695 ± 0.0048 | 74.5 | 5580.9 ± 0.0 |
| llama3_2_1b | fullft | 2e-05 | 3 | 1235814400 | 100.00000 | 1.774 ± 0.046 | 0.2370 ± 0.0141 | N/A (cluster only) | 10512.4 ± 2.3 |
| llama3_2_1b | lora_r1_attnonly | 0.0002 | 3 | 212992 | 0.01723 | 1.866 ± 0.041 | 0.1678 ± 0.0039 | 854.3 | 5527.5 ± 0.0 |
| llama3_2_1b | lora_r8_attnonly | 0.0002 | 3 | 1703936 | 0.13769 | 1.786 ± 0.044 | 0.2141 ± 0.0113 | 6678.5 | 5544.6 ± 0.0 |

## samsum -- llama3_2_1b

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | ROUGE-L | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_1b | fitln | 0.0013 | 3 | 34816 | 0.00282 | 5.277 ± 0.071 | 0.3502 ± 0.0018 | 74.5 | 5042.8 ± 14.6 |
| llama3_2_1b | fullft | 1e-05 | 3 | 1235814400 | 100.00000 | 4.414 ± 0.082 | 0.3950 ± 0.0022 | N/A (cluster only) | 10517.2 ± 2.6 |
| llama3_2_1b | lora_r1_attnonly | 0.0002 | 3 | 212992 | 0.01723 | 4.517 ± 0.051 | 0.3929 ± 0.0008 | 854.3 | 4990.9 ± 14.4 |
| llama3_2_1b | lora_r8_attnonly | 5e-05 | 3 | 1703936 | 0.13769 | 4.452 ± 0.069 | 0.3995 ± 0.0012 | 6678.5 | 5007.9 ± 14.4 |

