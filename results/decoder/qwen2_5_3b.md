## sst2 -- qwen2_5_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_3b | fitln | 0.0013 | 3 | 75776 | 0.00246 | 0.9385 ± 0.0163 | 160.6 | 7067.4 ± 8.7 |
| qwen2_5_3b | fullft | 1e-05 | 3 | 3085938688 | 100.00000 | 0.9595 ± 0.0035 | N/A (cluster only) | 25193.8 ± 10.3 |
| qwen2_5_3b | lora_r1_attnonly | 0.0002 | 3 | 460800 | 0.01493 | 0.9591 ± 0.0013 | 1842.9 | 7033.5 ± 9.9 |
| qwen2_5_3b | lora_r8_attnonly | 5e-05 | 3 | 3686400 | 0.11932 | 0.9610 ± 0.0000 | 14443.4 | 7070.4 ± 9.9 |

## rte -- qwen2_5_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_3b | fitln | 0.0013 | 3 | 75776 | 0.00246 | 0.8664 ± 0.0036 | 160.6 | 9935.4 ± 0.0 |
| qwen2_5_3b | fullft | 1e-05 | 3 | 3085938688 | 100.00000 | 0.8965 ± 0.0075 | N/A (cluster only) | 25096.3 ± 72.4 |
| qwen2_5_3b | lora_r1_attnonly | 5e-05 | 3 | 460800 | 0.01493 | 0.8712 ± 0.0055 | 1842.9 | 9884.8 ± 0.0 |
| qwen2_5_3b | lora_r8_attnonly | 0.0002 | 3 | 3686400 | 0.11932 | 0.8809 ± 0.0072 | 14443.4 | 9921.7 ± 0.0 |

## mrpc -- qwen2_5_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_3b | fitln | 0.0013 | 3 | 75776 | 0.00246 | 0.8064 ± 0.0025 | 160.6 | 8225.5 ± 0.1 |
| qwen2_5_3b | fullft | 2e-05 | 3 | 3085938688 | 100.00000 | 0.8758 ± 0.0116 | N/A (cluster only) | 25140.3 ± 27.3 |
| qwen2_5_3b | lora_r1_attnonly | 0.0002 | 3 | 460800 | 0.01493 | 0.8775 ± 0.0153 | 1842.9 | 8178.6 ± 0.5 |
| qwen2_5_3b | lora_r8_attnonly | 0.0001 | 3 | 3686400 | 0.11932 | 0.8758 ± 0.0174 | 14443.4 | 8215.5 ± 0.5 |

## cola -- qwen2_5_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_3b | fitln | 0.0013 | 3 | 75776 | 0.00246 | 0.8213 ± 0.0022 | 160.6 | 6958.7 ± 0.0 |
| qwen2_5_3b | fullft | 2e-05 | 3 | 3085938688 | 100.00000 | 0.8549 ± 0.0125 | N/A (cluster only) | 25205.1 ± 9.7 |
| qwen2_5_3b | lora_r1_attnonly | 0.0002 | 3 | 460800 | 0.01493 | 0.8552 ± 0.0010 | 1842.9 | 6914.2 ± 0.0 |
| qwen2_5_3b | lora_r8_attnonly | 0.0002 | 3 | 3686400 | 0.11932 | 0.8533 ± 0.0091 | 14443.4 | 6951.1 ± 0.0 |

## alpaca_mini -- qwen2_5_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | ROUGE-L | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_3b | fitln | 0.0013 | 3 | 75776 | 0.00246 | 3.659 ± 0.033 | 0.3911 ± 0.0084 | 160.6 | 9935.7 ± 0.5 |
| qwen2_5_3b | fullft | 2e-05 | 3 | 3085938688 | 100.00000 | 3.421 ± 0.059 | 0.4152 ± 0.0077 | N/A (cluster only) | 25162.3 ± 23.6 |
| qwen2_5_3b | lora_r1_attnonly | 0.0002 | 3 | 460800 | 0.01493 | 3.488 ± 0.042 | 0.4103 ± 0.0079 | 1842.9 | 9885.1 ± 0.5 |
| qwen2_5_3b | lora_r8_attnonly | 5e-05 | 3 | 3686400 | 0.11932 | 3.463 ± 0.055 | 0.4127 ± 0.0063 | 14443.4 | 9922.0 ± 0.5 |

## gsm8k -- qwen2_5_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | Exact Match | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_3b | fitln | 0.0013 | 3 | 75776 | 0.00246 | 1.395 ± 0.017 | 0.6778 ± 0.0053 | 160.6 | 9935.4 ± 0.0 |
| qwen2_5_3b | fullft | 1e-05 | 3 | 3085938688 | 100.00000 | 1.330 ± 0.019 | 0.6790 ± 0.0027 | N/A (cluster only) | 25164.8 ± 7.8 |
| qwen2_5_3b | lora_r1_attnonly | 0.0002 | 3 | 460800 | 0.01493 | 1.333 ± 0.019 | 0.6717 ± 0.0087 | 1842.9 | 9884.8 ± 0.0 |
| qwen2_5_3b | lora_r8_attnonly | 0.0002 | 3 | 3686400 | 0.11932 | 1.331 ± 0.019 | 0.6831 ± 0.0045 | 14443.4 | 9921.7 ± 0.0 |

## samsum -- qwen2_5_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | ROUGE-L | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5_3b | fitln | 0.0013 | 3 | 75776 | 0.00246 | 4.944 ± 0.134 | 0.3616 ± 0.0040 | 160.6 | 9302.5 ± 17.6 |
| qwen2_5_3b | fullft | 2e-05 | 3 | 3085938688 | 100.00000 | 3.988 ± 0.074 | 0.4135 ± 0.0009 | N/A (cluster only) | 25127.2 ± 60.0 |
| qwen2_5_3b | lora_r1_attnonly | 0.0001 | 3 | 460800 | 0.01493 | 4.058 ± 0.081 | 0.4113 ± 0.0008 | 1842.9 | 9253.5 ± 17.9 |
| qwen2_5_3b | lora_r8_attnonly | 5e-05 | 3 | 3686400 | 0.11932 | 4.038 ± 0.078 | 0.4132 ± 0.0033 | 14443.4 | 9290.4 ± 17.9 |

