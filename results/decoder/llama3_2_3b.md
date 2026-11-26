## sst2 -- llama3_2_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_3b | fitln | 0.0013 | 3 | 89088 | 0.00277 | 0.9534 ± 0.0058 | 184.1 | 7081.9 ± 7.9 |
| llama3_2_3b | fullft | 1e-05 | 3 | 3212749824 | 100.00000 | 0.9511 ± 0.0107 | N/A (cluster only) | 26088.6 ± 0.4 |
| llama3_2_3b | lora_r1_attnonly | 0.0002 | 3 | 573440 | 0.01785 | 0.9599 ± 0.0064 | 2274.9 | 7036.8 ± 7.5 |
| llama3_2_3b | lora_r8_attnonly | 0.0001 | 3 | 4587520 | 0.14259 | 0.9614 ± 0.0067 | 17955.2 | 7087.1 ± 15.1 |

## rte -- llama3_2_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_3b | fitln | 0.001 | 3 | 89088 | 0.00277 | 0.8351 ± 0.0021 | 184.1 | 9571.4 ± 0.0 |
| llama3_2_3b | fullft | 2e-05 | 3 | 3212749824 | 100.00000 | 0.8628 ± 0.0191 | N/A (cluster only) | 26120.3 ± 2.0 |
| llama3_2_3b | lora_r1_attnonly | 0.0002 | 3 | 573440 | 0.01785 | 0.8496 ± 0.0150 | 2274.9 | 9517.9 ± 0.0 |
| llama3_2_3b | lora_r8_attnonly | 0.0001 | 3 | 4587520 | 0.14259 | 0.8568 ± 0.0075 | 17955.2 | 9563.9 ± 0.0 |

## mrpc -- llama3_2_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_3b | fitln | 0.0013 | 3 | 89088 | 0.00277 | 0.8333 ± 0.0112 | 184.1 | 7882.3 ± 22.2 |
| llama3_2_3b | fullft | 2e-05 | 3 | 3212749824 | 100.00000 | 0.8415 ± 0.0375 | N/A (cluster only) | 26104.8 ± 4.1 |
| llama3_2_3b | lora_r1_attnonly | 0.0002 | 3 | 573440 | 0.01785 | 0.8423 ± 0.0037 | 2274.9 | 7834.9 ± 22.0 |
| llama3_2_3b | lora_r8_attnonly | 0.0002 | 3 | 4587520 | 0.14259 | 0.8562 ± 0.0099 | 17955.2 | 7880.8 ± 22.1 |

## cola -- llama3_2_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Test Accuracy | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_3b | fitln | 0.0013 | 3 | 89088 | 0.00277 | 0.8380 ± 0.0085 | 184.1 | 6984.6 ± 0.2 |
| llama3_2_3b | fullft | 1e-05 | 3 | 3212749824 | 100.00000 | 0.8626 ± 0.0074 | N/A (cluster only) | 26088.2 ± 1.0 |
| llama3_2_3b | lora_r1_attnonly | 0.0002 | 3 | 573440 | 0.01785 | 0.8568 ± 0.0096 | 2274.9 | 6940.9 ± 0.0 |
| llama3_2_3b | lora_r8_attnonly | 0.0002 | 3 | 4587520 | 0.14259 | 0.8498 ± 0.0106 | 17955.2 | 6986.8 ± 0.0 |

## alpaca_mini -- llama3_2_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | ROUGE-L | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_3b | fitln | 0.0013 | 3 | 89088 | 0.00277 | 3.679 ± 0.058 | 0.3865 ± 0.0052 | 184.1 | 9571.4 ± 0.0 |
| llama3_2_3b | fullft | 1e-05 | 3 | 3212749824 | 100.00000 | 3.464 ± 0.073 | 0.4088 ± 0.0058 | N/A (cluster only) | 26110.4 ± 9.2 |
| llama3_2_3b | lora_r1_attnonly | 0.0001 | 3 | 573440 | 0.01785 | 3.518 ± 0.075 | 0.4022 ± 0.0089 | 2274.9 | 9518.2 ± 0.5 |
| llama3_2_3b | lora_r8_attnonly | 5e-05 | 3 | 4587520 | 0.14259 | 3.500 ± 0.074 | 0.4050 ± 0.0078 | 17955.2 | 9564.2 ± 0.5 |

## gsm8k -- llama3_2_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | Exact Match | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_3b | fitln | 0.0013 | 3 | 89088 | 0.00277 | 1.822 ± 0.029 | 0.3000 ± 0.0103 | 184.1 | 9571.4 ± 0.0 |
| llama3_2_3b | fullft | 2e-05 | 3 | 3212749824 | 100.00000 | 1.647 ± 0.036 | 0.4210 ± 0.0249 | N/A (cluster only) | 26106.1 ± 4.3 |
| llama3_2_3b | lora_r1_attnonly | 0.0002 | 3 | 573440 | 0.01785 | 1.682 ± 0.038 | 0.4036 ± 0.0090 | 2274.9 | 9517.9 ± 0.0 |
| llama3_2_3b | lora_r8_attnonly | 0.0002 | 3 | 4587520 | 0.14259 | 1.655 ± 0.042 | 0.4377 ± 0.0096 | 17955.2 | 9563.9 ± 0.0 |

## samsum -- llama3_2_3b

| Model | Method | Best LR | Seeds | Trainable Params | % | Eval Perplexity | ROUGE-L | Checkpoint Size (KB) | Peak GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3_2_3b | fitln | 0.0013 | 3 | 89088 | 0.00277 | 4.231 ± 0.069 | 0.3936 ± 0.0008 | 184.1 | 8995.9 ± 15.7 |
| llama3_2_3b | fullft | 1e-05 | 3 | 3212749824 | 100.00000 | 3.905 ± 0.067 | 0.4189 ± 0.0025 | N/A (cluster only) | 26113.2 ± 13.7 |
| llama3_2_3b | lora_r1_attnonly | 0.0001 | 3 | 573440 | 0.01785 | 3.940 ± 0.070 | 0.4193 ± 0.0020 | 2274.9 | 8944.8 ± 15.6 |
| llama3_2_3b | lora_r8_attnonly | 0.0001 | 3 | 4587520 | 0.14259 | 3.919 ± 0.059 | 0.4215 ± 0.0026 | 17955.2 | 8990.7 ± 15.6 |

