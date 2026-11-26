# Swap latency

Resident swap: the new task is already in device memory. Copy overwrites the model's own
parameters, pointer only changes which tensors are used, and costs no data movement. Cold swap:
the new task is copied in from host memory. Merge (lora1 and lora8 only): fusing the adapter
into the base weights. The comparable regimes are resident copy and cold swap; resident
pointer is near zero for every method. Times are means in ms; n is the number of switches.


## Encoder

### bert-large

| Method | device | n | resident copy (ms) | cold swap (ms) | merge (ms) | resident pointer (ms) | swap kind |
|---|---|---|---|---|---|---|---|
| fitln | cuda | 300 | **0.0642** | **0.4312** | - | 0.0119 | copy |
| bitfit | cuda | 300 | **0.0759** | **1.5002** | - | 0.0312 | copy |
| lora1 | cuda | 300 | **0.0888** | **1.0564** | 19.6404 | 0.0001 | pointer |
| lora8 | cuda | 300 | **0.0691** | **1.3148** | 9.9414 | 0.0001 | pointer |
| fullft | cuda | 300 | **0.6564** | **15.1163** | - | - | copy |

### deberta-v3-large

| Method | device | n | resident copy (ms) | cold swap (ms) | merge (ms) | resident pointer (ms) | swap kind |
|---|---|---|---|---|---|---|---|
| fitln | cuda | 300 | **0.0661** | **0.4318** | - | 0.0120 | copy |
| bitfit | cuda | 300 | **0.0860** | **1.5384** | - | 0.0317 | copy |
| lora1 | cuda | 300 | **0.0701** | **1.0666** | 8.5675 | 0.0001 | pointer |
| lora8 | cuda | 300 | **0.0732** | **1.3252** | 8.8417 | 0.0002 | pointer |
| fullft | cuda | 300 | **0.8497** | **18.6667** | - | - | copy |

### modernbert-large

| Method | device | n | resident copy (ms) | cold swap (ms) | merge (ms) | resident pointer (ms) | swap kind |
|---|---|---|---|---|---|---|---|
| fitln | cuda | 300 | **0.0328** | **0.2283** | - | 0.0091 | copy |
| bitfit | cuda | 300 | **0.0357** | **0.0804** | - | 0.0051 | copy |
| lora1 | cuda | 300 | **0.0519** | **0.4694** | 8.1579 | 0.0001 | pointer |
| lora8 | cuda | 300 | **0.0582** | **0.6157** | 8.2961 | 0.0001 | pointer |
| fullft | cuda | 300 | **0.7407** | **15.6397** | - | - | copy |

### roberta-large

| Method | device | n | resident copy (ms) | cold swap (ms) | merge (ms) | resident pointer (ms) | swap kind |
|---|---|---|---|---|---|---|---|
| fitln | cuda | 300 | **0.0474** | **0.3577** | - | 0.0108 | copy |
| bitfit | cuda | 300 | **0.0705** | **1.0723** | - | 0.0228 | copy |
| lora1 | cuda | 300 | **0.0712** | **1.0597** | 18.7000 | 0.0001 | pointer |
| lora8 | cuda | 300 | **0.0681** | **1.3169** | 10.0215 | 0.0002 | pointer |
| fullft | cuda | 300 | **0.7532** | **14.1436** | - | 0.0194 | copy |

Merge (lora1 and lora8 only) is not a hot-swap number: merging several tasks into the same
base weights needs one backbone per task.

## Decoder

### llama-3.2-1b

| Method | device | n | resident copy (ms) | cold swap (ms) | merge (ms) | resident pointer (ms) | swap kind |
|---|---|---|---|---|---|---|---|
| fitln | cuda | 300 | **0.0686** | **0.1378** | - | 0.0078 | copy |
| lora1 | cuda | 300 | **0.0652** | **0.9461** | 8.6849 | 0.0001 | pointer |
| lora8 | cuda | 300 | **0.0628** | **1.1913** | 8.3375 | 0.0001 | pointer |
| fullft | cuda | 300 | **2.1976** | **45.6038** | - | - | copy |

### llama-3.2-3b

| Method | device | n | resident copy (ms) | cold swap (ms) | merge (ms) | resident pointer (ms) | swap kind |
|---|---|---|---|---|---|---|---|
| fitln | cuda | 300 | **0.0446** | **0.2282** | - | 0.0085 | copy |
| lora1 | cuda | 300 | **0.0795** | **1.7206** | 10.9638 | 0.0001 | pointer |
| lora8 | cuda | 300 | **0.0876** | **2.1481** | 10.2793 | 0.0001 | pointer |
| fullft | cuda | 300 | **6.2569** | **116.6989** | - | 0.0321 | copy |

### qwen2.5-1.5b

| Method | device | n | resident copy (ms) | cold swap (ms) | merge (ms) | resident pointer (ms) | swap kind |
|---|---|---|---|---|---|---|---|
| fitln | cuda | 300 | **0.0611** | **0.2238** | - | 0.0086 | copy |
| lora1 | cuda | 300 | **0.0811** | **1.6240** | 8.4601 | 0.0001 | pointer |
| lora8 | cuda | 300 | **0.0793** | **2.1601** | 8.8485 | 0.0002 | pointer |
| fullft | cuda | 300 | **2.7849** | **58.2607** | - | - | copy |

### qwen2.5-3b

| Method | device | n | resident copy (ms) | cold swap (ms) | merge (ms) | resident pointer (ms) | swap kind |
|---|---|---|---|---|---|---|---|
| fitln | cuda | 300 | **0.0812** | **0.2897** | - | 0.0152 | copy |
| lora1 | cuda | 300 | **0.0972** | **2.1617** | 9.9465 | 0.0001 | pointer |
| lora8 | cuda | 300 | **0.0922** | **2.6559** | 9.3952 | 0.0001 | pointer |
| fullft | cuda | 300 | **5.5315** | **112.9106** | - | 0.0292 | copy |

### qwen3.5-2b-base

| Method | device | n | resident copy (ms) | cold swap (ms) | merge (ms) | resident pointer (ms) | swap kind |
|---|---|---|---|---|---|---|---|
| fitln | cuda | 300 | **0.0744** | **0.1958** | - | 0.0134 | copy |
| lora1 | cuda | 300 | **0.0710** | **1.4469** | 8.9412 | 0.0001 | pointer |
| lora8 | cuda | 300 | **0.0763** | **1.8528** | 9.1977 | 0.0001 | pointer |
| fullft | cuda | 300 | **3.4913** | **68.7385** | - | 0.0295 | copy |

### qwen3.5-4b-base

| Method | device | n | resident copy (ms) | cold swap (ms) | merge (ms) | resident pointer (ms) | swap kind |
|---|---|---|---|---|---|---|---|
| fitln | cuda | 300 | **0.0461** | **0.2580** | - | 0.0091 | copy |
| lora1 | cuda | 300 | **0.0856** | **1.9924** | 10.4223 | 0.0002 | pointer |
| lora8 | cuda | 300 | **0.0994** | **2.5233** | 10.3556 | 0.0001 | pointer |
| fullft | cuda | 300 | **8.2228** | **152.8041** | - | 0.0319 | copy |

Merge (lora1 and lora8 only) is not a hot-swap number: merging several tasks into the same
base weights needs one backbone per task.

## Pairwise significance of cold swap (Welch's t-test, BH-corrected per model)

| Arch | Model | A vs B | mean A (ms) | mean B (ms) | t | p (normal approx.) | q (BH) | sig. |
|---|---|---|---|---|---|---|---|---|
| encoder | bert-large | bitfit vs fitln | 1.5002 | 0.4312 | 582.62 | 0 | 0 | * |
| encoder | bert-large | bitfit vs fullft | 1.5002 | 15.1163 | -4289.06 | 0 | 0 | * |
| encoder | bert-large | bitfit vs lora1 | 1.5002 | 1.0564 | 81.85 | 0 | 0 | * |
| encoder | bert-large | bitfit vs lora8 | 1.5002 | 1.3148 | 85.42 | 0 | 0 | * |
| encoder | bert-large | fitln vs fullft | 0.4312 | 15.1163 | -4240.23 | 0 | 0 | * |
| encoder | bert-large | fitln vs lora1 | 0.4312 | 1.0564 | -111.70 | 0 | 0 | * |
| encoder | bert-large | fitln vs lora8 | 0.4312 | 1.3148 | -343.17 | 0 | 0 | * |
| encoder | bert-large | fullft vs lora1 | 15.1163 | 1.0564 | 2279.90 | 0 | 0 | * |
| encoder | bert-large | fullft vs lora8 | 15.1163 | 1.3148 | 3778.69 | 0 | 0 | * |
| encoder | bert-large | lora1 vs lora8 | 1.0564 | 1.3148 | -45.21 | 0 | 0 | * |
| encoder | deberta-v3-large | bitfit vs fitln | 1.5384 | 0.4318 | 348.37 | 0 | 0 | * |
| encoder | deberta-v3-large | bitfit vs fullft | 1.5384 | 18.6667 | -4397.97 | 0 | 0 | * |
| encoder | deberta-v3-large | bitfit vs lora1 | 1.5384 | 1.0666 | 162.18 | 0 | 0 | * |
| encoder | deberta-v3-large | bitfit vs lora8 | 1.5384 | 1.3252 | 61.97 | 0 | 0 | * |
| encoder | deberta-v3-large | fitln vs fullft | 0.4318 | 18.6667 | -5819.16 | 0 | 0 | * |
| encoder | deberta-v3-large | fitln vs lora1 | 0.4318 | 1.0666 | -359.76 | 0 | 0 | * |
| encoder | deberta-v3-large | fitln vs lora8 | 0.4318 | 1.3252 | -350.69 | 0 | 0 | * |
| encoder | deberta-v3-large | fullft vs lora1 | 18.6667 | 1.0666 | 6149.60 | 0 | 0 | * |
| encoder | deberta-v3-large | fullft vs lora8 | 18.6667 | 1.3252 | 5098.94 | 0 | 0 | * |
| encoder | deberta-v3-large | lora1 vs lora8 | 1.0666 | 1.3252 | -117.26 | 0 | 0 | * |
| encoder | roberta-large | bitfit vs fitln | 1.0723 | 0.3577 | 686.21 | 0 | 0 | * |
| encoder | roberta-large | bitfit vs fullft | 1.0723 | 14.1436 | -7791.67 | 0 | 0 | * |
| encoder | roberta-large | bitfit vs lora1 | 1.0723 | 1.0597 | 9.76 | 0 | 0 | * |
| encoder | roberta-large | bitfit vs lora8 | 1.0723 | 1.3169 | -109.58 | 0 | 0 | * |
| encoder | roberta-large | fitln vs fullft | 0.3577 | 14.1436 | -9782.48 | 0 | 0 | * |
| encoder | roberta-large | fitln vs lora1 | 0.3577 | 1.0597 | -764.46 | 0 | 0 | * |
| encoder | roberta-large | fitln vs lora8 | 0.3577 | 1.3169 | -470.56 | 0 | 0 | * |
| encoder | roberta-large | fullft vs lora1 | 14.1436 | 1.0597 | 8156.57 | 0 | 0 | * |
| encoder | roberta-large | fullft vs lora8 | 14.1436 | 1.3169 | 5287.33 | 0 | 0 | * |
| encoder | roberta-large | lora1 vs lora8 | 1.0597 | 1.3169 | -118.13 | 0 | 0 | * |
| encoder | modernbert-large | bitfit vs fitln | 0.0804 | 0.2283 | -153.49 | 0 | 0 | * |
| encoder | modernbert-large | bitfit vs fullft | 0.0804 | 15.6397 | -9143.13 | 0 | 0 | * |
| encoder | modernbert-large | bitfit vs lora1 | 0.0804 | 0.4694 | -274.14 | 0 | 0 | * |
| encoder | modernbert-large | bitfit vs lora8 | 0.0804 | 0.6157 | -466.09 | 0 | 0 | * |
| encoder | modernbert-large | fitln vs fullft | 0.2283 | 15.6397 | -10555.31 | 0 | 0 | * |
| encoder | modernbert-large | fitln vs lora1 | 0.2283 | 0.4694 | -215.74 | 0 | 0 | * |
| encoder | modernbert-large | fitln vs lora8 | 0.2283 | 0.6157 | -520.17 | 0 | 0 | * |
| encoder | modernbert-large | fullft vs lora1 | 15.6397 | 0.4694 | 8458.30 | 0 | 0 | * |
| encoder | modernbert-large | fullft vs lora8 | 15.6397 | 0.6157 | 9459.49 | 0 | 0 | * |
| encoder | modernbert-large | lora1 vs lora8 | 0.4694 | 0.6157 | -114.25 | 0 | 0 | * |
| decoder | llama-3.2-1b | fitln vs fullft | 0.1378 | 45.6038 | -8174.66 | 0 | 0 | * |
| decoder | llama-3.2-1b | fitln vs lora1 | 0.1378 | 0.9461 | -428.01 | 0 | 0 | * |
| decoder | llama-3.2-1b | fitln vs lora8 | 0.1378 | 1.1913 | -601.81 | 0 | 0 | * |
| decoder | llama-3.2-1b | fullft vs lora1 | 45.6038 | 0.9461 | 8221.08 | 0 | 0 | * |
| decoder | llama-3.2-1b | fullft vs lora8 | 45.6038 | 1.1913 | 8246.28 | 0 | 0 | * |
| decoder | llama-3.2-1b | lora1 vs lora8 | 0.9461 | 1.1913 | -191.63 | 0 | 0 | * |
| decoder | llama-3.2-3b | fitln vs fullft | 0.2282 | 116.6989 | -18395.75 | 0 | 0 | * |
| decoder | llama-3.2-3b | fitln vs lora1 | 0.2282 | 1.7206 | -991.76 | 0 | 0 | * |
| decoder | llama-3.2-3b | fitln vs lora8 | 0.2282 | 2.1481 | -1170.74 | 0 | 0 | * |
| decoder | llama-3.2-3b | fullft vs lora1 | 116.6989 | 1.7206 | 17748.17 | 0 | 0 | * |
| decoder | llama-3.2-3b | fullft vs lora8 | 116.6989 | 2.1481 | 17593.41 | 0 | 0 | * |
| decoder | llama-3.2-3b | lora1 vs lora8 | 1.7206 | 2.1481 | -199.91 | 0 | 0 | * |
| decoder | qwen2.5-3b | fitln vs fullft | 0.2897 | 112.9106 | -9328.49 | 0 | 0 | * |
| decoder | qwen2.5-3b | fitln vs lora1 | 0.2897 | 2.1617 | -385.59 | 0 | 0 | * |
| decoder | qwen2.5-3b | fitln vs lora8 | 0.2897 | 2.6559 | -487.11 | 0 | 0 | * |
| decoder | qwen2.5-3b | fullft vs lora1 | 112.9106 | 2.1617 | 9720.47 | 0 | 0 | * |
| decoder | qwen2.5-3b | fullft vs lora8 | 112.9106 | 2.6559 | 9676.13 | 0 | 0 | * |
| decoder | qwen2.5-3b | lora1 vs lora8 | 2.1617 | 2.6559 | -178.63 | 0 | 0 | * |
| decoder | qwen3.5-4b-base | fitln vs fullft | 0.2580 | 152.8041 | -104077.22 | 0 | 0 | * |
| decoder | qwen3.5-4b-base | fitln vs lora1 | 0.2580 | 1.9924 | -778.91 | 0 | 0 | * |
| decoder | qwen3.5-4b-base | fitln vs lora8 | 0.2580 | 2.5233 | -1264.46 | 0 | 0 | * |
| decoder | qwen3.5-4b-base | fullft vs lora1 | 152.8041 | 1.9924 | 58463.78 | 0 | 0 | * |
| decoder | qwen3.5-4b-base | fullft vs lora8 | 152.8041 | 2.5233 | 67851.15 | 0 | 0 | * |
| decoder | qwen3.5-4b-base | lora1 vs lora8 | 1.9924 | 2.5233 | -191.13 | 0 | 0 | * |
| decoder | qwen2.5-1.5b | fitln vs fullft | 0.2238 | 58.2607 | -10280.53 | 0 | 0 | * |
| decoder | qwen2.5-1.5b | fitln vs lora1 | 0.2238 | 1.6240 | -669.42 | 0 | 0 | * |
| decoder | qwen2.5-1.5b | fitln vs lora8 | 0.2238 | 2.1601 | -591.52 | 0 | 0 | * |
| decoder | qwen2.5-1.5b | fullft vs lora1 | 58.2607 | 1.6240 | 9892.08 | 0 | 0 | * |
| decoder | qwen2.5-1.5b | fullft vs lora8 | 58.2607 | 2.1601 | 8969.41 | 0 | 0 | * |
| decoder | qwen2.5-1.5b | lora1 vs lora8 | 1.6240 | 2.1601 | -157.21 | 0 | 0 | * |
| decoder | qwen3.5-2b-base | fitln vs fullft | 0.1958 | 68.7385 | -12597.51 | 0 | 0 | * |
| decoder | qwen3.5-2b-base | fitln vs lora1 | 0.1958 | 1.4469 | -792.30 | 0 | 0 | * |
| decoder | qwen3.5-2b-base | fitln vs lora8 | 0.1958 | 1.8528 | -1694.35 | 0 | 0 | * |
| decoder | qwen3.5-2b-base | fullft vs lora1 | 68.7385 | 1.4469 | 11919.42 | 0 | 0 | * |
| decoder | qwen3.5-2b-base | fullft vs lora8 | 68.7385 | 1.8528 | 12143.94 | 0 | 0 | * |
| decoder | qwen3.5-2b-base | lora1 vs lora8 | 1.4469 | 1.8528 | -226.08 | 0 | 0 | * |

## Summary

- Provenance: device cuda, 300 switches per cell, 13 of 44 cells
  with real pretrained weights and the others with synthetic tensors of the registry sizes.
- Resident pointer swap is 0.0001-0.0321 ms for every method and separates nothing.
- Cold swap, decoders: fitln 0.14-0.29 ms, lora1 0.95-2.16 ms, lora8 1.19-2.66 ms,
  fullft 45.6-152.8 ms. Encoders: fitln 0.23-0.43 ms, fullft 14.1-18.7 ms.
- Resident copy, decoders: fitln 0.045-0.081 ms, fullft 2.20-8.22 ms.
- LoRA copy into fixed buffers (decoders, lora1): 0.0652-0.0972 ms, against 0.0001-0.0002 ms for
  the adapter switch.
- LoRA merge, rank 1: 8.2-19.6 ms (encoders), 8.5-11.0 ms (decoders).
