# Capacity: maximum resident tasks per GPU memory budget

Weights in bf16, 10% of GPU memory reserved. Encoder task head: not included. FullFT: one full model copy per task. Parameter counts are those of the paper tables.

## Encoder

### bert-large

| Method | 24GB (e.g. RTX 4090 / L4) | 40GB (e.g. A100-40GB) | 80GB (e.g. A100/H100-80GB) |
|---|---|---|---|
| fitln (delta=96.0KB) | 229012 | 386298 | 779514 |
| bitfit (delta=527.3KB) | 41690 | 70323 | 141906 |
| lora1 (delta=287.1KB) | 76574 | 129165 | 260644 |
| lora8 (delta=2.3MB) | 9380 | 15822 | 31928 |
| fullft (delta=648.5MB) | 34 | 56 | 113 |

### roberta-large

| Method | 24GB (e.g. RTX 4090 / L4) | 40GB (e.g. A100-40GB) | 80GB (e.g. A100/H100-80GB) |
|---|---|---|---|
| fitln (delta=96.0KB) | 228707 | 385993 | 779209 |
| bitfit (delta=527.3KB) | 41634 | 70267 | 141850 |
| lora1 (delta=287.1KB) | 76472 | 129063 | 260542 |
| lora8 (delta=2.3MB) | 9367 | 15810 | 31916 |
| fullft (delta=677.1MB) | 32 | 54 | 108 |

### deberta-v3-large

| Method | 24GB (e.g. RTX 4090 / L4) | 40GB (e.g. A100-40GB) | 80GB (e.g. A100/H100-80GB) |
|---|---|---|---|
| fitln (delta=96.0KB) | 227079 | 384365 | 777581 |
| bitfit (delta=527.3KB) | 41338 | 69971 | 141554 |
| lora1 (delta=287.1KB) | 75927 | 128519 | 259997 |
| lora8 (delta=2.3MB) | 9301 | 15743 | 31849 |
| fullft (delta=829.7MB) | 26 | 44 | 88 |

### modernbert-large

| Method | 24GB (e.g. RTX 4090 / L4) | 40GB (e.g. A100-40GB) | 80GB (e.g. A100/H100-80GB) |
|---|---|---|---|
| fitln (delta=56.0KB) | 390674 | 660308 | 1334392 |
| bitfit (delta=568.4KB) | 38492 | 65059 | 131476 |
| lora1 (delta=671.9KB) | 32562 | 55035 | 111220 |
| lora8 (delta=5.3MB) | 4000 | 6761 | 13664 |
| fullft (delta=753.4MB) | 29 | 48 | 97 |

## Decoder

### llama-3.2-1b

| Method | 24GB (e.g. RTX 4090 / L4) | 40GB (e.g. A100-40GB) | 80GB (e.g. A100/H100-80GB) |
|---|---|---|---|
| fitln (delta=68.0KB) | 297461 | 519512 | 1074641 |
| lora1 (delta=416.0KB) | 48621 | 84917 | 175655 |
| lora8 (delta=3.2MB) | 6092 | 10639 | 22008 |
| fullft (delta=2.3GB) | 9 | 15 | 31 |

### qwen2.5-1.5b

| Method | 24GB (e.g. RTX 4090 / L4) | 40GB (e.g. A100-40GB) | 80GB (e.g. A100/H100-80GB) |
|---|---|---|---|
| fitln (delta=87.0KB) | 225763 | 399320 | 833214 |
| lora1 (delta=531.2KB) | 36972 | 65394 | 136451 |
| lora8 (delta=4.2MB) | 4613 | 8159 | 17025 |
| fullft (delta=2.9GB) | 7 | 12 | 25 |

### qwen3.5-2b-base

| Method | 24GB (e.g. RTX 4090 / L4) | 40GB (e.g. A100-40GB) | 80GB (e.g. A100/H100-80GB) |
|---|---|---|---|
| fitln (delta=100.0KB) | 189773 | 340768 | 718255 |
| lora1 (delta=900.4KB) | 21076 | 37846 | 79771 |
| lora8 (delta=7.0MB) | 2633 | 4728 | 9966 |
| fullft (delta=3.5GB) | 6 | 10 | 20 |

### llama-3.2-3b

| Method | 24GB (e.g. RTX 4090 / L4) | 40GB (e.g. A100-40GB) | 80GB (e.g. A100/H100-80GB) |
|---|---|---|---|
| fitln (delta=174.0KB) | 94136 | 180914 | 397861 |
| lora1 (delta=1.1MB) | 14635 | 28128 | 61858 |
| lora8 (delta=8.8MB) | 1827 | 3511 | 7722 |
| fullft (delta=6.0GB) | 3 | 6 | 12 |

### qwen2.5-3b

| Method | 24GB (e.g. RTX 4090 / L4) | 40GB (e.g. A100-40GB) | 80GB (e.g. A100/H100-80GB) |
|---|---|---|---|
| fitln (delta=148.0KB) | 112257 | 214280 | 469339 |
| lora1 (delta=900.4KB) | 18452 | 35222 | 77146 |
| lora8 (delta=7.0MB) | 2305 | 4400 | 9638 |
| fullft (delta=5.8GB) | 3 | 6 | 12 |

### qwen3.5-4b-base

| Method | 24GB (e.g. RTX 4090 / L4) | 40GB (e.g. A100-40GB) | 80GB (e.g. A100/H100-80GB) |
|---|---|---|---|
| fitln (delta=165.0KB) | 87433 | 178945 | 407726 |
| lora1 (delta=1.7MB) | 8216 | 16815 | 38314 |
| lora8 (delta=13.7MB) | 1027 | 2102 | 4790 |
| fullft (delta=7.8GB) | 2 | 4 | 9 |
