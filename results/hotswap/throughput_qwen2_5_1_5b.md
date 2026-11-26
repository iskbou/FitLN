# Throughput: Qwen/Qwen2.5-1.5B

Single process, batch_size=8, seq_len=256, dtype=fp16 for all methods.

| Method | Regime | Merged | Trainable params | samples/sec | ms/batch (mean) | ms/batch (median) | std |
|---|---|---|---|---|---|---|---|
| fitln | inference | False | 44,544 | 422.3 | 18.944 | 18.934 | 0.043 |
| fitln | train_step | False | 44,544 | 163.3 | 48.998 | 48.969 | 0.270 |
| fullft | inference | False | 1,543,714,304 | 421.9 | 18.964 | 18.971 | 0.060 |
| fullft | train_step | False | 1,543,714,304 | 92.3 | 86.632 | 86.624 | 0.053 |
| lora1 | inference | False | 272,384 | 334.6 | 23.908 | 23.903 | 0.036 |
| lora1 | train_step | False | 272,384 | 128.0 | 62.510 | 62.586 | 0.428 |
| lora1 | inference | True | 272,384 | 422.3 | 18.942 | 18.940 | 0.042 |
| lora8 | inference | False | 2,179,072 | 294.7 | 27.144 | 27.150 | 0.057 |
| lora8 | train_step | False | 2,179,072 | 115.8 | 69.093 | 69.123 | 0.153 |
| lora8 | inference | True | 2,179,072 | 423.2 | 18.905 | 18.903 | 0.041 |
