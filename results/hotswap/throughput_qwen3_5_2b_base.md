# Throughput: Qwen/Qwen3.5-2B-Base

Single process, batch_size=8, seq_len=256, dtype=fp16 for all methods.

| Method | Regime | Merged | Trainable params | samples/sec | ms/batch (mean) | ms/batch (median) | std |
|---|---|---|---|---|---|---|---|
| fitln | inference | False | 51,200 | 250.9 | 31.883 | 31.729 | 0.899 |
| fitln | train_step | False | 51,200 | 99.6 | 80.332 | 80.209 | 1.216 |
| fullft | inference | False | 1,881,825,088 | 249.9 | 32.013 | 32.011 | 0.960 |
| fullft | train_step | False | 1,881,825,088 | 61.0 | 131.131 | 131.351 | 1.040 |
| lora1 | inference | False | 461,376 | 199.4 | 40.124 | 39.905 | 0.936 |
| lora1 | train_step | False | 461,376 | 79.8 | 100.213 | 99.350 | 1.766 |
| lora1 | inference | True | 461,376 | 252.4 | 31.690 | 30.715 | 1.524 |
| lora8 | inference | False | 3,691,008 | 195.1 | 41.004 | 40.602 | 2.082 |
| lora8 | train_step | False | 3,691,008 | 72.1 | 110.968 | 111.327 | 0.817 |
| lora8 | inference | True | 3,691,008 | 225.5 | 35.482 | 35.557 | 0.284 |
