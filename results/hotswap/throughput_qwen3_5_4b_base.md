# Throughput: Qwen/Qwen3.5-4B-Base

Single process, batch_size=8, seq_len=256, dtype=fp16 for all methods.

| Method | Regime | Merged | Trainable params | samples/sec | ms/batch (mean) | ms/batch (median) | std |
|---|---|---|---|---|---|---|---|
| fitln | inference | False | 84,480 | 148.6 | 53.833 | 53.822 | 0.125 |
| fitln | train_step | False | 84,480 | 57.0 | 140.440 | 140.464 | 0.170 |
| fullft | inference | False | 4,205,751,296 | 148.7 | 53.783 | 53.780 | 0.157 |
| fullft | train_step | False | 4,205,751,296 | 33.0 | 242.128 | 241.985 | 0.563 |
| lora1 | inference | False | 898,560 | 116.4 | 68.748 | 68.704 | 0.148 |
| lora1 | train_step | False | 898,560 | 44.8 | 178.504 | 178.515 | 0.191 |
| lora1 | inference | True | 898,560 | 148.8 | 53.756 | 53.757 | 0.162 |
| lora8 | inference | False | 7,188,480 | 110.7 | 72.260 | 72.228 | 0.142 |
| lora8 | train_step | False | 7,188,480 | 42.3 | 189.336 | 189.283 | 0.273 |
| lora8 | inference | True | 7,188,480 | 148.7 | 53.792 | 53.763 | 0.154 |
