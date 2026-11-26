# Throughput: Qwen/Qwen2.5-3B

Single process, batch_size=8, seq_len=256, dtype=fp16 for all methods.

| Method | Regime | Merged | Trainable params | samples/sec | ms/batch (mean) | ms/batch (median) | std |
|---|---|---|---|---|---|---|---|
| fitln | inference | False | 75,776 | 241.9 | 33.070 | 33.098 | 0.143 |
| fitln | train_step | False | 75,776 | 101.2 | 79.076 | 79.053 | 0.153 |
| fullft | inference | False | 3,085,938,688 | 241.8 | 33.089 | 33.100 | 0.124 |
| fullft | train_step | False | 3,085,938,688 | 52.9 | 151.145 | 151.159 | 0.132 |
| lora1 | inference | False | 460,800 | 200.3 | 39.931 | 39.904 | 0.110 |
| lora1 | train_step | False | 460,800 | 80.6 | 99.260 | 99.261 | 0.072 |
| lora1 | inference | True | 460,800 | 241.9 | 33.078 | 33.063 | 0.140 |
| lora8 | inference | False | 3,686,400 | 189.4 | 42.240 | 42.203 | 0.315 |
| lora8 | train_step | False | 3,686,400 | 75.1 | 106.507 | 106.516 | 0.122 |
| lora8 | inference | True | 3,686,400 | 241.9 | 33.069 | 33.103 | 0.142 |
