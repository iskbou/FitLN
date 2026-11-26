# Throughput: meta-llama/Llama-3.2-1B

Single process, batch_size=8, seq_len=256, dtype=fp16 for all methods.

| Method | Regime | Merged | Trainable params | samples/sec | ms/batch (mean) | ms/batch (median) | std |
|---|---|---|---|---|---|---|---|
| fitln | inference | False | 34,816 | 586.7 | 13.635 | 13.628 | 0.062 |
| fitln | train_step | False | 34,816 | 229.0 | 34.929 | 34.924 | 0.103 |
| fullft | inference | False | 1,235,814,400 | 585.4 | 13.665 | 13.615 | 0.369 |
| fullft | train_step | False | 1,235,814,400 | 124.7 | 64.155 | 64.156 | 0.080 |
| lora1 | inference | False | 212,992 | 471.5 | 16.966 | 16.944 | 0.067 |
| lora1 | train_step | False | 212,992 | 181.0 | 44.197 | 44.180 | 0.100 |
| lora1 | inference | True | 212,992 | 584.8 | 13.680 | 13.675 | 0.043 |
| lora8 | inference | False | 1,703,936 | 434.2 | 18.424 | 18.423 | 0.056 |
| lora8 | train_step | False | 1,703,936 | 165.7 | 48.273 | 48.278 | 0.069 |
| lora8 | inference | True | 1,703,936 | 585.0 | 13.675 | 13.661 | 0.061 |
