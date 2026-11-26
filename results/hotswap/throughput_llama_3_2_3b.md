# Throughput: meta-llama/Llama-3.2-3B

Single process, batch_size=8, seq_len=256, dtype=fp16 for all methods.

| Method | Regime | Merged | Trainable params | samples/sec | ms/batch (mean) | ms/batch (median) | std |
|---|---|---|---|---|---|---|---|
| fitln | inference | False | 89,088 | 231.5 | 34.558 | 34.554 | 0.088 |
| fitln | train_step | False | 89,088 | 100.9 | 79.323 | 79.326 | 0.105 |
| fullft | inference | False | 3,212,749,824 | 231.2 | 34.609 | 34.612 | 0.094 |
| fullft | train_step | False | 3,212,749,824 | 52.6 | 152.225 | 152.240 | 0.091 |
| lora1 | inference | False | 573,440 | 188.4 | 42.456 | 42.454 | 0.113 |
| lora1 | train_step | False | 573,440 | 78.0 | 102.608 | 102.582 | 0.185 |
| lora1 | inference | True | 573,440 | 233.4 | 34.279 | 34.297 | 0.083 |
| lora8 | inference | False | 4,587,520 | 174.3 | 45.887 | 45.891 | 0.095 |
| lora8 | train_step | False | 4,587,520 | 71.7 | 111.647 | 111.609 | 0.215 |
| lora8 | inference | True | 4,587,520 | 234.2 | 34.166 | 34.179 | 0.069 |
