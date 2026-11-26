**Hardware:** Tesla T4 (Kaggle), torch 2.11.0+cu130, torch.bfloat16  
**Model:** random-init nanochat GPT 12L/768d, 176.2M params  
**Workload:** 128 requests, prompts 100-1000 tokens, outputs 128-256 tokens, max_model_len 2048

| | requests timed | output tok/s |
|---|---|---|
| one request at a time | 16 | 40.8 |
| continuous batching | 128 | **254.2** (6.2x) |

KV memory with all requests resident: 3072.0 MB if each request reserves max_model_len=2048 tokens, 1086.8 MB paged (2.8x less; mean 716 tokens per request).
