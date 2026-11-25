**Hardware:** x86_64 (4 threads), torch 2.15.0+cu132, torch.float32  
**Model:** random-init nanochat GPT 6L/256d, 8.0M params  
**Workload:** 24 requests, prompts 50-300 tokens, outputs 32-64 tokens, max_model_len 1024

| | time | output tok/s |
|---|---|---|
| one request at a time | 5.15s | 218.9 |
| continuous batching | 1.30s | 868.0 |

KV memory with all requests resident: 144.0 MB if each request reserves max_model_len=1024 tokens, 28.7 MB paged (5.0x less; mean 197 tokens per request).
