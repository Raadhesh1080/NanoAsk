"""
Throughput benchmark: continuous batching vs. one-request-at-a-time, and paged vs. contiguous KV memory.

    python scripts/bench.py --num-requests 32
    python scripts/bench.py --checkpoint ~/.cache/nanochat/chatsft_checkpoints/d20 --device cuda
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nanoask.engine.llm_engine import EngineConfig, LLMEngine  # noqa: E402
from nanoask.engine.model_runner import kv_bytes_per_block  # noqa: E402
from nanoask.engine.sequence import SamplingParams  # noqa: E402
from nanoask.model.gpt import GPT, GPTConfig  # noqa: E402
from nanoask.model.loader import load_checkpoint, pick_device  # noqa: E402


class _Tok:
    bos_token_id = 0
    stop_token_ids: set = set()

    def encode_special(self, s):
        return 0

    def decode(self, ids):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    ap.add_argument("--num-requests", type=int, default=32)
    ap.add_argument("--min-input", type=int, default=50)
    ap.add_argument("--max-input", type=int, default=300)
    ap.add_argument("--max-output", type=int, default=64)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    args = ap.parse_args()
    dev = pick_device(args.device)
    if args.checkpoint:
        model, tok, _ = load_checkpoint(args.checkpoint, device=str(dev))
    else:
        cfg = GPTConfig(sequence_len=1024, vocab_size=4096, n_layer=6, n_head=4, n_kv_head=2, n_embd=256)
        torch.manual_seed(0)
        model = GPT(cfg)
        model.init_weights()
        model = model.to(dev).eval()
        tok = _Tok()
    V = model.config.vocab_size
    rng = random.Random(0)
    prompts = [[rng.randrange(10, V) for _ in range(rng.randint(args.min_input, args.max_input))]
               for _ in range(args.num_requests)]
    outs = [rng.randint(args.max_output // 2, args.max_output) for _ in prompts]
    sps = [SamplingParams(temperature=0.8, max_tokens=o, ignore_eos=True) for o in outs]
    total_out = sum(outs)
    print(f"device={dev}  model={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params  "
          f"requests={len(prompts)}  output tokens={total_out}")

    def make(max_num_seqs):
        return LLMEngine(model, tok, EngineConfig(max_num_seqs=max_num_seqs, max_model_len=1024, kv_cache_mb=256))

    eng = make(args.max_num_seqs)
    eng.generate([prompts[0][:16]], SamplingParams(max_tokens=4))  # warmup
    t = time.perf_counter()
    for p, sp in zip(prompts, sps):
        eng.generate([p], sp)
    seq_time = time.perf_counter() - t

    eng = make(args.max_num_seqs)
    t = time.perf_counter()
    eng.generate(prompts, sps)
    bat_time = time.perf_counter() - t
    print(f"sequential : {seq_time:6.2f}s  {total_out / seq_time:8.1f} tok/s")
    print(f"continuous : {bat_time:6.2f}s  {total_out / bat_time:8.1f} tok/s   "
          f"(x{seq_time / bat_time:.1f}, max batch {eng.stats.max_batch_size})")

    # memory: contiguous max_model_len reservation vs paged blocks actually touched
    bs = eng.config.block_size
    per_tok = kv_bytes_per_block(model, bs, model.dtype) / bs
    contiguous = len(prompts) * eng.config.max_model_len * per_tok
    paged = sum(-(-(len(p) + o) // bs) * bs for p, o in zip(prompts, outs)) * per_tok
    print(f"KV memory for all requests at once: contiguous {contiguous / 2**20:.1f} MB vs paged "
          f"{paged / 2**20:.1f} MB ({contiguous / paged:.1f}x less)")


if __name__ == "__main__":
    main()
