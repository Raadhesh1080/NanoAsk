"""
Throughput benchmark: continuous batching vs. one-request-at-a-time, and paged vs. contiguous KV memory.

    python scripts/bench.py                                   # CPU-sized preset
    python scripts/bench.py --preset gpu --device cuda --out results/bench_t4.md
    python scripts/bench.py --checkpoint ~/.cache/nanochat/chatsft_checkpoints/d20 --device cuda

Memory baseline: "contiguous" = what a server without paging reserves, max_model_len tokens of
KV cache per request. "paged" = blocks actually touched (prompt + output, rounded up to whole
blocks). The ratio therefore depends on max_model_len and on the request lengths, and both are
printed with the result.
"""
from __future__ import annotations

import argparse
import platform
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

PRESETS = {
    # random-weight models: throughput does not depend on what the weights are
    "cpu": dict(model=dict(n_layer=6, n_head=4, n_kv_head=2, n_embd=256, vocab_size=4096),
                num_requests=24, min_input=50, max_input=300, max_output=64, max_model_len=1024,
                sequential_requests=24),
    "gpu": dict(model=dict(n_layer=12, n_head=12, n_kv_head=4, n_embd=768, vocab_size=32768),
                num_requests=128, min_input=100, max_input=1000, max_output=256, max_model_len=2048,
                sequential_requests=16),
}


class _Tok:
    bos_token_id = 0
    stop_token_ids: set = set()

    def encode_special(self, s):
        return 0

    def decode(self, ids):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="cpu")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device")
    ap.add_argument("--num-requests", type=int)
    ap.add_argument("--min-input", type=int)
    ap.add_argument("--max-input", type=int)
    ap.add_argument("--max-output", type=int)
    ap.add_argument("--max-model-len", type=int)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--skip-sequential", action="store_true", help="only run the batched pass")
    ap.add_argument("--sequential-requests", type=int,
                    help="how many requests the one-at-a-time baseline serves (it reports a rate, so a sample suffices)")
    ap.add_argument("--out", help="also write the results as a Markdown table here")
    args = ap.parse_args()
    pre = PRESETS[args.preset]
    for key in ("num_requests", "min_input", "max_input", "max_output", "max_model_len", "sequential_requests"):
        if getattr(args, key) is None:
            setattr(args, key, pre[key])

    dev = pick_device(args.device)
    if args.checkpoint:
        model, tok, _ = load_checkpoint(args.checkpoint, device=str(dev))
        model_desc = args.checkpoint
    else:
        cfg = GPTConfig(sequence_len=args.max_model_len, **pre["model"])
        torch.manual_seed(0)
        model = GPT(cfg)
        model.init_weights()
        model = model.to(dev, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32).eval()
        tok = _Tok()
        model_desc = f"random-init nanochat GPT {cfg.n_layer}L/{cfg.n_embd}d"
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    max_len = min(args.max_model_len, model.config.sequence_len)
    V = model.config.vocab_size
    rng = random.Random(0)
    prompts = [[rng.randrange(10, V) for _ in range(rng.randint(args.min_input, args.max_input))]
               for _ in range(args.num_requests)]
    outs = [rng.randint(args.max_output // 2, args.max_output) for _ in prompts]
    sps = [SamplingParams(temperature=0.8, max_tokens=o, ignore_eos=True) for o in outs]
    total_out = sum(outs)
    hw = torch.cuda.get_device_name(dev) if dev.type == "cuda" else f"{platform.processor() or 'cpu'} ({torch.get_num_threads()} threads)"
    print(f"hardware: {hw} | torch {torch.__version__} | dtype {model.dtype}")
    print(f"model: {model_desc}, {n_params:.1f}M params | requests: {len(prompts)}, "
          f"prompt {args.min_input}-{args.max_input} tok, output {args.max_output // 2}-{args.max_output} tok "
          f"({total_out} total) | max_model_len={max_len}")

    def make():
        return LLMEngine(model, tok, EngineConfig(max_num_seqs=args.max_num_seqs, max_model_len=max_len,
                                                  max_num_batched_tokens=max(4096, args.max_input),
                                                  kv_cache_mb=1024))

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    eng = make()
    print(f"KV cache: {eng.runner.num_kvcache_blocks} blocks ({eng.runner.kv_cache_bytes / 2**30:.2f} GiB)", flush=True)
    eng.generate([prompts[0][:32]], SamplingParams(max_tokens=8))  # warmup

    def run_batched(engine, ps, ss, label):
        """Like engine.generate, but prints progress so long runs don't look frozen."""
        for p_, s_ in zip(ps, ss):
            engine.add_request(p_, s_)
        t0 = last = time.perf_counter()
        while not engine.is_finished():
            engine.step()
            if time.perf_counter() - last > 15:
                last = time.perf_counter()
                done = len(ps) - len(engine.scheduler.waiting) - len(engine.scheduler.running)
                print(f"  {label}: {done}/{len(ps)} requests done, {last - t0:.0f}s", flush=True)

    seq_time = seq_out = None
    if not args.skip_sequential:
        n_seq = min(args.sequential_requests, len(prompts))
        seq_out = sum(outs[:n_seq])
        print(f"one request at a time: {n_seq} requests ...", flush=True)
        sync()
        t = time.perf_counter()
        for i, (p, sp) in enumerate(zip(prompts[:n_seq], sps[:n_seq]), 1):
            eng.generate([p], sp)
            if i % 4 == 0:
                print(f"  sequential: {i}/{n_seq} requests done, {time.perf_counter() - t:.0f}s", flush=True)
        sync()
        seq_time = time.perf_counter() - t

    print(f"continuous batching: {len(prompts)} requests ...", flush=True)
    eng = make()
    sync()
    t = time.perf_counter()
    run_batched(eng, prompts, sps, "batched")
    sync()
    bat_time = time.perf_counter() - t

    bs = eng.config.block_size
    per_tok = kv_bytes_per_block(model, bs, model.dtype) / bs
    contiguous = len(prompts) * max_len * per_tok
    paged = sum(-(-(len(p) + o) // bs) * bs for p, o in zip(prompts, outs)) * per_tok
    mean_len = sum(len(p) + o for p, o in zip(prompts, outs)) / len(prompts)

    lines = []
    seq_rate = seq_out / seq_time if seq_time else None
    bat_rate = total_out / bat_time
    if seq_time:
        lines.append(f"sequential : {seq_rate:9.1f} output tok/s  ({seq_out} tokens in {seq_time:.2f}s, "
                     f"{args.sequential_requests if args.sequential_requests < len(prompts) else len(prompts)} requests)")
    lines.append(f"continuous : {bat_rate:9.1f} output tok/s  ({total_out} tokens in {bat_time:.2f}s)"
                 + (f"   x{bat_rate / seq_rate:.1f}" if seq_time else "")
                 + f"   max batch {eng.stats.max_batch_size}, preemptions {eng.scheduler.num_preemptions}")
    lines.append(f"KV memory, all requests resident: contiguous {contiguous / 2**20:.1f} MB "
                 f"(max_model_len={max_len} tokens reserved per request) vs paged {paged / 2**20:.1f} MB "
                 f"(mean {mean_len:.0f} tokens used per request, block size {bs}) -> {contiguous / paged:.1f}x less")
    print("\n".join(lines))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        md = [f"**Hardware:** {hw}, torch {torch.__version__}, {model.dtype}  ",
              f"**Model:** {model_desc}, {n_params:.1f}M params  ",
              f"**Workload:** {len(prompts)} requests, prompts {args.min_input}-{args.max_input} tokens, "
              f"outputs {args.max_output // 2}-{args.max_output} tokens, max_model_len {max_len}", "",
              "| | requests timed | output tok/s |", "|---|---|---|"]
        if seq_time:
            md.append(f"| one request at a time | {min(args.sequential_requests, len(prompts))} | {seq_rate:.1f} |")
        md.append(f"| continuous batching | {len(prompts)} | **{bat_rate:.1f}**"
                  + (f" ({bat_rate / seq_rate:.1f}x)" if seq_time else "") + " |")
        md += ["", f"KV memory with all requests resident: {contiguous / 2**20:.1f} MB if each request reserves "
               f"max_model_len={max_len} tokens, {paged / 2**20:.1f} MB paged "
               f"({contiguous / paged:.1f}x less; mean {mean_len:.0f} tokens per request)."]
        Path(args.out).write_text("\n".join(md) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
