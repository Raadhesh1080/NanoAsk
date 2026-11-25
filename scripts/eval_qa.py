"""
End-to-end QA evaluation on hand-written questions (not produced by the training generator).

Reports, per question and overall:
  retrieval@1   the passage holding the answer is ranked first
  retrieval@k   ...is anywhere in the prompt
  llm           the raw model answer contains the expected fact
  auto          the served answer (grounding check + extractive fallback) contains it
  extractive    the model-free answer contains it
  grounded      the raw model answer passed the grounding check

    python scripts/eval_qa.py                         # uses checkpoints/tiny
    python scripts/eval_qa.py --checkpoint ~/.cache/nanochat/chatsft_checkpoints/d20 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nanoask.engine.llm_engine import EngineConfig, LLMEngine  # noqa: E402
from nanoask.engine.sequence import SamplingParams  # noqa: E402
from nanoask.model.loader import load_checkpoint  # noqa: E402
from nanoask.rag.pipeline import RAGPipeline, RetrievalConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# (question, substring that a correct answer must contain)
QA = [
    ("How much is the hotel reimbursement per night?", "180 euros"),
    ("How many sick days do employees get?", "10 paid sick days"),
    ("How many vacation days do full-time staff receive each year?", "24 days"),
    ("When can employees work remotely?", "three days per week"),
    ("What is the daily meal allowance when travelling?", "45 euros"),
    ("How often are engineer laptops replaced?", "every three years"),
    ("How much is the home office stipend?", "600 euros"),
    ("What does chunked prefill do?", "several pieces"),
    ("Why does the KV cache use so much memory?", "exhausts GPU memory"),
    ("How does prefix caching find reusable blocks?", "hash"),
    ("What happens when the scheduler runs out of memory?", "preempts"),
    ("What does reciprocal rank fusion compute?", "1 / (k + rank)"),
    ("What is BM25 good at?", "names, codes and rare keywords"),
    ("What does a reranker do?", "reads the question and passage together"),
    ("What is group-query attention?", "shares each key/value head"),
    ("How do rotary position embeddings work?", "rotate pairs"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "tiny"))
    ap.add_argument("--tokenizer-dir")
    ap.add_argument("--device")
    ap.add_argument("--docs", default=str(ROOT / "data" / "sample_docs"))
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    model, tok, meta = load_checkpoint(args.checkpoint, args.tokenizer_dir, device=args.device)
    train = meta.get("train", {})
    k, ctx_chars = train.get("k", 4), train.get("ctx_chars", 6000)
    rag = RAGPipeline(retrieval=RetrievalConfig(final_k=k))
    for f in sorted(Path(args.docs).glob("*")):
        if f.is_file():
            rag.ingest_file(f.name, f.read_bytes())
    engine = LLMEngine(model, tok, EngineConfig(max_model_len=model.config.sequence_len))

    hits = [rag.retrieve(q) for q, _ in QA]
    prompts = []
    for (q, _), h in zip(QA, hits):
        ids = tok.render_for_completion(rag.build_messages(q, h, max_context_chars=ctx_chars))
        prompts.append(ids[-(engine.config.max_model_len - args.max_tokens - 1):])
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens, stop_token_ids=tok.stop_token_ids)
    outs = engine.generate(prompts, sp)   # all questions in one continuous batch

    keys = ["retrieval@1", "retrieval@k", "llm", "auto", "extractive", "grounded"]
    totals = dict.fromkeys(keys, 0)
    print(f"{'question':58s} " + " ".join(f"{c[:6]:>6s}" for c in keys))
    for (q, fact), h, o in zip(QA, hits, outs):
        raw = o["text"].strip()
        ext = rag.extractive_answer(q, h)
        g = rag.ground_answer(q, raw, h)
        auto = (g["answer"] if g["grounded"] else ext)
        row = {
            "retrieval@1": bool(h) and fact in h[0].chunk.text,
            "retrieval@k": any(fact in x.chunk.text for x in h),
            "llm": fact in raw, "auto": fact in auto, "extractive": fact in ext, "grounded": g["grounded"],
        }
        for c in keys:
            totals[c] += row[c]
        print(f"{q[:58]:58s} " + " ".join(f"{'✓' if row[c] else '·':>6s}" for c in keys))
        if args.verbose:
            print(f"    llm : {raw[:150]}\n    auto: {auto[:150]}")
    n = len(QA)
    print(f"{'TOTAL':58s} " + " ".join(f"{totals[c] / n:6.0%}" for c in keys))


if __name__ == "__main__":
    main()
