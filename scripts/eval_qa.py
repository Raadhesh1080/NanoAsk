"""
End-to-end QA evaluation on the handwritten question set (data/eval/qa.jsonl, 88 questions over
the 8 documents in data/sample_docs).

Leak check: if the checkpoint directory has train_sentence_hashes.json (written by
scripts/train_tiny.py), every sentence of the test documents is checked against the training
sentences and the overlap is printed. A non-zero overlap means the score is contaminated.

Columns (fraction of questions whose answer contains the expected fact):
  ret@1      the top retrieved passage contains the fact
  ret@k      some passage in the prompt contains it
  llm        raw model answer
  auto       served answer, strict rule (model answer kept only if >= as relevant as extraction)
  auto0.5    served answer with the old lenient rule (model answer kept if >= half as relevant)
  ext@1      model-free extractive answer, first sentence only (same length as a model answer)
  ext@3      model-free extractive answer, up to 3 sentences (the server's default)
  grounded   raw model answer passed the strict grounding check

    python scripts/eval_qa.py
    python scripts/eval_qa.py --distractors data/pypi_corpus      # 225 unrelated docs in the index too
    python scripts/eval_qa.py --checkpoint ~/.cache/nanochat/chatsft_checkpoints/d20 --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nanoask.engine.llm_engine import EngineConfig, LLMEngine  # noqa: E402
from nanoask.engine.sequence import SamplingParams  # noqa: E402
from nanoask.model.loader import load_checkpoint  # noqa: E402
from nanoask.rag.pipeline import RAGPipeline, RetrievalConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLS = ["ret@1", "ret@k", "llm", "auto", "auto0.5", "ext@1", "ext@3", "grounded"]


def sentence_hash(sent: str) -> str:  # must match scripts/train_tiny.py
    return hashlib.sha1(" ".join(sent.lower().split()).encode()).hexdigest()[:16]


def leak_check(ckpt: str, docs_dir: Path) -> str:
    f = Path(ckpt) / "train_sentence_hashes.json"
    if not f.exists():
        return "UNKNOWN (checkpoint has no train_sentence_hashes.json; cannot prove the test docs were unseen)"
    train = set(json.loads(f.read_text())["hashes"])
    sents = [s.strip() for p in docs_dir.glob("*") if p.is_file()
             for s in re.split(r"(?<=[.!?])\s+|\n+", p.read_text()) if len(s.split()) >= 5]
    overlap = sum(sentence_hash(s) in train for s in sents)
    verdict = "CLEAN" if overlap == 0 else "CONTAMINATED"
    return f"{verdict}: {overlap} of {len(sents)} test-document sentences occur in the training data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "tiny"))
    ap.add_argument("--tokenizer-dir")
    ap.add_argument("--device")
    ap.add_argument("--docs", default=str(ROOT / "data" / "sample_docs"))
    ap.add_argument("--qa", default=str(ROOT / "data" / "eval" / "qa.jsonl"))
    ap.add_argument("--distractors", help="extra unrelated documents to add to the index")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--json", help="write per-question results here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    qa = [json.loads(l) for l in open(args.qa) if l.strip()]
    print(f"Leak check: {leak_check(args.checkpoint, Path(args.docs))}")
    model, tok, meta = load_checkpoint(args.checkpoint, args.tokenizer_dir, device=args.device)
    train = meta.get("train", {})
    k, ctx_chars = train.get("k", 4), train.get("ctx_chars", 6000)
    rag = RAGPipeline(retrieval=RetrievalConfig(final_k=k))
    files = [f for f in sorted(Path(args.docs).glob("*")) if f.is_file()]
    if args.distractors:
        files += [f for f in sorted(Path(args.distractors).glob("*")) if f.is_file()]
    rag.add_chunks([c for f in files for c in _chunks(rag, f)])
    print(f"index: {len(files)} documents, {len(rag.chunks)} chunks | model: {args.checkpoint} | k={k}")
    engine = LLMEngine(model, tok, EngineConfig(max_model_len=model.config.sequence_len))

    hits = [rag.retrieve(r["q"]) for r in qa]
    prompts = []
    for r, h in zip(qa, hits):
        ids = tok.render_for_completion(rag.build_messages(r["q"], h, max_context_chars=ctx_chars))
        prompts.append(ids[-(engine.config.max_model_len - args.max_tokens - 1):])
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens, stop_token_ids=tok.stop_token_ids)
    outs = engine.generate(prompts, sp)  # all questions in one continuous batch

    rows, by_doc = [], defaultdict(lambda: defaultdict(int))
    for r, h, o in zip(qa, hits, outs):
        fact, raw = r["a"], o["text"].strip()
        ext3 = rag.extractive_answer(r["q"], h)
        ext1 = rag.extractive_answer(r["q"], h, max_sentences=1)
        strict = rag.ground_answer(r["q"], raw, h, 1.0)
        lenient = rag.ground_answer(r["q"], raw, h, 0.5)
        auto = strict["answer"] if strict["grounded"] else ext3
        auto05 = lenient["answer"] if lenient["grounded"] else ext3
        row = {"ret@1": bool(h) and fact in h[0].chunk.text, "ret@k": any(fact in x.chunk.text for x in h),
               "llm": fact in raw, "auto": fact in auto, "auto0.5": fact in auto05,
               "ext@1": fact in ext1, "ext@3": fact in ext3, "grounded": strict["grounded"]}
        rows.append({**r, **row, "llm_answer": raw, "auto_answer": auto})
        by_doc[r["doc"]]["n"] += 1
        for c in COLS:
            by_doc[r["doc"]][c] += row[c]
        if args.verbose:
            print(f"{r['q']}\n    llm : {raw[:140]}\n    auto: {auto[:140]}")

    n = len(rows)
    print(f"\n{'document':22s} {'n':>3s} " + " ".join(f"{c:>8s}" for c in COLS))
    for d, s in sorted(by_doc.items()):
        print(f"{d:22s} {s['n']:3d} " + " ".join(f"{s[c] / s['n']:8.0%}" for c in COLS))
    print(f"{'TOTAL':22s} {n:3d} " + " ".join(f"{sum(r[c] for r in rows) / n:8.0%}" for c in COLS))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


def _chunks(rag, path):
    from nanoask.rag.documents import load_path, make_chunks
    title, text = load_path(path)
    return make_chunks(title, text, rag.chunker, path.name)


if __name__ == "__main__":
    main()
