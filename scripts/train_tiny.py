"""
Train a tiny nanochat-architecture model that answers from retrieved context, on CPU, in minutes.

This is a *demo* model so the whole stack can be exercised end-to-end without a GPU. It learns
one skill: given the RAG prompt (numbered passages + question), copy the most relevant sentence
and cite its passage number. It is trained on a corpus of real PyPI package READMEs
(scripts/build_corpus.py) and evaluated on the unrelated handwritten documents in
data/sample_docs, which it never sees (scripts/eval_qa.py checks this). It is still far weaker
than a real nanochat SFT checkpoint (d20/d32, trained on GPUs with nanochat's speedrun.sh);
pass one of those to the server with --checkpoint.

Steps (mirrors nanochat's pipeline at toy scale):
  1. train a BPE tokenizer with rustbpe            (nanochat/tokenizer.py)
  2. build synthetic RAG conversations with the real retrieval pipeline (askme-style)
  3. supervised fine-tuning with loss masked to assistant tokens (nanochat/chat_sft.py)

    python scripts/build_corpus.py
    python scripts/train_tiny.py --docs data/pypi_corpus --steps 4000 --n-layer 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nanoask.model.gpt import GPT, GPTConfig  # noqa: E402
from nanoask.model.loader import save_checkpoint  # noqa: E402
from nanoask.model.tokenizer import Tokenizer  # noqa: E402
from nanoask.rag.bm25 import STOPWORDS  # noqa: E402
from nanoask.rag.documents import load_path  # noqa: E402
from nanoask.rag.pipeline import RAGPipeline, RetrievalConfig  # noqa: E402

TEMPLATES = ["What does the document say about {}?", "{}?", "Tell me about {}.", "What about {}?",
             "Explain {}.", "Can you tell me about {}?"]
QWORDS = ["What is", "What are", "How does", "How do", "Why does", "When can", "What do", "Which",
          "Is there", "Do", "Does", "Can"]


def make_question(sent: str, words: list[str], content: list[str], rng: random.Random) -> str:
    """Mix of keyword-bag questions and natural-looking ones (question word + ordered phrase),
    so the model learns to match the question against the context rather than a fixed template."""
    r = rng.random()
    if r < 0.35:
        kw = rng.sample(content, min(len(content), rng.randint(2, 4)))
        return rng.choice(TEMPLATES).format(" ".join(kw))
    if r < 0.55 and re.search(r"\d", sent):
        # number-bearing sentence: ask for the number without revealing it
        rest = [w for w in words if not re.search(r"\d", w)]
        start = rng.randrange(max(1, len(rest) - 5))
        phrase = " ".join(rest[start:start + rng.randint(3, 6)]).lower()
        return f"{rng.choice(['How much', 'How many', 'How long', 'What is the limit for'])} {phrase}?"
    # natural: question word + a contiguous window of the sentence, with some words dropped
    start = rng.randrange(max(1, len(words) - 6))
    window = words[start:start + rng.randint(4, 8)]
    window = [w for w in window if rng.random() > 0.2] or window
    q = f"{rng.choice(QWORDS)} {' '.join(window).lower()}?"
    return q[0].upper() + q[1:]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def sentence_hash(sent: str) -> str:
    return hashlib.sha1(" ".join(sent.lower().split()).encode()).hexdigest()[:16]


def make_examples(rag: RAGPipeline, chunk_ids: list[int], rng: random.Random, k: int, ctx_chars: int,
                  n: int, per_sentence: int):
    """Sample up to n conversations from sentences of the given chunks."""
    cands = []
    for ci in chunk_ids:
        for sent in split_sentences(rag.chunks[ci].text):
            words = re.findall(r"[A-Za-z0-9\-]+", sent)
            content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 2]
            if 6 <= len(words) <= 60 and len(content) >= 3 and not sent.startswith("#"):
                cands.append((sent, words, content))
    rng.shuffle(cands)
    examples = []
    for sent, words, content in cands:
        if len(examples) >= n:
            break
        for _ in range(per_sentence):  # several paraphrased questions per sentence
            q = make_question(sent, words, content, rng)
            hits = rag.retrieve(q, k)
            rng.shuffle(hits)  # citation number must be read from the context, not memorised
            idx = next((i for i, h in enumerate(hits, 1) if sent in h.chunk.text), None)
            if idx is None:
                continue
            msgs = rag.build_messages(q, hits, max_context_chars=ctx_chars)
            if sent not in msgs[-1]["content"]:
                continue
            examples.append((msgs, f"{sent} [{idx}]"))
    return examples


def render(tok: Tokenizer, msgs, answer, max_len):
    prompt = tok.render_for_completion(msgs)
    ans = tok.encode(answer) + [tok.encode_special("<|assistant_end|>")]
    ids = (prompt + ans)[:max_len + 1]
    mask = ([0] * len(prompt) + [1] * len(ans))[:max_len + 1]
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/pypi_corpus")
    ap.add_argument("--out", default="checkpoints/tiny")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=192)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ctx-chars", type=int, default=1800)
    ap.add_argument("--q-per-sentence", type=int, default=3)
    ap.add_argument("--num-examples", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    rng = random.Random(0)
    torch.manual_seed(0)

    files = sorted(p for p in Path(args.docs).glob("*") if p.is_file())
    texts = [load_path(p)[1] for p in files]
    rag = RAGPipeline(retrieval=RetrievalConfig(final_k=args.k))
    for p in files:
        rag.ingest_file(p.name, p.read_bytes())
    print(f"{len(files)} docs, {len(rag.chunks)} chunks")

    # 1. tokenizer (train on docs + the prompt template so the RAG scaffolding is cheap to encode)
    t0 = time.time()
    scaffold = rag.build_messages("question", [], max_context_chars=10)[0]["content"]
    tok = Tokenizer.train_from_iterator(iter(texts * 3 + [scaffold] * 50), args.vocab)
    print(f"tokenizer: vocab {tok.get_vocab_size()} in {time.time() - t0:.1f}s")

    # 2. data: validation uses held-out *documents*, so val loss measures generalisation
    doc_ids = sorted({c.doc_id for c in rag.chunks})
    rng.shuffle(doc_ids)
    val_docs = set(doc_ids[: max(1, len(doc_ids) // 10)])
    tr_chunks = [i for i, c in enumerate(rag.chunks) if c.doc_id not in val_docs]
    va_chunks = [i for i, c in enumerate(rag.chunks) if c.doc_id in val_docs]
    t0 = time.time()
    train = make_examples(rag, tr_chunks, rng, args.k, args.ctx_chars, args.num_examples, args.q_per_sentence)
    val = make_examples(rag, va_chunks, rng, args.k, args.ctx_chars, max(64, args.num_examples // 20), 1)
    lens = [len(render(tok, m, a, 10**9)[0]) for m, a in train[:200]]
    print(f"{len(train)} train / {len(val)} val (held-out docs) conversations, "
          f"mean {sum(lens) / len(lens):.0f} tokens, built in {time.time() - t0:.0f}s")

    # fingerprints of every training sentence, so eval can prove the test docs were never seen
    os.makedirs(args.out, exist_ok=True)
    hashes = sorted({sentence_hash(s) for t in texts for s in split_sentences(t)})
    with open(os.path.join(args.out, "train_sentence_hashes.json"), "w") as f:
        json.dump({"docs_dir": args.docs, "num_docs": len(files), "hashes": hashes}, f)

    # 3. model (nanochat architecture, GQA + sliding windows) + masked SFT
    cfg = GPTConfig(sequence_len=args.seq_len, vocab_size=tok.get_vocab_size(), n_layer=args.n_layer,
                    n_head=args.n_embd // 64, n_kv_head=max(1, args.n_embd // 128), n_embd=args.n_embd,
                    window_pattern="SL")
    model = GPT(cfg)
    model.init_weights()
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {cfg}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def batch(pool):
        rows = [render(tok, *rng.choice(pool), args.seq_len) for _ in range(args.batch)]
        T = max(len(r[0]) for r in rows) - 1
        x = torch.zeros(args.batch, T, dtype=torch.long)
        y = torch.full((args.batch, T), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(rows):
            n = len(ids) - 1
            x[i, :n] = torch.tensor(ids[:-1])
            tgt = torch.tensor(ids[1:])
            y[i, :n] = torch.where(torch.tensor(mask[1:]) == 1, tgt, torch.full_like(tgt, -1))
        return x, y

    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / 50) * (1 - 0.9 * step / args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch(train)
        loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            with torch.no_grad():
                vl = model(*batch(val)).item()
            print(f"step {step:5d} | loss {loss.item():.3f} | val {vl:.3f} | lr {lr:.2e} | {time.time() - t0:.0f}s", flush=True)
        if step % 500 == 0 or step == args.steps:
            for old in Path(args.out).glob("*_0*.*"):  # keep only the newest checkpoint
                old.unlink()
            save_checkpoint(args.out, model, tok, step, {"train": vars(args)})
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
