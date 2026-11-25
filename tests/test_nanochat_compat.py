"""
Compatibility with the real nanochat code base.

Builds a model with nanochat's own GPT class, saves it with nanochat's own checkpoint_manager,
trains + saves a tokenizer with nanochat's own RustBPETokenizer, then loads both through
nanoask and checks that logits, greedy generations and chat rendering match exactly.

Needs a checkout of https://github.com/karpathy/nanochat; set NANOCHAT_REPO or place it at
../refs/nanochat. Skipped otherwise.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(os.environ.get("NANOCHAT_REPO", Path(__file__).resolve().parents[2] / "refs" / "nanochat"))
pytestmark = pytest.mark.skipif(not (REPO / "nanochat" / "gpt.py").exists(), reason="nanochat checkout not found")


@pytest.fixture(scope="module")
def nanochat_ckpt(tmp_path_factory):
    sys.path.insert(0, str(REPO))
    from nanochat.checkpoint_manager import save_checkpoint
    from nanochat.common import COMPUTE_DTYPE
    from nanochat.gpt import GPT as NCGPT, GPTConfig as NCConfig
    from nanochat.tokenizer import RustBPETokenizer

    assert COMPUTE_DTYPE == torch.float32, "test assumes nanochat runs in fp32 on CPU"
    out = tmp_path_factory.mktemp("nanochat")
    text = (Path(__file__).resolve().parents[1] / "data" / "sample_docs" / "rag.md").read_text()
    nc_tok = RustBPETokenizer.train_from_iterator(iter([text] * 5), 600)
    nc_tok.save(str(out / "tokenizer"))

    cfg = NCConfig(sequence_len=256, vocab_size=nc_tok.get_vocab_size(), n_layer=4, n_head=4, n_kv_head=2,
                   n_embd=64, window_pattern="SL")
    torch.manual_seed(0)
    with torch.device("meta"):
        model = NCGPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    with torch.no_grad():  # nanochat zero-inits projections; make every weight matter
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    model.eval()
    # simulate a torch.compile'd training run: nanochat checkpoints carry this prefix
    state = {"_orig_mod." + k: v for k, v in model.state_dict().items()}
    save_checkpoint(str(out / "d4"), 1234, state, None, {"model_config": cfg.__dict__, "step": 1234})
    yield out, model, nc_tok
    sys.path.remove(str(REPO))


def test_load_nanochat_checkpoint_matches_logits(nanochat_ckpt):
    from nanoask.model.loader import load_checkpoint

    out, nc_model, nc_tok = nanochat_ckpt
    model, tok, meta = load_checkpoint(str(out / "d4"), str(out / "tokenizer"), device="cpu")
    assert meta["step"] == 1234
    ids = torch.tensor([nc_tok.encode("Hybrid search runs BM25 and dense retrieval", prepend="<|bos|>")])
    with torch.no_grad():
        torch.testing.assert_close(model(ids), nc_model(ids), atol=1e-5, rtol=1e-5)


def test_engine_generation_matches_nanochat(nanochat_ckpt):
    from nanoask.engine.llm_engine import EngineConfig, LLMEngine
    from nanoask.engine.sequence import SamplingParams
    from nanoask.model.loader import load_checkpoint

    out, nc_model, nc_tok = nanochat_ckpt
    model, tok, _ = load_checkpoint(str(out / "d4"), str(out / "tokenizer"), device="cpu")
    prompts = ["Reciprocal rank fusion", "HNSW graphs make search", "A reranker reads the question"]
    ids = [nc_tok.encode(p, prepend="<|bos|>") for p in prompts]
    expected = [list(nc_model.generate(i, max_tokens=10, temperature=0)) for i in ids]
    eng = LLMEngine(model, tok, EngineConfig(block_size=4, num_kvcache_blocks=64))
    got = eng.generate(ids, SamplingParams(temperature=0, max_tokens=10, ignore_eos=True))
    assert [g["token_ids"] for g in got] == expected


def test_chat_rendering_matches_nanochat(nanochat_ckpt):
    from nanoask.model.tokenizer import Tokenizer

    out, _, nc_tok = nanochat_ckpt
    tok = Tokenizer.from_directory(str(out / "tokenizer"))
    conv = {"messages": [{"role": "system", "content": "Be brief."},
                         {"role": "user", "content": "What is BM25?"},
                         {"role": "assistant", "content": "A ranking function."},
                         {"role": "user", "content": "And HNSW?"},
                         {"role": "assistant", "content": ""}]}
    assert tok.render_for_completion(conv["messages"][:-1]) == nc_tok.render_for_completion(conv)
