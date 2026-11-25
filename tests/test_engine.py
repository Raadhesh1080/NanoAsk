"""Engine correctness: paged + batched inference must match nanochat's dense forward."""
import os

import pytest
import torch

from nanoask.engine.block_manager import BlockManager
from nanoask.engine.llm_engine import EngineConfig, LLMEngine
from nanoask.engine.sequence import SamplingParams, Sequence
from nanoask.model.gpt import GPT, GPTConfig

# NANOASK_TEST_DEVICE=cuda runs these equivalence tests on a GPU (see notebooks/gpu_benchmark.ipynb)
DEVICE = torch.device(os.environ.get("NANOASK_TEST_DEVICE", "cpu"))


class FakeTok:
    """Minimal tokenizer stand-in so engine tests don't depend on BPE training."""
    bos_token_id = 0
    stop_token_ids = {0, 1}

    def encode_special(self, s):
        return 1

    def decode(self, ids):
        return " ".join(map(str, ids))


def tiny_model(window_pattern="SL", n_kv_head=2, seq_len=128):
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=seq_len, vocab_size=96, n_layer=4, n_head=4, n_kv_head=n_kv_head,
                    n_embd=64, window_pattern=window_pattern)
    m = GPT(cfg)
    m.init_weights()
    # nanochat zero-inits projections; randomise everything so the test is meaningful
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        m.smear_lambda.fill_(0.5)
    return m.to(DEVICE).eval()


def greedy_dense(model, prompt, n):
    ids = list(prompt)
    for _ in range(n):
        logits = model(torch.tensor([ids], device=DEVICE))[0, -1]
        ids.append(int(logits.argmax()))
    return ids[len(prompt):]


@pytest.mark.parametrize("pattern", ["L", "SL"])
def test_paged_batched_matches_dense(pattern):
    # seq_len=128 -> short window = 128 so pattern only matters if windows < context; use small seq_len
    model = tiny_model(pattern, seq_len=128)
    model.window_sizes = [8 if w < 128 else w for w in model.window_sizes]  # force a tiny sliding window
    prompts = [[2, 5, 7, 9, 11, 3, 4], [10, 20, 30], list(range(2, 40)), [5, 5, 5, 5, 5, 6]]
    n_new = 12
    expected = [greedy_dense(model, p, n_new) for p in prompts]

    eng = LLMEngine(model, FakeTok(), EngineConfig(block_size=4, num_kvcache_blocks=64, max_num_batched_tokens=16))
    sp = SamplingParams(temperature=0, max_tokens=n_new, ignore_eos=True)
    outs = eng.generate(prompts, sp)
    assert [o["token_ids"] for o in outs] == expected
    # chunked prefill (budget 16 < 38-token prompt) and batching both happened
    assert eng.stats.max_batch_size == len(prompts)


def test_logits_close():
    model = tiny_model("L")
    prompt = list(range(3, 30))
    dense = model(torch.tensor([prompt], device=DEVICE))[0, -1]
    eng = LLMEngine(model, FakeTok(), EngineConfig(block_size=4, num_kvcache_blocks=32))
    seq = Sequence(prompt, SamplingParams(temperature=0, max_tokens=1))
    seq.block_table = list(range(seq.num_blocks))
    seq.num_scheduled_tokens = len(prompt)
    ids, pos, prev, ctx = eng.runner.prepare([seq], True)
    paged = model.compute_logits(model.forward_paged(ids, pos, prev, ctx))[-1]
    torch.testing.assert_close(paged, dense, atol=1e-4, rtol=1e-4)


def test_prefix_cache_reuse_and_correctness():
    model = tiny_model("L")
    shared = list(range(2, 34))            # 32 tokens = 8 full blocks of 4
    p1, p2 = shared + [40, 41], shared + [50, 51, 52]
    sp = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
    eng = LLMEngine(model, FakeTok(), EngineConfig(block_size=4, num_kvcache_blocks=64))
    eng.generate([p1], sp)
    out2 = eng.generate([p2], sp)[0]["token_ids"]
    assert eng.scheduler.block_manager.prefix_hits >= 8
    assert out2 == greedy_dense(model, p2, 6)


def test_preemption_under_memory_pressure():
    model = tiny_model("L")
    prompts = [[i + 2] * 6 for i in range(6)]
    sp = SamplingParams(temperature=0, max_tokens=20, ignore_eos=True)
    # 6 seqs * 26 tokens / block 4 = ~42 blocks needed, give it 16
    eng = LLMEngine(model, FakeTok(), EngineConfig(block_size=4, num_kvcache_blocks=16))
    outs = eng.generate(prompts, sp)
    assert eng.scheduler.num_preemptions > 0
    for p, o in zip(prompts, outs):
        assert o["token_ids"] == greedy_dense(model, p, 20)
    assert eng.scheduler.block_manager.num_free_blocks == 16  # everything freed


def test_block_manager_refcounts():
    bm = BlockManager(num_blocks=8, block_size=4)
    Sequence.block_size = 4
    a = Sequence(list(range(10)))
    assert bm.can_allocate(a) == 0
    bm.allocate(a, 0)
    a.num_scheduled_tokens = 10
    bm.hash_blocks(a)
    b = Sequence(list(range(10)))
    assert bm.can_allocate(b) == 2       # two full blocks shared
    bm.allocate(b, 2)
    assert bm.blocks[a.block_table[0]].ref_count == 2
    bm.deallocate(a)
    bm.deallocate(b)
    assert bm.num_free_blocks == 8
