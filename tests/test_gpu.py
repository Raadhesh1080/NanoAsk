"""
bf16 path (what the server uses on CUDA). Runs on CPU in bf16 by default, so the reduced-precision
code path is exercised everywhere; NANOASK_TEST_DEVICE=cuda runs it on a GPU.
bf16 paged and dense attention accumulate in different orders, so logits are compared with a
tolerance rather than requiring identical greedy tokens.
"""
import os

import torch

from nanoask.engine.llm_engine import EngineConfig, LLMEngine
from nanoask.engine.sequence import SamplingParams, Sequence
from nanoask.model.gpt import GPT, GPTConfig
from tests.test_engine import FakeTok

DEVICE = torch.device(os.environ.get("NANOASK_TEST_DEVICE", "cpu"))


def bf16_model():
    torch.manual_seed(0)
    m = GPT(GPTConfig(sequence_len=128, vocab_size=96, n_layer=4, n_head=4, n_kv_head=2, n_embd=64, window_pattern="SL"))
    m.init_weights()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m.to(DEVICE, dtype=torch.bfloat16).eval()


def test_bf16_paged_logits_close_to_dense():
    model = bf16_model()
    eng = LLMEngine(model, FakeTok(), EngineConfig(block_size=4, num_kvcache_blocks=32))
    assert eng.runner.kv_cache.dtype == torch.bfloat16 and eng.runner.kv_cache.device.type == DEVICE.type
    prompt = list(range(3, 30))
    with torch.no_grad():
        dense = model(torch.tensor([prompt], device=DEVICE))[0, -1]
    seq = Sequence(prompt, SamplingParams(temperature=0, max_tokens=1))
    seq.block_table = list(range(seq.num_blocks))
    seq.num_scheduled_tokens = len(prompt)
    with torch.no_grad():
        ids, pos, prev, ctx = eng.runner.prepare([seq], True)
        paged = model.compute_logits(model.forward_paged(ids, pos, prev, ctx))[-1]
    torch.testing.assert_close(paged, dense, atol=0.1, rtol=0.05)


def test_bf16_batched_generation_runs_and_frees_memory():
    model = bf16_model()
    eng = LLMEngine(model, FakeTok(), EngineConfig(block_size=4, num_kvcache_blocks=24))
    prompts = [[i + 2] * (5 + i) for i in range(6)]
    outs = eng.generate(prompts, SamplingParams(temperature=0.8, max_tokens=12, ignore_eos=True))
    assert all(len(o["token_ids"]) == 12 for o in outs)
    assert eng.scheduler.block_manager.num_free_blocks == 24
