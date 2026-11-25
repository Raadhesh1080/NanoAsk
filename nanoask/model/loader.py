"""
Load a nanochat checkpoint (or a nanoask tiny checkpoint, which uses the same format).

nanochat layout (see nanochat/checkpoint_manager.py):
    <ckpt_dir>/model_<step:06d>.pt     state dict (keys may carry a "_orig_mod." prefix)
    <ckpt_dir>/meta_<step:06d>.json    {"model_config": {...GPTConfig kwargs...}, ...}
    <tokenizer_dir>/tokenizer.pkl       pickled tiktoken Encoding
e.g. ~/.cache/nanochat/chatsft_checkpoints/d20 and ~/.cache/nanochat/tokenizer
"""
from __future__ import annotations

import json
import os
import re

import torch

from nanoask.model.gpt import GPT, GPTConfig
from nanoask.model.tokenizer import Tokenizer


def pick_device(device: str | None = None) -> torch.device:
    if device:
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"--device {device} requested, but no CUDA GPU is available to PyTorch "
                               f"(torch {torch.__version__}). On Colab, switch the runtime to a T4 GPU.")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_last_step(ckpt_dir: str) -> int:
    steps = [int(m.group(1)) for f in os.listdir(ckpt_dir) if (m := re.match(r"model_(\d+)\.pt$", f))]
    if not steps:
        raise FileNotFoundError(f"no model_*.pt in {ckpt_dir}")
    return max(steps)


def load_checkpoint(ckpt_dir: str, tokenizer_dir: str | None = None, step: int | None = None,
                    device: str | None = None) -> tuple[GPT, Tokenizer, dict]:
    dev = pick_device(device)
    step = find_last_step(ckpt_dir) if step is None else step
    state = torch.load(os.path.join(ckpt_dir, f"model_{step:06d}.pt"), map_location="cpu")
    with open(os.path.join(ckpt_dir, f"meta_{step:06d}.json")) as f:
        meta = json.load(f)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    cfg_kwargs = dict(meta["model_config"])
    cfg_kwargs.setdefault("window_pattern", "L")      # old nanochat checkpoints
    cfg = GPTConfig(**{k: v for k, v in cfg_kwargs.items() if k in GPTConfig.__dataclass_fields__})
    state.setdefault("resid_lambdas", torch.ones(cfg.n_layer))
    state.setdefault("x0_lambdas", torch.zeros(cfg.n_layer))

    # bf16 on CUDA (what nanochat trains in), fp32 elsewhere
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = GPT(cfg)
    model.load_state_dict({k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()}, strict=True)
    model.to(device=dev, dtype=dtype).eval()

    tok_dir = tokenizer_dir or os.path.join(ckpt_dir, "tokenizer")
    if not os.path.exists(os.path.join(tok_dir, "tokenizer.pkl")):
        tok_dir = os.path.join(os.path.expanduser("~/.cache/nanochat"), "tokenizer")
    tokenizer = Tokenizer.from_directory(tok_dir)
    assert tokenizer.get_vocab_size() == cfg.vocab_size, \
        f"tokenizer vocab {tokenizer.get_vocab_size()} != model vocab {cfg.vocab_size}"
    return model, tokenizer, meta


def save_checkpoint(ckpt_dir: str, model: GPT, tokenizer: Tokenizer, step: int, extra: dict | None = None):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_{step:06d}.pt"))
    meta = {"model_config": model.config.__dict__, "step": step, **(extra or {})}
    with open(os.path.join(ckpt_dir, f"meta_{step:06d}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    tokenizer.save(os.path.join(ckpt_dir, "tokenizer"))
