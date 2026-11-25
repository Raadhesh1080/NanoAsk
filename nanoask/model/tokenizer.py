"""
GPT-4 style BPE tokenizer, adapted from nanochat/tokenizer.py.

Training uses `rustbpe` (same as nanochat); inference uses `tiktoken`.
The special tokens and the chat rendering format are kept identical to nanochat so that
real nanochat checkpoints + tokenizer.pkl files can be dropped in unchanged.
"""
from __future__ import annotations

import os
import pickle
from functools import lru_cache

import tiktoken

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

# nanochat split pattern (\p{N}{1,2} instead of GPT-4's \p{N}{1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, enc: tiktoken.Encoding, bos_token: str = "<|bos|>"):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    # ---------------------------------------------------------------- construction
    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size: int) -> "Tokenizer":
        import rustbpe  # only needed for training

        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, "vocab_size too small"
        tok = rustbpe.Tokenizer()
        tok.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        ranks = {bytes(k): v for k, v in tok.get_mergeable_ranks()}
        offset = len(ranks)
        special = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(name="rustbpe", pat_str=tok.get_pattern(),
                                mergeable_ranks=ranks, special_tokens=special)
        return cls(enc)

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "Tokenizer":
        """Loads a nanochat-format tokenizer directory (tokenizer.pkl)."""
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def save(self, tokenizer_dir: str):
        os.makedirs(tokenizer_dir, exist_ok=True)
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.enc, f)

    # ---------------------------------------------------------------- basics
    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    @lru_cache(maxsize=32)
    def encode_special(self, text: str) -> int:
        return self.enc.encode_single_token(text)

    def encode(self, text: str, prepend: int | str | None = None) -> list[int]:
        ids = self.enc.encode_ordinary(text)
        if prepend is not None:
            pid = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.insert(0, pid)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    @property
    def stop_token_ids(self) -> set[int]:
        return {self.encode_special("<|assistant_end|>"), self.bos_token_id}

    # ---------------------------------------------------------------- chat format
    def render_for_completion(self, messages: list[dict]) -> list[int]:
        """
        Renders a conversation the way nanochat does and primes the assistant:
        <|bos|><|user_start|>...<|user_end|><|assistant_start|>...<|assistant_end|>...<|assistant_start|>
        A leading system message is merged into the first user message (nanochat convention).
        """
        messages = [dict(m) for m in messages]
        if messages and messages[0]["role"] == "system":
            sys_msg = messages.pop(0)
            assert messages and messages[0]["role"] == "user"
            messages[0]["content"] = sys_msg["content"] + "\n\n" + messages[0]["content"]
        ids = [self.bos_token_id]
        for i, m in enumerate(messages):
            expected = "user" if i % 2 == 0 else "assistant"
            assert m["role"] == expected, f"message {i} should be from {expected}"
            tag = m["role"]
            ids.append(self.encode_special(f"<|{tag}_start|>"))
            ids.extend(self.encode(m["content"]))
            ids.append(self.encode_special(f"<|{tag}_end|>"))
        ids.append(self.encode_special("<|assistant_start|>"))
        return ids
