"""Request/sequence state, adapted from nano-vllm/engine/sequence.py."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count


@dataclass
class SamplingParams:
    temperature: float = 0.7      # 0 => greedy (nano-vllm forbids greedy; we allow it)
    top_k: int | None = 50
    max_tokens: int = 128
    ignore_eos: bool = False
    stop_token_ids: set[int] = field(default_factory=set)


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 16
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams | None = None):
        sp = sampling_params or SamplingParams()
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0      # tokens whose K/V already sit in the cache
        self.num_scheduled_tokens = 0   # tokens to process in the current step
        self.is_prefill = True
        self.block_table: list[int] = []
        self.sampling_params = sp
        self.finish_reason: str | None = None

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
