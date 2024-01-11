from typing import Optional
from vllm._C import ops
import torch


def self_extend_forward(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 1024,
):
    grouped_positions = (positions - group_size_2) // group_size_1 + group_size_2
    final_positions = torch.where(
        positions < group_size_2, positions, grouped_positions
    )
    ops.rotary_embedding(
        final_positions,
        query,
        key,
        self.head_size,
        self.cos_sin_cache,
        self.is_neox_style,
    )
    return query, key
