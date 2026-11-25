from __future__ import annotations
import math
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRAQKV(nn.Module):

    def __init__(self, base_linear: nn.Linear, rank: int=8, alpha: float=16.0) -> None:
        super().__init__()
        self.register_parameter('base_weight', base_linear.weight)
        if base_linear.bias is not None:
            self.register_parameter('base_bias', base_linear.bias)
        else:
            self.base_bias = None
        self.base_weight.requires_grad_(False)
        if self.base_bias is not None:
            self.base_bias.requires_grad_(False)
        (out_dim, in_dim) = self.base_weight.shape
        if out_dim % in_dim != 0 or out_dim // in_dim != 3:
            raise ValueError('Expected merged QKV of shape (3h, h).')
        self.h = in_dim
        self.rank = rank
        self.scale = alpha / rank

        def _pair() -> tuple[nn.Parameter, nn.Parameter]:
            B = nn.Parameter(torch.empty(self.h, rank))
            A = nn.Parameter(torch.empty(rank, self.h))
            nn.init.kaiming_uniform_(B, a=math.sqrt(5))
            nn.init.zeros_(A)
            return (A, B)
        (self.A_q, self.B_q) = _pair()
        (self.A_k, self.B_k) = _pair()
        (self.A_v, self.B_v) = _pair()

    @staticmethod
    def _matmul_cast(x_2d: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.dtype != x_2d.dtype:
            A = A.to(x_2d.dtype)
        if B.dtype != x_2d.dtype:
            B = B.to(x_2d.dtype)
        return x_2d @ B @ A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (*prefix, h) = x.shape
        if h != self.h:
            raise ValueError(f'Expected last-dim = {self.h}, got {h}')
        x_2d = x.reshape(-1, h)
        lora_q = self._matmul_cast(x_2d, self.A_q, self.B_q).reshape(*prefix, self.h)
        lora_k = self._matmul_cast(x_2d, self.A_k, self.B_k).reshape(*prefix, self.h)
        lora_v = self._matmul_cast(x_2d, self.A_v, self.B_v).reshape(*prefix, self.h)
        W0 = self.base_weight.to(x.dtype)
        b0 = self.base_bias.to(x.dtype) if self.base_bias is not None else None
        base_out = F.linear(x, W0, b0).view(*prefix, 3, self.h)
        lora_out = torch.stack((lora_q, lora_k, lora_v), dim=-2) * self.scale
        return (base_out + lora_out).reshape(*prefix, 3 * self.h)

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, 'lora_state.safetensors')
        torch.save({k: v.cpu() for (k, v) in self.state_dict().items() if k.startswith(('A_', 'B_'))}, path)
        logging.info('LoRA state saved → %s', path)
