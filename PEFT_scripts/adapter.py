from __future__ import annotations
import os
import logging
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdapterTuningWrapper(nn.Module):

    def __init__(self, base_mlp: nn.Module, hidden_dim: int, bottleneck_dim: int, dropout_prob: float=0.0, use_layer_norm: bool=False):
        super().__init__()
        self.base_mlp = base_mlp
        for p in self.base_mlp.parameters():
            p.requires_grad_(False)
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.up_proj.weight)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.ln = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

    def peft_placeholder_names(self) -> List[str]:
        return [n for (n, p) in self.named_parameters() if p.requires_grad]

    def peft_param_count(self) -> int:
        return sum((p.numel() for (_, p) in self.named_parameters() if p.requires_grad))

    def forward(self, hidden_states: torch.Tensor, *args, **kw):
        base_out = self.base_mlp(hidden_states, *args, **kw)
        if isinstance(base_out, tuple):
            (base_out, residue) = (base_out[0], base_out[1:])
        else:
            residue = ()
        z = self.down_proj(base_out)
        z = F.relu(z, inplace=False)
        z = self.dropout(z)
        up = self.up_proj(z)
        out = self.ln(base_out + up)
        return (out, *residue) if residue else out

class ModelAdapter(nn.Module):

    def __init__(self, base_model: nn.Module, *, bottleneck_dim: Optional[int]=None, adapter_bottleneck_dim: Optional[int]=None, adapter_dim: Optional[int]=None, adapter_dropout: float=0.0, use_layer_norm: bool=False):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = adapter_bottleneck_dim or adapter_dim
        if bottleneck_dim is None:
            raise ValueError('Provide `bottleneck_dim` (or alias).')
        self.add_module('backbone', base_model)
        self.config = base_model.config
        self.gpt_neox = base_model.gpt_neox
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        h = self.config.hidden_size
        for layer in self.gpt_neox.layers:
            layer.mlp = AdapterTuningWrapper(base_mlp=layer.mlp, hidden_dim=h, bottleneck_dim=bottleneck_dim, dropout_prob=adapter_dropout, use_layer_norm=use_layer_norm)

    def forward(self, *a, **k):
        return self.backbone(*a, **k)

    def generate(self, *a, **k):
        return self.backbone.generate(*a, **k)

    def peft_placeholder_names(self) -> List[str]:
        return [n for (n, p) in self.named_parameters() if p.requires_grad]

    def peft_param_count(self) -> int:
        return sum((p.numel() for (_, p) in self.named_parameters() if p.requires_grad))

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, 'adapter_state.safetensors'))
        self.config.save_pretrained(path)
        logging.info('Adapter checkpoint written to %s | trainable params=%d', path, self.peft_param_count())
