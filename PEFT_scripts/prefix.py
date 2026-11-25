from __future__ import annotations
import os
import logging
from typing import Any, Dict, List
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput

class ModelPrefixOnly(nn.Module):

    def __init__(self, backbone: nn.Module, *, prefix_length: int=10, embed_dim: int | None=None, dropout_prob: float=0.0):
        super().__init__()
        self.add_module('backbone', backbone)
        self.config = backbone.config
        self.gpt_neox = backbone.gpt_neox
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        embed_dim = embed_dim or self.config.hidden_size
        self.prefix_length = int(prefix_length)
        self.prefix_emb = nn.Parameter(torch.empty(self.prefix_length, embed_dim))
        nn.init.normal_(self.prefix_emb, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()

    def peft_placeholder_names(self) -> List[str]:
        return [n for (n, p) in self.named_parameters() if p.requires_grad]

    def peft_param_count(self) -> int:
        return sum((p.numel() for (_, p) in self.named_parameters() if p.requires_grad))

    def forward(self, *, input_ids: torch.LongTensor | None=None, attention_mask: torch.Tensor | None=None, labels: torch.LongTensor | None=None, **kwargs: Any) -> CausalLMOutput:
        if input_ids is None:
            raise ValueError('`input_ids` must be provided')
        single_sample = input_ids.dim() == 1
        if single_sample:
            input_ids = input_ids.unsqueeze(0)
        (B, T) = input_ids.shape
        device = input_ids.device
        dtype = self.prefix_emb.dtype
        prefix = self.dropout(self.prefix_emb).to(dtype).unsqueeze(0).expand(B, -1, -1)
        tok_emb = self.backbone.gpt_neox.embed_in(input_ids).to(dtype)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        pref_mask = torch.ones(B, self.prefix_length, dtype=torch.long, device=device)
        attention_mask = torch.cat([pref_mask, attention_mask], dim=1)
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            pref_lbl = labels.new_full((B, self.prefix_length), -100)
            labels = torch.cat([pref_lbl, labels], dim=1)
        if kwargs.get('position_ids', None) is None:
            pos_pref = torch.arange(self.prefix_length, device=device).unsqueeze(0)
            pos_real = torch.arange(T, device=device).unsqueeze(0) + self.prefix_length
            kwargs['position_ids'] = torch.cat([pos_pref.expand(B, -1), pos_real.expand(B, -1)], dim=1)
        kwargs.pop('input_ids', None)
        kwargs.pop('inputs_embeds', None)
        kwargs.pop('attention_mask', None)
        kwargs.pop('labels', None)
        out = self.backbone(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)
        if single_sample:
            out.loss = None if out.loss is None else out.loss.squeeze(0)
            out.logits = out.logits.squeeze(0)
            if out.hidden_states is not None:
                out.hidden_states = tuple((h.squeeze(0) for h in out.hidden_states))
            if out.attentions is not None:
                out.attentions = tuple((a.squeeze(0) for a in out.attentions))
        return out

    def generate(self, *args: Any, **kwargs: Any):
        return self.backbone.generate(*args, **kwargs)

    def save_pretrained(self, path: str | os.PathLike) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, 'prefix_state.safetensors'))
        self.config.save_pretrained(path)
        logging.info('Prefix checkpoint saved to %s | trainable params=%d', path, self.peft_param_count())
