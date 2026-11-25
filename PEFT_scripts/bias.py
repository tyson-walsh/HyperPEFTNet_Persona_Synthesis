from __future__ import annotations
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple
import torch
import torch.nn as nn

def _is_bias_vector(name: str, p: torch.nn.Parameter) -> bool:
    return name.endswith('.bias') and p.ndim == 1

def count_trainable_params(module: nn.Module) -> int:
    return sum((int(p.numel()) for p in module.parameters() if p.requires_grad))

def strip_backbone_prefix(name: str) -> str:
    return name[len('backbone.'):] if name.startswith('backbone.') else name

def add_backbone_prefix(name: str) -> str:
    return name if name.startswith('backbone.') else f'backbone.{name}'

class BiasTuningModel(nn.Module):

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.add_module('backbone', base_model)
        self.config = getattr(base_model, 'config', None)
        self.gpt_neox = getattr(base_model, 'gpt_neox', None)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        n_trainable_tensors = 0
        for (name, p) in self.backbone.named_parameters():
            if _is_bias_vector(name, p):
                p.requires_grad_(True)
                n_trainable_tensors += 1
        if n_trainable_tensors == 0:
            raise RuntimeError('BiasTuningModel: found 0 trainable bias tensors. Backbone may not match expected HF GPT-NeoX/Pythia structure.')

    def forward(self, *args: Any, **kwargs: Any):
        return self.backbone(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any):
        return self.backbone.generate(*args, **kwargs)

    def __getattr__(self, name: str):
        if name not in {'backbone', 'config', 'gpt_neox'} and 'backbone' in self.__dict__.get('_modules', {}):
            bb = self.__dict__['_modules']['backbone']
            if hasattr(bb, name):
                return getattr(bb, name)
        return super().__getattr__(name)

    def peft_placeholder_names(self) -> List[str]:
        return [n for (n, p) in self.named_parameters() if p.requires_grad]

    def peft_param_count(self) -> int:
        return sum((int(p.numel()) for (_, p) in self.named_parameters() if p.requires_grad))

    def peft_state_dict(self) -> 'OrderedDict[str, torch.Tensor]':
        out: 'OrderedDict[str, torch.Tensor]' = OrderedDict()
        for (name, p) in self.named_parameters():
            if p.requires_grad:
                out[name] = p.detach().cpu()
        return out

    def load_peft_state_dict(self, sd: Mapping[str, torch.Tensor], *, strict_shapes: bool=True) -> Tuple[List[str], List[str]]:
        expected = {n: p for (n, p) in self.named_parameters() if p.requires_grad}
        missing: List[str] = []
        unexpected: List[str] = []
        for (k, v) in sd.items():
            if k not in expected:
                unexpected.append(k)
                continue
            tgt = expected[k]
            if strict_shapes and tuple(v.shape) != tuple(tgt.shape):
                raise RuntimeError(f'Shape mismatch for {k}: ckpt={tuple(v.shape)} model={tuple(tgt.shape)}')
            with torch.no_grad():
                tgt.copy_(v.to(dtype=tgt.dtype, device=tgt.device))
        for k in expected.keys():
            if k not in sd:
                missing.append(k)
        return (missing, unexpected)

    def save_pretrained(self, path: str, *, safe_serialization: bool=True) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        placeholders = self.peft_state_dict()
        if len(placeholders) == 0:
            raise RuntimeError('Refusing to save: bias placeholders dict is empty.')
        pt_path = out / 'peft_placeholders.safetensors'
        if safe_serialization:
            try:
                from safetensors.torch import save_file
                save_file(dict(placeholders), pt_path.as_posix())
            except Exception:
                torch.save(placeholders, pt_path.as_posix())
        else:
            torch.save(placeholders, pt_path.as_posix())
        cfg = getattr(self.backbone, 'config', None)
        if cfg is not None and hasattr(cfg, 'to_json_file'):
            cfg.to_json_file((out / 'config.json').as_posix())
        logging.info('Bias-only placeholders saved to %s | n_tensors=%d n_params=%d', pt_path.as_posix(), len(placeholders), self.peft_param_count())

def create_bias_tuning_model(backbone: nn.Module) -> BiasTuningModel:
    return BiasTuningModel(backbone)

def summarize_bias_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum((int(p.numel()) for p in model.parameters()))
    trainable = count_trainable_params(model)
    return {'total': int(total), 'trainable': int(trainable)}

def load_bias_tuning_model(backbone: nn.Module, placeholders_path: str, *, map_location: str='cpu') -> BiasTuningModel:
    model = BiasTuningModel(backbone)
    p = Path(placeholders_path)
    if not p.exists():
        raise FileNotFoundError(placeholders_path)
    sd: Dict[str, torch.Tensor]
    try:
        from safetensors.torch import load_file
        sd = dict(load_file(p.as_posix()))
    except Exception:
        obj = torch.load(p.as_posix(), map_location=map_location)
        if not isinstance(obj, dict):
            raise TypeError(f'Unsupported checkpoint object type: {type(obj)}')
        sd = {str(k): v for (k, v) in obj.items() if torch.is_tensor(v)}
    (missing, unexpected) = model.load_peft_state_dict(sd, strict_shapes=True)
    if missing:
        logging.warning('Bias placeholders missing %d keys (first 5): %s', len(missing), missing[:5])
    if unexpected:
        logging.warning('Bias placeholders had %d unexpected keys (first 5): %s', len(unexpected), unexpected[:5])
    return model
