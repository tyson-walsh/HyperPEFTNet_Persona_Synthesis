from __future__ import annotations
import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed, utils as hf_utils
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def _silence_warnings() -> None:
    import warnings
    noisy = ['Deterministic behavior was enabled.*cublas', 'stateless\\.functional_call.*deprecated', 'torch\\.cuda\\.amp\\.GradScaler\\(.*\\).*deprecated', 'torch\\.cuda\\.amp\\.autocast\\(.*\\).*deprecated', 'non-deterministic algorithm.*attention_backward', 'Converting a tensor with requires_grad=True to a scalar']
    libs = '(?i).*?(torch|transformers).*'
    for pat in noisy:
        warnings.filterwarnings('ignore', message=pat, category=UserWarning, module=libs)
        warnings.filterwarnings('ignore', message=pat, category=FutureWarning, module=libs)
    warnings.filterwarnings('ignore', category=FutureWarning, module=libs)
    warnings.filterwarnings('ignore', category=UserWarning, module=libs)
_silence_warnings()
REPLY_SEP_TOKEN = '<|reply|>'
REPLY_END_TOKEN = '<|eoreply|>'
TRAIN_BOUNDARY_ROWS: bool = False
REPLY_SEP_ID: Optional[int] = None
REPLY_END_ID: Optional[int] = None
ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / 'data_scripts'), str(ROOT / 'PEFT_scripts')])
OPT_DIR_DEFAULT = './results'
from dataset_10000 import RedditConversationDataset10000
from adapter import ModelAdapter
from bias import BiasTuningModel
from hypernetwork_structure_160M import ModelLoRAQKV
from hypernetwork_structure_160M import ModelPrefixOnly
PEFT_CFG: Dict[str, Dict[str, Any]] = {'lora': {'lora_rank': 32, 'lora_alpha': 32.0}, 'base': {}, 'adapter': {'adapter_bottleneck': 256, 'adapter_dropout': 0.0, 'adapter_ln': False}, 'bias': {'bias_mode': 'all_bias_terms'}, 'prefix': {'prefix_length': 8, 'prefix_dropout': 0.0}}

def _ppl(ce: float) -> float:
    return math.exp(ce) if ce < 50 else float('inf')

def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())

def _core_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, torch.nn.DataParallel) else m

def _count_trainable_params(m: torch.nn.Module) -> int:
    core = _core_model(m)
    return sum((int(p.numel()) for p in core.parameters() if p.requires_grad))

def _trainable_names(m: torch.nn.Module) -> List[str]:
    core = _core_model(m)
    return [n for (n, p) in core.named_parameters() if p.requires_grad]

def _json_dump(path: Path, payload: Any) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2, default=str))
    except Exception as e:
        logging.warning('Failed writing %s: %s', path.as_posix(), e)

def _maybe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _safe_rm_tree(p: Path) -> None:
    try:
        shutil.rmtree(p.as_posix())
    except Exception as e:
        logging.warning('Failed removing %s: %s', p.as_posix(), e)

def _rotate_checkpoints(root: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    subs = [d for d in root.glob('step_*') if d.is_dir()]
    subs_sorted: List[Tuple[int, Path]] = []
    for d in subs:
        m = re.search('step_(\\d+)', d.name)
        if not m:
            continue
        subs_sorted.append((int(m.group(1)), d))
    subs_sorted.sort(key=lambda x: x[0])
    if len(subs_sorted) <= keep_last:
        return
    for (_, d) in subs_sorted[:-keep_last]:
        _safe_rm_tree(d)

def _amp_dtype_for_device(device: torch.device) -> Tuple[torch.dtype, bool]:
    if device.type != 'cuda':
        return (torch.float32, False)
    bf16_ok = torch.cuda.is_bf16_supported()
    if bf16_ok:
        return (torch.bfloat16, False)
    return (torch.float16, True)

def _align_logits_and_labels(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.size(1) != labels.size(1):
        m = min(logits.size(1), labels.size(1))
        return (logits[:, :m, :], labels[:, :m])
    return (logits, labels)

def _ce_loss(out, labels: torch.Tensor) -> torch.Tensor:
    (l, y) = _align_logits_and_labels(out.logits[..., :-1, :], labels[..., 1:])
    return F.cross_entropy(l.reshape(-1, l.size(-1)), y.reshape(-1), ignore_index=-100)

def _boundary_ce(logits: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    reply_mask = labels != -100
    has = reply_mask.any(dim=1)
    if int(has.sum().item()) == 0:
        return torch.tensor(0.0, device=logits.device)
    first_idx = reply_mask.float().argmax(dim=1)
    rows = torch.nonzero(has, as_tuple=False).squeeze(-1)
    fi = first_idx[rows]
    prev = (fi - 1).clamp_min(0)
    sel_logits = logits[rows, prev, :]
    sel_targets = labels[rows, fi].clamp_min(0)
    return F.cross_entropy(sel_logits, sel_targets, ignore_index=-100)

def _make_packed_batch(batch: Dict[str, torch.Tensor], pad_id: int, seq_len: int) -> None:
    if REPLY_SEP_ID is None or REPLY_END_ID is None:
        raise RuntimeError('Special token IDs not initialized.')
    if seq_len < 8:
        raise ValueError(f'seq_len too small: {seq_len}')
    reply_budget = 128
    input_ids = batch['input_ids']
    attn = batch.get('attention_mask', (input_ids != pad_id).long())
    tgt_labels = batch['labels']
    (B, _) = input_ids.size()
    packed_ids = input_ids.new_full((B, seq_len), pad_id)
    packed_attn = attn.new_zeros((B, seq_len))
    packed_lbls = input_ids.new_full((B, seq_len), -100)
    for i in range(B):
        ctx_len_true = int(attn[i].sum().item())
        ctx_cap = max(0, min(ctx_len_true, max(0, seq_len - reply_budget - 2)))
        tgt_row = tgt_labels[i]
        tgt_tokens = tgt_row[tgt_row != -100]
        room = max(0, seq_len - ctx_cap - 2)
        tlen = int(min(room, int(tgt_tokens.numel())))
        if tlen <= 0:
            ctx_cap = max(0, min(ctx_cap, seq_len - 3))
            room = max(0, seq_len - ctx_cap - 2)
            tlen = int(min(room, int(tgt_tokens.numel())))
        pos = 0
        if ctx_cap > 0:
            packed_ids[i, pos:pos + ctx_cap] = input_ids[i, :ctx_cap]
            pos += ctx_cap
        packed_ids[i, pos] = REPLY_SEP_ID
        pos += 1
        if tlen > 0:
            packed_ids[i, pos:pos + tlen] = tgt_tokens[:tlen]
            packed_lbls[i, pos:pos + tlen] = tgt_tokens[:tlen]
            pos += tlen
        if pos > seq_len - 1:
            ctx_cap = max(0, min(ctx_len_true, seq_len - 3))
            room = max(0, seq_len - ctx_cap - 2)
            tlen = int(min(room, int(tgt_tokens.numel())))
            pos = 0
            if ctx_cap > 0:
                packed_ids[i, pos:pos + ctx_cap] = input_ids[i, :ctx_cap]
                pos += ctx_cap
            packed_ids[i, pos] = REPLY_SEP_ID
            pos += 1
            if tlen > 0:
                packed_ids[i, pos:pos + tlen] = tgt_tokens[:tlen]
                packed_lbls[i, pos:pos + tlen] = tgt_tokens[:tlen]
                pos += tlen
        packed_ids[i, pos] = REPLY_END_ID
        packed_lbls[i, pos] = REPLY_END_ID
        pos += 1
        packed_attn[i, :pos] = 1
    batch['input_ids'] = packed_ids
    batch['attention_mask'] = packed_attn
    batch['labels'] = packed_lbls

def _build_scheduler(opt: torch.optim.Optimizer, steps: int, warm_frac: float=0.05):
    warm = max(1, int(steps * warm_frac))
    rest = max(1, steps - warm)
    return torch.optim.lr_scheduler.SequentialLR(opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warm), torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rest)], milestones=[warm])

def _enable_gradient_checkpointing(m: torch.nn.Module) -> None:
    fn = getattr(m, 'gradient_checkpointing_enable', None)
    if callable(fn):
        fn()
        return
    bb = getattr(m, 'backbone', None)
    if bb is not None and callable(getattr(bb, 'gradient_checkpointing_enable', None)):
        bb.gradient_checkpointing_enable()

def _wrap_model(variant: str, backbone: torch.nn.Module, cfg: Dict[str, Any]) -> torch.nn.Module:
    if variant == 'lora':
        rows = None
        if TRAIN_BOUNDARY_ROWS:
            if REPLY_SEP_ID is None or REPLY_END_ID is None:
                raise RuntimeError('TRAIN_BOUNDARY_ROWS is set but the reply-boundary token ids are not initialized; _init_tokenizer_and_special_ids must run first')
            rows = [int(REPLY_SEP_ID), int(REPLY_END_ID)]
            logging.info('[lora] boundary rows LEARNABLE for ids %s, matching the hypernetwork arm', rows)
        model = ModelLoRAQKV(backbone, rank=int(cfg['lora_rank']), alpha=float(cfg['lora_alpha']), train_token_rows=rows)
        return model
    if variant == 'adapter':
        model = ModelAdapter(backbone, bottleneck_dim=int(cfg['adapter_bottleneck']), adapter_dropout=float(cfg['adapter_dropout']), use_layer_norm=bool(cfg.get('adapter_ln', False)))
        for (name, p) in model.named_parameters():
            nl = name.lower()
            if '.up_proj.weight' in nl or '.up_proj.bias' in nl:
                with torch.no_grad():
                    p.zero_()
            needs_grad = '.down_proj.' in nl or '.up_proj.' in nl
            p.requires_grad_(needs_grad)
        return model
    if variant == 'bias':
        model = BiasTuningModel(backbone)
        for (name, p) in model.named_parameters():
            p.requires_grad_(name.endswith('.bias'))
        if _count_trainable_params(model) == 0:
            raise RuntimeError('[bias] 0 trainable params after strict selection.')
        return model
    if variant == 'prefix':
        model = ModelPrefixOnly(backbone, prefix_length=int(cfg['prefix_length']), dropout_prob=float(cfg.get('prefix_dropout', 0.0)))
        for (name, p) in model.named_parameters():
            p.requires_grad_('prefix_key' in name or 'prefix_value' in name)
        if _count_trainable_params(model) == 0:
            raise RuntimeError('[prefix] 0 trainable params after enabling prefix_key/prefix_value.')
        return model
    if variant == 'base':
        for p in backbone.parameters():
            p.requires_grad_(True)
        return backbone
    raise ValueError(f'Unknown variant: {variant}')

def _expected_trainable_params(cfg: Any, variant: str) -> Optional[int]:
    if variant == 'base':
        return None
    hidden = int(getattr(cfg, 'hidden_size', getattr(cfg, 'n_embd', 0)) or 0)
    n_layers = int(getattr(cfg, 'num_hidden_layers', getattr(cfg, 'n_layer', 0)) or 0)
    if hidden <= 0 or n_layers <= 0:
        return None
    if variant == 'lora':
        r = int(PEFT_CFG['lora']['lora_rank'])
        return int(6 * hidden * r * n_layers)
    if variant == 'adapter':
        b = int(PEFT_CFG['adapter']['adapter_bottleneck'])
        per_layer = 2 * hidden * b + b + hidden
        return int(per_layer * n_layers)
    if variant == 'prefix':
        plen = int(PEFT_CFG['prefix']['prefix_length'])
        return int(2 * plen * hidden * n_layers)
    return None

def _audit_trainable(model: torch.nn.Module, variant: str) -> None:
    core = _core_model(model)
    total = sum((int(p.numel()) for p in core.parameters()))
    trainable = sum((int(p.numel()) for p in core.parameters() if p.requires_grad))
    names = _trainable_names(core)
    exp = _expected_trainable_params(getattr(core, 'config', None), variant)
    logging.info('[%s] trainable=%d / %d (%.4f%%) | n_tensors=%d | expected=%s', variant, trainable, total, 100.0 * trainable / max(1, total), len(names), str(exp) if exp is not None else 'n/a')
    for n in names[:12]:
        logging.info('[%s] trainable: %s', variant, n)
    if len(names) > 12:
        logging.info('[%s] trainable: ... (+%d more)', variant, len(names) - 12)
    if exp is not None and trainable != exp:
        logging.warning('[%s] trainable param-count mismatch (%d vs expected %d)', variant, trainable, exp)

def _extract_trainable_placeholders(model: torch.nn.Module) -> 'OrderedDict[str, torch.Tensor]':
    core = _core_model(model)
    out: 'OrderedDict[str, torch.Tensor]' = OrderedDict()
    for (name, p) in core.named_parameters():
        if p.requires_grad:
            out[name] = p.detach().cpu()
    return out

def _save_tensor_dict(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    _maybe_mkdir(path.parent)
    try:
        from safetensors.torch import save_file
        save_file({str(k): v.contiguous().cpu() for (k, v) in tensors.items()}, path.as_posix())
        return
    except Exception:
        torch.save(OrderedDict(((str(k), v.contiguous().cpu()) for (k, v) in tensors.items())), path.as_posix())

def _load_tensor_dict_any(path: Path) -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path.as_posix())
    try:
        from safetensors.torch import load_file
        sd = load_file(path.as_posix())
        if isinstance(sd, dict):
            return {str(k): v for (k, v) in sd.items() if torch.is_tensor(v)}
    except Exception:
        pass
    obj = torch.load(path.as_posix(), map_location='cpu')
    if isinstance(obj, dict):
        return {str(k): v for (k, v) in obj.items() if torch.is_tensor(v)}
    raise TypeError(f'Unsupported checkpoint object type: {type(obj)}')

def _verify_placeholders_file(path: Path, model: torch.nn.Module) -> None:
    sd = _load_tensor_dict_any(path)
    if len(sd) == 0:
        raise RuntimeError(f'Saved placeholders are empty: {path.as_posix()}')
    core = _core_model(model)
    live = {n: p for (n, p) in core.named_parameters() if p.requires_grad}
    live_keys = set(live.keys())
    sd_keys = set(sd.keys())
    if sd_keys != live_keys:
        missing = sorted(list(live_keys - sd_keys))[:20]
        extra = sorted(list(sd_keys - live_keys))[:20]
        raise RuntimeError(f'Placeholder key mismatch.\n  file={path.as_posix()}\n  missing(first20)={missing}\n  extra(first20)={extra}\n  n_live={len(live_keys)} n_file={len(sd_keys)}')
    for (k, v) in sd.items():
        if tuple(v.shape) != tuple(live[k].shape):
            raise RuntimeError(f'Shape mismatch for {k}: file={tuple(v.shape)} live={tuple(live[k].shape)}')
    size_b = path.stat().st_size
    logging.info('Verified placeholders: %s | n_tensors=%d | size=%.1f KB', path.name, len(sd), size_b / 1024.0)

def _eval_split(model: torch.nn.Module, loader: DataLoader, device: torch.device, pad_id: int, seq_len: int, use_amp: bool, amp_dtype: torch.dtype, *, max_batches: Optional[int]=None) -> Tuple[float, float]:
    model.eval()
    ce_sum = 0.0
    tok_sum = 0.0
    bce_sum = 0.0
    samp_sum = 0.0
    with torch.no_grad():
        for (b_idx, batch) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            batch = {k: v.to(device) for (k, v) in batch.items()}
            _make_packed_batch(batch, pad_id, seq_len)
            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss_t = getattr(out, 'loss', None)
                if loss_t is None:
                    loss_t = _ce_loss(out, batch['labels'])
                else:
                    loss_t = loss_t.mean()
            nt = int((batch['labels'][..., 1:] != -100).sum().item())
            if nt > 0:
                ce = float(loss_t.item())
                ce_sum += ce * nt
                tok_sum += nt
            bce = float(_boundary_ce(out.logits, batch['input_ids'], batch['labels']).item())
            bce_sum += bce * batch['input_ids'].size(0)
            samp_sum += batch['input_ids'].size(0)
    model.train()
    return (ce_sum / max(1.0, tok_sum), bce_sum / max(1.0, samp_sum))

@dataclass
class StepMetrics:
    step: int
    ce: float
    ppl: float
    lr: float
    elapsed_sec: float

def _param_fingerprint(model: torch.nn.Module, k: int=8) -> Dict[str, Any]:
    core = _core_model(model)
    items: List[Tuple[str, torch.Tensor]] = []
    for (name, p) in core.named_parameters():
        if p.requires_grad:
            items.append((name, p.detach().float().cpu()))
        if len(items) >= k:
            break
    fp: Dict[str, Any] = {}
    for (n, t) in items:
        fp[n] = {'mean_abs': float(t.abs().mean().item()), 'l2': float(t.norm().item())}
    return fp

def _param_fingerprint_delta(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = sorted(set(a.keys()) & set(b.keys()))
    for k in keys:
        out[k] = {'mean_abs_delta': float(b[k]['mean_abs'] - a[k]['mean_abs']), 'l2_delta': float(b[k]['l2'] - a[k]['l2'])}
    return out

def _train_variant(*, model: torch.nn.Module, variant: str, vdir: Path, dl_tr: DataLoader, dl_val: DataLoader, device: torch.device, pad_id: int, seq_len: int, steps: int, grad_accum: int, lr: float, weight_decay: float, log_interval: int, save_every: int, keep_last: int, eval_every: int, eval_batches: int, final_eval_batches: int, use_amp: bool, amp_dtype: torch.dtype, use_grad_scaler: bool, tok: Any, args: argparse.Namespace) -> Dict[str, Any]:
    _maybe_mkdir(vdir)
    ckpt_root = vdir / 'checkpoints'
    _maybe_mkdir(ckpt_root)
    core = _core_model(model)
    trainable = [(n, p) for (n, p) in core.named_parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError(f'[{variant}] 0 trainable params; refusing to proceed.')
    trainable_param_list = [p for (_, p) in trainable]
    if variant == 'base':
        emb_lr = float(lr) * 0.1
        decay_params: List[torch.nn.Parameter] = []
        no_decay_params: List[torch.nn.Parameter] = []
        emb_params: List[torch.nn.Parameter] = []
        for (name, p) in trainable:
            lname = name.lower()
            if 'embed_in' in lname or 'embed_out' in lname or 'lm_head' in lname:
                emb_params.append(p)
            elif p.ndim == 1 or lname.endswith('.bias'):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        param_groups: List[Dict[str, Any]] = []
        if decay_params:
            param_groups.append({'params': decay_params, 'weight_decay': float(weight_decay), 'lr': float(lr)})
        if no_decay_params:
            param_groups.append({'params': no_decay_params, 'weight_decay': 0.0, 'lr': float(lr)})
        if emb_params:
            param_groups.append({'params': emb_params, 'weight_decay': 0.0, 'lr': float(emb_lr)})
        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), lr=float(lr), weight_decay=0.0)
    else:
        decay_params: List[torch.nn.Parameter] = []
        no_decay_params: List[torch.nn.Parameter] = []
        for (name, p) in trainable:
            lname = name.lower()
            if p.ndim == 1 or lname.endswith('.bias'):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        param_groups: List[Dict[str, Any]] = []
        if decay_params:
            param_groups.append({'params': decay_params, 'weight_decay': float(weight_decay), 'lr': float(lr)})
        if no_decay_params:
            param_groups.append({'params': no_decay_params, 'weight_decay': 0.0, 'lr': float(lr)})
        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), lr=float(lr), weight_decay=0.0)
    sched = _build_scheduler(opt, steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and use_grad_scaler)
    _json_dump(vdir / 'args.json', {**vars(args), 'variant': variant, 'peft_cfg': PEFT_CFG.get(variant, {})})
    try:
        tok.save_pretrained(vdir)
    except Exception as e:
        logging.warning('[%s] tokenizer save_pretrained failed: %s', variant, e)
    try:
        (getattr(core, 'config', None) or getattr(core, 'backbone', None).config).save_pretrained(vdir)
    except Exception:
        pass
    fp0 = _param_fingerprint(model, k=8)
    if variant != 'base' and getattr(args, 'smoke_save', False):
        smoke_path = vdir / 'SMOKE_peft_placeholders.safetensors'
        ph0 = _extract_trainable_placeholders(model)
        if len(ph0) == 0:
            raise RuntimeError(f'[{variant}] SMOKE save: 0 trainable tensors; refusing to proceed.')
        _save_tensor_dict(smoke_path, ph0)
        _verify_placeholders_file(smoke_path, model)
        try:
            smoke_path.unlink()
        except Exception:
            pass
        logging.info('[%s] SMOKE save/verify passed', variant)

    def _write_train_state(step_i: int, extra: Optional[Dict[str, Any]]=None) -> None:
        payload: Dict[str, Any] = {'variant': variant, 'step': int(step_i), 'steps_target': int(steps), 'timestamp': _now_iso()}
        if extra:
            payload.update(extra)
        _json_dump(vdir / 'train_state.json', payload)

    def _save_checkpoint(step_i: int, metrics: Dict[str, Any]) -> None:
        sub = ckpt_root / f'step_{step_i:07d}'
        _maybe_mkdir(sub)
        if variant == 'base':
            try:
                core.save_pretrained(sub, safe_serialization=True)
            except Exception as e:
                logging.warning('[%s] save_pretrained failed at step %d: %s', variant, step_i, e)
        else:
            ph = _extract_trainable_placeholders(model)
            if len(ph) == 0:
                raise RuntimeError(f'[{variant}] checkpoint step {step_i}: extracted 0 trainable tensors')
            ph_path = sub / 'peft_placeholders.safetensors'
            _save_tensor_dict(ph_path, ph)
            _verify_placeholders_file(ph_path, model)
            root_ph = vdir / 'peft_placeholders.safetensors'
            try:
                shutil.copy2(ph_path.as_posix(), root_ph.as_posix())
            except Exception:
                _save_tensor_dict(root_ph, ph)
                _verify_placeholders_file(root_ph, model)
        _json_dump(sub / 'metrics.json', metrics)
        _json_dump(sub / 'train_state.json', {'variant': variant, 'step': int(step_i), 'timestamp': _now_iso()})
        (vdir / 'LATEST').write_text(sub.name + '\n')
        _rotate_checkpoints(ckpt_root, keep_last)
        fp1 = _param_fingerprint(model, k=8)
        delta = _param_fingerprint_delta(fp0, fp1)
        logging.info('[%s] checkpoint step=%d fingerprint_delta=%s', variant, step_i, json.dumps(delta)[:800])
    ce_running = 0.0
    tok_running = 0.0
    step = 0
    t0 = time.time()
    model.train()
    opt.zero_grad(set_to_none=True)
    while step < steps:
        progressed = False
        for (b_idx, batch) in enumerate(dl_tr):
            batch = {k: v.to(device) for (k, v) in batch.items()}
            _make_packed_batch(batch, pad_id, seq_len)
            nt = int((batch['labels'][..., 1:] != -100).sum().item())
            if nt == 0:
                continue
            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss_t = getattr(out, 'loss', None)
                if loss_t is None:
                    loss_t = _ce_loss(out, batch['labels'])
                else:
                    loss_t = loss_t.mean()
                loss = loss_t / float(grad_accum)
            ce_running += float(loss_t.item()) * float(nt)
            tok_running += float(nt)
            if use_amp and use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (b_idx + 1) % grad_accum == 0:
                if use_amp and use_grad_scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable_param_list, 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_param_list, 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                progressed = True
                if step % log_interval == 0 or step == steps:
                    mce = ce_running / max(1.0, tok_running)
                    lr_now = float(sched.get_last_lr()[0])
                    elapsed = time.time() - t0
                    logging.info('[%-6s] step %5d/%d | CE %.4f | PPL %.1f | lr %.3e | elapsed %.1fs', variant, step, steps, mce, _ppl(mce), lr_now, elapsed)
                    _write_train_state(step, {'running_ce': mce, 'running_ppl': _ppl(mce), 'lr': lr_now, 'elapsed': elapsed})
                if eval_every > 0 and (step % eval_every == 0 or step == steps):
                    (val_ce, val_bce) = _eval_split(model, dl_val, device, pad_id, seq_len, use_amp, amp_dtype, max_batches=None if eval_batches <= 0 else eval_batches)
                    logging.info('[%-6s] eval@%d | Val CE %.4f (PPL %.1f) | ValBoundaryCE %.4f (PPL %.1f)', variant, step, val_ce, _ppl(val_ce), val_bce, _ppl(val_bce))
                if save_every > 0 and (step % save_every == 0 or step == steps):
                    metrics = {'variant': variant, 'step': int(step), 'running_ce': float(ce_running / max(1.0, tok_running)), 'running_ppl': float(_ppl(ce_running / max(1.0, tok_running))), 'lr': float(sched.get_last_lr()[0]), 'elapsed_sec': float(time.time() - t0)}
                    try:
                        _save_checkpoint(step, metrics)
                    except Exception as e:
                        logging.error('[%s] checkpoint failed at step %d: %s', variant, step, e)
                        raise
                if step >= steps:
                    break
        if not progressed:
            raise RuntimeError(f'[{variant}] No usable batches; aborting (targets empty?)')
    (final_val_ce, final_val_bce) = _eval_split(model, dl_val, device, pad_id, seq_len, use_amp, amp_dtype, max_batches=None if final_eval_batches <= 0 else final_eval_batches)
    final_metrics = {'variant': variant, 'final_step': int(step), 'final_val_ce': float(final_val_ce), 'final_val_ppl': float(_ppl(final_val_ce)), 'final_val_boundary_ce': float(final_val_bce), 'final_val_boundary_ppl': float(_ppl(final_val_bce)), 'elapsed_sec': float(time.time() - t0)}
    _json_dump(vdir / 'metrics.json', final_metrics)
    _write_train_state(step, {'final_metrics': final_metrics})
    return final_metrics

def _load_optuna_lr(opt_dir: str, variant: str, default: float) -> float:
    p = Path(opt_dir) / f'opt_nonhypernet_{variant}_10000.json'
    if p.is_file():
        try:
            blob = json.loads(p.read_text())
            if isinstance(blob, dict) and variant in blob and isinstance(blob[variant], dict):
                bt = blob[variant].get('best_trial', None)
                if isinstance(bt, dict) and 'lr' in bt:
                    return float(bt['lr'])
            if isinstance(blob, dict) and 'best_trial' in blob and isinstance(blob['best_trial'], dict):
                if 'lr' in blob['best_trial']:
                    return float(blob['best_trial']['lr'])
        except Exception:
            pass
    return float(default)

def _env_flag(name: str, default: bool=False) -> bool:
    v = os.environ.get(name, '')
    if v is None or v.strip() == '':
        return default
    v = v.strip().lower()
    if v in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if v in ('0', 'false', 'no', 'n', 'off'):
        return False
    logging.warning('[env] %s=%r is not a valid bool; using default=%s', name, v, default)
    return default

def _init_tokenizer_and_special_ids(base_ckpt: str):
    tok = AutoTokenizer.from_pretrained(base_ckpt, local_files_only=True)
    add_special: List[str] = []
    if REPLY_SEP_TOKEN not in tok.get_vocab():
        add_special.append(REPLY_SEP_TOKEN)
    if REPLY_END_TOKEN not in tok.get_vocab():
        add_special.append(REPLY_END_TOKEN)
    if add_special:
        tok.add_special_tokens({'additional_special_tokens': add_special})
    global REPLY_SEP_ID, REPLY_END_ID
    REPLY_SEP_ID = tok.convert_tokens_to_ids(REPLY_SEP_TOKEN)
    REPLY_END_ID = tok.convert_tokens_to_ids(REPLY_END_TOKEN)
    if tok.pad_token_id is None:
        tok.add_special_tokens({'pad_token': '[PAD]'})
        tok.pad_token_id = tok.convert_tokens_to_ids('[PAD]')
    return tok

def _run_demo_preflight(*, args: argparse.Namespace, device: torch.device, tok) -> None:
    logging.info('[demo] DEMO_MODE=true → running preflight only (no parquet read, no training).')
    variants = [v.strip() for v in str(args.variants).split(',') if v.strip()]
    allowed = {'base', 'lora', 'adapter', 'bias', 'prefix'}
    for v in variants:
        if v not in allowed:
            raise ValueError(f'Unknown variant in --variants: {v}')
    out_root = Path(args.models_output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        vdir = out_root / f'nonhypernet_{variant}_model_10000'
        _maybe_mkdir(vdir)
        cfg = AutoConfig.from_pretrained(args.base_ckpt, local_files_only=True)
        cfg.use_cache = False
        backbone = AutoModelForCausalLM.from_pretrained(args.base_ckpt, config=cfg, local_files_only=True)
        backbone.resize_token_embeddings(len(tok))
        if variant != 'base':
            for p in backbone.parameters():
                p.requires_grad_(False)
        model = _wrap_model(variant, backbone, PEFT_CFG[variant]).to(device)
        if torch.cuda.device_count() > 1:
            logging.info('[%s] using DataParallel across %d GPUs', variant, torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
        _audit_trainable(model, variant)
        lr_val = float(args.lr) if float(args.lr) > 0 else _load_optuna_lr(args.opt_results_dir, variant, 0.0004)
        logging.info('[%s] lr=%.3e', variant, lr_val)
        if variant != 'base':
            smoke_path = vdir / 'SMOKE_peft_placeholders.safetensors'
            ph0 = _extract_trainable_placeholders(model)
            if len(ph0) == 0:
                raise RuntimeError(f'[{variant}] DEMO smoke: 0 trainable tensors; refusing to proceed.')
            _save_tensor_dict(smoke_path, ph0)
            _verify_placeholders_file(smoke_path, model)
            try:
                smoke_path.unlink()
            except Exception:
                pass
            logging.info('[%s] SMOKE save/verify passed (demo mode)', variant)
        else:
            logging.info('[base] demo mode: skipping save_pretrained to keep demo fast.')
        del model, backbone
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    logging.info('[demo] DEMO_MODE complete. Exiting now.')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_parquet', required=True)
    ap.add_argument('--val_parquet', required=True)
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--models_output_dir', required=True)
    ap.add_argument('--train_steps', type=int, default=10000)
    ap.add_argument('--batch_size', type=int, default=48)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--lr', type=float, default=0.0)
    ap.add_argument('--lr_base', type=float, default=0.0, help='Override LR for base full fine-tune only (0 = use optuna/default).')
    ap.add_argument('--lora_rank', type=int, default=0, help='Override LoRA rank (0 = keep 32).')
    ap.add_argument('--lora_alpha', type=float, default=0.0, help='Override LoRA alpha (0 = match rank).')
    ap.add_argument('--weight_decay', type=float, default=0.0001)
    ap.add_argument('--log_interval', type=int, default=500)
    ap.add_argument('--fp32', action='store_true')
    ap.add_argument('--save_every', type=int, default=1000)
    ap.add_argument('--keep_last', type=int, default=3)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--eval_batches', type=int, default=50, help='0 = full validation set; otherwise cap to this many batches for periodic eval.')
    ap.add_argument('--final_eval_batches', type=int, default=0, help='Cap the final validation evaluation to N batches (0 = full val set).')
    ap.add_argument('--smoke_save', action='store_true', help='Save+verify PEFT placeholders before training starts.')
    ap.add_argument('--save_optimizer', action='store_true', help='(Unused stub) keep for parity with other scripts.')
    ap.add_argument('--log_file', default='', help='Optional extra log file (stdout logging always enabled).')
    ap.add_argument('--opt_results_dir', default=OPT_DIR_DEFAULT)
    ap.add_argument('--train_boundary_rows', action='store_true', help='LoRA variant only. Learn the reply-separator and reply-end embedding rows as zero-initialized deltas, the same convention train_hypernet_160M.py uses. Adds 3,072 trainable parameters. OFF by default: every shipped arm trained without this.')
    ap.add_argument('--seed', type=int, default=142, help='RNG seed. Was hardcoded at 142; exposed 2026-08-10 for multi-seed runs.')
    ap.add_argument('--checklist', default='./log_files/train_nonhypernet_checklist_10000.txt')
    ap.add_argument('--variants', default='lora,adapter,bias,prefix,base', help='Comma-separated list in training order, subset of: base,lora,adapter,bias,prefix')
    args = ap.parse_args()
    global TRAIN_BOUNDARY_ROWS
    TRAIN_BOUNDARY_ROWS = bool(getattr(args, 'train_boundary_rows', False))
    if TRAIN_BOUNDARY_ROWS:
        logging.info('[cfg] train_boundary_rows ON: the LoRA head will learn its two reply-boundary rows, 3072 extra parameters')
    if int(getattr(args, 'lora_rank', 0)) > 0:
        _r = int(args.lora_rank)
        _a = float(args.lora_alpha) if float(getattr(args, 'lora_alpha', 0.0)) > 0 else float(_r)
        PEFT_CFG['lora']['lora_rank'] = _r
        PEFT_CFG['lora']['lora_alpha'] = _a
        logging.info('[lora] RANK OVERRIDE r=%d alpha=%.1f -> expected trainable %d', _r, _a, 55296 * _r)
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        try:
            Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(args.log_file, mode='a'))
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', handlers=handlers)
    hf_utils.logging.set_verbosity_error()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(int(getattr(args, 'seed', 142)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and (not args.fp32)
    (amp_dtype, use_grad_scaler) = _amp_dtype_for_device(device)
    tok = _init_tokenizer_and_special_ids(args.base_ckpt)
    pad_id = int(tok.pad_token_id)
    if _env_flag('DEMO_MODE', default=False):
        _run_demo_preflight(args=args, device=device, tok=tok)
        return
    df_tr = pd.read_parquet(args.train_parquet)
    df_val = pd.read_parquet(args.val_parquet)
    ds_tr = RedditConversationDataset10000(df_tr, tok, max_length=int(args.max_len), mode='seq2seq')
    ds_val = RedditConversationDataset10000(df_val, tok, max_length=int(args.max_len), mode='seq2seq')
    dl_tr = DataLoader(ds_tr, batch_size=int(args.batch_size), shuffle=True, num_workers=2, pin_memory=device.type == 'cuda', persistent_workers=True)
    dl_val = DataLoader(ds_val, batch_size=int(args.batch_size), shuffle=False, num_workers=2, pin_memory=device.type == 'cuda', persistent_workers=True)
    ck = Path(args.checklist)
    ck.parent.mkdir(parents=True, exist_ok=True)
    done = set(ck.read_text().splitlines()) if ck.exists() else set()
    out_root = Path(args.models_output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in str(args.variants).split(',') if v.strip()]
    allowed = {'base', 'lora', 'adapter', 'bias', 'prefix'}
    for v in variants:
        if v not in allowed:
            raise ValueError(f'Unknown variant in --variants: {v}')
    for variant in variants:
        if variant in done:
            logging.info('[%s] already in checklist; skipping.', variant)
            continue
        vdir = out_root / f'nonhypernet_{variant}_model_10000'
        _maybe_mkdir(vdir)
        cfg = AutoConfig.from_pretrained(args.base_ckpt, local_files_only=True)
        cfg.use_cache = False
        backbone = AutoModelForCausalLM.from_pretrained(args.base_ckpt, config=cfg, local_files_only=True)
        backbone.resize_token_embeddings(len(tok))
        if variant != 'base':
            for p in backbone.parameters():
                p.requires_grad_(False)
        model = _wrap_model(variant, backbone, PEFT_CFG[variant]).to(device)
        if variant == 'base':
            _enable_gradient_checkpointing(model)
        if torch.cuda.device_count() > 1:
            logging.info('[%s] using DataParallel across %d GPUs', variant, torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
        _audit_trainable(model, variant)
        _init_ph = os.environ.get('HN_INIT_PLACEHOLDERS', '').strip()
        if _init_ph and variant != 'base':
            _sd = _load_tensor_dict_any(Path(_init_ph))
            _live = {n: p for (n, p) in _core_model(model).named_parameters() if p.requires_grad}
            if set(_sd) != set(_live) or any((tuple(_sd[k].shape) != tuple(_live[k].shape) for k in _live)):
                raise RuntimeError(f'[{variant}] HN_INIT_PLACEHOLDERS={_init_ph} does not match the live trainable set')
            with torch.no_grad():
                for (_k, _p) in _live.items():
                    _p.copy_(_sd[_k].to(device=_p.device, dtype=_p.dtype))
            logging.info('[%s] started %d trainable tensors from %s', variant, len(_live), _init_ph)
            print(f'[{variant}] started {len(_live)} trainable tensors from {_init_ph}', flush=True)
        if float(args.lr) > 0:
            lr_val = float(args.lr)
        elif variant == 'base' and float(args.lr_base) > 0:
            lr_val = float(args.lr_base)
        else:
            default_lrs = {'base': 5e-05, 'lora': 0.0004, 'adapter': 0.0001, 'bias': 0.0004, 'prefix': 0.0004}
            lr_val = _load_optuna_lr(args.opt_results_dir, variant, default_lrs.get(variant, 0.0004))
        if variant == 'adapter' and float(args.lr) <= 0 and (lr_val > 0.0002):
            logging.warning('[adapter] lr=%.3e is high for direct adapter tuning; clamping to 1.000e-04 (override with --lr to bypass).', lr_val)
            lr_val = 0.0001
        if variant == 'base' and float(args.lr) <= 0 and (float(args.lr_base) <= 0) and (lr_val > 0.0001):
            logging.warning('[base] lr=%.3e is high for full FT; clamping to 1.000e-04 (override with --lr_base to bypass).', lr_val)
            lr_val = 0.0001
        logging.info('[%s] lr=%.3e', variant, lr_val)
        metrics = _train_variant(model=model, variant=variant, vdir=vdir, dl_tr=dl_tr, dl_val=dl_val, device=device, pad_id=pad_id, seq_len=int(args.max_len), steps=int(args.train_steps), grad_accum=int(args.grad_accum), lr=lr_val, weight_decay=float(args.weight_decay), log_interval=int(args.log_interval), save_every=int(args.save_every), keep_last=int(args.keep_last), eval_every=int(args.eval_every), eval_batches=int(args.eval_batches), final_eval_batches=int(args.final_eval_batches), use_amp=use_amp, amp_dtype=amp_dtype, use_grad_scaler=use_grad_scaler, tok=tok, args=args)
        logging.info('[%s] DONE | metrics=%s', variant, json.dumps(metrics))
        if variant == 'base':
            try:
                core = _core_model(model)
                core.save_pretrained(vdir, safe_serialization=True)
            except Exception as e:
                logging.warning('[%s] final save_pretrained failed: %s', variant, e)
        if variant != 'base':
            ph_path = vdir / 'peft_placeholders.safetensors'
            if not ph_path.exists():
                ph = _extract_trainable_placeholders(model)
                _save_tensor_dict(ph_path, ph)
            _verify_placeholders_file(ph_path, model)
        with ck.open('a') as fh:
            fh.write(f'{variant}\n')
        del model, backbone
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    logging.info('All requested variants complete.')
if __name__ == '__main__':
    _t0 = time.time()
    main()
    print(f'Script runtime: {time.time() - _t0:.1f}s')
