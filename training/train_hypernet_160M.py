from __future__ import annotations
import argparse
import atexit
import json
import math
import os
import random
import re
import sys
if os.environ.get('HN_USE_UNIFIED_MEMORY', '0') == '1':
    try:
        try:
            os.chdir('/tmp')
        except OSError:
            pass
        import rmm
        from rmm.allocators.torch import rmm_torch_allocator
        import torch as _torch_for_rmm
        rmm.reinitialize(pool_allocator=True, managed_memory=True, logging=False)
        _torch_for_rmm.cuda.memory.change_current_allocator(rmm_torch_allocator)
        if int(os.environ.get('LOCAL_RANK', '0')) == 0:
            print('[rmm] unified memory allocator active (RMM %s)' % rmm.__version__, flush=True)
    except Exception as _rmm_err:
        if int(os.environ.get('LOCAL_RANK', '0')) == 0:
            print('[rmm] failed to init: %s; falling back to the default allocator' % _rmm_err, flush=True)
import time
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import builtins
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed as hf_set_seed, utils as hf_utils
os.environ.setdefault('NCCL_DEBUG', 'WARN')
os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')
os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
if 'NCCL_BLOCKING_WAIT' in os.environ:
    try:
        del os.environ['NCCL_BLOCKING_WAIT']
    except Exception:
        pass
_LOG_PATCHED = False

def _setup_logging(tag: str, rank: Optional[int]=None, log_file: Optional[str]=None) -> None:
    global _LOG_PATCHED
    if _LOG_PATCHED:
        return
    import time as _time
    _orig_print = builtins.print
    r = 0 if rank is None else int(rank)
    debug_all = str(os.environ.get('DEBUG_RANKS', '0')).lower() in {'1', 'true', 'yes', 'on'}
    tag_rank = f'{tag} r{r}'
    _fp = None
    if log_file:
        try:
            lf = Path(log_file)
            lf.parent.mkdir(parents=True, exist_ok=True)
            _fp = open(lf, mode='a', buffering=1, encoding='utf-8')
        except Exception:
            _fp = None
    _fsync_every = int(os.environ.get('LOG_FSYNC_EVERY', '0'))
    _fsync_count = 0

    def _log_print(*args, **kwargs):
        if r != 0 and (not debug_all):
            return
        ts = _time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime())
        ms = f'{int(_time.time() % 1 * 1000):03d}'
        msg = ' '.join((str(a) for a in args))
        line = f'{ts},{ms} INFO     [{tag_rank}] {msg}'
        try:
            sys.stdout.write(line + '\n')
            sys.stdout.flush()
        except Exception:
            _orig_print(line, flush=True)
        if _fp is not None:
            try:
                _fp.write(line + '\n')
                _fp.flush()
                nonlocal _fsync_count
                _fsync_count += 1
                if _fsync_every > 0 and _fsync_count % _fsync_every == 0:
                    try:
                        os.fsync(_fp.fileno())
                    except Exception:
                        pass
            except Exception:
                pass
    builtins.print = _log_print
    _LOG_PATCHED = True

    def _close_fp():
        try:
            if _fp is not None:
                _fp.flush()
                _fp.close()
        except Exception:
            pass
    atexit.register(_close_fp)

    def _log_print(*args, **kwargs):
        if r != 0 and (not debug_all):
            return
        ts = _time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime())
        ms = f'{int(_time.time() % 1 * 1000):03d}'
        msg = ' '.join((str(a) for a in args))
        line = f'{ts},{ms} INFO     [{tag}] {msg}'
        try:
            sys.stdout.write(line + '\n')
            sys.stdout.flush()
        except Exception:
            _orig_print(line, flush=True)
        if _fp is not None:
            try:
                _fp.write(line + '\n')
                _fp.flush()
                nonlocal _fsync_count
                _fsync_count += 1
                if _fsync_every > 0 and _fsync_count % _fsync_every == 0:
                    try:
                        os.fsync(_fp.fileno())
                    except Exception:
                        pass
            except Exception:
                pass
    builtins.print = _log_print
    _LOG_PATCHED = True

    def _close_fp():
        try:
            if _fp is not None:
                _fp.flush()
                _fp.close()
        except Exception:
            pass
    atexit.register(_close_fp)
HAVE_FIXED_STRUCTURE = False
from hypernetwork_structure_160M import HypernetGlobalOnlyDataset10000, FlatHypernetwork, PEFTHypernet, ModelLoRAQKV, ModelAdapter, BiasTuningModel, count_trainable_params, RedditConversationDataset10000
HAVE_PREFIX = False
try:
    if HAVE_FIXED_STRUCTURE:
        from hypernetwork_structure_160M_FIXED import ModelPrefixOnly
    else:
        from hypernetwork_structure_160M import ModelPrefixOnly
    HAVE_PREFIX = True
except Exception:
    HAVE_PREFIX = False
REPLY_SEP_TOKEN = '<|reply|>'
REPLY_SEP_ID: Optional[int] = None
REPLY_END_TOKEN = '<|eoreply|>'
REPLY_END_ID: Optional[int] = None
REPLY_BUDGET = 128
BOS_ID: Optional[int] = None
EOS_ID: Optional[int] = None
_PACK_SANITY_PRINTED = False

def _is_main() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def _world() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def _ddp_broadcast_int(value: int, device: torch.device, src: int=0) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return int(value)
    t = torch.tensor([int(value)], device=device, dtype=torch.int64)
    dist.broadcast(t, src=int(src))
    return int(t.item())

def _ddp_broadcast_float(value: float, device: torch.device, src: int=0) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.broadcast(t, src=int(src))
    return float(t.item())

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    hf_set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False

def _seed_prefix_theta(peft_model: nn.Module, args: Any) -> None:
    mode = str(getattr(args, 'prefix_theta_init', 'random') or 'random')
    if mode == 'random':
        return
    params: List[Tuple[str, nn.Parameter]] = [(n, p) for (n, p) in peft_model.named_parameters() if n.endswith(('prefix_key', 'prefix_value'))]
    if not params:
        raise RuntimeError('--prefix_theta_init set, but the prefix head exposes no prefix_key/prefix_value parameters')
    if mode == 'zero':
        with torch.no_grad():
            for (_, p) in params:
                p.zero_()
        if _is_main():
            print(f'[prefix-theta] {len(params)} placeholder tensors zeroed (neutral base)', flush=True)
        return
    src = str(getattr(args, 'prefix_theta_from', '') or '').strip()
    if not src or not os.path.isfile(src):
        raise ValueError(f'--prefix_theta_init vanilla needs --prefix_theta_from to point at a readable peft_placeholders.safetensors; got {src!r}')
    from safetensors.torch import load_file as _load_safetensors
    van = _load_safetensors(src)
    by_suffix: Dict[str, torch.Tensor] = {}
    for (k, v) in van.items():
        by_suffix.setdefault(k.split('backbone.')[-1], v)
    (hit, miss) = (0, [])
    with torch.no_grad():
        for (n, p) in params:
            cand = by_suffix.get(n.split('backbone.')[-1], None)
            if cand is None or tuple(cand.shape) != tuple(p.shape):
                miss.append(n)
                continue
            p.copy_(cand.to(device=p.device, dtype=p.dtype))
            hit += 1
    if miss:
        raise ValueError(f'prefix theta substitution incomplete: matched {hit} of {len(params)} tensors from {src}; unmatched={miss[:8]}')
    if _is_main():
        rms = float(torch.sqrt(sum(((p.float() ** 2).sum() for (_, p) in params)) / max(1, sum((p.numel() for (_, p) in params)))).item())
        print(f'[prefix-theta] {hit} placeholder tensors copied from the trained shared head • RMS {rms:.6f} • {src}', flush=True)

def _init_token_rows_from_id(backbone: nn.Module, new_token_ids: List[int], src_id: int) -> None:
    if not new_token_ids:
        return
    try:
        in_emb = backbone.get_input_embeddings()
        out_emb = backbone.get_output_embeddings()
    except Exception:
        return
    W_in = getattr(in_emb, 'weight', None)
    if W_in is None:
        return
    W_out = getattr(out_emb, 'weight', None) if out_emb is not None else None
    b_out = getattr(out_emb, 'bias', None) if out_emb is not None else None
    src_id = int(src_id)
    with torch.no_grad():
        for tid in new_token_ids:
            tid = int(tid)
            if tid < 0 or tid >= int(W_in.size(0)):
                continue
            if 0 <= src_id < int(W_in.size(0)):
                W_in[tid].copy_(W_in[src_id])
            if W_out is not None and tid < int(W_out.size(0)) and (0 <= src_id < int(W_out.size(0))):
                W_out[tid].copy_(W_out[src_id])
            if b_out is not None and tid < int(b_out.size(0)) and (0 <= src_id < int(b_out.size(0))):
                b_out[tid].copy_(b_out[src_id])

def _ppl(x: float) -> float:
    return float('inf') if x > 50 else math.exp(x)

def _sdp_no_flash() -> Any:
    try:
        from torch.backends.cuda import sdp_kernel
        return sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    except Exception:
        return nullcontext()

def _steps_per_epoch_exact(loader: DataLoader, *, batch_size: int, microbatch_size: int, grad_accum: int, sampler: Optional[DistributedSampler]) -> int:
    if sampler is not None and hasattr(sampler, 'num_samples'):
        n_per_rank = int(sampler.num_samples)
    else:
        n_per_rank = int(len(loader.dataset))
    if n_per_rank <= 0:
        return 1
    (full_batches, rem) = divmod(n_per_rank, int(batch_size))
    mb_per_full = int(math.ceil(batch_size / float(microbatch_size)))
    total_microbatches = full_batches * mb_per_full
    if rem > 0:
        total_microbatches += int(math.ceil(rem / float(microbatch_size)))
    steps = int(math.ceil(total_microbatches / float(max(1, grad_accum))))
    return max(1, steps)

def _make_concat_inputs(batch: Dict[str, torch.Tensor], pad_id: int, L: int) -> None:
    assert REPLY_SEP_ID is not None
    assert REPLY_END_ID is not None
    if 'context_input_ids' in batch and 'context_attention_mask' in batch and ('target_input_ids' in batch) and ('target_attention_mask' in batch):
        ctx_ids = batch['context_input_ids']
        ctx_attn = batch['context_attention_mask']
        tgt_ids = batch['target_input_ids']
        tgt_attn = batch['target_attention_mask']
    elif 'input_ids' in batch and 'attention_mask' in batch and ('labels' in batch):
        ctx_ids = batch['input_ids']
        ctx_attn = batch['attention_mask']
        tgt_ids = batch['labels']
        tgt_attn = (tgt_ids != -100).to(dtype=torch.long)
    else:
        raise KeyError(f'Batch must contain either (context_input_ids/context_attention_mask + target_input_ids/target_attention_mask) or (input_ids/attention_mask + labels). Keys present: {sorted(batch.keys())}')
    (B, _) = ctx_ids.size()
    dev = ctx_ids.device
    new_input = torch.full((B, L), pad_id, dtype=torch.long, device=dev)
    new_attn = torch.zeros((B, L), dtype=torch.long, device=dev)
    new_labels = torch.full((B, L), -100, dtype=torch.long, device=dev)
    reply_budget = int(REPLY_BUDGET)
    reply_budget = max(1, reply_budget)
    stripped_ctx_eos = 0
    stripped_tgt_bos = 0
    stripped_tgt_eos = 0
    for j in range(B):
        ctx_len_true = int(ctx_attn[j].sum().item())
        ctx_tokens = ctx_ids[j, :ctx_len_true]
        if EOS_ID is not None:
            eos = int(EOS_ID)
            while ctx_tokens.numel() > 0 and int(ctx_tokens[-1].item()) == eos:
                ctx_tokens = ctx_tokens[:-1]
                stripped_ctx_eos += 1
        ctx_len = int(ctx_tokens.numel())
        tgt_tokens = tgt_ids[j][tgt_attn[j].bool()]
        if BOS_ID is not None:
            bos = int(BOS_ID)
            while tgt_tokens.numel() > 0 and int(tgt_tokens[0].item()) == bos:
                tgt_tokens = tgt_tokens[1:]
                stripped_tgt_bos += 1
        if EOS_ID is not None:
            eos = int(EOS_ID)
            while tgt_tokens.numel() > 0 and int(tgt_tokens[-1].item()) == eos:
                tgt_tokens = tgt_tokens[:-1]
                stripped_tgt_eos += 1
        tgt_len_true = int(tgt_tokens.numel())
        ctx_cap = min(ctx_len, max(0, L - reply_budget - 2))
        room = max(0, L - ctx_cap - 2)
        tlen = int(min(room, tgt_len_true))
        if tlen <= 0:
            ctx_cap = min(ctx_cap, max(0, L - 3))
            room = max(0, L - ctx_cap - 2)
            tlen = int(min(room, tgt_len_true))
        pos = 0
        if ctx_cap > 0:
            new_input[j, :ctx_cap] = ctx_tokens[:ctx_cap]
            pos += ctx_cap
        new_input[j, pos] = int(REPLY_SEP_ID)
        pos += 1
        if tlen > 0:
            new_input[j, pos:pos + tlen] = tgt_tokens[:tlen]
            new_labels[j, pos:pos + tlen] = tgt_tokens[:tlen]
            pos += tlen
        new_input[j, pos] = int(REPLY_END_ID)
        new_labels[j, pos] = int(REPLY_END_ID)
        pos += 1
        new_attn[j, :pos] = 1
    global _PACK_SANITY_PRINTED
    if not _PACK_SANITY_PRINTED and _is_main():
        print(f'[pack] strip_special_tokens: ctx_eos={stripped_ctx_eos} tgt_bos={stripped_tgt_bos} tgt_eos={stripped_tgt_eos} (BOS_ID={BOS_ID} EOS_ID={EOS_ID})', flush=True)
        _PACK_SANITY_PRINTED = True
    batch.update(input_ids=new_input, attention_mask=new_attn, labels=new_labels)

def _placeholder_splits_by_layer(peft_model: nn.Module, *, exclude_param_ids: Optional[List[int]]=None) -> List[int]:
    layer_pat = re.compile('\\.layers\\.(\\d+)\\.')
    order: List[int] = []
    sizes: Dict[int, int] = defaultdict(int)
    excl = set((int(x) for x in exclude_param_ids or []))
    for (name, p) in peft_model.named_parameters():
        if not p.requires_grad:
            continue
        if bool(getattr(p, '_hn_skip_placeholder', False)):
            continue
        if id(p) in excl:
            continue
        m = layer_pat.search(name)
        gid = int(m.group(1)) if m else -1
        if gid not in sizes:
            order.append(gid)
        sizes[gid] += int(p.numel())
    return [sizes[g] for g in order]

@torch.no_grad()
def _boundary_ce_logits(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, *, device: torch.device, pad_id: int, use_amp: bool, amp_dtype: torch.dtype, global_features: torch.Tensor) -> Tuple[torch.Tensor, int]:
    if input_ids.dim() != 2:
        raise ValueError(f'input_ids must be (B,S); got {tuple(input_ids.shape)}')
    if attention_mask is None or attention_mask.dim() != 2:
        raise ValueError(f'attention_mask must be (B,S); got {(None if attention_mask is None else tuple(attention_mask.shape))}')
    if labels.dim() != 2:
        raise ValueError(f'labels must be (B,S); got {tuple(labels.shape)}')
    if input_ids.size(0) != attention_mask.size(0) or input_ids.size(1) != attention_mask.size(1):
        raise ValueError('input_ids and attention_mask must have the same (B,S) shape')
    if labels.size(0) != input_ids.size(0):
        raise ValueError('labels and input_ids must share batch size')
    reply_mask = labels != -100
    has_reply = reply_mask.any(dim=1)
    if not bool(has_reply.any().item()):
        return (input_ids.new_tensor(0.0, dtype=torch.float32), 0)
    first_idx = reply_mask.float().argmax(dim=1)
    last_ctx = (first_idx - 1).clamp_min(0)
    rows = torch.nonzero(has_reply, as_tuple=False).squeeze(-1)
    core = model.module if hasattr(model, 'module') else model
    if hasattr(core, 'logits_at_positions'):
        with _sdp_no_flash():
            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                logits_last = core.logits_at_positions(input_ids=input_ids, attention_mask=attention_mask, positions=last_ctx, global_features=global_features, use_cache=False, return_dict=True)
                slice_logits = logits_last[rows, :]
                targets = input_ids[rows, first_idx[rows]]
                ce = F.cross_entropy(slice_logits, targets, reduction='mean')
        return (ce, int(rows.numel()))
    cut = int(first_idx.max().item())
    cut = max(1, cut)
    ctx_ids = input_ids[:, :cut]
    ctx_attn = attention_mask[:, :cut]
    with _sdp_no_flash():
        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            out = model(input_ids=ctx_ids, attention_mask=ctx_attn, labels=None, global_features=global_features, use_cache=False, return_dict=True)
            logits = out.logits
            slice_logits = logits[rows, last_ctx[rows], :]
            targets = input_ids[rows, first_idx[rows]]
            ce = F.cross_entropy(slice_logits, targets, reduction='mean')
    return (ce, int(rows.numel()))

@contextmanager
def _temporary_hyper_clamp(model: nn.Module, clamp_value: float):
    core = model.module if hasattr(model, 'module') else model
    if not hasattr(core, 'hypernet') or not hasattr(core.hypernet, 'clamp'):
        yield
        return
    old = float(core.hypernet.clamp)
    core.hypernet.clamp = float(clamp_value)
    try:
        yield
    finally:
        core.hypernet.clamp = old

@torch.no_grad()
def _context_only_ce(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, *, device: torch.device, use_amp: bool, amp_dtype: torch.dtype, global_features: torch.Tensor, disable_injection: bool=False) -> Tuple[torch.Tensor, int]:
    if input_ids.dim() != 2:
        raise ValueError(f'input_ids must be (B,S); got {tuple(input_ids.shape)}')
    if attention_mask is None or attention_mask.dim() != 2:
        raise ValueError(f'attention_mask must be (B,S); got {(None if attention_mask is None else tuple(attention_mask.shape))}')
    if input_ids.size(0) != attention_mask.size(0) or input_ids.size(1) != attention_mask.size(1):
        raise ValueError('input_ids and attention_mask must have the same (B,S) shape')
    (B, S) = input_ids.size()
    is_sep = input_ids == REPLY_SEP_ID
    if not bool(is_sep.any().item()):
        return (input_ids.new_tensor(0.0, dtype=torch.float32), 0)
    sep_idx = is_sep.float().argmax(dim=1, keepdim=True)
    max_ctx_len = int(sep_idx.max().item())
    max_ctx_len = max(1, max_ctx_len)
    ids = input_ids[:, :max_ctx_len]
    attn = attention_mask[:, :max_ctx_len]
    pos = torch.arange(max_ctx_len, device=device).unsqueeze(0).expand(B, max_ctx_len)
    ctx_mask = (pos < sep_idx) & (attn > 0)
    labels_ctx = ids.clone()
    labels_ctx[~ctx_mask] = -100
    labels_ctx[:, 0] = -100
    nt = int((labels_ctx != -100).sum().item())
    if nt == 0:
        return (input_ids.new_tensor(0.0, dtype=torch.float32), 0)
    ctx_mgr = _temporary_hyper_clamp(model, 0.0) if disable_injection else nullcontext()
    with ctx_mgr:
        with _sdp_no_flash():
            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out = model(input_ids=ids, attention_mask=attn, labels=labels_ctx, global_features=global_features, use_cache=False, return_dict=True, return_loss_only=True)
                ce = out.loss
    return (ce, nt)

def _clamp_schedule(step: int, start: int, duration: int, cmin: float, cmax: float) -> float:
    if step < start:
        return cmin
    if duration <= 0:
        return cmax
    f = min(1.0, max(0.0, (step - start) / float(duration)))
    return cmin + 0.5 * (1.0 - math.cos(math.pi * f)) * (cmax - cmin)

def _wrap_ddp(model: nn.Module, device: torch.device) -> nn.Module:
    if _world() <= 1:
        return model
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False, broadcast_buffers=False, gradient_as_bucket_view=True)
    if hasattr(ddp_model, '_set_static_graph'):
        try:
            ddp_model._set_static_graph()
        except Exception:
            pass
    return ddp_model

def _score_from_metrics(args: argparse.Namespace, metrics: Dict[str, float]) -> float:
    m = str(getattr(args, 'best_metric', 'ce')).lower()
    if m == 'ce':
        return float(metrics['ce'])
    if m == 'bnd_apply':
        return float(metrics['bnd_apply'])
    if m == 'combo':
        lam = float(getattr(args, 'combo_lambda', 0.0))
        return float(metrics['ce']) + lam * float(metrics['bnd_apply'])
    if m == 'ctx':
        return float(metrics['ctx'])
    if m == 'guard':
        return float(metrics['guard'])
    if m == 'guard_gap':
        return float(metrics['guard']) - float(metrics['ctx'])
    return float(metrics['ce'])

@torch.no_grad()
def _reply_region_breakdown_ce(*, logits: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor, reply_end_id: int) -> Dict[str, Tuple[float, int]]:
    if logits.dim() != 3:
        raise ValueError(f'logits must be (B,S,V); got {tuple(logits.shape)}')
    if input_ids.dim() != 2 or labels.dim() != 2:
        raise ValueError(f'input_ids/labels must be (B,S); got {tuple(input_ids.shape)} / {tuple(labels.shape)}')
    if input_ids.shape != labels.shape:
        raise ValueError('input_ids and labels must have same shape')
    if logits.size(0) != input_ids.size(0) or logits.size(1) != input_ids.size(1):
        raise ValueError('logits and input_ids must share (B,S)')
    (B, S, V) = logits.size()
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_input = input_ids[:, 1:]
    sup = shift_labels != -100
    end_mask = sup & (shift_labels == int(reply_end_id))
    has = sup.any(dim=1)
    start_idx = torch.zeros((B,), dtype=torch.long, device=logits.device)
    if bool(has.any().item()):
        start_idx = sup.float().argmax(dim=1)
    start_mask = torch.zeros_like(sup)
    if bool(has.any().item()):
        rows = torch.nonzero(has, as_tuple=False).squeeze(-1)
        start_mask[rows, start_idx[rows]] = True
    mid_mask = sup & ~start_mask & ~end_mask

    def _ce_from_mask(mask: torch.Tensor) -> Tuple[float, int]:
        n = int(mask.sum().item())
        if n <= 0:
            return (0.0, 0)
        sl = shift_logits[mask]
        yl = shift_labels[mask]
        ce = F.cross_entropy(sl, yl, reduction='mean')
        return (float(ce.detach().float().cpu().item()), n)
    (ce_start, n_start) = _ce_from_mask(start_mask)
    (ce_mid, n_mid) = _ce_from_mask(mid_mask)
    (ce_end, n_end) = _ce_from_mask(end_mask)
    return {'resp_start': (ce_start, n_start), 'resp_mid': (ce_mid, n_mid), 'resp_end': (ce_end, n_end)}

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, pad_id: int, max_len: int, use_amp: bool, amp_dtype: torch.dtype, max_batches: int=0, ddp_reduce: bool=True, progress_every: int=0, heartbeat_secs: int=60) -> Dict[str, float]:
    model.eval()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    tot_tokens = 0.0
    sum_ce = 0.0
    (sum_bnd_apply, cnt_bnd_apply) = (0.0, 0.0)
    (sum_guard, cnt_guard) = (0.0, 0.0)
    (sum_ctx, cnt_ctx) = (0.0, 0.0)
    (sum_start, cnt_start) = (0.0, 0.0)
    (sum_mid, cnt_mid) = (0.0, 0.0)
    (sum_end, cnt_end) = (0.0, 0.0)
    t_last = time.time()

    def _boundary_ce_from_logits(logits: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
        reply_mask = labels != -100
        has_reply = reply_mask.any(dim=1)
        if not bool(has_reply.any().item()):
            return (logits.new_tensor(0.0, dtype=torch.float32), 0)
        first_idx = reply_mask.float().argmax(dim=1)
        last_ctx = (first_idx - 1).clamp_min(0)
        rows = torch.nonzero(has_reply, as_tuple=False).squeeze(-1)
        slice_logits = logits[rows, last_ctx[rows], :]
        targets = input_ids[rows, first_idx[rows]]
        ce = F.cross_entropy(slice_logits, targets, reduction='mean')
        return (ce, int(rows.numel()))

    def _context_only_ce_from_logits(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if input_ids.dim() != 2 or attention_mask.dim() != 2:
            return (logits.new_tensor(0.0, dtype=torch.float32), 0)
        (B, _S) = input_ids.size()
        is_sep = input_ids == REPLY_SEP_ID
        if not bool(is_sep.any().item()):
            return (logits.new_tensor(0.0, dtype=torch.float32), 0)
        sep_idx = is_sep.float().argmax(dim=1, keepdim=True)
        max_ctx_len = int(sep_idx.max().item())
        max_ctx_len = max(1, max_ctx_len)
        ids = input_ids[:, :max_ctx_len]
        attn = attention_mask[:, :max_ctx_len]
        lg = logits[:, :max_ctx_len, :]
        pos = torch.arange(max_ctx_len, device=device).unsqueeze(0).expand(B, max_ctx_len)
        ctx_mask = (pos < sep_idx) & (attn > 0)
        labels_ctx = ids.clone()
        labels_ctx[~ctx_mask] = -100
        labels_ctx[:, 0] = -100
        shift_logits = lg[:, :-1, :]
        shift_labels = labels_ctx[:, 1:]
        sup = shift_labels != -100
        nt = int(sup.sum().item())
        if nt <= 0:
            return (logits.new_tensor(0.0, dtype=torch.float32), 0)
        ce = F.cross_entropy(shift_logits[sup], shift_labels[sup], reduction='mean')
        return (ce, nt)
    for (i, batch) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device) for (k, v) in batch.items()}
        _make_concat_inputs(batch, pad_id, max_len)
        with _sdp_no_flash():
            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], global_features=batch['global_features'], use_cache=False, return_dict=True, return_loss_only=False)
                loss = out.loss
                logits = out.logits
        nt = float((batch['labels'] != -100).sum().item())
        if nt > 0:
            sum_ce += float(loss.item()) * nt
            tot_tokens += nt
        if REPLY_END_ID is not None and logits is not None:
            br = _reply_region_breakdown_ce(logits=logits, input_ids=batch['input_ids'], labels=batch['labels'], reply_end_id=int(REPLY_END_ID))
            (ce_s, n_s) = br['resp_start']
            (ce_m, n_m) = br['resp_mid']
            (ce_e, n_e) = br['resp_end']
            if n_s > 0:
                sum_start += float(ce_s) * float(n_s)
                cnt_start += float(n_s)
            if n_m > 0:
                sum_mid += float(ce_m) * float(n_m)
                cnt_mid += float(n_m)
            if n_e > 0:
                sum_end += float(ce_e) * float(n_e)
                cnt_end += float(n_e)
        (ce_b, kept) = _boundary_ce_from_logits(logits=logits, input_ids=batch['input_ids'], labels=batch['labels'])
        if kept > 0:
            sum_bnd_apply += float(ce_b.item()) * float(kept)
            cnt_bnd_apply += float(kept)
        (ce_guard, nt_guard) = _context_only_ce_from_logits(logits=logits, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        if nt_guard > 0:
            sum_guard += float(ce_guard.item()) * float(nt_guard)
            cnt_guard += float(nt_guard)
        (ce_ctx, nt_ctx) = _context_only_ce(model=model, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], device=device, use_amp=use_amp, amp_dtype=amp_dtype, global_features=batch['global_features'], disable_injection=True)
        if nt_ctx > 0:
            sum_ctx += float(ce_ctx.item()) * float(nt_ctx)
            cnt_ctx += float(nt_ctx)
        if progress_every and (i + 1) % progress_every == 0 and _is_main():
            print(f'[eval r{rank}] processed {i + 1} batches', flush=True)
        if heartbeat_secs and _is_main():
            now = time.time()
            if now - t_last >= float(heartbeat_secs):
                mb = int(max_batches) if max_batches else int(len(loader))
                print(f'[eval r{rank}] heartbeat: {i + 1}/{mb} batches', flush=True)
                t_last = now
    if ddp_reduce and dist.is_available() and dist.is_initialized():
        buf = torch.tensor([sum_ce, tot_tokens, sum_bnd_apply, cnt_bnd_apply, sum_guard, cnt_guard, sum_ctx, cnt_ctx, sum_start, cnt_start, sum_mid, cnt_mid, sum_end, cnt_end], dtype=torch.float64, device=device)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        (sum_ce, tot_tokens, sum_bnd_apply, cnt_bnd_apply, sum_guard, cnt_guard, sum_ctx, cnt_ctx, sum_start, cnt_start, sum_mid, cnt_mid, sum_end, cnt_end) = buf.tolist()
    ce = sum_ce / max(1.0, tot_tokens)
    bnd_apply = sum_bnd_apply / max(1.0, cnt_bnd_apply)
    guard = sum_guard / max(1.0, cnt_guard)
    ctx = sum_ctx / max(1.0, cnt_ctx)
    resp_start = sum_start / max(1.0, cnt_start)
    resp_mid = sum_mid / max(1.0, cnt_mid)
    resp_end = sum_end / max(1.0, cnt_end)
    return {'ce': ce, 'ppl': _ppl(ce), 'bnd_apply': bnd_apply, 'guard': guard, 'ctx': ctx, 'resp_start': resp_start, 'resp_mid': resp_mid, 'resp_end': resp_end}

def _tok_words(tok: AutoTokenizer, ids: List[int]) -> List[str]:
    text = tok.decode(ids, skip_special_tokens=True)
    return [w for w in text.strip().split() if w]

def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter((tuple(tokens[i:i + n]) for i in range(0, max(0, len(tokens) - n + 1))))

def _bleu_corpus(preds: List[List[str]], refs: List[List[str]], max_n: int=2) -> Dict[str, float]:
    eps = 1e-09
    P = []
    for n in range(1, max_n + 1):
        (clip, total) = (0, 0)
        for (pr, rf) in zip(preds, refs):
            pc = _ngram_counts(pr, n)
            rc = _ngram_counts(rf, n)
            clip += sum((min(v, rc.get(k, 0)) for (k, v) in pc.items()))
            total += max(1, sum(pc.values()))
        P.append(max(eps, clip / max(eps, total)))
    pred_len = sum((len(p) for p in preds))
    ref_len = sum((len(r) for r in refs))
    if pred_len == 0:
        bp = 0.0
    elif pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_len / max(1, pred_len))
    bleu1 = bp * P[0]
    if max_n >= 2:
        bleu2 = bp * math.exp(0.5 * (math.log(P[0]) + math.log(P[1])))
    else:
        bleu2 = bleu1
    return {'bleu1': float(bleu1), 'bleu2': float(bleu2)}

@torch.no_grad()
def evaluate_ctx_generation(model: nn.Module, loader: DataLoader, tokenizer: AutoTokenizer, device: torch.device, *, max_samples: int=200, max_new_tokens: int=64, do_sample: bool=False, top_p: float=1.0, temperature: float=1.0, progress_every: int=0, heartbeat_secs: int=60) -> Dict[str, float]:
    model.eval()
    eos_id = REPLY_END_ID if REPLY_END_ID is not None else tokenizer.eos_token_id
    preds_words: List[List[str]] = []
    refs_words: List[List[str]] = []
    first_token_match = 0
    total_used = 0
    avg_len = 0.0
    t_last = time.time()
    core = model.module if hasattr(model, 'module') else model
    for batch in loader:
        if total_used >= max_samples:
            break
        B = batch['input_ids'].size(0)
        for j in range(B):
            if total_used >= max_samples:
                break
            ctx_ids = batch['input_ids'][j].unsqueeze(0).to(device)
            ctx_attn = batch['attention_mask'][j].unsqueeze(0).to(device)
            gvec = batch['global_features'][j].unsqueeze(0).to(device)
            real = int(ctx_attn.sum().item())
            real = max(1, real)
            ctx_trim = ctx_ids[:, :real]
            sep = torch.full((1, 1), REPLY_SEP_ID, dtype=ctx_trim.dtype, device=device)
            seed = torch.cat([ctx_trim, sep], dim=1)
            seed_attn = torch.ones(seed.size(0), seed.size(1), dtype=ctx_attn.dtype, device=device)
            lab = batch['labels'][j].to(device)
            gold_ids = lab[lab != -100].tolist()
            gold_first = gold_ids[0] if len(gold_ids) > 0 else None
            with _sdp_no_flash():
                gen_ids = core.generate_ctx(input_ids=seed, attention_mask=seed_attn, global_features=gvec, max_new_tokens=max_new_tokens, eos_token_id=eos_id, do_sample=do_sample, top_p=top_p, temperature=temperature)
            gen_row = gen_ids[0] if gen_ids else []
            avg_len += float(len(gen_row))
            if gold_first is not None and len(gen_row) > 0 and (int(gen_row[0]) == int(gold_first)):
                first_token_match += 1
            preds_words.append(_tok_words(tokenizer, gen_row))
            refs_words.append(_tok_words(tokenizer, gold_ids))
            total_used += 1
            if progress_every and total_used % int(progress_every) == 0:
                print(f'[ctxgen] processed {total_used}/{max_samples} samples', flush=True)
            if heartbeat_secs and _is_main():
                now = time.time()
                if now - t_last >= float(heartbeat_secs):
                    print(f'[ctxgen] heartbeat: {total_used}/{max_samples} samples', flush=True)
                    t_last = now
    if total_used == 0:
        return {'first_token_acc': 0.0, 'bleu1': 0.0, 'bleu2': 0.0, 'avg_len': 0.0}
    bleu = _bleu_corpus(preds_words, refs_words, max_n=2)
    return {'first_token_acc': float(first_token_match) / float(total_used), 'bleu1': float(bleu['bleu1']), 'bleu2': float(bleu['bleu2']), 'avg_len': float(avg_len) / float(total_used)}

def _preflight_contract_tests(peft_model: nn.Module, hyper: nn.Module, *, expected_placeholders: Optional[int]=None) -> None:

    def _hyper_out_dim(h: nn.Module) -> int:
        if hasattr(h, 'head') and isinstance(h.head, nn.Linear):
            return int(h.head.out_features)
        if hasattr(h, 'head_V') and isinstance(h.head_V, nn.Linear):
            return int(h.head_V.out_features)
        try:
            with torch.no_grad():
                in_dim = None
                for m in getattr(h, 'hidden', []):
                    if isinstance(m, nn.Linear):
                        in_dim = int(m.in_features)
                        break
                in_dim = int(in_dim or 1)
                dtype = next(h.parameters()).dtype
                dev = next(h.parameters()).device
                y = h(torch.zeros(1, in_dim, dtype=dtype, device=dev))
            return int(y.numel())
        except Exception:
            return -1
    P = int(expected_placeholders) if expected_placeholders is not None else count_trainable_params(peft_model)
    out_dim = _hyper_out_dim(hyper)
    if out_dim != P:
        raise RuntimeError(f'[preflight] hypernet out_dim {out_dim} != PEFT placeholder scalars {P}')
    print(f'[preflight] contract OK: Δθ dims {out_dim} match PEFT placeholders {P}', flush=True)

@torch.no_grad()
def _golden_batch_smoke(model: nn.Module, tokenizer: AutoTokenizer, ds: Dataset, device: torch.device, pad_id: int, max_len: int, use_amp: bool, amp_dtype: torch.dtype) -> None:
    if len(ds) == 0:
        raise RuntimeError('[preflight] dataset is empty')
    idxs = random.sample(range(len(ds)), k=min(2, len(ds)))
    items = [ds[i] for i in idxs]

    def _stack(key: str):
        return torch.stack([it[key] for it in items], dim=0).to(device)
    batch = {'input_ids': _stack('input_ids'), 'attention_mask': _stack('attention_mask'), 'labels': _stack('labels'), 'global_features': torch.stack([it['global_features'] for it in items], dim=0).to(device)}
    _make_concat_inputs(batch, pad_id, max_len)
    with _sdp_no_flash():
        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], global_features=batch['global_features'], use_cache=False, return_dict=True, return_loss_only=True)
            tf_ce = out.loss
    if not torch.isfinite(tf_ce):
        raise RuntimeError('[preflight] TF CE is not finite')
    (ce_b, kept) = _boundary_ce_logits(model=model, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], device=device, pad_id=pad_id, use_amp=use_amp, amp_dtype=amp_dtype, global_features=batch['global_features'])
    if kept == 0 or not torch.isfinite(ce_b):
        raise RuntimeError('[preflight] boundary CE failed')
    (ce_guard, n_guard) = _context_only_ce(model=model, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], device=device, use_amp=use_amp, amp_dtype=amp_dtype, global_features=batch['global_features'], disable_injection=False)
    (ce_ctx, n_ctx) = _context_only_ce(model=model, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], device=device, use_amp=use_amp, amp_dtype=amp_dtype, global_features=batch['global_features'], disable_injection=True)
    if n_guard == 0 or n_ctx == 0 or (not torch.isfinite(ce_guard)) or (not torch.isfinite(ce_ctx)):
        raise RuntimeError('[preflight] guard/ctx CE failed')
    core = model.module if hasattr(model, 'module') else model
    real = batch['attention_mask'].sum(dim=1).min().item()
    real = int(max(1, real))
    seed = torch.cat([batch['input_ids'][:, :real], torch.full((batch['input_ids'].size(0), 1), REPLY_SEP_ID, dtype=batch['input_ids'].dtype, device=device)], dim=1)
    seed_attn = torch.ones(seed.size(0), seed.size(1), dtype=batch['attention_mask'].dtype, device=device)
    with _sdp_no_flash():
        gen = core.generate_ctx(input_ids=seed, attention_mask=seed_attn, global_features=batch['global_features'], max_new_tokens=16, eos_token_id=REPLY_END_ID if REPLY_END_ID is not None else tokenizer.eos_token_id, do_sample=False, top_p=1.0, temperature=1.0)
    if not isinstance(gen, list) or len(gen) != seed.size(0):
        raise RuntimeError('[preflight] generate_ctx returned unexpected structure')
    print('[preflight] golden batch passed (TF CE, boundary CE, guard/ctx CE, generate_ctx)', flush=True)

@torch.no_grad()
def _author_swap_sanity(model: nn.Module, tokenizer: AutoTokenizer, ds: Dataset, device: torch.device, pad_id: int, max_len: int, use_amp: bool, amp_dtype: torch.dtype, *, samples: int=32) -> None:
    if len(ds) < 2:
        print('[preflight] author-swap sanity skipped (dataset too small)')
        return
    n = min(samples, len(ds))
    idxs = random.sample(range(len(ds)), k=n)
    items = [ds[i] for i in idxs]

    def _stack(key: str) -> torch.Tensor:
        return torch.stack([it[key] for it in items], dim=0).to(device)
    batch = {'input_ids': _stack('input_ids'), 'attention_mask': _stack('attention_mask'), 'labels': _stack('labels'), 'global_features': _stack('global_features')}
    _make_concat_inputs(batch, pad_id, max_len)
    core = model.module if hasattr(model, 'module') else model
    with _sdp_no_flash():
        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            out_orig = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], global_features=batch['global_features'], use_cache=False, return_dict=True, return_loss_only=True)
            ce_orig = out_orig.loss
            delta_orig = core.hypernet(batch['global_features'])
            perm = torch.randperm(batch['global_features'].size(0), device=device)
            g_swapped = batch['global_features'][perm]
            out_swap = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], global_features=g_swapped, use_cache=False, return_dict=True, return_loss_only=True)
            ce_swap = out_swap.loss
            delta_swap = core.hypernet(g_swapped)
    d_orig = delta_orig.view(delta_orig.size(0), -1)
    d_swap = delta_swap.view(delta_swap.size(0), -1)
    l2_orig = d_orig.norm(dim=1).mean().item()
    l2_swap = d_swap.norm(dim=1).mean().item()
    l2_diff = (d_orig - d_swap).norm(dim=1).mean().item()
    print(f'[preflight] author-swap sanity: CE(orig) {ce_orig.item():.4f} • CE(swapped) {ce_swap.item():.4f} (Δ={ce_swap.item() - ce_orig.item():+.4f}) • ‖Δθ(orig)‖₂/row {l2_orig:.3e} • ‖Δθ(swapped)‖₂/row {l2_swap:.3e} • ‖ΔΔθ‖₂/row {l2_diff:.3e}', flush=True)

def _feature_importance_absgrad(model: nn.Module, loader: DataLoader, device: torch.device, pad_id: int, max_len: int, use_amp: bool, amp_dtype: torch.dtype, *, batches: int) -> Tuple[np.ndarray, int]:
    model.eval()
    Kdim: Optional[int] = None
    total: Optional[np.ndarray] = None
    count = 0
    it = iter(loader)
    for _ in range(batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        gf = batch['global_features'].to(device).detach().clone()
        Kdim = int(gf.size(1))
        gf.requires_grad_(True)
        b = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device), 'labels': batch['labels'].to(device), 'global_features': gf}
        _make_concat_inputs(b, pad_id, max_len)
        with _sdp_no_flash():
            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out = model(input_ids=b['input_ids'], attention_mask=b['attention_mask'], labels=b['labels'], global_features=gf, use_cache=False, return_dict=True, return_loss_only=True)
                loss = out.loss
        grads = torch.autograd.grad(loss, gf, retain_graph=False, create_graph=False, allow_unused=False)[0]
        imp = grads.detach().abs().mean(dim=0).float().cpu().numpy()
        if total is None:
            total = imp
        else:
            total += imp
        count += 1
    if total is None or Kdim is None or count == 0:
        return (np.zeros((0,), dtype=np.float32), 0)
    return ((total / float(count)).astype(np.float32), Kdim)

def _load_init_feature_list(path: str) -> List[str]:
    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(f'init_features_json not found: {p}')
    obj = json.loads(p.read_text())
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if all((isinstance(x, str) for x in obj)):
            return [str(x) for x in obj]
        if all((isinstance(x, dict) for x in obj)):
            out = []
            for d in obj:
                if 'feature' in d:
                    out.append(str(d['feature']))
                elif 'name' in d:
                    out.append(str(d['name']))
                else:
                    raise ValueError(f'init_features_json list[dict] missing feature/name keys: keys={sorted(d.keys())}')
            return out
    if isinstance(obj, dict):
        if 'feature_names' in obj and isinstance(obj['feature_names'], list):
            return [str(x) for x in obj['feature_names']]
        if 'names' in obj and isinstance(obj['names'], list):
            return [str(x) for x in obj['names']]
    raise ValueError('init_features_json must be list[str], list[dict], or dict with feature_names/names')

def _is_list_valued_series(s: pd.Series) -> bool:
    try:
        x = s.dropna().head(64).tolist()
    except Exception:
        return False
    for v in x:
        if isinstance(v, (list, tuple, np.ndarray)):
            return True
    return False

def _topk_by_variance(gdf: pd.DataFrame, candidates: List[str], k: int) -> List[str]:
    scores = []
    for c in candidates:
        try:
            s = pd.to_numeric(gdf[c], errors='coerce')
            v = float(s.var(skipna=True))
        except Exception:
            v = float('nan')
        if math.isfinite(v):
            scores.append((v, c))
    scores.sort(key=lambda t: (-t[0], t[1]))
    return [c for (_, c) in scores[:k]]

def _resolve_feature_profile(args: argparse.Namespace, df_tr: 'pd.DataFrame', gdf: 'pd.DataFrame') -> Dict[str, Any]:
    profile = getattr(args, 'feature_profile', 'all')
    profile = str(profile).lower()
    include = list(args.feature_include or [])
    exclude = list(args.feature_exclude or [])
    g_cols = [c for c in gdf.columns if str(c).startswith('gstat_')]
    rng = random.Random(args.seed)
    scalar_candidates = []
    list_candidates = []
    for c in g_cols:
        if _is_list_valued_series(gdf[c]):
            list_candidates.append(c)
        else:
            scalar_candidates.append(c)
    init_list: List[str] = []
    if getattr(args, 'init_features_json', ''):
        init_list = _load_init_feature_list(args.init_features_json)
    resolved_include: List[str]
    resolved_exclude: List[str]
    if profile == 'all':
        resolved_include = include
        resolved_exclude = exclude
    elif profile == 'top10':
        resolved_include = ['gstat_punct_ratio', 'gstat_user_len_mean', 'gstat_readability_fk', 'gstat_ellipsis_ratio', 'gstat_user_ttr', 'gstat_emoticon_ratio', 'gstat_rep_bigram_ratio', 'gstat_profanity_ratio', 'gstat_firstperson_ratio', 'gstat_secondperson_ratio']
        resolved_exclude = exclude
    elif profile == 'random10':
        cand = scalar_candidates if bool(args.no_lists) else scalar_candidates + list_candidates
        if len(cand) <= 10:
            chosen = list(cand)
        else:
            chosen = rng.sample(list(cand), 10)
        resolved_include = chosen
        resolved_exclude = exclude
    elif profile == 'drop_readability':
        resolved_include = include or ['gstat_*']
        drop_keys = {'gstat_readability_fk', 'gstat_readability_ari', 'gstat_readability_flesch'}
        resolved_exclude = sorted(set(exclude) | drop_keys)
    else:
        m = re.fullmatch('k(\\d+)(?:v\\d+)?', profile)
        if m:
            k = int(m.group(1))
            if k <= 0:
                raise ValueError(f'feature_profile={profile!r} implies non-positive K={k}')
            if profile == 'k6' and (not init_list):
                resolved_include = ['gstat_user_ttr', 'gstat_user_post_rate', 'gstat_user_subreddit_entropy', 'gstat_user_sr_max_share', 'gstat_punct_ratio', 'gstat_question_ratio']
            elif profile == 'k6v2' and (not init_list):
                resolved_include = ['gstat_user_ttr', 'gstat_firstperson_ratio', 'gstat_user_subreddit_entropy', 'gstat_user_sr_max_share', 'gstat_punct_ratio', 'gstat_question_ratio']
            elif profile == 'k10' and (not init_list):
                resolved_include = ['gstat_punct_ratio', 'gstat_user_len_mean', 'gstat_readability_fk', 'gstat_ellipsis_ratio', 'gstat_user_ttr', 'gstat_emoticon_ratio', 'gstat_rep_bigram_ratio', 'gstat_profanity_ratio', 'gstat_firstperson_ratio', 'gstat_secondperson_ratio']
            elif init_list:
                resolved_include = list(init_list[:k])
            else:
                cand = scalar_candidates if bool(args.no_lists) else scalar_candidates + list_candidates
                resolved_include = _topk_by_variance(gdf, list(cand), k)
            resolved_exclude = exclude
        else:
            resolved_include = include
            resolved_exclude = exclude
            if _is_main():
                print(f"[ablate] unknown feature_profile={profile!r}; falling back to 'all' (no automatic change)")
    cols = set([str(c) for c in gdf.columns])
    missing = [c for c in resolved_include if '*' not in c and '?' not in c and (c not in cols)]
    if missing:
        raise ValueError(f'feature_profile={profile!r} missing required gstat_* columns: {missing}')
    if bool(args.no_lists):
        bad = [c for c in resolved_include if c in list_candidates]
        if bad:
            raise ValueError(f'feature_profile={profile!r} selected list-valued features but --no_lists is set: {bad}')
    if _is_main():
        print(f'[ablate] feature_profile={profile} → include={resolved_include} exclude={resolved_exclude}')
    args.feature_include = resolved_include
    args.feature_exclude = resolved_exclude
    return {'feature_profile': profile, 'resolved_include': resolved_include, 'resolved_exclude': resolved_exclude, 'init_features_json': str(getattr(args, 'init_features_json', '') or ''), 'n_train_users': int(df_tr['target_user_id'].nunique()) if 'target_user_id' in df_tr.columns else None, 'n_authors_rows': int(len(gdf))}

def _resolve_hyper_profile(args: argparse.Namespace) -> Dict[str, Any]:
    profile = getattr(args, 'hyper_profile', 'base')
    profile = str(profile).lower()
    orig = {'hyper_width': args.hyper_width, 'hyper_depth': args.hyper_depth, 'hyper_head_rank': args.hyper_head_rank}
    if profile == 'base':
        pass
    elif profile == 'narrow':
        args.hyper_width = 32
        args.hyper_depth = 6
        args.hyper_head_rank = 8
    elif profile == 'wide':
        args.hyper_width = 128
        args.hyper_depth = 9
        args.hyper_head_rank = 32
    elif _is_main():
        print(f'[ablate] unknown hyper_profile={profile!r}; using explicit CLI values')
    if _is_main():
        print(f"[ablate] hyper_profile={profile} → width {orig['hyper_width']}→{args.hyper_width}, depth {orig['hyper_depth']}→{args.hyper_depth}, head_rank {orig['hyper_head_rank']}→{args.hyper_head_rank}")
    return {'hyper_profile': profile, 'hyper_width': args.hyper_width, 'hyper_depth': args.hyper_depth, 'hyper_head_rank': args.hyper_head_rank}

def _write_ablation_manifest(out_dir: Path, args: argparse.Namespace, feat_info: Dict[str, Any], hyp_info: Dict[str, Any], steps_per_epoch: int, total_steps: int) -> None:
    if not _is_main():
        return
    payload = {'feature_profile': feat_info.get('feature_profile'), 'feature_include': feat_info.get('resolved_include'), 'feature_exclude': feat_info.get('resolved_exclude'), 'init_features_json': feat_info.get('init_features_json', ''), 'hyper_profile': hyp_info.get('hyper_profile'), 'hyper_width': hyp_info.get('hyper_width'), 'hyper_depth': hyp_info.get('hyper_depth'), 'hyper_head_rank': hyp_info.get('hyper_head_rank'), 'seed': int(args.seed), 'train_epochs': int(args.train_epochs), 'steps_per_epoch': int(steps_per_epoch), 'total_optimizer_steps': int(total_steps), 'batch_size': int(args.batch_size), 'grad_accum': int(args.grad_accum), 'microbatch_size': int(args.microbatch_size), 'peft_head': args.peft_head, 'base_ckpt': args.base_ckpt, 'best_metric': str(getattr(args, 'best_metric', 'ce')), 'combo_lambda': float(getattr(args, 'combo_lambda', 0.0))}
    path = out_dir / 'ablation_manifest.json'
    try:
        path.write_text(json.dumps(payload, indent=2))
        print(f'[ablate] wrote ablation_manifest.json → {path}')
    except Exception as e:
        print(f'[ablate] failed to write ablation_manifest.json: {e}')

def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        try:
            state['cuda'] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state

def _restore_rng_state(state: Dict[str, Any]) -> None:
    try:
        if 'python' in state:
            random.setstate(state['python'])
    except Exception:
        pass
    try:
        if 'numpy' in state:
            np.random.set_state(state['numpy'])
    except Exception:
        pass
    try:
        if 'torch' in state:
            torch.set_rng_state(state['torch'])
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            if 'cuda' in state:
                torch.cuda.set_rng_state_all(state['cuda'])
        except Exception:
            pass

def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for st in optimizer.state.values():
        for (k, v) in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device, non_blocking=True)

def _special_row_state(peft_model: nn.Module) -> Optional[List[torch.Tensor]]:
    if hasattr(peft_model, 'special_row_params') and callable(getattr(peft_model, 'special_row_params')):
        try:
            params = list(peft_model.special_row_params())
            if params:
                return [p.detach().cpu() for p in params]
        except Exception:
            return None
    return None

def _load_special_row_state(peft_model: nn.Module, tensors: Optional[List[torch.Tensor]], device: torch.device) -> None:
    if tensors is None:
        return
    if not (hasattr(peft_model, 'special_row_params') and callable(getattr(peft_model, 'special_row_params'))):
        return
    try:
        params = list(peft_model.special_row_params())
    except Exception:
        return
    if len(params) != len(tensors):
        if _is_main():
            print(f'[ckpt] warning: special_row_params len mismatch (ckpt={len(tensors)} vs model={len(params)}); skipping', flush=True)
        return
    with torch.no_grad():
        for (p, t) in zip(params, tensors):
            p.copy_(t.to(device=device, dtype=p.dtype))

def _atomic_torch_save(obj: Any, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(tmp), _use_new_zipfile_serialization=False)
    os.replace(str(tmp), str(path))

def _prune_step_checkpoints(ckpt_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    pats = list(ckpt_dir.glob('checkpoint_step_*.pt'))
    if len(pats) <= keep_last:
        return
    pats.sort(key=lambda p: p.name)
    drop = pats[:max(0, len(pats) - keep_last)]
    for p in drop:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            try:
                p.unlink()
            except Exception:
                pass

def save_training_checkpoint(*, out_dir: Path, model: nn.Module, peft_model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, scaler: GradScaler, step: int, epoch: int, batch_in_epoch: int, best_score: float, accum_bucket: int, keep_last: int) -> None:
    if not _is_main():
        return
    ckpt_dir = Path(out_dir) / 'train_checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    core = model.module if hasattr(model, 'module') else model

    def _cpuify(x: Any) -> Any:
        if torch.is_tensor(x):
            return x.detach().cpu()
        if isinstance(x, dict):
            return {k: _cpuify(v) for (k, v) in x.items()}
        if isinstance(x, list):
            return [_cpuify(v) for v in x]
        if isinstance(x, tuple):
            return tuple((_cpuify(v) for v in x))
        return x
    hyper_state = core.hypernet.state_dict()
    hyper_state = {k: v.detach().cpu() for (k, v) in hyper_state.items()}
    opt_state = _cpuify(optimizer.state_dict())
    sched_state = _cpuify(scheduler.state_dict())
    scaler_state = None
    if getattr(scaler, 'is_enabled', lambda : False)():
        try:
            scaler_state = _cpuify(scaler.state_dict())
        except Exception:
            scaler_state = scaler.state_dict()
    payload: Dict[str, Any] = {'step': int(step), 'epoch': int(epoch), 'batch_in_epoch': int(batch_in_epoch), 'best_score': float(best_score), 'accum_bucket': int(accum_bucket), 'hypernet_state': hyper_state, 'special_rows': _special_row_state(peft_model), 'optimizer_state': opt_state, 'scheduler_state': sched_state, 'scaler_state': scaler_state, 'rng_state': _capture_rng_state(), 'saved_at_unix': float(time.time())}
    last_path = ckpt_dir / 'checkpoint_last.pt'
    _atomic_torch_save(payload, last_path)
    step_path = ckpt_dir / f'checkpoint_step_{int(step):07d}.pt'
    _atomic_torch_save(payload, step_path)
    _prune_step_checkpoints(ckpt_dir, int(keep_last))
    print(f'[ckpt] saved training checkpoint → {last_path} (and {step_path.name})', flush=True)

def load_training_checkpoint(*, ckpt_path: str, model: nn.Module, peft_model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, scaler: GradScaler, device: torch.device) -> Dict[str, Any]:
    ckpt_p = Path(str(ckpt_path))
    if not ckpt_p.exists():
        raise FileNotFoundError(f'resume_from not found: {ckpt_p}')
    payload = torch.load(str(ckpt_p), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f'checkpoint is not a dict: {ckpt_p}')
    core = model.module if hasattr(model, 'module') else model
    if 'hypernet_state' in payload and isinstance(payload['hypernet_state'], dict):
        core.hypernet.load_state_dict(payload['hypernet_state'], strict=True)
    _load_special_row_state(peft_model, payload.get('special_rows', None), device=device)
    if 'optimizer_state' in payload:
        optimizer.load_state_dict(payload['optimizer_state'])
        _optimizer_to(optimizer, device)
    if 'scheduler_state' in payload:
        try:
            scheduler.load_state_dict(payload['scheduler_state'])
        except Exception as e:
            if _is_main():
                print(f'[ckpt] warning: failed to load scheduler state: {e}', flush=True)
    if payload.get('scaler_state', None) is not None and getattr(scaler, 'is_enabled', lambda : False)():
        try:
            scaler.load_state_dict(payload['scaler_state'])
        except Exception as e:
            if _is_main():
                print(f'[ckpt] warning: failed to load GradScaler state: {e}', flush=True)
    if isinstance(payload.get('rng_state', None), dict):
        _restore_rng_state(payload['rng_state'])
    info = {'step': int(payload.get('step', 0)), 'epoch': int(payload.get('epoch', 0)), 'batch_in_epoch': int(payload.get('batch_in_epoch', 0)), 'best_score': float(payload.get('best_score', float('inf'))), 'accum_bucket': int(payload.get('accum_bucket', 0)), 'saved_at_unix': float(payload.get('saved_at_unix', 0.0))}
    if _is_main():
        ts = info['saved_at_unix']
        when = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)) if ts > 0 else 'unknown'
        print(f"[ckpt] resumed from {ckpt_p} (saved_at={when}) step={info['step']} epoch={info['epoch']} batch_in_epoch={info['batch_in_epoch']} best_score={info['best_score']:.6f}", flush=True)
    return info

def train(args: argparse.Namespace) -> None:
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if torch.cuda.is_available() and world > 1:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{local_rank}')
        rank = dist.get_rank()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda' and device.index is None:
            torch.cuda.set_device(0)
        rank = 0
    sidecar_path = args.log_file.strip() if args.log_file else str(Path(args.out_dir) / 'train_hypernet_160M.log')
    _setup_logging(f'hyper{args.peft_head}', rank=rank, log_file=sidecar_path)
    hf_utils.logging.set_verbosity_error()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if _is_main():
        print(f'[seed] using seed={args.seed}')
    _seed_everything(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base_ckpt, local_files_only=True, trust_remote_code=True)
    tok.padding_side = 'right'
    added = False
    if REPLY_SEP_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({'additional_special_tokens': [REPLY_SEP_TOKEN]})
        added = True
    if REPLY_END_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({'additional_special_tokens': [REPLY_END_TOKEN]})
        added = True
    global REPLY_SEP_ID, REPLY_END_ID
    REPLY_SEP_ID = tok.convert_tokens_to_ids(REPLY_SEP_TOKEN)
    REPLY_END_ID = tok.convert_tokens_to_ids(REPLY_END_TOKEN)
    if tok.pad_token_id is None or tok.pad_token_id == tok.eos_token_id:
        tok.add_special_tokens({'pad_token': '[PAD]'})
        tok.pad_token_id = tok.convert_tokens_to_ids('[PAD]')
        added = True
    global BOS_ID, EOS_ID
    BOS_ID = tok.bos_token_id
    EOS_ID = tok.eos_token_id
    backbone = AutoModelForCausalLM.from_pretrained(args.base_ckpt, local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True)
    old_n = int(backbone.get_input_embeddings().weight.size(0))
    new_n = int(len(tok))
    if new_n != old_n:
        if new_n > old_n:
            backbone.resize_token_embeddings(new_n)
            src_id = int(EOS_ID) if EOS_ID is not None else 0
            new_token_ids = list(range(old_n, new_n))
            _init_token_rows_from_id(backbone, new_token_ids, src_id=src_id)
            if _is_main():
                print(f'[tok] resized token embeddings {old_n}→{new_n} • init ids={new_token_ids} from src_id={src_id}', flush=True)
        elif _is_main():
            print(f'[tok] tokenizer len {new_n} < model embeddings {old_n} (padded vocab); keeping embeddings as-is', flush=True)
    backbone.config.use_cache = False
    _resume_path = str(getattr(args, 'resume_from', '') or '').strip()
    resuming = bool(_resume_path) and os.path.isfile(_resume_path)
    if _resume_path and (not resuming) and _is_main():
        print(f'[tok] --resume_from {_resume_path} does not exist yet; treating this as a cold start', flush=True)
    src_id = int(EOS_ID) if EOS_ID is not None else 0
    init_ids = [int(REPLY_SEP_ID), int(REPLY_END_ID), int(tok.pad_token_id)]
    init_ids = [tid for tid in init_ids if tid != src_id]
    _init_token_rows_from_id(backbone, init_ids, src_id=src_id)
    if _is_main():
        print(f"[tok] init special token rows from src_id={src_id} for ids={init_ids} ({('resumed' if resuming else 'cold')} start; seeded on every start since 2026-10-12)", flush=True)
    if args.peft_head == 'lora':
        peft_model = ModelLoRAQKV(backbone, rank=args.lora_rank, alpha=args.lora_alpha, train_token_rows=[int(REPLY_SEP_ID), int(REPLY_END_ID)])
    elif args.peft_head == 'adapter':
        peft_model = ModelAdapter(backbone, bottleneck_dim=args.adapter_bottleneck, adapter_dropout=args.adapter_dropout, use_layer_norm=args.adapter_ln)
    elif args.peft_head == 'bias':
        peft_model = BiasTuningModel(backbone)
    elif args.peft_head == 'prefix':
        if not HAVE_PREFIX:
            raise RuntimeError('prefix head requested, but ModelPrefixOnly is not available in hypernetwork_structure_160M.py')
        peft_model = ModelPrefixOnly(backbone, prefix_length=args.prefix_length, dropout_prob=args.prefix_dropout)
        _seed_prefix_theta(peft_model, args)
    else:
        raise ValueError(f'Unknown --peft_head {args.peft_head}')
    df_tr = pd.read_parquet(args.train_parquet)
    df_va = pd.read_parquet(args.val_parquet)
    gdf = pd.read_parquet(args.author_parquet)
    if 'target_user_id' not in gdf.columns:
        raise KeyError("author_parquet must include 'target_user_id'")
    if not any((str(c).startswith('gstat_') for c in gdf.columns)):
        raise KeyError('author_parquet must provide gstat_* columns built by the feature builder.')
    if args.max_train_rows > 0 and len(df_tr) > args.max_train_rows:
        df_tr = df_tr.sample(n=args.max_train_rows, random_state=args.seed).reset_index(drop=True)
    if args.max_val_rows > 0 and len(df_va) > args.max_val_rows:
        df_va = df_va.sample(n=args.max_val_rows, random_state=args.seed + 1).reset_index(drop=True)
    feat_info = _resolve_feature_profile(args, df_tr=df_tr, gdf=gdf)
    hyp_info = _resolve_hyper_profile(args)
    ds_tr = HypernetGlobalOnlyDataset10000(df=df_tr, tokenizer=tok, author_features_df=gdf, max_length=args.max_len, add_special_tokens=False, pretokenize=False, feature_include=args.feature_include or ['gstat_*'], feature_exclude=args.feature_exclude or [], include_lists=not args.no_lists, zscore=args.zscore_global, zscore_center_only=args.zscore_center_only)
    ds_va = HypernetGlobalOnlyDataset10000(df=df_va, tokenizer=tok, author_features_df=gdf, max_length=args.max_len, add_special_tokens=False, pretokenize=False, feature_include=args.feature_include or ['gstat_*'], feature_exclude=args.feature_exclude or [], include_lists=not args.no_lists, zscore=args.zscore_global, zscore_center_only=args.zscore_center_only)
    pad_id = tok.pad_token_id
    rank_id = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    sampler_tr = DistributedSampler(ds_tr, num_replicas=_world(), rank=rank_id, shuffle=True) if _world() > 1 else None
    ds_va_eval = ds_va
    _nb = int(getattr(args, 'eval_batches', 0) or 0)
    if _nb > 0:
        _need = _nb * int(args.batch_size) * max(1, _world())
        if _need < len(ds_va):
            _g = torch.Generator().manual_seed(int(args.seed))
            _idx = torch.randperm(len(ds_va), generator=_g)[:_need].tolist()
            ds_va_eval = torch.utils.data.Subset(ds_va, sorted(_idx))
            if _is_main():
                _nu = None
                try:
                    _df = getattr(ds_va, 'df', None)
                    if _df is not None and 'target_user_id' in getattr(_df, 'columns', []):
                        _nu_all = int(_df['target_user_id'].nunique())
                        _nu = int(_df['target_user_id'].iloc[sorted(_idx)].nunique())
                except Exception:
                    _nu = None
                _cov = '' if _nu is None else f' covering {_nu:,} of {_nu_all:,} distinct users ({100.0 * _nu / max(1, _nu_all):.2f}%)'
                print(f'[val] eval subset: {len(ds_va_eval):,} of {len(ds_va):,} rows (seed {int(args.seed)}, uniform random, fixed across steps){_cov}', flush=True)
        elif _is_main():
            print(f'[val] eval subset: --eval_batches {_nb} covers the whole split; using all {len(ds_va):,} rows', flush=True)
    elif _is_main():
        print(f'[val] eval subset: --eval_batches 0 (no cap); using all {len(ds_va):,} rows', flush=True)
    sampler_va = DistributedSampler(ds_va_eval, num_replicas=_world(), rank=rank_id, shuffle=False) if _world() > 1 else None
    dl_tr_kwargs = dict(dataset=ds_tr, batch_size=args.batch_size, shuffle=sampler_tr is None, sampler=sampler_tr, num_workers=args.num_workers, pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0, drop_last=True)
    if args.num_workers > 0:
        dl_tr_kwargs['prefetch_factor'] = args.prefetch_factor
    dl_va_kwargs = dict(dataset=ds_va_eval, batch_size=args.batch_size, shuffle=False, sampler=sampler_va, num_workers=args.num_workers, pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0)
    if args.num_workers > 0:
        dl_va_kwargs['prefetch_factor'] = args.prefetch_factor
    dl_tr = DataLoader(**dl_tr_kwargs)
    dl_va = DataLoader(**dl_va_kwargs)
    steps_per_epoch = _steps_per_epoch_exact(dl_tr, batch_size=args.batch_size, microbatch_size=args.microbatch_size, grad_accum=args.grad_accum, sampler=sampler_tr)
    n_samples = len(ds_tr)
    n_users = df_tr['target_user_id'].nunique()
    batches_per_epoch = len(dl_tr)
    if _is_main():
        print(f'[data] train samples {n_samples:,} • train users {n_users:,} • batches/epoch {batches_per_epoch:,} • steps/epoch {steps_per_epoch:,}', flush=True)
    if args.train_epochs <= 0:
        raise ValueError(f'--train_epochs must be >=1 for epoch-based training; got {args.train_epochs}')
    total_steps = int(max(1, args.train_epochs) * steps_per_epoch)
    if _is_main():
        print(f'[train] epochs={args.train_epochs} × steps/epoch={steps_per_epoch} → total optimizer steps={total_steps}', flush=True)
    exclude_ids: List[int] = []
    if hasattr(peft_model, 'special_row_params') and callable(getattr(peft_model, 'special_row_params')):
        try:
            for p in list(peft_model.special_row_params()):
                exclude_ids.append(id(p))
        except Exception:
            exclude_ids = []
    P = count_trainable_params(peft_model, exclude_param_ids=exclude_ids)
    Kdim = ds_tr.g_dim
    out_dir = Path(args.out_dir)
    if _is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'feature_names.json').write_text(json.dumps(getattr(ds_tr, 'feature_names', []), indent=2))
        stats_payload = getattr(ds_tr, 'feature_stats_payload', {'names': getattr(ds_tr, 'feature_names', []), 'mean': None, 'std': None})
        (out_dir / 'feature_stats.json').write_text(json.dumps(stats_payload, indent=2))
        _write_ablation_manifest(out_dir, args, feat_info=feat_info, hyp_info=hyp_info, steps_per_epoch=steps_per_epoch, total_steps=total_steps)
    group_splits = _placeholder_splits_by_layer(peft_model, exclude_param_ids=exclude_ids) if args.hyper_group_by_layer else None
    hyper = FlatHypernetwork(input_dim=Kdim, output_dim=P, width=args.hyper_width, depth=args.hyper_depth, dropout=args.hyper_dropout, clamp=args.clamp_min, head_rank=args.hyper_head_rank, group_splits=group_splits)
    if group_splits is not None and _is_main():
        print(f'[hyper] group head enabled • groups={len(group_splits)} • sum={sum(group_splits)} ≟ P={P}')
    wrapper = PEFTHypernet(peft_model, hyper, clamp=args.clamp_min).to(device)
    model = _wrap_ddp(wrapper, device)
    if _is_main():
        print('[preflight] running contract checks and golden batch…', flush=True)
    _preflight_contract_tests(peft_model, hyper, expected_placeholders=P)
    use_amp = device.type == 'cuda' and (not args.fp32)
    amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    _golden_batch_smoke(model=model, tokenizer=tok, ds=ds_tr, device=device, pad_id=pad_id, max_len=args.max_len, use_amp=use_amp, amp_dtype=amp_dtype)
    if _is_main():
        pass
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if _is_main():
        core = model.module if hasattr(model, 'module') else model
        hn_params = sum((p.numel() for p in core.hypernet.parameters()))
        peft_placeholders = P
        try:
            base_params = sum((p.numel() for p in peft_model.backbone.parameters()))
        except Exception:
            base_params = sum((p.numel() for p in backbone.parameters()))
        extra_trainable = 0
        if hasattr(peft_model, 'special_row_params') and callable(getattr(peft_model, 'special_row_params')):
            try:
                extra_trainable = int(sum((int(p.numel()) for p in peft_model.special_row_params())))
            except Exception:
                extra_trainable = 0
        trainable_params = int(hn_params + extra_trainable)
        ratio_hn_vs_base = trainable_params / max(1, base_params)
        print(f'[peft] backbone {base_params / 1000000.0:.2f} M • placeholders {peft_placeholders / 1000000.0:.2f} M • hypernet {hn_params / 1000000.0:.2f} M • special_tokens {extra_trainable / 1000000.0:.4f} M • trainable {trainable_params / 1000000.0:.2f} M ({100.0 * ratio_hn_vs_base:.1f}% of backbone)')
        if hasattr(core.hypernet, 'head_rank') and getattr(core.hypernet, 'head_rank', 0) > 0:
            r = core.hypernet.head_rank
            if hasattr(core.hypernet, 'head_U') and hasattr(core.hypernet, 'head_V'):
                u_params = sum((p.numel() for p in core.hypernet.head_U.parameters()))
                v_params = sum((p.numel() for p in core.hypernet.head_V.parameters()))
                head_params = u_params + v_params
                print(f'[peft] factorized head rank={r} • head params {head_params / 1000000.0:.2f} M • hypernet {hn_params / 1000000.0:.2f} M', flush=True)
    tgt = model.module if hasattr(model, 'module') else model
    params = list(tgt.hypernet.parameters())
    if hasattr(peft_model, 'special_row_params') and callable(getattr(peft_model, 'special_row_params')):
        extra = list(peft_model.special_row_params())
        if extra:
            params.extend(extra)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    warm = max(1, int(total_steps * args.warmup_frac))
    rest = max(1, total_steps - warm)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warm), torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rest, eta_min=args.lr * float(args.cosine_floor_frac))], milestones=[warm])
    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    if not str(getattr(args, 'resume_from', '')).strip():
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if _is_main():
            print('[preflight] one-step learning probe (verifies Δθ and reply-region CE start/mid/end)…', flush=True)
        _preflight_one_step_probe(model=model, peft_model=peft_model, tokenizer=tok, ds=ds_tr, device=device, pad_id=pad_id, max_len=args.max_len, use_amp=use_amp, amp_dtype=amp_dtype, lr=args.lr)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    if _is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'args.json').write_text(json.dumps(vars(args), indent=2))
    resume_path = str(getattr(args, 'resume_from', '') or '').strip()
    start_epoch = 0
    start_batch = 0
    step = 0
    best_score = float('inf')
    accum_bucket = 0
    if resume_path:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        info = load_training_checkpoint(ckpt_path=resume_path, model=model, peft_model=peft_model, optimizer=opt, scheduler=sched, scaler=scaler, device=device)
        step = int(info.get('step', 0))
        best_score = float(info.get('best_score', float('inf')))
        start_epoch = int(info.get('epoch', 0))
        start_batch = int(info.get('batch_in_epoch', 0))
        accum_bucket = int(info.get('accum_bucket', 0))
        if accum_bucket != 0:
            if _is_main():
                print(f'[ckpt] warning: accum_bucket={accum_bucket} on resume; resetting to 0', flush=True)
            accum_bucket = 0
        opt.zero_grad(set_to_none=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    model.train()
    opt.zero_grad(set_to_none=True)
    last_hb_ts = time.time()
    last_ckpt_ts = time.time()
    last_ckpt_step = int(step)
    prev_delta_mean: Optional[torch.Tensor] = None
    for epoch_idx in range(start_epoch, args.train_epochs):
        if sampler_tr is not None:
            sampler_tr.set_epoch(epoch_idx)
        for (batch_idx, batch) in enumerate(dl_tr):
            if epoch_idx == start_epoch and batch_idx < start_batch:
                continue
            clamp = _clamp_schedule(step, start=args.unclamp_start, duration=args.unclamp_duration, cmin=args.clamp_min, cmax=args.clamp_max)
            (model.module if hasattr(model, 'module') else model).hypernet.clamp = float(clamp)
            batch = {k: v.to(device) for (k, v) in batch.items()}
            _make_concat_inputs(batch, pad_id, args.max_len)
            B = batch['input_ids'].size(0)
            mb = max(1, int(args.microbatch_size))
            accum = max(1, int(args.grad_accum))
            s0 = 0
            while s0 < B:
                e0 = min(B, s0 + mb)
                sub = {k: v[s0:e0] if torch.is_tensor(v) and v.size(0) == B else v for (k, v) in batch.items()}
                sync_now = (accum_bucket + 1) % accum == 0
                outer_ctx = model.no_sync() if _world() > 1 and (not sync_now) else nullcontext()
                with outer_ctx:
                    with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                        with _sdp_no_flash():
                            out = model(input_ids=sub['input_ids'], attention_mask=sub['attention_mask'], labels=sub['labels'], global_features=sub['global_features'], use_cache=False, return_dict=True, return_loss_only=True)
                            loss_tf = out.loss
                        (ce_b, kept) = _boundary_ce_logits(model=model, input_ids=sub['input_ids'], attention_mask=sub['attention_mask'], labels=sub['labels'], device=device, pad_id=pad_id, use_amp=use_amp, amp_dtype=amp_dtype, global_features=sub['global_features'])
                        loss_b = ce_b if kept > 0 else ce_b.new_zeros(())
                        delta = (model.module if hasattr(model, 'module') else model).hypernet(sub['global_features'])
                        loss_reg = delta.pow(2).mean()
                        drift_pen = 0.0
                        if args.ctx_drift_coef > 0.0:
                            (ce_ctx_mb, _) = _context_only_ce(model=model, input_ids=sub['input_ids'], attention_mask=sub['attention_mask'], device=device, use_amp=use_amp, amp_dtype=amp_dtype, global_features=sub['global_features'], disable_injection=True)
                            (ce_guard_mb, _) = _context_only_ce(model=model, input_ids=sub['input_ids'], attention_mask=sub['attention_mask'], device=device, use_amp=use_amp, amp_dtype=amp_dtype, global_features=sub['global_features'], disable_injection=False)
                            drift_pen = torch.relu(ce_guard_mb - ce_ctx_mb) * float(args.ctx_drift_coef)
                        loss = loss_tf + args.boundary_ce_weight * loss_b + args.l2_delta_coef * loss_reg + (drift_pen if isinstance(drift_pen, torch.Tensor) else 0.0)
                        loss = loss / float(accum)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                s0 = e0
                accum_bucket += 1
                if _is_main() and int(getattr(args, 'heartbeat_secs', 0)) > 0:
                    _now = time.time()
                    if _now - last_hb_ts >= float(args.heartbeat_secs):
                        try:
                            _lr_now = float(sched.get_last_lr()[0])
                        except Exception:
                            _lr_now = float(args.lr)
                        print(f'[hb] epoch {epoch_idx + 1}/{args.train_epochs} • batch {batch_idx + 1}/{batches_per_epoch} • micro {e0}/{B} • step {step}/{total_steps} • accum {accum_bucket}/{accum} • clamp ±{clamp:.3f} • lr {_lr_now:.2e}', flush=True)
                        last_hb_ts = _now
                if accum_bucket % accum == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    step += 1
                    accum_bucket = 0
                    if _is_main() and (step % args.log_interval == 0 or step == 1):
                        with torch.no_grad():
                            d = delta.detach()
                            row_l2 = d.view(d.size(0), -1).norm(dim=1).mean().item()
                            zero_pct = (d == 0).float().mean().item()
                            if d.size(0) > 1:
                                v = d.var(dim=0, unbiased=False)
                                k = min(10, v.numel())
                                var_top10 = torch.topk(v, k, sorted=False).values.mean().item()
                            else:
                                var_top10 = 0.0
                            cur_mean = d.mean(dim=0).flatten()
                            cur_norm = cur_mean.norm().item() + 1e-12
                            if prev_delta_mean is None:
                                dot_prev = 0.0
                                cos_prev = 0.0
                            else:
                                dot_prev = float(torch.dot(cur_mean, prev_delta_mean).item())
                                cos_prev = float(dot_prev / (cur_norm * (prev_delta_mean.norm().item() + 1e-12)))
                            prev_delta_mean = cur_mean.clone()
                            (ctx_ce_mb, _) = _context_only_ce(model=model, input_ids=sub['input_ids'], attention_mask=sub['attention_mask'], device=device, use_amp=use_amp, amp_dtype=amp_dtype, global_features=sub['global_features'], disable_injection=True)
                            (guard_ce_mb, _) = _context_only_ce(model=model, input_ids=sub['input_ids'], attention_mask=sub['attention_mask'], device=device, use_amp=use_amp, amp_dtype=amp_dtype, global_features=sub['global_features'], disable_injection=False)
                            ctx_now = float(ctx_ce_mb.item())
                            guard_now = float(guard_ce_mb.item())
                        ce_now = float(loss_tf.detach().item())
                        b_now = float(loss_b.detach().item()) if torch.is_tensor(loss_b) else float(loss_b)
                        lr_now = sched.get_last_lr()[0]
                        core_hn = (model.module if hasattr(model, 'module') else model).hypernet
                        splits = getattr(core_hn, 'group_splits', None)
                        grp_top = ''
                        if splits:
                            ptr = 0
                            gl2 = []
                            for s_seg in splits:
                                seg = d[:, ptr:ptr + s_seg].reshape(d.size(0), -1)
                                gl2.append(seg.norm(dim=1).mean())
                                ptr += s_seg
                            gl2_t = torch.stack(gl2)
                            k_top = int(min(3, gl2_t.numel()))
                            (vals, idxs) = torch.topk(gl2_t, k=k_top, largest=True)
                            grp_top = ' '.join([f'{int(i)}:{float(v):.3e}' for (v, i) in zip(vals.tolist(), idxs.tolist())])
                        print(f'step {step:6d}/{total_steps} • CE(TF) {ce_now:.4f} • PPL(TF) {_ppl(ce_now):.1f} • CE(CTX) {ctx_now:.4f} • PPL(CTX) {_ppl(ctx_now):.1f} • CE(guard) {guard_now:.4f} • PPL(guard) {_ppl(guard_now):.1f} • CE(RESP_START) {b_now:.4f} • clamp ±{clamp:.3f} • ‖δθ‖₂/row {row_l2:.4e} • δ0% {100 * zero_pct:.2f}% • var_top10 {var_top10:.2e} • gL2_top3 {grp_top} • δθ·prev {dot_prev:.2f} • cos(prev) {cos_prev:.3f} • lr {lr_now:.2e} • micro={mb} • accum={accum}')
                    if step % args.eval_every == 0 or step == total_steps:
                        if sampler_va is not None:
                            sampler_va.set_epoch(step)
                        do_reduce = bool(args.eval_reduce)
                        if _world() > 1 and (not do_reduce):
                            if dist.get_rank() == 0:
                                metrics = evaluate(model, dl_va, device, pad_id, args.max_len, use_amp, amp_dtype, max_batches=args.eval_batches, ddp_reduce=False, progress_every=args.eval_progress_every, heartbeat_secs=args.eval_heartbeat_secs)
                            else:
                                metrics = None
                            if dist.is_available() and dist.is_initialized():
                                dist.barrier()
                        else:
                            metrics = evaluate(model, dl_va, device, pad_id, args.max_len, use_amp, amp_dtype, max_batches=args.eval_batches, ddp_reduce=True, progress_every=args.eval_progress_every, heartbeat_secs=args.eval_heartbeat_secs)
                        if _is_main() and metrics is not None:
                            score = _score_from_metrics(args, metrics)
                            print(f"[eval  {step:6d}] CE(TF) {metrics['ce']:.4f} • PPL(TF) {metrics['ppl']:.1f} • CE(CTX) {metrics['ctx']:.4f} • CE(guard) {metrics['guard']:.4f} • CE(RESP_START) {metrics['resp_start']:.4f} • CE(RESP_MID) {metrics['resp_mid']:.4f} • CE(RESP_END) {metrics['resp_end']:.4f} • score({args.best_metric}) {score:.4f}", flush=True)
                            ctx = evaluate_ctx_generation(model=model, loader=dl_va, tokenizer=tok, device=device, max_samples=min(200, len(ds_va)), max_new_tokens=64, do_sample=False, top_p=1.0, temperature=1.0, heartbeat_secs=int(getattr(args, 'eval_heartbeat_secs', 60)))
                            print(f"[ctx   {step:6d}] first_tok_acc {ctx['first_token_acc']:.3f} • BLEU-1 {ctx['bleu1']:.3f} • BLEU-2 {ctx['bleu2']:.3f} • avg_len {ctx['avg_len']:.1f}", flush=True)
                            if score < best_score:
                                best_score = score
                                ckpt_path = out_dir / 'best_hypernet.safetensors'
                                (model.module if hasattr(model, 'module') else model).save_hypernet_ckpt(str(ckpt_path))
                                tok.save_pretrained(out_dir)
                                best_blob = dict(metrics)
                                best_blob.update({'best_metric': str(args.best_metric), 'combo_lambda': float(args.combo_lambda), 'best_score': float(best_score), 'ctx_first_token_acc': ctx['first_token_acc'], 'ctx_bleu1': ctx['bleu1'], 'ctx_bleu2': ctx['bleu2'], 'ctx_avg_len': ctx['avg_len']})
                                (out_dir / 'best_metrics.json').write_text(json.dumps(best_blob, indent=2))
                                print(f'[ckpt] saved → {ckpt_path}', flush=True)
                    if step >= total_steps:
                        break
            if accum_bucket == 0:
                _do_ckpt = False
                if int(getattr(args, 'ckpt_every_steps', 0)) > 0:
                    if step - last_ckpt_step >= int(args.ckpt_every_steps):
                        _do_ckpt = True
                if float(getattr(args, 'ckpt_every_minutes', 0.0)) > 0.0:
                    if time.time() - last_ckpt_ts >= float(args.ckpt_every_minutes) * 60.0:
                        _do_ckpt = True
                if dist.is_available() and dist.is_initialized():
                    _flag = 1 if _is_main() and _do_ckpt else 0
                    _flag = _ddp_broadcast_int(_flag, device=device, src=0)
                    _do_ckpt = bool(int(_flag) == 1)
                if _do_ckpt:
                    next_epoch = int(epoch_idx)
                    next_batch = int(batch_idx + 1)
                    if next_batch >= int(batches_per_epoch):
                        next_epoch = int(epoch_idx + 1)
                        next_batch = 0
                    if _is_main():
                        print(f'[ckpt] begin periodic checkpoint • step {int(step)} • next_epoch {int(next_epoch)} • next_batch {int(next_batch)}', flush=True)
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    if _is_main():
                        print('[ckpt] all ranks reached checkpoint barrier', flush=True)
                    save_training_checkpoint(out_dir=out_dir, model=model, peft_model=peft_model, optimizer=opt, scheduler=sched, scaler=scaler, step=int(step), epoch=int(next_epoch), batch_in_epoch=int(next_batch), best_score=float(best_score), accum_bucket=int(accum_bucket), keep_last=int(getattr(args, 'ckpt_keep_last', 3)))
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    if _is_main():
                        print('[ckpt] checkpoint barrier complete', flush=True)
                    _new_ts = time.time() if _is_main() else 0.0
                    if dist.is_available() and dist.is_initialized():
                        _new_ts = _ddp_broadcast_float(_new_ts, device=device, src=0)
                    last_ckpt_ts = float(_new_ts)
                    last_ckpt_step = int(step)
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    save_training_checkpoint(out_dir=out_dir, model=model, peft_model=peft_model, optimizer=opt, scheduler=sched, scaler=scaler, step=int(step), epoch=int(args.train_epochs), batch_in_epoch=0, best_score=float(best_score), accum_bucket=0, keep_last=int(getattr(args, 'ckpt_keep_last', 3)))
    if _is_main():
        final_path = out_dir / 'final_hypernet.safetensors'
        (model.module if hasattr(model, 'module') else model).save_hypernet_ckpt(str(final_path))
        print(f'[done] wrote {final_path}', flush=True)
        try:
            fi_batches = max(1, int(args.fi_batches))
            (imp, Kdim_imp) = _feature_importance_absgrad(model=model, loader=dl_va, device=device, pad_id=pad_id, max_len=args.max_len, use_amp=use_amp, amp_dtype=amp_dtype, batches=fi_batches)
            names: List[str] = []
            feat_names_path = out_dir / 'feature_names.json'
            if feat_names_path.exists():
                try:
                    names = json.loads(feat_names_path.read_text())
                except Exception:
                    names = []
            if not names or len(names) != Kdim_imp:
                names = [f'g[{i}]' for i in range(Kdim_imp)]
            order = np.argsort(-imp)
            topk = int(args.fi_topk) if args.fi_topk > 0 else len(order)
            payload = [{'rank': int(r + 1), 'feature': names[idx], 'absgrad': float(imp[idx])} for (r, idx) in enumerate(order[:topk])]
            (out_dir / 'feature_importance.json').write_text(json.dumps(payload, indent=2))
            print(f'[fi] wrote feature_importance.json (top {topk}/{Kdim_imp})', flush=True)
        except Exception as e:
            print(f'[fi] feature importance failed: {e}')
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Flat hypernetwork training on Pythia-160M with dynamic global-static features (gstat_*).')
    ap.add_argument('--train_parquet', type=str, required=True)
    ap.add_argument('--val_parquet', type=str, required=True)
    ap.add_argument('--author_parquet', type=str, required=True, help='One row per user with gstat_* columns from the feature builder.')
    ap.add_argument('--max_train_rows', type=int, default=0)
    ap.add_argument('--max_val_rows', type=int, default=0)
    ap.add_argument('--feature_include', type=str, nargs='*', default=['gstat_*'], help='Glob patterns to include (default: gstat_*)')
    ap.add_argument('--feature_exclude', type=str, nargs='*', default=['gstat_user_sent_*', 'gstat_gap_sentiment', 'gstat_beast_sent_*'], help='Glob patterns to exclude')
    ap.add_argument('--no_lists', action='store_true', help='Exclude list-valued gstat_* (use scalars only).')
    ap.add_argument('--init_features_json', type=str, default='', help='Optional JSON with ordered feature names (list[str] or list[dict]) used by k{N} profiles.')
    ap.add_argument('--feature_profile', type=str, default='k6v2', choices=['all', 'top10', 'k30', 'k20', 'k10', 'k6', 'k6v2', 'k5', 'k3', 'random10', 'drop_readability'], help="High-level feature selection profile. 'k{N}' uses init_features_json if provided, else top-N by variance.")
    ap.add_argument('--base_ckpt', type=str, default='EleutherAI/pythia-160m')
    choices = ['lora', 'adapter', 'bias'] + (['prefix'] if HAVE_PREFIX else [])
    ap.add_argument('--peft_head', type=str, default='lora', choices=choices)
    ap.add_argument('--lora_rank', type=int, default=32)
    ap.add_argument('--lora_alpha', type=float, default=32.0)
    ap.add_argument('--adapter_bottleneck', type=int, default=256)
    ap.add_argument('--adapter_dropout', type=float, default=0.0)
    ap.add_argument('--adapter_ln', action='store_true', default=False)
    ap.add_argument('--prefix_length', type=int, default=10)
    ap.add_argument('--prefix_dropout', type=float, default=0.0)
    ap.add_argument('--prefix_theta_init', default='random', choices=['random', 'zero', 'vanilla'], help="What the FROZEN prefix placeholders hold under the generated delta. 'random' is the historical behavior, nn.init.normal_(std=0.02). 'zero' is the neutral base the other three heads already get. 'vanilla' copies the trained shared prefix head named by --prefix_theta_from. Only the hypernetwork trains in a hyper run, so whatever this sets is what the arm carries for all 21,846 updates.")
    ap.add_argument('--prefix_theta_from', default='', help='Path to a trained shared-head peft_placeholders.safetensors. Required by, and only read under, --prefix_theta_init vanilla.')
    ap.add_argument('--hyper_width', type=int, default=64)
    ap.add_argument('--hyper_depth', type=int, default=9)
    ap.add_argument('--hyper_dropout', type=float, default=0.0)
    ap.add_argument('--hyper_head_rank', type=int, default=16)
    ap.add_argument('--hyper_group_by_layer', action='store_true', default=False)
    ap.add_argument('--hyper_profile', type=str, default='base', choices=['base', 'narrow', 'wide'], help='High-level hypernetwork capacity profile.')
    ap.add_argument('--clamp_min', type=float, default=0.05)
    ap.add_argument('--clamp_max', type=float, default=0.1)
    ap.add_argument('--unclamp_start', type=int, default=500)
    ap.add_argument('--unclamp_duration', type=int, default=2500)
    ap.add_argument('--train_epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--microbatch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--weight_decay', type=float, default=5e-05)
    ap.add_argument('--warmup_frac', type=float, default=0.05)
    ap.add_argument('--cosine_floor_frac', type=float, default=0.3, help='Cosine decay floor as a fraction of peak LR. Default 0.3 is the\nhistorical value. train_nonhypernet_10000.py decays to 0, so pass 0.0\nfor a schedule that matches the non-hyper baselines.')
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--boundary_ce_weight', type=float, default=0.1)
    ap.add_argument('--l2_delta_coef', type=float, default=0.0001)
    ap.add_argument('--log_interval', type=int, default=1000)
    ap.add_argument('--eval_every', type=int, default=600)
    ap.add_argument('--eval_batches', type=int, default=0)
    ap.add_argument('--eval_reduce', type=int, default=1)
    ap.add_argument('--eval_progress_every', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=12)
    ap.add_argument('--prefetch_factor', type=int, default=4)
    ap.add_argument('--fp32', action='store_true', default=False)
    ap.add_argument('--best_metric', type=str, default='ce', choices=['ce', 'bnd_apply', 'combo', 'ctx', 'guard', 'guard_gap'], help='Metric to minimize for best_hypernet checkpoint selection.')
    ap.add_argument('--combo_lambda', type=float, default=0.0, help='Lambda for combo metric: ce + lambda*bnd_apply.')
    ap.add_argument('--fi_batches', type=int, default=24)
    ap.add_argument('--fi_topk', type=int, default=200)
    ap.add_argument('--zscore_global', action='store_true', default=False)
    ap.add_argument('--zscore_center_only', action='store_true', default=False)
    ap.add_argument('--ctx_drift_coef', type=float, default=0.0)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--log_file', type=str, default='')
    ap.add_argument('--resume_from', type=str, default='', help='Optional path to a training checkpoint (.pt) to resume from (see out_dir/train_checkpoints/checkpoint_last.pt).')
    ap.add_argument('--ckpt_every_steps', type=int, default=1000, help='Save a full training checkpoint (model+optim+sched+scaler) every N optimizer steps (0 disables).')
    ap.add_argument('--ckpt_every_minutes', type=float, default=30.0, help='Also save a training checkpoint at least every N minutes (0 disables).')
    ap.add_argument('--ckpt_keep_last', type=int, default=3, help='Keep only the last N numbered step checkpoints (checkpoint_last.pt is always overwritten).')
    ap.add_argument('--heartbeat_secs', type=int, default=60, help='Emit a training heartbeat log line at least every N seconds (0 disables).')
    ap.add_argument('--eval_heartbeat_secs', type=int, default=60, help='Emit an eval heartbeat log line at least every N seconds (0 disables).')
    ap.add_argument('--seed', type=int, default=142)
    ap.add_argument('-f', '--f', help=argparse.SUPPRESS)
    (args, _unknown) = ap.parse_known_args()
    return args

@torch.no_grad()
def _print_reply_region_breakdown_from_batch(*, model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device, use_amp: bool, amp_dtype: torch.dtype) -> None:
    core = model.module if hasattr(model, 'module') else model
    with _sdp_no_flash():
        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=None, global_features=batch['global_features'], use_cache=False, return_dict=True)
            logits = out.logits
    if REPLY_END_ID is None:
        print('[probe] REPLY_END_ID is None; cannot compute reply-region breakdown', flush=True)
        return
    br = _reply_region_breakdown_ce(logits=logits, input_ids=batch['input_ids'], labels=batch['labels'], reply_end_id=int(REPLY_END_ID))
    (ce_s, n_s) = br['resp_start']
    (ce_m, n_m) = br['resp_mid']
    (ce_e, n_e) = br['resp_end']
    print(f'[probe] region CE: start {ce_s:.4f} (n={n_s}) • mid {ce_m:.4f} (n={n_m}) • end {ce_e:.4f} (n={n_e})', flush=True)

def _preflight_one_step_probe(*, model: nn.Module, peft_model: nn.Module, tokenizer: AutoTokenizer, ds: Dataset, device: torch.device, pad_id: int, max_len: int, use_amp: bool, amp_dtype: torch.dtype, lr: float) -> None:
    if len(ds) == 0:
        raise RuntimeError('[probe] dataset empty')
    core = model.module if hasattr(model, 'module') else model
    idxs = [0, min(1, len(ds) - 1)]
    items = [ds[i] for i in idxs]

    def _stack(key: str) -> torch.Tensor:
        return torch.stack([it[key] for it in items], dim=0).to(device)
    batch = {'input_ids': _stack('input_ids'), 'attention_mask': _stack('attention_mask'), 'labels': _stack('labels'), 'global_features': torch.stack([it['global_features'] for it in items], dim=0).to(device)}
    _make_concat_inputs(batch, pad_id, max_len)
    if _is_main():
        print('[probe] BEFORE one-step update:', flush=True)
        _print_reply_region_breakdown_from_batch(model=model, batch=batch, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
    d0_l2 = None
    if _is_main():
        with torch.no_grad():
            d0 = core.hypernet(batch['global_features'])
            d0_l2 = d0.view(d0.size(0), -1).norm(dim=1).mean().item()
    params = list(core.hypernet.parameters())
    if hasattr(peft_model, 'special_row_params') and callable(getattr(peft_model, 'special_row_params')):
        extra = list(peft_model.special_row_params())
        if extra:
            params.extend(extra)
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0, betas=(0.9, 0.95))
    model.train()
    opt.zero_grad(set_to_none=True)
    with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
        with _sdp_no_flash():
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], global_features=batch['global_features'], use_cache=False, return_dict=True, return_loss_only=True)
            loss = out.loss
    if not torch.isfinite(loss):
        raise RuntimeError('[probe] loss is not finite')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    model.eval()
    if _is_main():
        with torch.no_grad():
            d1 = core.hypernet(batch['global_features'])
            d1_l2 = d1.view(d1.size(0), -1).norm(dim=1).mean().item()
        if d0_l2 is not None:
            print(f'[probe] Δθ L2/row: before {d0_l2:.4e} → after {d1_l2:.4e}', flush=True)
        print('[probe] AFTER one-step update:', flush=True)
        _print_reply_region_breakdown_from_batch(model=model, batch=batch, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
        _author_swap_sanity(model=model, tokenizer=tokenizer, ds=ds, device=device, pad_id=pad_id, max_len=max_len, use_amp=use_amp, amp_dtype=amp_dtype, samples=32)
if __name__ == '__main__':
    args = parse_args()
    t0 = time.time()
    try:
        train(args)
    finally:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
    if _is_main():
        print(f'Runtime: {time.time() - t0:.1f}s', flush=True)
