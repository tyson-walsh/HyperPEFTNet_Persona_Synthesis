from __future__ import annotations
import argparse
import fnmatch
import os
import sys
import textwrap
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
_SANITY_PRINTED = False
_DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / 'data'
_DATA_ROOT = Path(os.environ.get('HYPERNETS_DATA_ROOT', _DEFAULT_DATA_ROOT))

def _is_main_process() -> bool:
    try:
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0
    except Exception:
        return True

class RedditConversationDataset10000(Dataset):

    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer, *, max_length: int=512, add_special_tokens: bool=True, pretokenize: bool=False) -> None:
        super().__init__()
        required = {'gid', 'group_label', 'text', 'target_user_id'}
        if not required.issubset(dataframe.columns):
            raise ValueError(f'DataFrame must contain {required} - found {dataframe.columns.tolist()}')
        df = dataframe.copy()
        df.sort_values(['gid', 'group_label'], inplace=True, ignore_index=True)
        items: List[Dict[str, Any]] = []
        for (gid, sub) in df.groupby('gid', sort=False):
            tgt_rows = sub[sub['group_label'].astype(str).str.endswith('_target')]
            if tgt_rows.empty:
                continue
            final_row = tgt_rows.iloc[-1]
            tgt_text = str(final_row['text']).strip()
            if not tgt_text:
                continue
            sub_sorted = sub.sort_values('group_label')
            cut_idx = sub_sorted.index.get_loc(final_row.name)
            ctx_lines = sub_sorted.iloc[:cut_idx]['text'].tolist()
            ctx_text = '\n'.join(map(str, ctx_lines)).strip()
            if not ctx_text:
                continue
            items.append(dict(gid=int(gid), target_user_id=int(final_row['target_user_id']), context_text=ctx_text, target_text=tgt_text))
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.add_special_tokens = bool(add_special_tokens)
        self._cache = [self._encode(it) for it in items] if pretokenize else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._cache[idx] if self._cache is not None else self._encode(self.items[idx])

    def _encode(self, item: Dict[str, Any]) -> Dict[str, Any]:
        ctx = item['context_text']
        tgt = item['target_text']
        enc_ctx = self.tokenizer(ctx, truncation=True, max_length=self.max_length, padding='max_length', add_special_tokens=self.add_special_tokens, return_tensors='pt')
        enc_tgt = self.tokenizer(tgt, truncation=True, max_length=self.max_length, padding='max_length', add_special_tokens=self.add_special_tokens, return_tensors='pt')
        input_ids = enc_ctx['input_ids'].squeeze(0)
        attention_mask = enc_ctx['attention_mask'].squeeze(0)
        labels = enc_tgt['input_ids'].squeeze(0)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels = labels.clone()
            labels[labels == pad_id] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'target_user_id': int(item['target_user_id']), 'gid': int(item['gid'])}

def _ensure_int_series(s: pd.Series) -> pd.Series:
    if str(s.dtype).startswith('int'):
        return s
    try:
        return s.astype('int64')
    except Exception:
        return pd.to_numeric(s, errors='coerce').fillna(0).astype('int64')

def _is_scalar(x: Any) -> bool:
    return isinstance(x, (int, float, np.generic))

def _flatten_numeric(v: Any) -> List[float]:
    if _is_scalar(v):
        return [float(v)]
    if isinstance(v, np.ndarray):
        return [float(x) for x in v.ravel().tolist()]
    if isinstance(v, (list, tuple)):
        out: List[float] = []
        for elem in v:
            out.extend(_flatten_numeric(elem))
        return out or [0.0]
    return [0.0]

def _match_any(name: str, globs: Sequence[str]) -> bool:
    return any((fnmatch.fnmatch(name, pat) for pat in globs))

class _FeatureSpec:

    def __init__(self, author_df: pd.DataFrame, include: Sequence[str], exclude: Sequence[str], include_lists: bool) -> None:
        all_cols = [c for c in author_df.columns if c.startswith('gstat_')]
        cols = [c for c in all_cols if _match_any(c, include)] if include else list(all_cols)
        if exclude:
            cols = [c for c in cols if not _match_any(c, exclude)]
        scalars: List[str] = []
        lists: List[str] = []
        for c in cols:
            dt = author_df[c].dtype
            if pd.api.types.is_object_dtype(dt):
                lists.append(c)
            elif len(author_df) and pd.api.types.is_list_like(author_df[c].iloc[0]):
                lists.append(c)
            else:
                scalars.append(c)
        if not include_lists:
            lists = []
        flat_names: List[str] = []
        parts: List[Tuple[str, int]] = []
        for c in sorted(scalars):
            flat_names.append(c)
            parts.append((c, 1))
        for c in sorted(lists):
            length = 0
            if len(author_df):
                series = author_df[c]
                sample = next((x for x in series if isinstance(x, (list, tuple, np.ndarray))), [])
                length = len(sample) if sample is not None else 0
            length = int(length or 0)
            if length <= 0:
                continue
            for i in range(length):
                flat_names.append(f'{c}[{i:02d}]')
            parts.append((c, length))
        self.cols_scalar = sorted(scalars)
        self.cols_list = sorted([c for (c, w) in parts if w > 1])
        self.parts = parts
        self.flat_names = flat_names
        self.dim = len(flat_names)

    def vectorize_row(self, row: pd.Series) -> List[float]:
        out: List[float] = []
        for (c, width) in self.parts:
            val = row.get(c, 0.0)
            flat = _flatten_numeric(val)
            if width == 1:
                out.append(float(flat[0] if flat else 0.0))
            else:
                if len(flat) < width:
                    flat = flat + [0.0] * (width - len(flat))
                out.extend([float(x) for x in flat[:width]])
        return out

class HypernetGlobalOnlyDataset10000(Dataset):

    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, *, author_features_df: pd.DataFrame, max_length: int=512, add_special_tokens: bool=True, pretokenize: bool=False, feature_include: Sequence[str] | None=None, feature_exclude: Sequence[str] | None=None, include_lists: bool=True, zscore: bool=False, zscore_center_only: bool=False) -> None:
        super().__init__()
        self._text_ds = RedditConversationDataset10000(dataframe=df, tokenizer=tokenizer, max_length=max_length, add_special_tokens=add_special_tokens, pretokenize=False)
        if 'gid' not in df.columns or 'target_user_id' not in df.columns:
            raise KeyError("Input dataframe must contain columns: 'gid' and 'target_user_id'")
        gdf = author_features_df.copy()
        if 'target_user_id' not in gdf.columns:
            raise KeyError("author_features_df must include 'target_user_id'")
        split_uids = set((int(x) for x in df['target_user_id'].unique().tolist()))
        gdf['target_user_id'] = _ensure_int_series(gdf['target_user_id'])
        gdf = gdf[gdf['target_user_id'].isin(split_uids)].drop_duplicates('target_user_id', keep='last').reset_index(drop=True)
        if gdf.empty:
            raise RuntimeError('Author features filtered to 0 rows for this split - check inputs.')
        include = list(feature_include or ['gstat_*'])
        default_excl = ['gstat_user_sent_*', 'gstat_gap_sentiment', 'gstat_beast_sent_*']
        exclude = list(feature_exclude or default_excl)

        def _truthy(s: str) -> bool:
            t = str(s).strip().lower()
            return t in {'1', 'true', 'yes', 'y', 'on'}
        graph_on = _truthy(os.environ.get('GRAPH', 'false'))
        persona_only_env = os.environ.get('PERSONA_ONLY', None)
        persona_only = graph_on if persona_only_env is None else _truthy(persona_only_env)
        persona_prefix = os.environ.get('PERSONA_PREFIX', 'gstat_psage')
        if persona_only:
            cand: List[str] = []
            if persona_prefix in gdf.columns:
                cand = [persona_prefix]
            else:
                cand = [c for c in gdf.columns if isinstance(c, str) and c.startswith(persona_prefix)]
            if cand:
                include = [cand[0]]
                exclude = []
                if _is_main_process():
                    print(f"[features] persona-only enabled • using '{cand[0]}' as the global vector")
            elif _is_main_process():
                print(f"[features] persona-only requested but '{persona_prefix}' not found; falling back to include={include} exclude={exclude}")
        self._spec = _FeatureSpec(gdf, include=include, exclude=exclude, include_lists=include_lists)
        if self._spec.dim <= 0:
            raise RuntimeError('Selected feature set is empty. Adjust include/exclude patterns.')
        self.feature_names: List[str] = list(self._spec.flat_names)
        base_cols = sorted({c for (c, _) in self._spec.parts})
        self._gdf = gdf[['target_user_id'] + base_cols].copy()
        self._g_index: Dict[int, int] = {int(uid): int(i) for (i, uid) in enumerate(self._gdf['target_user_id'].tolist())}
        self._do_z = bool(zscore)
        self._center_only = bool(zscore_center_only)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        if self._do_z:
            (self._mean, self._std) = self._compute_feature_stats()
        self._cache = [self._merge(self._text_ds[i]) for i in range(len(self._text_ds))] if pretokenize else None
        self._quick_sanity()

    @property
    def g_dim(self) -> int:
        return int(self._spec.dim)

    @property
    def feature_stats_payload(self) -> Dict[str, Any]:
        names = list(self.feature_names)
        k = len(names)
        if self._do_z and self._mean is not None and (self._std is not None):
            mean = [float(x) for x in self._mean.tolist()]
            std = [float(x) for x in self._std.tolist()]
        else:
            mean = [0.0] * k
            std = [1.0] * k
        return {'names': names, 'mean': mean, 'std': std}

    def __len__(self) -> int:
        return len(self._text_ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._cache[idx] if self._cache is not None else self._merge(self._text_ds[idx])

    def _compute_feature_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        K = int(self._spec.dim)
        n = int(len(self._gdf))
        if n == 0 or K == 0:
            return (np.zeros((K,), dtype=np.float32), np.ones((K,), dtype=np.float32))
        mean = np.zeros((K,), dtype=np.float64)
        m2 = np.zeros((K,), dtype=np.float64)
        count = 0
        for (_, row) in self._gdf.iterrows():
            v = np.asarray(self._spec.vectorize_row(row), dtype=np.float64)
            if v.size != K:
                if v.size < K:
                    v = np.pad(v, (0, K - v.size))
                else:
                    v = v[:K]
            count += 1
            delta = v - mean
            mean += delta / float(count)
            m2 += delta * (v - mean)
        var = m2 / float(max(1, count - 1))
        std = np.sqrt(np.maximum(var, 1e-12))
        if self._center_only:
            std = np.ones_like(std)
        return (mean.astype(np.float32), std.astype(np.float32))

    def _build_gvec(self, uid: int) -> np.ndarray:
        ridx = self._g_index.get(int(uid), None)
        if ridx is None:
            vec = np.zeros((self._spec.dim,), dtype=np.float32)
        else:
            row = self._gdf.iloc[ridx]
            vals = self._spec.vectorize_row(row)
            if len(vals) != self._spec.dim:
                vals = (vals + [0.0] * self._spec.dim)[:self._spec.dim]
            vec = np.asarray(vals, dtype=np.float32)
        if self._do_z and self._mean is not None and (self._std is not None):
            vec = (vec - self._mean) / (self._std + 1e-06)
        return vec

    def _merge(self, base: Dict[str, Any]) -> Dict[str, Any]:
        uid = int(base['target_user_id'])
        g_vec = self._build_gvec(uid)
        sample = dict(base)
        sample['global_features'] = torch.from_numpy(g_vec)
        return sample

    def _quick_sanity(self) -> None:
        global _SANITY_PRINTED
        if _SANITY_PRINTED or not _is_main_process():
            return
        n = len(self._text_ds)
        if n == 0:
            print('[sanity] Empty base text dataset.')
            _SANITY_PRINTED = True
            return
        idxs = list(range(n))
        if len(idxs) > 1024:
            import random as _rnd
            idxs = _rnd.sample(idxs, 1024)
        zero_ct = 0
        for i in idxs:
            uid = int(self._text_ds[i]['target_user_id'])
            vec = self._build_gvec(uid)
            if not np.any(vec != 0.0):
                zero_ct += 1
        rate = zero_ct / max(1, len(idxs))
        print(f'[sanity] g-dim={self._spec.dim} • features={len(self._spec.parts)} cols (scalars={len(self._spec.cols_scalar)}, lists={len(self._spec.cols_list)}) • all-zero vectors in probe={rate:.1%}')
        _SANITY_PRINTED = True

class FlatHypernetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, *, width: int=64, depth: int=9, dropout: float=0.0, clamp: float=0.05, head_rank: int=0, group_splits: Optional[Sequence[int]]=None) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError('depth must be >= 2 (at least one hidden layer and one output layer)')
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.clamp = float(clamp)
        self.head_rank = int(head_rank)
        self.group_splits = [int(x) for x in group_splits] if group_splits else None
        if self.group_splits is not None and sum(self.group_splits) != self.output_dim:
            raise ValueError(f'group_splits must sum to output_dim; got {sum(self.group_splits)} vs {self.output_dim}')
        layers: List[nn.Module] = []
        in_dim = self.input_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = width
        self.hidden = nn.Sequential(*layers)
        if self.head_rank > 0:
            if self.group_splits is not None:
                self.head_U_list = nn.ModuleList([nn.Linear(in_dim, self.head_rank, bias=False) for _ in self.group_splits])
                self.head_V_list = nn.ModuleList([nn.Linear(self.head_rank, s) for s in self.group_splits])
            else:
                self.head_U = nn.Linear(in_dim, self.head_rank, bias=False)
                self.head_V = nn.Linear(self.head_rank, self.output_dim)
        elif self.group_splits is not None:
            self.head_list = nn.ModuleList([nn.Linear(in_dim, s) for s in self.group_splits])
        else:
            self.head = nn.Linear(in_dim, self.output_dim)
        self._mc_dropout_rate: float = 0.0
        self.reset_parameters()

    def set_mc_dropout_rate(self, rate: float) -> None:
        self._mc_dropout_rate = float(max(0.0, min(1.0, rate)))

    def reset_parameters(self) -> None:
        for m in self.hidden.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    fan_in = m.weight.size(1)
                    bound = 1.0 / max(1, fan_in) ** 0.5
                    nn.init.uniform_(m.bias, -bound, bound)
        if hasattr(self, 'head'):
            nn.init.zeros_(self.head.weight)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)
        elif hasattr(self, 'head_U') and hasattr(self, 'head_V'):
            nn.init.kaiming_uniform_(self.head_U.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.head_V.weight)
            if self.head_V.bias is not None:
                nn.init.zeros_(self.head_V.bias)
        elif hasattr(self, 'head_U_list') and hasattr(self, 'head_V_list'):
            for U in self.head_U_list:
                nn.init.kaiming_uniform_(U.weight, mode='fan_in', nonlinearity='relu')
            for V in self.head_V_list:
                nn.init.zeros_(V.weight)
                if V.bias is not None:
                    nn.init.zeros_(V.bias)
        elif hasattr(self, 'head_list'):
            for H in self.head_list:
                nn.init.zeros_(H.weight)
                if H.bias is not None:
                    nn.init.zeros_(H.bias)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        h = self.hidden(g)
        if self._mc_dropout_rate > 0.0:
            h = F.dropout(h, p=self._mc_dropout_rate, training=True)
        if self.group_splits is not None:
            if self.head_rank > 0:
                chunks: List[torch.Tensor] = []
                for (U, V) in zip(self.head_U_list, self.head_V_list):
                    r = U(h)
                    chunks.append(V(r))
                delta = torch.cat(chunks, dim=-1) if chunks else h.new_zeros(h.size(0), 0)
            else:
                delta = torch.cat([H(h) for H in self.head_list], dim=-1)
        elif hasattr(self, 'head'):
            delta = self.head(h)
        else:
            r = self.head_U(h)
            delta = self.head_V(r)
        return torch.clamp(delta, -self.clamp, self.clamp)

def _infer_layer_index_from_name(param_name: str) -> int:
    marker = '.layers.'
    idx = param_name.find(marker)
    if idx == -1:
        return 0
    s = param_name
    start = idx + len(marker)
    end = start
    while end < len(s) and s[end].isdigit():
        end += 1
    if end == start:
        return 0
    try:
        return int(s[start:end])
    except ValueError:
        return 0

def build_peft_layer_groups(model: nn.Module) -> Tuple[int, List[int], Dict[str, Any]]:
    placeholders: List[Tuple[str, torch.nn.Parameter]] = [(name, p) for (name, p) in model.named_parameters() if p.requires_grad]
    total = 0
    group_sizes: Dict[int, int] = defaultdict(int)
    group_params: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for (name, p) in placeholders:
        numel = int(p.numel())
        layer_idx = _infer_layer_index_from_name(name)
        group_sizes[layer_idx] += numel
        group_params[layer_idx].append({'name': name, 'shape': [int(x) for x in p.shape], 'numel': numel})
        total += numel
    layer_ids = sorted(group_sizes.keys())
    group_splits = [int(group_sizes[i]) for i in layer_ids]
    payload_groups: List[Dict[str, Any]] = []
    for layer_idx in layer_ids:
        payload_groups.append({'layer': int(layer_idx), 'size': int(group_sizes[layer_idx]), 'params': group_params[layer_idx]})
    payload: Dict[str, Any] = {'total': int(total), 'groups': payload_groups}
    return (int(total), group_splits, payload)
try:
    from torch.func import functional_call as _fcall
except Exception:
    from torch.nn.utils.stateless import functional_call as _fcall
try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
    _HAVE_SAFETENSORS = True
except Exception:
    _HAVE_SAFETENSORS = False

class LoRAQKV(nn.Module):

    def __init__(self, base_linear: nn.Linear, rank: int=8, alpha: float=16.0):
        super().__init__()
        self.base_weight = base_linear.weight
        self.base_bias = base_linear.bias
        self.base_weight.requires_grad_(False)
        if self.base_bias is not None:
            self.base_bias.requires_grad_(False)
        (out_features, in_features) = self.base_weight.shape
        if out_features % in_features != 0 or out_features // in_features != 3:
            raise ValueError('Expected merged QKV weight of shape (3h, h).')
        self.h = int(in_features)
        self.rank = int(rank)
        self.scale = float(alpha) / max(1, self.rank)

        def _pair() -> Tuple[nn.Parameter, nn.Parameter]:
            A = nn.Parameter(torch.empty(self.h, self.rank))
            B = nn.Parameter(torch.empty(self.rank, self.h))
            nn.init.normal_(A, std=0.0001)
            nn.init.zeros_(B)
            return (A, B)
        (self.A_q, self.B_q) = _pair()
        (self.A_k, self.B_k) = _pair()
        (self.A_v, self.B_v) = _pair()
        self._dA_q: Optional[torch.Tensor] = None
        self._dB_q: Optional[torch.Tensor] = None
        self._dA_k: Optional[torch.Tensor] = None
        self._dB_k: Optional[torch.Tensor] = None
        self._dA_v: Optional[torch.Tensor] = None
        self._dB_v: Optional[torch.Tensor] = None

    def set_delta_views(self, delta_flat: torch.Tensor, layout: Dict[str, Tuple[int, int, Tuple[int, ...]]]) -> None:
        B = int(delta_flat.size(0))

        def _view(tag: str) -> Optional[torch.Tensor]:
            if tag not in layout:
                return None
            (off, n, shape) = layout[tag]
            return delta_flat[:, off:off + n].view(B, *shape)
        self._dA_q = _view('A_q')
        self._dB_q = _view('B_q')
        self._dA_k = _view('A_k')
        self._dB_k = _view('B_k')
        self._dA_v = _view('A_v')
        self._dB_v = _view('B_v')

    def clear_delta_views(self) -> None:
        self._dA_q = self._dB_q = None
        self._dA_k = self._dB_k = None
        self._dA_v = self._dB_v = None

    def _has_batch_deltas(self) -> bool:
        return any((t is not None for t in (self._dA_q, self._dB_q, self._dA_k, self._dB_k, self._dA_v, self._dB_v)))

    def _batched_update(self, x2: torch.Tensor, A: torch.Tensor, B: torch.Tensor, dA: Optional[torch.Tensor], dB: Optional[torch.Tensor]) -> torch.Tensor:
        (Bsz, T, H) = x2.size()
        dtype = x2.dtype
        A_T = A.t().unsqueeze(0).expand(Bsz, -1, -1).to(dtype)
        B_T = B.t().unsqueeze(0).expand(Bsz, -1, -1).to(dtype)
        if dA is not None:
            A_T = A_T + dA.permute(0, 2, 1).to(dtype)
        if dB is not None:
            B_T = B_T + dB.permute(0, 2, 1).to(dtype)
        tmp = torch.bmm(x2, B_T)
        upd = torch.bmm(tmp, A_T)
        return upd * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (*prefix, h) = x.shape
        if h != self.h:
            raise ValueError(f'Expected last dim={self.h}, got {h}')
        base = F.linear(x, self.base_weight, self.base_bias)
        if not self._has_batch_deltas():
            x2 = x.reshape(-1, h)

            def _update(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
                A_cast = A.to(dtype=x2.dtype)
                B_cast = B.to(dtype=x2.dtype)
                return (x2 @ B_cast.t() @ A_cast.t()).reshape(*prefix, self.h) * self.scale
            lq = _update(self.A_q, self.B_q)
            lk = _update(self.A_k, self.B_k)
            lv = _update(self.A_v, self.B_v)
            upd = torch.stack((lq, lk, lv), dim=-2)
            return (base.view(*prefix, 3, self.h) + upd).reshape(*prefix, 3 * self.h).to(x.dtype)
        Bsz = int(x.size(0))
        x_bt = x.view(Bsz, -1, self.h)
        lq = self._batched_update(x_bt, self.A_q, self.B_q, self._dA_q, self._dB_q)
        lk = self._batched_update(x_bt, self.A_k, self.B_k, self._dA_k, self._dB_k)
        lv = self._batched_update(x_bt, self.A_v, self.B_v, self._dA_v, self._dB_v)
        upd = torch.stack((lq, lk, lv), dim=2)
        out = base.view(Bsz, -1, 3, self.h) + upd
        return out.view(*prefix, 3 * self.h).to(x.dtype)

class ModelLoRAQKV(nn.Module):

    def __init__(self, base_model: nn.Module, *, rank: int=8, alpha: float=16.0, train_token_rows: Optional[List[int]]=None, train_pad_row: bool=False) -> None:
        super().__init__()
        self.add_module('backbone', base_model)
        self.config = base_model.config
        self.gpt_neox = getattr(base_model, 'gpt_neox', None)
        if self.gpt_neox is None:
            raise ValueError('Expected attribute `gpt_neox` on base_model; this head targets GPT-NeoX-like models.')
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for layer in self.gpt_neox.layers:
            qkv = layer.attention.query_key_value
            if isinstance(qkv, LoRAQKV):
                continue
            if not isinstance(qkv, nn.Linear):
                raise TypeError(f'Expected nn.Linear for attention.query_key_value, got {type(qkv).__name__}')
            layer.attention.query_key_value = LoRAQKV(qkv, rank=rank, alpha=alpha)
        self._special_row_params: List[torch.nn.Parameter] = []
        self._special_row_hooks: List[Any] = []
        self._special_token_ids: List[int] = []
        if train_token_rows is not None and len(train_token_rows) > 0:
            allow_ids = [int(x) for x in train_token_rows]
            if train_pad_row:
                pass
            self._enable_token_row_training(allow_ids)

    def _enable_token_row_training(self, token_ids: List[int]) -> None:
        if not (hasattr(self.backbone, 'get_input_embeddings') and callable(self.backbone.get_input_embeddings)):
            raise RuntimeError('Backbone does not expose get_input_embeddings()')
        if not (hasattr(self.backbone, 'get_output_embeddings') and callable(self.backbone.get_output_embeddings)):
            raise RuntimeError('Backbone does not expose get_output_embeddings()')
        in_mod = self.backbone.get_input_embeddings()
        out_mod = self.backbone.get_output_embeddings()
        if in_mod is None or not hasattr(in_mod, 'weight'):
            raise RuntimeError('Backbone get_input_embeddings() has no weight')
        if out_mod is None or not hasattr(out_mod, 'weight'):
            raise RuntimeError('Backbone get_output_embeddings() has no weight')
        W_in = in_mod.weight
        W_out = out_mod.weight
        W_in.requires_grad_(False)
        W_out.requires_grad_(False)
        hidden = int(W_in.size(1))
        if int(W_out.size(1)) != hidden:
            raise RuntimeError('Unexpected lm_head weight shape: expected (vocab, hidden)')
        seen = set()
        ids: List[int] = []
        for t in token_ids:
            ti = int(t)
            if ti not in seen:
                seen.add(ti)
                ids.append(ti)
        self._special_token_ids = ids
        n_ids = int(len(ids))
        if n_ids <= 0:
            return
        vocab = int(W_in.size(0))
        bad = [t for t in ids if t < 0 or t >= vocab]
        if bad:
            raise ValueError(f'Token ids out of range for embedding matrix of size {vocab}: {bad}')
        self.special_in_delta = nn.Parameter(torch.zeros(n_ids, hidden, dtype=W_in.dtype, device=W_in.device))
        self.special_out_delta = nn.Parameter(torch.zeros(n_ids, hidden, dtype=W_out.dtype, device=W_out.device))
        setattr(self.special_in_delta, '_hn_skip_placeholder', True)
        setattr(self.special_out_delta, '_hn_skip_placeholder', True)
        self._special_row_params = [self.special_in_delta, self.special_out_delta]
        ids_list = list(ids)

        def _embed_hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
            try:
                x = inputs[0]
                if x is None or not torch.is_tensor(x):
                    return output
                if output is None or not torch.is_tensor(output):
                    return output
                if x.dim() != 2 or output.dim() != 3:
                    return output
                out = output
                for (i, tid) in enumerate(ids_list):
                    mask = (x == tid).unsqueeze(-1).to(dtype=out.dtype)
                    out = out + mask * self.special_in_delta[i].view(1, 1, -1)
                return out
            except Exception:
                return output

        def _lm_head_hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
            try:
                h = inputs[0]
                if h is None or not torch.is_tensor(h):
                    return output
                if output is None or not torch.is_tensor(output):
                    return output
                if h.dim() != 3 or output.dim() != 3:
                    return output
                out2 = output.clone()
                for (i, tid) in enumerate(ids_list):
                    add = (h * self.special_out_delta[i].view(1, 1, -1)).sum(dim=-1)
                    out2[..., tid] = out2[..., tid] + add.to(dtype=out2.dtype)
                return out2
            except Exception:
                return output
        self._special_row_hooks.append(in_mod.register_forward_hook(_embed_hook))
        self._special_row_hooks.append(out_mod.register_forward_hook(_lm_head_hook))

    def special_row_params(self) -> List[torch.nn.Parameter]:
        return list(self._special_row_params)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone.generate(*args, **kwargs)

class AdapterTuningWrapper(nn.Module):

    def __init__(self, base_mlp: nn.Module, hidden_dim: int, bottleneck_dim: int, dropout_prob: float=0.0, use_layer_norm: bool=False) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        for p in self.base_mlp.parameters():
            p.requires_grad_(False)
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.up_proj.weight)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()
        self.ln = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        base_out = self.base_mlp(hidden_states, *args, **kwargs)
        if isinstance(base_out, tuple):
            (base_primary, rest) = (base_out[0], base_out[1:])
        else:
            (base_primary, rest) = (base_out, ())
        up = self.up_proj(self.dropout(F.relu(self.down_proj(base_primary))))
        out = self.ln(base_primary + up)
        return (out, *rest) if rest else out

class ModelAdapter(nn.Module):

    def __init__(self, base_model: nn.Module, *, bottleneck_dim: int, adapter_dropout: float=0.0, use_layer_norm: bool=False) -> None:
        super().__init__()
        self.add_module('backbone', base_model)
        self.config = base_model.config
        self.gpt_neox = getattr(base_model, 'gpt_neox', None)
        if self.gpt_neox is None:
            raise ValueError('Expected attribute `gpt_neox` on base_model; this head targets GPT‑NeoX‑like models.')
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        h = int(self.config.hidden_size)
        for layer in self.gpt_neox.layers:
            layer.mlp = AdapterTuningWrapper(base_mlp=layer.mlp, hidden_dim=h, bottleneck_dim=int(bottleneck_dim), dropout_prob=float(adapter_dropout), use_layer_norm=bool(use_layer_norm))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone.generate(*args, **kwargs)

class BiasTuningModel(nn.Module):

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.add_module('backbone', base_model)
        self.config = base_model.config
        self.gpt_neox = getattr(base_model, 'gpt_neox', None)
        for (name, p) in self.backbone.named_parameters():
            p.requires_grad_('bias' in name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone.generate(*args, **kwargs)

class ModelPrefixOnly(nn.Module):

    def __init__(self, base_model: nn.Module, *, prefix_length: int=10, dropout_prob: float=0.0) -> None:
        super().__init__()
        self.add_module('backbone', base_model)
        self.config = base_model.config
        self.gpt_neox = getattr(base_model, 'gpt_neox', None)
        if self.gpt_neox is None:
            raise ValueError('Expected attribute `gpt_neox` on base_model; this head targets GPT‑NeoX‑like models.')
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.prefix_length = int(prefix_length)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()
        hidden_size = int(self.config.hidden_size)
        num_heads = int(self.config.num_attention_heads)
        head_dim = hidden_size // num_heads
        num_layers = len(self.gpt_neox.layers)
        for (layer_idx, layer) in enumerate(self.gpt_neox.layers):
            k = nn.Parameter(torch.zeros(num_heads, self.prefix_length, head_dim))
            v = nn.Parameter(torch.zeros(num_heads, self.prefix_length, head_dim))
            nn.init.normal_(k, mean=0.0, std=0.02)
            nn.init.normal_(v, mean=0.0, std=0.02)
            layer.register_parameter('prefix_key', k)
            layer.register_parameter('prefix_value', v)

    def _infer_batch_device(self, **kwargs: Any) -> Tuple[int, torch.device]:
        if 'input_ids' in kwargs and kwargs['input_ids'] is not None:
            x = kwargs['input_ids']
            return (int(x.size(0)), x.device)
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            x = kwargs['attention_mask']
            return (int(x.size(0)), x.device)
        if 'labels' in kwargs and kwargs['labels'] is not None:
            x = kwargs['labels']
            return (int(x.size(0)), x.device)
        dev = next(self.backbone.parameters()).device
        return (1, dev)

    def _current_autocast_dtype(self, device: torch.device) -> torch.dtype:
        try:
            if torch.is_autocast_enabled():
                if device.type == 'cuda':
                    return torch.get_autocast_gpu_dtype()
                else:
                    return torch.get_autocast_cpu_dtype()
        except Exception:
            pass
        return next(self.backbone.parameters()).dtype

    def _build_prefix_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        legacy: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.gpt_neox.layers:
            k_param: torch.Tensor = getattr(layer, 'prefix_key')
            v_param: torch.Tensor = getattr(layer, 'prefix_value')
            k = self.dropout(k_param).to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1, -1)
            v = self.dropout(v_param).to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1, -1)
            legacy.append((k, v))
        try:
            from transformers.cache_utils import StaticCache, DynamicCache
            try:
                return StaticCache.from_legacy_cache(tuple(legacy))
            except Exception:
                return DynamicCache.from_legacy_cache(tuple(legacy))
        except Exception:
            return tuple(legacy)

    def _augment_attention_mask(self, attn: Optional[torch.Tensor], batch: int, device: torch.device) -> Optional[torch.Tensor]:
        if attn is None or self.prefix_length <= 0:
            return attn
        prefix = torch.ones(batch, self.prefix_length, dtype=attn.dtype, device=device)
        return torch.cat([prefix, attn], dim=1)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        (B, dev) = self._infer_batch_device(**kwargs)
        dtype = self._current_autocast_dtype(dev)
        cache = self._build_prefix_cache(B, dtype, dev)
        attn = kwargs.get('attention_mask', None)
        kwargs['attention_mask'] = self._augment_attention_mask(attn, B, dev)
        kwargs['past_key_values'] = cache
        return self.backbone(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        (B, dev) = self._infer_batch_device(**kwargs)
        dtype = self._current_autocast_dtype(dev)
        cache = self._build_prefix_cache(B, dtype, dev)
        attn = kwargs.get('attention_mask', None)
        kwargs['attention_mask'] = self._augment_attention_mask(attn, B, dev)
        kwargs['past_key_values'] = cache
        return self.backbone.generate(*args, **kwargs)

def count_trainable_params(module: nn.Module, *, exclude_param_ids: Optional[Sequence[int]]=None) -> int:
    excl = set((int(x) for x in exclude_param_ids or []))
    total = 0
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if id(p) in excl:
            continue
        total += int(p.numel())
    return total

def _merge_outputs(outputs: List[Any]) -> Any:
    import types
    if not outputs:
        return None

    def _join(ts: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(ts, 0) if ts[0].dim() == 0 else torch.cat(ts, 0)
    first = outputs[0]
    try:
        if hasattr(first, 'to_dict'):
            template = first.to_dict()
            merged: Dict[str, Any] = {}
            for k in template:
                vals = [getattr(o, k) for o in outputs]
                if torch.is_tensor(vals[0]):
                    merged[k] = _join(vals)
                elif isinstance(vals[0], tuple) and all((torch.is_tensor(t) for t in vals[0])):
                    merged[k] = tuple((_join(ts) for ts in zip(*vals)))
                else:
                    merged[k] = vals
            if hasattr(type(first), 'from_dict'):
                return type(first).from_dict(merged)
            return type(first)(**merged)
    except Exception:
        pass
    if isinstance(first, dict):
        merged = {}
        for k in first:
            vals = [o[k] for o in outputs]
            if torch.is_tensor(vals[0]):
                merged[k] = _join(vals)
            elif isinstance(vals[0], tuple) and all((torch.is_tensor(t) for t in vals[0])):
                merged[k] = tuple((_join(ts) for ts in zip(*vals)))
            else:
                merged[k] = vals
        try:
            from transformers.modeling_outputs import CausalLMOutput
            return CausalLMOutput(**merged)
        except Exception:
            return types.SimpleNamespace(**merged)
    if hasattr(first, '__dict__'):
        merged = {}
        for k in list(vars(first).keys()):
            vals = [getattr(o, k) for o in outputs]
            if torch.is_tensor(vals[0]):
                merged[k] = _join(vals)
            elif isinstance(vals[0], tuple) and all((torch.is_tensor(t) for t in vals[0])):
                merged[k] = tuple((_join(ts) for ts in zip(*vals)))
            else:
                merged[k] = vals
        return types.SimpleNamespace(**merged)
    return outputs if len(outputs) > 1 else outputs[0]

class PEFTHypernet(nn.Module):

    def __init__(self, base_peft_model: nn.Module, hypernet: FlatHypernetwork, *, clamp: Optional[float]=None, delta_scale: float=1.0, use_lora_fast_path: bool=True) -> None:
        super().__init__()
        self.backbone = base_peft_model
        self.hypernet = hypernet
        if clamp is not None:
            self.hypernet.clamp = float(clamp)
        self.delta_scale = float(delta_scale)
        self.use_lora_fast_path = bool(use_lora_fast_path)
        self._placeholders: 'OrderedDict[str, torch.nn.Parameter]' = OrderedDict()
        for (name, p) in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            if bool(getattr(p, '_hn_skip_placeholder', False)):
                continue
            p.requires_grad_(False)
            self._placeholders[name] = p
        if not self._placeholders:
            raise ValueError('No trainable PEFT placeholders detected. Ensure your PEFT head exposes trainable parameters.')
        dev = next(self.backbone.parameters()).device
        self._theta_bar: 'OrderedDict[str, torch.Tensor]' = OrderedDict(((name, p.detach().clone().to(dev)) for (name, p) in self._placeholders.items()))
        total = int(sum((p.numel() for p in self._placeholders.values())))
        out_dim = int(self._infer_hyper_out_dim())
        if out_dim != total:
            raise ValueError(f'Hypernetwork emits {out_dim} scalars, but {total} PEFT parameters were detected.')
        self._lora_modules: List[LoRAQKV] = []
        self._param_to_lora: Dict[int, Tuple[int, str]] = {}
        for m in self.backbone.modules():
            if isinstance(m, LoRAQKV):
                idx = len(self._lora_modules)
                self._lora_modules.append(m)
                for tag in ('A_q', 'B_q', 'A_k', 'B_k', 'A_v', 'B_v'):
                    self._param_to_lora[id(getattr(m, tag))] = (idx, tag)
        self._flat_offsets: Dict[int, Tuple[int, int, Tuple[int, ...]]] = {}
        self._lora_layouts: List[Dict[str, Tuple[int, int, Tuple[int, ...]]]] = []
        self._rebuild_offsets_and_layouts()
        self._rank_mask: Optional[int] = None

    def set_delta_scale(self, scale: float) -> None:
        s = float(scale)
        try:
            import math as _math
            if not _math.isfinite(s):
                raise ValueError
        except Exception:
            if not np.isfinite(s):
                raise ValueError('delta_scale must be finite')
        self.delta_scale = s

    def set_use_lora_fast_path(self, enabled: bool) -> None:
        self.use_lora_fast_path = bool(enabled)

    def set_rank_mask(self, rank: Optional[int]=None) -> None:
        self._rank_mask = int(rank) if rank is not None else None

    def _apply_rank_mask(self, delta_all: torch.Tensor) -> torch.Tensor:
        if self._rank_mask is None or not self._lora_layouts:
            return delta_all
        delta_all = delta_all.clone()
        B = int(delta_all.size(0))
        for layout in self._lora_layouts:
            for (tag, (offset, numel, shape)) in layout.items():
                if len(shape) != 2:
                    continue
                sl = delta_all[:, offset:offset + numel].view(B, *shape)
                if tag.startswith('A_'):
                    if self._rank_mask < shape[1]:
                        sl[:, :, self._rank_mask:] = 0.0
                elif tag.startswith('B_'):
                    if self._rank_mask < shape[0]:
                        sl[:, self._rank_mask:, :] = 0.0
        return delta_all

    def _infer_hyper_out_dim(self) -> int:
        if hasattr(self.hypernet, 'output_dim'):
            return int(self.hypernet.output_dim)
        try:
            with torch.no_grad():
                in_dim = getattr(self.hypernet, 'input_dim', None)
                if in_dim is None:
                    for m in self.hypernet.hidden.modules():
                        if isinstance(m, nn.Linear):
                            in_dim = int(m.in_features)
                            break
                in_dim = int(in_dim or 1)
                dtype = next(self.hypernet.parameters()).dtype
                dev = next(self.hypernet.parameters()).device
                y = self.hypernet(torch.zeros(1, in_dim, dtype=dtype, device=dev))
            return int(y.numel())
        except Exception:
            return -1

    def _rebuild_offsets_and_layouts(self) -> None:
        self._flat_offsets = {}
        off = 0
        for (_, p) in self._placeholders.items():
            n = int(p.numel())
            self._flat_offsets[id(p)] = (int(off), n, tuple(p.shape))
            off += n
        self._lora_layouts = [dict() for _ in self._lora_modules]
        for (param_id, (mod_idx, tag)) in self._param_to_lora.items():
            if param_id in self._flat_offsets:
                self._lora_layouts[mod_idx][tag] = self._flat_offsets[param_id]

    def _ensure_theta_on(self, device: torch.device) -> None:
        for k in list(self._theta_bar.keys()):
            t = self._theta_bar[k]
            if t.device != device:
                self._theta_bar[k] = t.to(device)

    @staticmethod
    def _reduce_loss_field(obj: Any) -> Any:
        try:
            if hasattr(obj, 'loss') and torch.is_tensor(obj.loss) and (obj.loss.numel() > 1):
                obj.loss = obj.loss.mean()
        except Exception:
            pass
        return obj

    def _set_lora_delta_views(self, delta_all: torch.Tensor) -> None:
        if not self._lora_modules:
            return
        for (idx, mod) in enumerate(self._lora_modules):
            layout = self._lora_layouts[idx] if idx < len(self._lora_layouts) else {}
            if layout:
                mod.set_delta_views(delta_all, layout)

    def _clear_lora_delta_views(self) -> None:
        for mod in self._lora_modules:
            mod.clear_delta_views()

    def _can_use_lora_batched(self) -> bool:
        return bool(self.use_lora_fast_path and self._lora_modules)

    def _apply_delta_scale(self, delta_all: torch.Tensor, scale: Optional[float]) -> torch.Tensor:
        s = float(self.delta_scale if scale is None else scale)
        if s != 1.0:
            delta_all = delta_all * s
        clamp_val = getattr(self.hypernet, 'clamp', 0.0)
        try:
            c = float(clamp_val)
        except Exception:
            c = 0.0
        if c and c > 0.0:
            delta_all = torch.clamp(delta_all, -c, c)
        return delta_all

    def save_hypernet_ckpt(self, path: str | os.PathLike) -> None:
        placeholder_names = list(self._placeholders.keys())
        meta_obj = {'placeholder_names': placeholder_names, 'placeholder_total': int(sum((int(p.numel()) for p in self._placeholders.values()))), 'delta_scale': float(self.delta_scale), 'hyper_input_dim': int(getattr(self.hypernet, 'input_dim', -1)), 'hyper_output_dim': int(getattr(self.hypernet, 'output_dim', -1)), 'hyper_clamp': float(getattr(self.hypernet, 'clamp', 0.0))}
        blob = {'__hyper_state__': self.hypernet.state_dict(), '__theta__': {k: v.detach().cpu() for (k, v) in self._theta_bar.items()}, '__meta__': meta_obj}
        if _HAVE_SAFETENSORS and str(path).endswith('.safetensors'):
            flat: Dict[str, torch.Tensor] = {}
            flat.update({f'hyper.{k}': v for (k, v) in blob['__hyper_state__'].items()})
            flat.update({f'theta.{k}': v for (k, v) in blob['__theta__'].items()})
            meta_str: Dict[str, str] = {}
            try:
                import json as _json
                meta_str['placeholder_names'] = _json.dumps(placeholder_names)
                meta_str['delta_scale'] = str(self.delta_scale)
                meta_str['hyper_input_dim'] = str(meta_obj['hyper_input_dim'])
                meta_str['hyper_output_dim'] = str(meta_obj['hyper_output_dim'])
                meta_str['hyper_clamp'] = str(meta_obj['hyper_clamp'])
            except Exception:
                meta_str = {}
            try:
                _st_save(flat, str(path), metadata=meta_str)
            except TypeError:
                _st_save(flat, str(path))
            except Exception:
                _st_save(flat, str(path))
        else:
            torch.save(blob, path)

    @staticmethod
    def _read_safetensors_metadata(path: str | os.PathLike) -> Dict[str, str]:
        if not _HAVE_SAFETENSORS:
            return {}
        try:
            from safetensors import safe_open as _safe_open
            with _safe_open(str(path), framework='pt', device='cpu') as f:
                md = f.metadata()
                return dict(md) if md is not None else {}
        except Exception:
            return {}

    def _resolve_ckpt_key(self, key: str, local_keys: Optional[set]=None) -> str:
        if local_keys is None:
            local_keys = set(self._placeholders.keys())
        if key in local_keys:
            return key
        cands: List[str] = [key]
        if key.startswith('module.'):
            cands.append(key[len('module.'):])
        if key.startswith('backbone.'):
            cands.append(key[len('backbone.'):])
        if key.startswith('backbone.backbone.'):
            cands.append('backbone.' + key[len('backbone.backbone.'):])
            cands.append(key[len('backbone.'):])
        if not key.startswith('backbone.'):
            cands.append('backbone.' + key)
        if key.startswith('module.') and (not key.startswith('module.backbone.')):
            cands.append('module.backbone.' + key[len('module.'):])
        stripped = key[len('module.'):] if key.startswith('module.') else key
        if not stripped.startswith('backbone.'):
            cands.append('backbone.' + stripped)
        if stripped.startswith('backbone.') and stripped[len('backbone.'):] in local_keys:
            cands.append(stripped[len('backbone.'):])
        for cand in cands:
            if cand in local_keys:
                return cand
        raise KeyError(f"Could not resolve checkpoint placeholder key '{key}' to any local placeholder name.")

    def _load_theta_state(self, theta_state: Dict[str, torch.Tensor], *, placeholder_order: Optional[Sequence[str]]=None, strict: bool=True) -> None:
        if not isinstance(theta_state, dict):
            raise TypeError('theta_state must be a dict of name->tensor')
        local_keys = set(self._placeholders.keys())
        resolved: Dict[str, torch.Tensor] = {}
        for (ckpt_k, t) in theta_state.items():
            lk = self._resolve_ckpt_key(str(ckpt_k), local_keys)
            if lk in resolved:
                raise ValueError(f"Duplicate resolved theta key: '{ckpt_k}' -> '{lk}'")
            resolved[lk] = t
        if strict:
            missing = sorted(local_keys - set(resolved.keys()))
            extra = sorted(set(resolved.keys()) - local_keys)
            if missing or extra:
                msg = f'Theta keys do not match local placeholders. missing={missing[:10]} extra={extra[:10]}'
                raise ValueError(msg)
        if placeholder_order:
            order_local: List[str] = []
            for ckpt_name in placeholder_order:
                try:
                    order_local.append(self._resolve_ckpt_key(str(ckpt_name), local_keys))
                except Exception:
                    order_local = []
                    break
            if order_local and set(order_local) == local_keys and (len(order_local) == len(local_keys)):
                self._placeholders = OrderedDict(((k, self._placeholders[k]) for k in order_local))
            elif strict and order_local:
                raise ValueError('Provided placeholder_order could not be reconciled with local placeholders.')
        dev = next(self.backbone.parameters()).device
        new_theta: 'OrderedDict[str, torch.Tensor]' = OrderedDict()
        with torch.no_grad():
            for (name, p) in self._placeholders.items():
                if name not in resolved:
                    if strict:
                        raise KeyError(f"Missing theta tensor for placeholder '{name}'")
                    continue
                t = resolved[name]
                if tuple(t.shape) != tuple(p.shape):
                    raise ValueError(f"Theta shape mismatch for '{name}': ckpt={tuple(t.shape)} model={tuple(p.shape)}")
                tt = t.to(device=dev, dtype=p.dtype)
                p.copy_(tt)
                new_theta[name] = tt.detach().clone()
        self._theta_bar = new_theta
        self._rebuild_offsets_and_layouts()

    @staticmethod
    def load_hypernet_ckpt(path: str | os.PathLike, model: 'PEFTHypernet') -> None:
        placeholder_order: Optional[List[str]] = None
        delta_scale: Optional[float] = None
        if _HAVE_SAFETENSORS and str(path).endswith('.safetensors'):
            meta = model._read_safetensors_metadata(path)
            if 'placeholder_names' in meta:
                try:
                    import json as _json
                    placeholder_order = list(_json.loads(meta['placeholder_names']))
                except Exception:
                    placeholder_order = None
            if 'delta_scale' in meta:
                try:
                    delta_scale = float(meta['delta_scale'])
                except Exception:
                    delta_scale = None
            flat = _st_load(str(path))
            h_state = {k.removeprefix('hyper.'): v for (k, v) in flat.items() if k.startswith('hyper.')}
            t_state = {k.removeprefix('theta.'): v for (k, v) in flat.items() if k.startswith('theta.')}
        else:
            blob = torch.load(path, map_location='cpu')
            if not isinstance(blob, dict) or '__hyper_state__' not in blob or '__theta__' not in blob:
                raise ValueError("Checkpoint missing required keys: '__hyper_state__' and '__theta__'")
            h_state = blob['__hyper_state__']
            t_state = blob['__theta__']
            meta = blob.get('__meta__', None)
            if isinstance(meta, dict):
                try:
                    if 'placeholder_names' in meta and isinstance(meta['placeholder_names'], (list, tuple)):
                        placeholder_order = [str(x) for x in meta['placeholder_names']]
                except Exception:
                    placeholder_order = None
                try:
                    if 'delta_scale' in meta:
                        delta_scale = float(meta['delta_scale'])
                except Exception:
                    delta_scale = None

        def _infer_head_spec(state: Dict[str, torch.Tensor]) -> Tuple[int, Optional[List[int]]]:
            keys = list(state.keys())
            has_group_lowrank = False
            has_group_full = False
            for k in keys:
                if k.startswith('head_U_list.') or k.startswith('head_V_list.'):
                    has_group_lowrank = True
                    break
            if not has_group_lowrank:
                for k in keys:
                    if k.startswith('head_list.'):
                        has_group_full = True
                        break
            if has_group_lowrank:
                idxs: set[int] = set()
                for k in keys:
                    if not (k.startswith('head_U_list.') or k.startswith('head_V_list.')):
                        continue
                    parts = str(k).split('.')
                    if len(parts) < 3:
                        continue
                    if parts[1].isdigit():
                        idxs.add(int(parts[1]))
                gids = sorted(idxs)
                head_rank = 0
                group_splits: List[int] = []
                if gids:
                    tU = state.get(f'head_U_list.{gids[0]}.weight', None)
                    if torch.is_tensor(tU) and tU.ndim == 2:
                        head_rank = int(tU.shape[0])
                    for g in gids:
                        tV = state.get(f'head_V_list.{g}.weight', None)
                        if torch.is_tensor(tV) and tV.ndim == 2:
                            group_splits.append(int(tV.shape[0]))
                if group_splits and sum(group_splits) > 0:
                    return (int(head_rank), [int(x) for x in group_splits])
                return (int(head_rank), None)
            if has_group_full:
                idxs: set[int] = set()
                for k in keys:
                    if not k.startswith('head_list.'):
                        continue
                    parts = str(k).split('.')
                    if len(parts) < 3:
                        continue
                    if parts[1].isdigit():
                        idxs.add(int(parts[1]))
                gids = sorted(idxs)
                group_splits: List[int] = []
                for g in gids:
                    tW = state.get(f'head_list.{g}.weight', None)
                    if torch.is_tensor(tW) and tW.ndim == 2:
                        group_splits.append(int(tW.shape[0]))
                if group_splits and sum(group_splits) > 0:
                    return (0, [int(x) for x in group_splits])
                return (0, None)
            tU = state.get('head_U.weight', None)
            if torch.is_tensor(tU) and tU.ndim == 2:
                return (int(tU.shape[0]), None)
            return (0, None)
        try:
            model.hypernet.load_state_dict(h_state, strict=True)
        except RuntimeError as e:
            msg = str(e)
            headish = 'head.weight' in msg or 'head.bias' in msg or 'head_U_list' in msg or ('head_V_list' in msg) or ('head_list' in msg) or ('head_U.weight' in msg) or ('head_V.weight' in msg)
            if not headish:
                raise
            (desired_head_rank, desired_group_splits) = _infer_head_spec(h_state)
            cur_head_rank = int(getattr(model.hypernet, 'head_rank', 0) or 0)
            cur_group_splits = getattr(model.hypernet, 'group_splits', None)
            cur_group_splits_list = [int(x) for x in cur_group_splits] if cur_group_splits else None
            desired_group_splits_list = [int(x) for x in desired_group_splits] if desired_group_splits else None
            if cur_head_rank == int(desired_head_rank) and cur_group_splits_list == desired_group_splits_list:
                raise
            try:
                dev = next(model.hypernet.parameters()).device
                dtype = next(model.hypernet.parameters()).dtype
            except Exception:
                dev = torch.device('cpu')
                dtype = torch.float32
            width = None
            n_linear = 0
            dropout = 0.0
            try:
                for m in model.hypernet.hidden:
                    if isinstance(m, nn.Linear):
                        n_linear += 1
                        if width is None:
                            width = int(m.out_features)
                    if isinstance(m, nn.Dropout) and dropout == 0.0:
                        dropout = float(m.p)
            except Exception:
                pass
            if width is None:
                width = 64
            depth = int(n_linear + 1) if n_linear > 0 else 2
            new_hyper = FlatHypernetwork(input_dim=int(getattr(model.hypernet, 'input_dim', 0) or 0), output_dim=int(getattr(model.hypernet, 'output_dim', 0) or 0), width=int(width), depth=int(depth), dropout=float(dropout), clamp=float(getattr(model.hypernet, 'clamp', 0.05)), head_rank=int(desired_head_rank), group_splits=desired_group_splits_list if desired_group_splits_list else None)
            new_hyper.to(device=dev, dtype=dtype)
            model.hypernet = new_hyper
            model.hypernet.load_state_dict(h_state, strict=True)
        model._load_theta_state(t_state, placeholder_order=placeholder_order, strict=True)
        if delta_scale is not None:
            try:
                model.delta_scale = float(delta_scale)
            except Exception:
                pass

    def forward_batched(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, global_features: torch.Tensor) -> SimpleNamespace:
        B = int(input_ids.size(0))
        outs: List[Any] = []
        for i in range(B):
            out = self.forward(input_ids=input_ids[i:i + 1], attention_mask=attention_mask[i:i + 1], labels=labels[i:i + 1] if labels is not None else None, global_features=global_features[i:i + 1])
            outs.append(out)
        logits = torch.cat([o.logits for o in outs], dim=0)
        loss = None
        if all((hasattr(o, 'loss') and o.loss is not None for o in outs)):
            loss = torch.stack([o.loss for o in outs]).mean()
        return SimpleNamespace(logits=logits, loss=loss)

    def forward_with_delta(self, delta_all: torch.Tensor, *, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.LongTensor]=None, **backbone_kwargs: Any) -> Any:
        if delta_all is None:
            raise ValueError('delta_all is required')
        self._ensure_theta_on(delta_all.device)
        sizes = [p.numel() for p in self._placeholders.values()]
        backbone_kwargs = dict(backbone_kwargs)
        backbone_kwargs['return_dict'] = True
        attn = backbone_kwargs.pop('attention_mask', attention_mask)
        if attn is not None:
            if not torch.is_tensor(attn) or attn.dim() != 2:
                raise ValueError('attention_mask must be (B,S)')
            if torch.all(attn == 1).item():
                attn = None
        if attn is not None:
            backbone_kwargs['attention_mask'] = attn
        outputs: List[Any] = []
        B = int(delta_all.size(0))
        for j in range(B):
            override: Dict[str, torch.Tensor] = {}
            ptr = 0
            for ((name, ref), sz) in zip(self._placeholders.items(), sizes):
                delta = delta_all[j, ptr:ptr + sz].view_as(ref).to(ref.dtype)
                override[name] = self._theta_bar[name] + delta
                ptr += sz
            outputs.append(_fcall(self.backbone, override, args=(), kwargs=dict(input_ids=None if input_ids is None else input_ids[j:j + 1], attention_mask=None if attn is None else attn[j:j + 1], labels=None if labels is None else labels[j:j + 1], **backbone_kwargs)))
        merged = _merge_outputs(outputs)
        return self._reduce_loss_field(merged)

    def forward(self, *, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.LongTensor]=None, global_features: torch.Tensor, **backbone_kwargs: Any) -> Any:
        if global_features is None:
            raise ValueError('`global_features` batch of g‑vectors is required')
        return_loss_only = bool(backbone_kwargs.pop('return_loss_only', False))
        per_call_scale = backbone_kwargs.pop('delta_scale', None)
        try:
            p = next(self.hypernet.parameters())
            if global_features.device != p.device or global_features.dtype != p.dtype:
                global_features = global_features.to(device=p.device, dtype=p.dtype)
        except Exception:
            pass
        delta_all = self.hypernet(global_features)
        delta_all = self._apply_delta_scale(delta_all, per_call_scale)
        delta_all = self._apply_rank_mask(delta_all)
        self._ensure_theta_on(delta_all.device)
        want_hidden = bool(backbone_kwargs.get('output_hidden_states', False))
        backbone_kwargs['return_dict'] = True
        backbone_kwargs.setdefault('output_attentions', False)
        backbone_kwargs['output_hidden_states'] = bool(want_hidden)
        backbone_kwargs.setdefault('use_cache', False)
        if self._can_use_lora_batched() and input_ids is not None and (not want_hidden):
            try:
                self._set_lora_delta_views(delta_all)
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **backbone_kwargs)
                if return_loss_only and labels is not None:
                    loss = out.loss if torch.is_tensor(out.loss) else None
                    loss = loss.mean() if torch.is_tensor(loss) else torch.tensor(0.0, device=delta_all.device)
                    return SimpleNamespace(loss=loss)
                return self._reduce_loss_field(out)
            finally:
                self._clear_lora_delta_views()
        sizes = [p.numel() for p in self._placeholders.values()]
        B = int(delta_all.size(0))
        if return_loss_only and labels is not None:
            sum_loss = delta_all.new_zeros((), dtype=torch.float32)
            counted = 0
            for j in range(B):
                lj = labels[j] if labels is not None else None
                try:
                    if lj is None or not torch.is_tensor(lj) or lj.dim() == 0:
                        continue
                    if not (lj != -100).any().item():
                        continue
                except Exception:
                    continue
                override: Dict[str, torch.Tensor] = {}
                ptr = 0
                for ((name, ref), sz) in zip(self._placeholders.items(), sizes):
                    delta = delta_all[j, ptr:ptr + sz].view_as(ref).to(ref.dtype)
                    override[name] = self._theta_bar[name] + delta
                    ptr += sz
                out = _fcall(self.backbone, override, args=(), kwargs=dict(input_ids=None if input_ids is None else input_ids[j:j + 1], attention_mask=None if attention_mask is None else attention_mask[j:j + 1], labels=labels[j:j + 1], **backbone_kwargs))
                val = out.loss.mean() if torch.is_tensor(out.loss) else None
                if val is not None and torch.isfinite(val):
                    sum_loss = sum_loss + val.to(sum_loss.dtype)
                    counted += 1
            avg_loss = sum_loss / float(max(1, counted))
            return SimpleNamespace(loss=avg_loss)
        outputs: List[Any] = []
        for j in range(B):
            override = {}
            ptr = 0
            for ((name, ref), sz) in zip(self._placeholders.items(), sizes):
                delta = delta_all[j, ptr:ptr + sz].view_as(ref).to(ref.dtype)
                override[name] = self._theta_bar[name] + delta
                ptr += sz
            outputs.append(_fcall(self.backbone, override, args=(), kwargs=dict(input_ids=None if input_ids is None else input_ids[j:j + 1], attention_mask=None if attention_mask is None else attention_mask[j:j + 1], labels=None if labels is None else labels[j:j + 1], **backbone_kwargs)))
        merged = _merge_outputs(outputs)
        return self._reduce_loss_field(merged)

    @torch.no_grad()
    def logits_at_positions(self, *, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor], positions: torch.LongTensor, global_features: torch.Tensor, use_cache: bool=False, return_dict: bool=True, **backbone_kwargs: Any) -> torch.Tensor:
        (B, S) = input_ids.size()
        if positions.dim() != 1 or positions.size(0) != B:
            raise ValueError('positions must be shape (B,) to match input_ids')
        per_call_scale = backbone_kwargs.pop('delta_scale', None)
        try:
            p = next(self.hypernet.parameters())
            if global_features.device != p.device or global_features.dtype != p.dtype:
                global_features = global_features.to(device=p.device, dtype=p.dtype)
        except Exception:
            pass
        delta_all = self.hypernet(global_features)
        delta_all = self._apply_delta_scale(delta_all, per_call_scale)
        delta_all = self._apply_rank_mask(delta_all)
        self._ensure_theta_on(delta_all.device)
        sizes = [p.numel() for p in self._placeholders.values()]
        rows: List[torch.Tensor] = []
        for j in range(B):
            override: Dict[str, torch.Tensor] = {}
            ptr = 0
            for ((name, ref), sz) in zip(self._placeholders.items(), sizes):
                delta = delta_all[j, ptr:ptr + sz].view_as(ref).to(ref.dtype)
                override[name] = self._theta_bar[name] + delta
                ptr += sz
            cut = int(positions[j].item()) + 1
            cut = max(1, min(cut, S))
            ids = input_ids[j:j + 1, :cut]
            attn = None if attention_mask is None else attention_mask[j:j + 1, :cut]
            out = _fcall(self.backbone, override, args=(), kwargs=dict(input_ids=ids, attention_mask=attn, labels=None, use_cache=bool(use_cache), return_dict=True, **backbone_kwargs))
            rows.append(out.logits[:, -1, :])
        return torch.cat(rows, dim=0)

    @torch.no_grad()
    def generate_ctx(self, *, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor], global_features: torch.Tensor, max_new_tokens: int=64, eos_token_id: Optional[int]=None, do_sample: bool=False, top_p: float=1.0, temperature: float=1.0, top_p_rule: str='legacy', **backbone_kwargs: Any) -> List[List[int]]:
        if top_p_rule not in ('legacy', 'hf'):
            raise ValueError(f"top_p_rule must be 'legacy' or 'hf', got {top_p_rule!r}")
        if input_ids.dim() != 2:
            raise ValueError('input_ids must be (B,S)')
        per_call_scale = backbone_kwargs.pop('delta_scale', None)
        (B, _) = input_ids.size()
        device = input_ids.device
        try:
            p = next(self.hypernet.parameters())
            if global_features.device != p.device or global_features.dtype != p.dtype:
                global_features = global_features.to(device=p.device, dtype=p.dtype)
        except Exception:
            pass
        delta_all = self.hypernet(global_features)
        delta_all = self._apply_delta_scale(delta_all, per_call_scale)
        delta_all = self._apply_rank_mask(delta_all)
        self._ensure_theta_on(delta_all.device)
        sizes = [p.numel() for p in self._placeholders.values()]

        def _forward(ids: torch.Tensor, attn: Optional[torch.Tensor]) -> Any:
            if self._can_use_lora_batched():
                try:
                    self._set_lora_delta_views(delta_all)
                    return self.backbone(input_ids=ids, attention_mask=attn, labels=None, use_cache=False, return_dict=True, **backbone_kwargs)
                finally:
                    self._clear_lora_delta_views()
            outputs: List[Any] = []
            B_loc = int(ids.size(0))
            for j in range(B_loc):
                override: Dict[str, torch.Tensor] = {}
                ptr = 0
                for ((name, ref), sz) in zip(self._placeholders.items(), sizes):
                    drow = delta_all[j:j + 1, ptr:ptr + sz].view_as(ref).to(ref.dtype)
                    override[name] = self._theta_bar[name] + drow
                    ptr += sz
                out = _fcall(self.backbone, override, args=(), kwargs=dict(input_ids=ids[j:j + 1], attention_mask=None if attn is None else attn[j:j + 1], labels=None, use_cache=False, return_dict=True, **backbone_kwargs))
                outputs.append(out)
            logits = torch.cat([o.logits for o in outputs], dim=0)
            return SimpleNamespace(logits=logits)
        seq = input_ids.clone()
        attn = attention_mask.clone() if attention_mask is not None else None
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated: List[List[int]] = [[] for _ in range(B)]
        max_new_tokens = int(max_new_tokens)
        for _ in range(max_new_tokens):
            out = _forward(seq, attn)
            logits = out.logits[:, -1, :]
            if temperature <= 0:
                temperature = 1.0
            if do_sample:
                logits_ = logits / float(temperature)
                probs = torch.softmax(logits_, dim=-1)
                if 0.0 < top_p < 1.0:
                    (sorted_probs, sorted_idx) = torch.sort(probs, dim=-1, descending=True)
                    cdf = torch.cumsum(sorted_probs, dim=-1)
                    mask = cdf - sorted_probs >= top_p if top_p_rule == 'hf' else cdf > top_p
                    mask[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                    sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)
                    next_sorted = torch.multinomial(sorted_probs, num_samples=1)
                    next_tokens = sorted_idx.gather(-1, next_sorted).squeeze(-1)
                else:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)
            seq = torch.cat([seq, next_tokens.unsqueeze(1)], dim=1)
            if attn is not None:
                ones = torch.ones(B, 1, dtype=attn.dtype, device=device)
                attn = torch.cat([attn, ones], dim=1)
            for j in range(B):
                if finished[j]:
                    continue
                tok = int(next_tokens[j].item())
                generated[j].append(tok)
                if eos_token_id is not None and tok == int(eos_token_id):
                    finished[j] = True
            if eos_token_id is not None and bool(finished.all().item()):
                break
        return generated

def _load_tokenizer_any(model_or_path: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_or_path, local_files_only=False, trust_remote_code=True)

def _smoke_text(df: pd.DataFrame, ds: RedditConversationDataset10000, tok: PreTrainedTokenizer) -> None:
    n_samples = len(ds)
    n_users = df['target_user_id'].nunique()
    per_user = Counter(map(int, df['target_user_id']))
    counts = list(per_user.values())
    lens_ctx: List[int] = []
    lens_tgt: List[int] = []
    step = max(1, n_samples // 20000)
    for i in range(0, n_samples, step):
        item = ds[i]
        lens_ctx.append(int(item['attention_mask'].sum()))
        lens_tgt.append(int((item['labels'] != -100).sum()))

    def _stats(arr: List[int]) -> Dict[str, Any]:
        return {'mean': float(sum(arr) / max(1, len(arr))), 'median': int(median(arr)) if arr else 0, 'max': int(max(arr)) if arr else 0}
    buckets = defaultdict(int)
    for L in lens_ctx:
        if L <= 64:
            buckets['≤64'] += 1
        elif L <= 128:
            buckets['≤128'] += 1
        elif L <= 256:
            buckets['≤256'] += 1
        else:
            buckets['>256'] += 1
    print(textwrap.dedent(f'\n            ------------------------ Text smoke-test ------------------------\n            Samples                      : {n_samples:,}\n            Users                        : {n_users:,}\n            Samples / user               : min={(min(counts) if counts else 0)}, median={(median(counts) if counts else 0)}, max={(max(counts) if counts else 0)}\n            Context token lengths        : {_stats(lens_ctx)}\n            Target  token lengths        : {_stats(lens_tgt)}\n            Context length buckets       : {dict(buckets)}\n            Tokenizer pad id             : {tok.pad_token_id}\n            -----------------------------------------------------------------\n            '))

def _smoke_global(df: pd.DataFrame, gdf: pd.DataFrame, ds: HypernetGlobalOnlyDataset10000) -> None:
    counts = Counter(map(int, df['target_user_id']))
    sample = ds[0] if len(ds) > 0 else {'global_features': torch.zeros(ds.g_dim)}
    report = f"\n    --------------------- Global-only smoke-test ----------------------\n    samples               : {len(ds):,}\n    unique users (split)  : {df['target_user_id'].nunique():,}\n    samples / user        : min={(min(counts.values()) if counts else 0)}, median={(median(counts.values()) if counts else 0)}, max={(max(counts.values()) if counts else 0)}\n    author table rows     : {len(gdf):,}\n    global vector dim     : {ds.g_dim}\n    first 12 feature names: {getattr(ds, 'feature_names', [])[:12]}\n    example g[:6]         : {(sample['global_features'][:6].tolist() if 'global_features' in sample else [])}\n    -------------------------------------------------------------------\n    "
    print(textwrap.dedent(report))
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Hypernetwork dataset smoke-test (global-only, dynamic gstat_*).')
    ap.add_argument('--parquet', type=Path, required=False, default=None, help="Split parquet with 'gid', 'group_label', 'text', 'target_user_id'.")
    ap.add_argument('--author_parquet', type=Path, required=False, default=None, help="Author table with 'target_user_id' + gstat_* columns.")
    ap.add_argument('--tokenizer', type=str, required=False, default='EleutherAI/pythia-160m', help='HF model ID or local tokenizer path.')
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--pretok', action='store_true')
    ap.add_argument('--feature_include', type=str, nargs='*', default=['gstat_*'])
    ap.add_argument('--feature_exclude', type=str, nargs='*', default=['gstat_user_sent_*', 'gstat_gap_sentiment', 'gstat_beast_sent_*'])
    ap.add_argument('--no_lists', action='store_true', help='Exclude list-valued gstat_* (keep scalars only).')
    ap.add_argument('-f', '--f', help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.parquet is None:
        for cand in [_DATA_ROOT / 'train_full_10000.parquet', _DATA_ROOT / 'train_data_10000.parquet']:
            if cand.exists():
                args.parquet = cand
                break
    if args.author_parquet is None:
        for cand in [_DATA_ROOT / 'author_static_10000.parquet', _DATA_ROOT / 'global_features_10000.parquet']:
            if cand.exists():
                args.author_parquet = cand
                break
    if args.parquet is None or not args.parquet.exists():
        sys.exit('Pass --parquet pointing to a valid split parquet.')
    if args.author_parquet is None or not args.author_parquet.exists():
        sys.exit('Pass --author_parquet pointing to a valid author feature parquet.')
    t0 = time.time()
    df = pd.read_parquet(args.parquet)
    gdf = pd.read_parquet(args.author_parquet)
    tok = _load_tokenizer_any(args.tokenizer)
    if tok.pad_token_id is None:
        tok.add_special_tokens({'pad_token': '[PAD]'})
        tok.pad_token_id = tok.convert_tokens_to_ids('[PAD]')
    text_ds = RedditConversationDataset10000(df, tok, max_length=args.max_len, pretokenize=args.pretok)
    _smoke_text(df, text_ds, tok)
    hyper_ds = HypernetGlobalOnlyDataset10000(df=df, tokenizer=tok, author_features_df=gdf, max_length=args.max_len, pretokenize=args.pretok, feature_include=args.feature_include, feature_exclude=args.feature_exclude, include_lists=not args.no_lists)
    _smoke_global(df, gdf, hyper_ds)
    print(f'Completed in {time.time() - t0:.1f} s')
