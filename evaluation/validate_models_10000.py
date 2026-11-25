from __future__ import annotations
import argparse
import json
import logging
import math
import os
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
        print('[rmm] unified memory allocator active (RMM %s)' % rmm.__version__, flush=True)
    except Exception as _rmm_err:
        print('[rmm] failed to init: %s; falling back to the default allocator' % _rmm_err, flush=True)
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
os.environ.setdefault('NUMBA_DISABLE_CACHE', '1')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
try:
    from umap import UMAP
except Exception as e:
    UMAP = None
    _UMAP_IMPORT_ERR = e
else:
    _UMAP_IMPORT_ERR = None

def _umap_method() -> str:
    return 'umap' if UMAP is not None else 'pca2_fallback'
try:
    from sklearn.neighbors import KNeighborsClassifier
except Exception as e:
    KNeighborsClassifier = None
    _KNN_IMPORT_ERR = e
else:
    _KNN_IMPORT_ERR = None
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed, utils as hf_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / 'data_scripts'), str(ROOT / 'training_scripts'), str(ROOT / 'PEFT_scripts')])
try:
    from hypernetwork_dataset_10000 import _collect_gstat_cols as _default_gcols
except ImportError:
    from hypernetwork_dataset_10000 import _collect_default_global_cols as _default_gcols
from hypernetwork_structure_160M import PEFTHypernet, FlatHypernetwork, ModelLoRAQKV, ModelAdapter, BiasTuningModel, ModelPrefixOnly, build_peft_layer_groups, count_trainable_params
_RESULT_ROOT = Path('results/embedding_analysis')
_disable_io = False
DEFAULT_REPLY_SEP_TOKEN = '<|reply|>'
DEFAULT_REPLY_END_TOKEN = '<|eoreply|>'
DEFAULT_PAD_TOKEN = '[PAD]'

@dataclass
class LoadedModel:
    model_key: str
    model_display: str
    kind: str
    variant: str
    module: nn.Module
    device: torch.device
    gdim: int = 0
    feat_names: List[str] = field(default_factory=list)
    zscore_global: bool = False
    zscore_center_only: bool = False
    feat_mean: Optional[np.ndarray] = None
    feat_std: Optional[np.ndarray] = None

def _configure_logging(log_path: Path) -> None:
    if getattr(_configure_logging, '_log_init', False):
        return
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(log_path, mode='a')], force=True)
    root = logging.getLogger()
    seen: set[tuple] = set()
    for h in root.handlers[:]:
        sig = (type(h), getattr(h, 'stream', None))
        if sig in seen:
            root.removeHandler(h)
        else:
            seen.add(sig)
    _configure_logging._log_init = True
sys.excepthook = lambda exc_type, exc, tb: logging.exception('Fatal', exc_info=(exc_type, exc, tb))

def _ensure_training_tokens(tok, *, reply_sep_token: str, reply_end_token: str, pad_token: str) -> Tuple[object, bool]:
    added = False
    n_added = tok.add_special_tokens({'additional_special_tokens': [str(reply_sep_token), str(reply_end_token)]})
    if int(n_added) > 0:
        added = True
    if tok.pad_token_id is None or tok.pad_token_id == tok.eos_token_id:
        n_added2 = tok.add_special_tokens({'pad_token': str(pad_token)})
        if int(n_added2) > 0:
            added = True
    tok.padding_side = 'right'
    return (tok, added)

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

def _maybe_resize_and_init_embeddings(model: nn.Module, tok: AutoTokenizer, *, special_token_ids: Optional[List[int]]=None) -> None:
    try:
        in_emb = model.get_input_embeddings()
        W = getattr(in_emb, 'weight', None)
        if W is None:
            return
        old_n = int(W.size(0))
    except Exception:
        return
    new_n = int(len(tok))
    src_id = int(tok.eos_token_id) if tok.eos_token_id is not None else 0
    if new_n > old_n:
        model.resize_token_embeddings(new_n)
        _init_token_rows_from_id(model, list(range(old_n, new_n)), src_id=src_id)
    elif new_n < old_n:
        pass
    if special_token_ids:
        init_ids = []
        for tid in special_token_ids:
            if tid is None:
                continue
            try:
                t = int(tid)
            except Exception:
                continue
            if t == src_id:
                continue
            if t < 0:
                continue
            init_ids.append(t)
        if init_ids:
            _init_token_rows_from_id(model, init_ids, src_id=src_id)

def _build_tokenizer(base_ckpt: str, *, reply_sep_token: str, reply_end_token: str, pad_token: str) -> object:
    tok = AutoTokenizer.from_pretrained(base_ckpt, local_files_only=True, trust_remote_code=True)
    (tok, _) = _ensure_training_tokens(tok, reply_sep_token=reply_sep_token, reply_end_token=reply_end_token, pad_token=pad_token)
    if tok.pad_token_id is None:
        raise ValueError('Tokenizer pad_token_id is still None after adding special tokens.')
    return tok

def _flatten_val(v) -> List[float]:
    if v is None:
        return [0.0]
    if isinstance(v, float) and math.isnan(v):
        return [0.0]
    if isinstance(v, (int, float, np.integer, np.floating)):
        return [float(v)]
    if isinstance(v, (list, tuple, np.ndarray)):
        out: List[float] = []
        for x in v:
            try:
                out.append(float(x))
            except Exception:
                out.append(0.0)
        return out if out else [0.0]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith('[') and s.endswith(']'):
            inner = s[1:-1].replace(',', ' ').split()
            vals: List[float] = []
            for tok in inner:
                try:
                    vals.append(float(tok))
                except Exception:
                    vals.append(0.0)
            return vals if vals else [0.0]
        try:
            return [float(s)]
        except Exception:
            return [0.0]
    return [0.0]

def _build_feature_matrix(df: pd.DataFrame, gcols_used: List[str], target_dim: int) -> np.ndarray:
    recs = df.to_dict(orient='records')
    mat = np.zeros((len(recs), target_dim), dtype=np.float32)
    for (i, rec) in enumerate(recs):
        g_vals: List[float] = []
        for col in gcols_used:
            g_vals.extend(_flatten_val(rec.get(col, 0.0)))
            if len(g_vals) >= target_dim:
                break
        if len(g_vals) < target_dim:
            g_vals.extend([0.0] * (target_dim - len(g_vals)))
        mat[i, :] = np.asarray(g_vals[:target_dim], dtype=np.float32)
    return mat

def _apply_feature_stats(G: np.ndarray, *, zscore_global: bool, zscore_center_only: bool, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    if not zscore_global:
        return G.astype(np.float32, copy=False)
    if mean is None or std is None:
        return G.astype(np.float32, copy=False)
    mu = np.asarray(mean, dtype=np.float32).reshape(-1)[:G.shape[1]]
    sd = np.asarray(std, dtype=np.float32).reshape(-1)[:G.shape[1]]
    sd = np.where(sd > 1e-12, sd, 1.0).astype(np.float32)
    out = G.astype(np.float32, copy=True)
    out -= mu
    if not zscore_center_only:
        out /= sd
    return out

def _load_any_state(path: Path) -> Dict[str, torch.Tensor]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        from safetensors.torch import load_file as _st_load
        if path.suffix == '.safetensors':
            return {k: v for (k, v) in _st_load(str(path)).items()}
    except Exception:
        pass
    blob = torch.load(str(path), map_location='cpu')
    if isinstance(blob, dict):
        if 'state_dict' in blob and isinstance(blob['state_dict'], dict):
            blob = blob['state_dict']
        out: Dict[str, torch.Tensor] = {}
        for (k, v) in blob.items():
            if torch.is_tensor(v):
                out[str(k)] = v
        if out:
            return out
    raise RuntimeError(f'Unrecognized checkpoint format: {path}')

def _strip_prefix(sd: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    for pref in prefixes:
        if sd and all((k.startswith(pref) for k in sd.keys())):
            return {k[len(pref):]: v for (k, v) in sd.items()}
    return sd

def _merge_vocab_tensor(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    out = dst.detach().clone()
    if dst.dim() != src.dim():
        return out
    if dst.dim() == 2:
        if dst.shape[1] != src.shape[1]:
            return out
        n = min(int(dst.shape[0]), int(src.shape[0]))
        out[:n].copy_(src[:n].to(dtype=dst.dtype))
        return out
    if dst.dim() == 1:
        n = min(int(dst.shape[0]), int(src.shape[0]))
        out[:n].copy_(src[:n].to(dtype=dst.dtype))
        return out
    return out

def _load_any_json_in_parents(start_dir: Path, filename: str, max_up: int=6) -> Dict[str, object]:
    d = Path(start_dir).expanduser().resolve()
    for _ in range(int(max_up) + 1):
        cand = d / filename
        if cand.is_file():
            try:
                obj = json.loads(cand.read_text())
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        if d.parent == d:
            break
        d = d.parent
    return {}

def _resolve_latest_step_dir(model_root: Path) -> Optional[Path]:
    model_root = Path(model_root).expanduser().resolve()
    latest = model_root / 'LATEST'
    if not latest.is_file():
        return None
    try:
        step = latest.read_text().strip()
    except Exception:
        return None
    if not step:
        return None
    direct = model_root / step
    if direct.exists():
        return direct
    in_ckpts = model_root / 'checkpoints' / step
    if in_ckpts.exists():
        return in_ckpts
    return None

def _resolve_hyper_ckpt(models_root: Path, variant: str, ckpt_override: Optional[str]) -> Path:
    if ckpt_override:
        _refuse_best_checkpoint(ckpt_override, flag='--hyper_ckpt')
        p = Path(ckpt_override).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f'--hyper_ckpt not found: {p}')
    root = Path(models_root).expanduser()
    py_root = root / 'pythia125m_flat'
    variant_l = variant.lower()
    if variant_l == 'lora':
        candidates = [py_root / 'lora_k6_final_w32d6hr4' / 'feat_k6_hyp_base_w32_d6_hr4', py_root / 'lora_10feat_ablation' / 'feat_top10_hyp_base', py_root / 'lora']
    elif variant_l in ('adapter', 'bias', 'prefix'):
        candidates = [py_root / variant_l]
    else:
        raise ValueError(f'Unknown hyper variant {variant!r}')
    for d in candidates:
        p = d / 'final_hypernet.safetensors'
        if p.is_file():
            return p
    stale = [str(d / 'best_hypernet.safetensors') for d in candidates if (d / 'best_hypernet.safetensors').is_file()]
    raise FileNotFoundError(f"No final_hypernet.safetensors for variant={variant_l!r} under {py_root}. Checked: {', '.join((str(c) for c in candidates))}. " + (f'best_hypernet.safetensors EXISTS at {stale} and is deliberately REFUSED: it is a mid-training snapshot chosen on a fraction of the users, not the model that saw the whole population. Wait for training to finish, or pass an explicit --hyper_ckpt pointing at a final_ checkpoint.' if stale else 'The arm is not finished; final_hypernet.safetensors is written only after the full training schedule completes.'))

def _refuse_best_checkpoint(path, *, flag: str) -> None:
    if path is None:
        return
    s = str(path)
    if not s:
        return
    import os
    if os.path.basename(s).startswith('best_'):
        raise SystemExit('REFUSED: %s points at %s.\n  best_* is a MID-TRAINING snapshot selected on 4.27%% of the users (427 of 10,000).\n  Score final_hypernet.safetensors, which saw all 10,000 users for the full schedule.\n  If the arm has no final_ checkpoint yet, it is not finished. Wait for it.' % (flag, s))

def _resolve_vanilla_placeholders(models_root: Path, variant: str, override: Optional[str]) -> Path:
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f'--vanilla_placeholders not found: {p}')
    root = Path(models_root).expanduser()
    nh_root = root / 'pythia125m_flat_nonhyper'
    variant_l = variant.lower()
    if variant_l == 'lora':
        candidates = [nh_root / 'nonhypernet_lora_model_10000', nh_root / 'lora']
    elif variant_l == 'adapter':
        candidates = [nh_root / 'nonhypernet_adapter_model_10000', nh_root / 'adapter']
    elif variant_l == 'bias':
        candidates = [nh_root / 'nonhypernet_bias_model_10000', nh_root / 'bias']
    elif variant_l == 'prefix':
        candidates = [nh_root / 'nonhypernet_prefix_model_10000', nh_root / 'prefix']
    else:
        raise ValueError(f'Unknown vanilla variant {variant!r}')
    for d in candidates:
        p = d / 'peft_placeholders.safetensors'
        if p.is_file():
            return p
        latest_dir = _resolve_latest_step_dir(d)
        if latest_dir is not None:
            p2 = latest_dir / 'peft_placeholders.safetensors'
            if p2.is_file():
                return p2
    raise FileNotFoundError(f"Could not find peft_placeholders.safetensors for vanilla variant={variant_l!r} under {nh_root}. Checked: {', '.join((str(c) for c in candidates))}")

def _resolve_baseline_ckpt(models_root: Path, override: Optional[str]) -> Path:
    if override:
        p = Path(override).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f'--baseline_ckpt not found: {p}')
    root = Path(models_root).expanduser()
    nh_root = root / 'pythia125m_flat_nonhyper'
    base_root = nh_root / 'nonhypernet_base_model_10000'
    latest_dir = _resolve_latest_step_dir(base_root)
    if latest_dir is not None and latest_dir.exists():
        return latest_dir
    candidates = [base_root / 'checkpoints' / 'step_0028000' / 'model.safetensors', base_root / 'checkpoints' / 'step_0028000', base_root]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find baseline checkpoint under {base_root}. Checked: {', '.join((str(c) for c in candidates))}")

def _build_backbone(base_ckpt: str, tok: AutoTokenizer, device: torch.device) -> nn.Module:
    cfg = AutoConfig.from_pretrained(base_ckpt, local_files_only=True, trust_remote_code=True)
    cfg.output_hidden_states = True
    cfg.use_cache = False
    if cfg.pad_token_id is None and tok.pad_token_id is not None:
        cfg.pad_token_id = int(tok.pad_token_id)
    backbone = AutoModelForCausalLM.from_pretrained(base_ckpt, local_files_only=True, trust_remote_code=True, config=cfg)
    special_ids: List[int] = []
    try:
        rid = tok.convert_tokens_to_ids(DEFAULT_REPLY_SEP_TOKEN)
        if isinstance(rid, int) and rid >= 0:
            special_ids.append(int(rid))
    except Exception:
        pass
    try:
        eid = tok.convert_tokens_to_ids(DEFAULT_REPLY_END_TOKEN)
        if isinstance(eid, int) and eid >= 0:
            special_ids.append(int(eid))
    except Exception:
        pass
    if tok.pad_token_id is not None:
        special_ids.append(int(tok.pad_token_id))
    _maybe_resize_and_init_embeddings(backbone, tok, special_token_ids=special_ids)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.to(device)
    backbone.eval()
    return backbone

def _infer_lora_cfg_from_placeholders(placeholders_path: Path) -> Tuple[int, float]:
    from safetensors.torch import load_file as _st_load
    flat = _st_load(str(placeholders_path))
    rank: Optional[int] = None
    for (k, t) in flat.items():
        if 'lora_A.weight' in k and isinstance(t, torch.Tensor) and (t.ndim == 2):
            rank = int(t.shape[0])
            break
    if rank is None:
        for (_, t) in flat.items():
            if isinstance(t, torch.Tensor) and t.ndim == 2:
                rank = int(min(t.shape))
                break
    if rank is None or rank <= 0:
        raise RuntimeError(f'Could not infer LoRA rank from placeholders: {placeholders_path}')
    alpha = float(rank)
    return (rank, alpha)

def _load_safetensors_subset_into_model(path: Path, model: nn.Module) -> Dict[str, int]:
    from safetensors.torch import load_file as _st_load
    sd = _st_load(str(path))
    model_keys = set(model.state_dict().keys())
    sd_keys = set(sd.keys())
    overlap = len(model_keys.intersection(sd_keys))
    res = model.load_state_dict(sd, strict=False)
    missing = len(getattr(res, 'missing_keys', []))
    unexpected = len(getattr(res, 'unexpected_keys', []))
    return {'keys_in_file': int(len(sd_keys)), 'keys_in_model': int(len(model_keys)), 'matched_keys': int(overlap), 'missing_keys': int(missing), 'unexpected_keys': int(unexpected)}

def _infer_expected_gdim(hyper_ckpt_path: Path) -> int:
    hyper_ckpt_path = Path(hyper_ckpt_path)
    try:
        from safetensors.torch import load_file as _st_load
        if hyper_ckpt_path.suffix == '.safetensors':
            flat = _st_load(str(hyper_ckpt_path))
            if 'hyper.hidden.0.weight' in flat:
                t = flat['hyper.hidden.0.weight']
                if isinstance(t, torch.Tensor) and t.ndim == 2:
                    return int(t.shape[1])
            k_max = -1
            for (k, t) in flat.items():
                if not (k.startswith('hyper.hidden.') and k.endswith('.weight')):
                    continue
                if not isinstance(t, torch.Tensor) or t.ndim != 2:
                    continue
                k_max = max(k_max, int(t.shape[1]))
            if k_max > 0:
                return k_max
            raise RuntimeError(f'Could not infer gdim from {hyper_ckpt_path}')
    except Exception:
        pass
    blob = torch.load(str(hyper_ckpt_path), map_location='cpu')
    h_state = blob.get('__hyper_state__', None)
    if isinstance(h_state, dict):
        t0 = h_state.get('hidden.0.weight', None)
        if isinstance(t0, torch.Tensor) and t0.ndim == 2:
            return int(t0.shape[1])
        k_max = -1
        for (k, t) in h_state.items():
            if not (isinstance(t, torch.Tensor) and t.ndim == 2):
                continue
            if k.startswith('hidden.') and k.endswith('.weight'):
                k_max = max(k_max, int(t.shape[1]))
        if k_max > 0:
            return k_max
    raise RuntimeError(f'Could not infer global-feature dimension from checkpoint {hyper_ckpt_path}')

def _placeholder_splits_by_layer(peft_model: nn.Module, *, exclude_param_ids: Optional[List[int]]=None) -> List[int]:
    import re
    from collections import defaultdict
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

def _peft_group_splits(peft_model: nn.Module) -> List[int]:
    try:
        return _placeholder_splits_by_layer(peft_model)
    except Exception:
        groups = build_peft_layer_groups(peft_model)
        splits = [int(g['size']) for g in groups]
        return splits

def _load_hyper_model(variant: str, models_root: Path, base_ckpt: str, tok: AutoTokenizer, device: torch.device, *, hyper_ckpt_override: Optional[str]) -> LoadedModel:
    ckpt_path = _resolve_hyper_ckpt(models_root, variant, hyper_ckpt_override)
    model_dir = Path(ckpt_path).expanduser().resolve().parent
    run_cfg = _load_any_json_in_parents(model_dir, 'args.json') or {}

    def _cfg_int(key: str, default: int) -> int:
        try:
            return int(run_cfg.get(key, default))
        except Exception:
            return int(default)

    def _cfg_float(key: str, default: float) -> float:
        try:
            return float(run_cfg.get(key, default))
        except Exception:
            return float(default)

    def _cfg_bool(key: str, default: bool) -> bool:
        v = run_cfg.get(key, default)
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, (int, float)):
            return bool(int(v) != 0)
        s = str(v).strip().lower()
        if s in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if s in {'0', 'false', 'no', 'n', 'off'}:
            return False
        return bool(default)
    gdim = _infer_expected_gdim(ckpt_path)
    clamp_min = _cfg_float('clamp_min', 0.05)
    clamp_max = _cfg_float('clamp_max', 0.1)
    clamp_use = float(clamp_max)
    hyper_width = _cfg_int('hyper_width', 64)
    hyper_depth = _cfg_int('hyper_depth', 9)
    hyper_dropout = _cfg_float('hyper_dropout', 0.0)
    hyper_head_rank = _cfg_int('hyper_head_rank', 16)
    hyper_group_by_layer = _cfg_bool('hyper_group_by_layer', False)
    backbone = _build_backbone(base_ckpt, tok, device)
    variant_l = variant.lower().strip()
    if variant_l == 'lora':
        lora_rank = _cfg_int('lora_rank', 32)
        lora_alpha = _cfg_float('lora_alpha', float(lora_rank))
        train_rows: List[int] = []
        try:
            rid = tok.convert_tokens_to_ids(DEFAULT_REPLY_SEP_TOKEN)
            if isinstance(rid, int) and rid >= 0:
                train_rows.append(int(rid))
        except Exception:
            pass
        try:
            eid = tok.convert_tokens_to_ids(DEFAULT_REPLY_END_TOKEN)
            if isinstance(eid, int) and eid >= 0:
                train_rows.append(int(eid))
        except Exception:
            pass
        peft = ModelLoRAQKV(backbone, rank=lora_rank, alpha=lora_alpha, train_token_rows=train_rows)
    elif variant_l == 'adapter':
        bottleneck = _cfg_int('adapter_bottleneck', _cfg_int('adapter_bottleneck_dim', 256))
        dropout = _cfg_float('adapter_dropout', 0.0)
        use_ln = _cfg_bool('adapter_ln', False)
        peft = ModelAdapter(backbone, bottleneck_dim=bottleneck, adapter_dropout=dropout, use_layer_norm=use_ln)
    elif variant_l == 'bias':
        peft = BiasTuningModel(backbone)
    elif variant_l == 'prefix':
        prefix_len = _cfg_int('prefix_length', 10)
        prefix_drop = _cfg_float('prefix_dropout', 0.0)
        peft = ModelPrefixOnly(backbone, prefix_length=prefix_len, dropout_prob=prefix_drop)
    else:
        raise ValueError(f'unknown variant: {variant}')
    exclude_ids: List[int] = []
    if hasattr(peft, 'special_row_params') and callable(getattr(peft, 'special_row_params')):
        try:
            for p in list(peft.special_row_params()):
                exclude_ids.append(id(p))
        except Exception:
            exclude_ids = []
    trainable_cnt = int(count_trainable_params(peft, exclude_param_ids=exclude_ids))
    group_splits = _placeholder_splits_by_layer(peft, exclude_param_ids=exclude_ids) if hyper_group_by_layer else None
    hyper = FlatHypernetwork(input_dim=gdim, output_dim=trainable_cnt, width=hyper_width, depth=hyper_depth, dropout=hyper_dropout, clamp=clamp_use, head_rank=hyper_head_rank, group_splits=group_splits)
    wrapper = PEFTHypernet(peft, hyper, clamp=clamp_use).to(device)
    PEFTHypernet.load_hypernet_ckpt(str(ckpt_path), wrapper)
    ckpt_last = model_dir / 'train_checkpoints' / 'checkpoint_last.pt'
    try:
        if ckpt_last.exists() and hasattr(peft, 'special_row_params') and callable(getattr(peft, 'special_row_params')):
            payload = torch.load(str(ckpt_last), map_location='cpu', weights_only=False)
            tensors = payload.get('special_rows', None) if isinstance(payload, dict) else None
            if isinstance(tensors, list) and tensors:
                params = list(peft.special_row_params())
                if len(params) == len(tensors):
                    with torch.no_grad():
                        for (p, t) in zip(params, tensors):
                            if torch.is_tensor(t):
                                p.copy_(t.to(device=device, dtype=p.dtype))
    except Exception as exc:
        logging.warning('[special-rows] could not restore from %s: %s', ckpt_last, exc)
    wrapper.hypernet.clamp = float(clamp_use)
    wrapper.eval()
    feat_names: List[str] = []
    feat_mean: Optional[np.ndarray] = None
    feat_std: Optional[np.ndarray] = None
    feat_names_path = model_dir / 'feature_names.json'
    if feat_names_path.is_file():
        try:
            payload = json.loads(feat_names_path.read_text())
            if isinstance(payload, list):
                feat_names = [str(x) for x in payload]
            elif isinstance(payload, dict):
                for key in ('features', 'feature_names', 'names'):
                    v = payload.get(key, None)
                    if isinstance(v, list) and v:
                        feat_names = [str(x) for x in v]
                        break
        except Exception:
            feat_names = []
    feat_stats_path = model_dir / 'feature_stats.json'
    if feat_stats_path.is_file():
        try:
            payload = json.loads(feat_stats_path.read_text())
            if isinstance(payload, dict):
                m = payload.get('mean', None)
                s = payload.get('std', None)
                if m is not None and s is not None:
                    feat_mean = np.asarray(m, dtype=np.float32).reshape(-1)
                    feat_std = np.asarray(s, dtype=np.float32).reshape(-1)
                if not feat_names:
                    names = payload.get('names', None)
                    if isinstance(names, list) and names:
                        feat_names = [str(x) for x in names]
        except Exception:
            pass
    try:
        zscore_global = bool(run_cfg.get('zscore_global', bool(feat_mean is not None)))
    except Exception:
        zscore_global = bool(feat_mean is not None)
    try:
        zscore_center_only = bool(run_cfg.get('zscore_center_only', False))
    except Exception:
        zscore_center_only = False
    model_key = f'hyperPEFT-{variant_l}'
    model_display = {'lora': 'HyperPEFT-LoRA', 'adapter': 'HyperPEFT-Adapter', 'bias': 'HyperPEFT-Bias', 'prefix': 'HyperPEFT-Prefix'}.get(variant_l, f'HyperPEFT-{variant_l}')
    logging.info('Loaded hyper model %s from %s • gdim=%d • P=%d • tok=%d', model_key, ckpt_path, int(gdim), int(trainable_cnt), int(len(tok)))
    return LoadedModel(model_key=model_key, model_display=model_display, kind='hyper', variant=variant_l, module=wrapper, device=device, gdim=int(gdim), feat_names=feat_names, zscore_global=bool(zscore_global), zscore_center_only=bool(zscore_center_only), feat_mean=feat_mean, feat_std=feat_std)

def _infer_vanilla_prefix_cfg(placeholders_path: Path) -> Tuple[int, float]:
    from safetensors.torch import load_file as _load_st
    sd = _load_st(str(placeholders_path))
    lens = {int(t.shape[1]) for (k, t) in sd.items() if ('prefix_key' in k or 'prefix_value' in k) and t.dim() == 3}
    if not lens:
        raise ValueError(f'no prefix_key/prefix_value tensors in {placeholders_path}; cannot infer prefix_length')
    if len(lens) != 1:
        raise ValueError(f'inconsistent prefix_length across tensors in {placeholders_path}: {sorted(lens)}')
    return (int(next(iter(lens))), 0.0)

def _infer_vanilla_adapter_cfg(placeholders_path: Path) -> Tuple[int, bool]:
    from safetensors.torch import load_file as _load_st
    sd = _load_st(str(placeholders_path))
    dims = {int(t.shape[0]) for (k, t) in sd.items() if k.endswith('down_proj.weight') and t.dim() == 2}
    if not dims:
        raise ValueError(f'no down_proj.weight tensors in {placeholders_path}; cannot infer bottleneck')
    if len(dims) != 1:
        raise ValueError(f'inconsistent adapter bottleneck in {placeholders_path}: {sorted(dims)}')
    has_ln = any(('layer_norm' in k.lower() or 'layernorm' in k.lower() for k in sd))
    return (int(next(iter(dims))), bool(has_ln))

def _load_vanilla_model(variant: str, models_root: Path, base_ckpt: str, tok, device: torch.device, *, placeholders_override: Optional[str]) -> LoadedModel:
    variant_l = variant.lower()
    placeholders_path = _resolve_vanilla_placeholders(models_root, variant_l, placeholders_override)
    if not placeholders_path.is_file():
        raise FileNotFoundError(f'Missing vanilla placeholders at {placeholders_path}')
    backbone = _build_backbone(base_ckpt, tok, device)
    if variant_l == 'lora':
        lora_rank = 32
        lora_alpha = 32.0
        cfg = _load_any_json_in_parents(placeholders_path.parent, 'args.json')
        if cfg:
            peft_cfg = cfg.get('peft_cfg', {}) if isinstance(cfg, dict) else {}
            if isinstance(peft_cfg, dict):
                try:
                    lora_rank = int(peft_cfg.get('lora_rank', lora_rank))
                except Exception:
                    pass
                try:
                    lora_alpha = float(peft_cfg.get('lora_alpha', lora_alpha))
                except Exception:
                    pass
            try:
                lora_rank = int(cfg.get('lora_rank', lora_rank))
            except Exception:
                pass
            try:
                lora_alpha = float(cfg.get('lora_alpha', lora_alpha))
            except Exception:
                pass
        if lora_rank <= 0:
            (rank_inf, alpha_inf) = _infer_lora_cfg_from_placeholders(placeholders_path)
            (lora_rank, lora_alpha) = (int(rank_inf), float(alpha_inf))
        if lora_alpha <= 0:
            lora_alpha = float(lora_rank)
        peft = ModelLoRAQKV(backbone, rank=int(lora_rank), alpha=float(lora_alpha))
    elif variant_l == 'adapter':
        (b_dim, use_ln) = _infer_vanilla_adapter_cfg(placeholders_path)
        peft = ModelAdapter(backbone, bottleneck_dim=int(b_dim), adapter_dropout=0.0, use_layer_norm=bool(use_ln))
    elif variant_l == 'bias':
        peft = BiasTuningModel(backbone)
    elif variant_l == 'prefix':
        (p_len, p_drop) = _infer_vanilla_prefix_cfg(placeholders_path)
        peft = ModelPrefixOnly(backbone, prefix_length=int(p_len), dropout_prob=float(p_drop))
    else:
        raise ValueError(f'Unsupported variant {variant!r}')
    peft.to(device)
    peft.eval()
    diag = _load_safetensors_subset_into_model(placeholders_path, peft)
    model_key = f'vanilla_{variant_l}'
    model_display = {'lora': 'Vanilla LoRA', 'adapter': 'Vanilla Adapter', 'bias': 'Vanilla Bias', 'prefix': 'Vanilla Prefix'}.get(variant_l, f'Vanilla {variant_l}')
    logging.info('Loaded vanilla model %s from %s • placeholders: %s • tok=%d', model_key, placeholders_path, diag, len(tok))
    return LoadedModel(model_key=model_key, model_display=model_display, kind='vanilla', variant=variant_l, module=peft, device=device)

def _load_baseline_model(models_root: Path, base_ckpt: str, tok, device: torch.device, *, baseline_override: Optional[str]) -> LoadedModel:
    ckpt_path = _resolve_baseline_ckpt(models_root, baseline_override)
    candidate_dirs: List[Path] = []
    if ckpt_path.is_dir():
        candidate_dirs.append(ckpt_path)
    elif ckpt_path.is_file():
        parent = ckpt_path.parent
        if (parent / 'config.json').is_file():
            candidate_dirs.append(parent)
    for d in candidate_dirs:
        try:
            model = AutoModelForCausalLM.from_pretrained(str(d), local_files_only=True, trust_remote_code=True)
            model.config.output_hidden_states = True
            model.config.use_cache = False
            if getattr(model.config, 'pad_token_id', None) is None and tok.pad_token_id is not None:
                model.config.pad_token_id = int(tok.pad_token_id)
            model_vocab = int(model.get_input_embeddings().weight.size(0))
            tok_vocab = int(len(tok))
            special_ids: List[int] = []
            try:
                rid = tok.convert_tokens_to_ids(DEFAULT_REPLY_SEP_TOKEN)
                if isinstance(rid, int) and rid >= 0:
                    special_ids.append(int(rid))
            except Exception:
                pass
            try:
                eid = tok.convert_tokens_to_ids(DEFAULT_REPLY_END_TOKEN)
                if isinstance(eid, int) and eid >= 0:
                    special_ids.append(int(eid))
            except Exception:
                pass
            if tok.pad_token_id is not None:
                special_ids.append(int(tok.pad_token_id))
            if tok_vocab > model_vocab:
                print(f'[baseline] resizing token embeddings {model_vocab}→{tok_vocab}')
                model.resize_token_embeddings(tok_vocab)
            _maybe_resize_and_init_embeddings(model, tok, special_token_ids=special_ids)
            for p in model.parameters():
                p.requires_grad_(False)
            model.to(device)
            model.eval()
            logging.info('Loaded baseline model from directory: %s • tok=%d', d, len(tok))
            return LoadedModel(model_key='baseline', model_display='Baseline', kind='baseline', variant='base', module=model, device=device)
        except Exception as e:
            logging.warning('[baseline] directory load failed for %s (%r); falling back to state-dict load.', d, e)
    model = _build_backbone(base_ckpt, tok, device)
    if ckpt_path.is_dir():
        cand = ckpt_path / 'model.safetensors'
        if not cand.is_file():
            raise RuntimeError(f'Baseline ckpt is a directory but model.safetensors is missing: {ckpt_path}')
        ckpt_file = cand
    else:
        ckpt_file = ckpt_path
    sd = _load_any_state(ckpt_file)
    sd = _strip_prefix(sd, ('module.', 'backbone.', 'model.'))
    cur_sd = model.state_dict()
    patched: List[str] = []
    for k in ('gpt_neox.embed_in.weight', 'embed_out.weight', 'embed_out.bias'):
        if k in sd and k in cur_sd and (tuple(sd[k].shape) != tuple(cur_sd[k].shape)):
            sd[k] = _merge_vocab_tensor(cur_sd[k], sd[k])
            patched.append(k)
    if patched:
        logging.warning('[baseline] vocab mismatch patched for %d key(s): %s', len(patched), ', '.join(patched))
    res = model.load_state_dict(sd, strict=False)
    missing = len(getattr(res, 'missing_keys', []))
    unexpected = len(getattr(res, 'unexpected_keys', []))
    if missing:
        logging.warning('[baseline] missing keys while loading fine-tuned weights: %d', missing)
    if unexpected:
        logging.warning('[baseline] unexpected keys while loading fine-tuned weights: %d', unexpected)
    model.to(device)
    model.eval()
    logging.info('Loaded baseline model from file: %s • tok=%d', ckpt_file, len(tok))
    return LoadedModel(model_key='baseline', model_display='Baseline', kind='baseline', variant='base', module=model, device=device)

def _pick_text_column(df: pd.DataFrame) -> str:
    for c in ('text', 'target_text', 'reply', 'response', 'content'):
        if c in df.columns:
            return c
    raise KeyError(f'test_parquet is missing a usable text column; found columns: {list(df.columns)[:50]}')

def _wrap_as_reply_block(text: str, *, reply_sep_token: str, reply_end_token: str) -> str:
    t = str(text).strip()
    if not t:
        return ''
    return f'{reply_sep_token}\n{t}\n{reply_end_token}'

def _build_user_prompts_from_test_parquet(test_df: pd.DataFrame, user_ids: List[int], *, texts_per_user: int, seed: int, fallback_prompt: str, group_suffix: str, wrap_reply_tokens: bool, reply_sep_token: str, reply_end_token: str) -> Dict[int, str]:
    want = set(map(int, user_ids))
    if not want:
        return {}
    required = {'target_user_id', 'group_label'}
    missing = required.difference(set(test_df.columns))
    if missing:
        raise KeyError(f'test_parquet missing required columns for hidden probe: {sorted(missing)}')
    text_col = _pick_text_column(test_df)
    df = test_df.copy()
    df['target_user_id'] = df['target_user_id'].astype('int64', errors='ignore')
    mask = df['group_label'].astype(str).str.endswith(str(group_suffix))
    df = df[mask]
    df = df[df['target_user_id'].isin(want)]
    if df.empty:
        return {uid: fallback_prompt for uid in want}
    sort_cols = [c for c in ('target_user_id', 'gid', 'group_label') if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    prompts: Dict[int, str] = {}
    for (uid, sub) in df.groupby('target_user_id', sort=False):
        texts = [str(t).strip() for t in sub[text_col].tolist() if isinstance(t, str) and t.strip()]
        if not texts:
            prompts[int(uid)] = fallback_prompt
            continue
        rng = np.random.default_rng(int(seed) + int(uid))
        if len(texts) > int(texts_per_user):
            idx = rng.choice(len(texts), size=int(texts_per_user), replace=False)
            picked = [texts[i] for i in sorted(idx)]
        else:
            picked = texts[:int(texts_per_user)]
        if wrap_reply_tokens:
            picked = [_wrap_as_reply_block(t, reply_sep_token=reply_sep_token, reply_end_token=reply_end_token) for t in picked]
            picked = [p for p in picked if p.strip()]
        prompts[int(uid)] = '\n'.join(picked) if picked else fallback_prompt
    for uid in want:
        prompts.setdefault(int(uid), fallback_prompt)
    return prompts

def _build_user_prompts_neutral(user_ids: List[int], *, mode: str, seed: int, base_prompt: str) -> Dict[int, str]:
    want = list(map(int, user_ids))
    out: Dict[int, str] = {}
    if mode == 'neutral':
        for uid in want:
            out[int(uid)] = str(base_prompt)
        return out
    if mode == 'neutral_uid':
        for uid in want:
            out[int(uid)] = f'{base_prompt}\n\n[USER_ID={int(uid)}]'
        return out
    if mode == 'neutral_nonce':
        pool = ['I enjoy reading.', 'The sky can be cloudy.', 'Numbers are interesting.', 'A short sentence follows.', 'This is a neutral statement.', 'Sometimes I take notes.', 'A simple thought appears.', 'This text is filler.', 'We can add a nonce.', 'Nothing personal here.']
        for uid in want:
            r = np.random.default_rng(int(seed) + int(uid))
            phrase = pool[int(r.integers(0, len(pool)))]
            out[int(uid)] = f'{base_prompt}\n\n{phrase}'
        return out
    raise ValueError(f'Unknown neutral hidden prompt mode: {mode!r}')

def _parse_neutral_prompt_pool(*, base_prompt: str, neutral_prompts: str) -> List[str]:
    s = str(neutral_prompts or '').strip()
    if not s:
        return [str(base_prompt)]
    parts = [p.strip() for p in s.split('|') if p.strip()]
    return parts if parts else [str(base_prompt)]

def _make_high_entropy_prompt_pool(*, base_prompt: str, neutral_prompts: str, pool_size: int) -> List[str]:
    pool = _parse_neutral_prompt_pool(base_prompt=str(base_prompt), neutral_prompts=str(neutral_prompts))
    if len(pool) > 1:
        return pool
    base_bank = [str(base_prompt), 'Explain a simple concept in a few sentences.', 'Write a short paragraph about a neutral topic.', 'Describe a routine daily activity.', 'Summarize what you did today in general terms.', 'Give a brief definition of a common word.', 'Describe a place you have visited.', 'Write a short note about an everyday object.', 'Explain how to do a basic task step by step.', 'State a few general facts about a topic you know.', 'Write two sentences about a memory.', 'Describe a hobby in a neutral tone.', 'Explain what you like about learning new things.', 'Write a short response to a generic question.', 'Describe something you saw recently.', 'Give a brief overview of a familiar subject.', 'Explain a basic idea without jargon.', 'Write a short answer with no strong opinions.', 'Describe a common social situation.', 'Write a neutral reply to a polite greeting.']
    target = int(max(1, pool_size))
    out: List[str] = []
    i = 0
    while len(out) < target:
        p = base_bank[i % len(base_bank)]
        out.append(f'{p}\n\n[NEUTRAL_PROMPT_ID={i}]')
        i += 1
    return out

def _expand_prompts_multi(user_ids: List[int], labels: List[str], *, prompt_pool: List[str], prompts_per_user: int, seed: int) -> Tuple[List[str], List[str], List[int], int]:
    if len(user_ids) != len(labels):
        raise ValueError('expand_prompts_multi: user_ids and labels length mismatch')
    pool = list(prompt_pool) if prompt_pool else ['Tell me something interesting.']
    P = int(max(1, prompts_per_user))
    prompts_rep: List[str] = []
    labels_rep: List[str] = []
    users_rep: List[int] = []
    for (uid, lab) in zip(user_ids, labels):
        uid_i = int(uid)
        r = np.random.default_rng(int(seed) + uid_i)
        for j in range(P):
            if len(pool) == 1:
                ptxt = pool[0]
            else:
                idx = int(r.integers(0, len(pool)))
                ptxt = pool[idx]
            prompts_rep.append(str(ptxt))
            labels_rep.append(str(lab))
            users_rep.append(uid_i)
    return (prompts_rep, labels_rep, users_rep, P)

def _aggregate_user_vectors_mean(X: np.ndarray, users_rep: List[int], *, min_norm: float) -> Tuple[np.ndarray, List[int]]:
    if X.ndim != 2:
        raise ValueError('aggregate_user_vectors_mean: X must be 2D')
    if X.shape[0] != len(users_rep):
        raise ValueError('aggregate_user_vectors_mean: X/users_rep length mismatch')
    from collections import defaultdict
    buckets: Dict[int, List[np.ndarray]] = defaultdict(list)
    for (v, uid) in zip(X, users_rep):
        buckets[int(uid)].append(v)
    users = sorted(buckets.keys())
    out = np.zeros((len(users), X.shape[1]), dtype=np.float32)
    for (i, uid) in enumerate(users):
        vs = buckets[int(uid)]
        m = np.mean(np.stack(vs, axis=0), axis=0).astype(np.float32)
        out[i, :] = m
    norms = np.linalg.norm(out, axis=1)
    keep = norms >= float(min_norm)
    if not keep.any():
        raise RuntimeError('aggregate_user_vectors_mean: produced zero usable vectors (all norms < min_norm).')
    out = out[keep].astype(np.float32)
    users = [u for (u, k) in zip(users, keep) if k]
    (out, _) = _l2_normalize_rows(out)
    return (out, users)

def _apply_hyper_g_mode(F_proc: np.ndarray, users_list: List[int], *, mode: str, seed: int) -> np.ndarray:
    m = str(mode).strip().lower()
    if m == 'normal':
        return F_proc
    if m == 'zero':
        return np.zeros_like(F_proc, dtype=np.float32)
    if m == 'shuffle':
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(F_proc.shape[0])
        return F_proc[perm].astype(np.float32, copy=False)
    raise ValueError(f'Unknown hyper g-mode: {mode!r}')

def _l2_normalize_rows(X: np.ndarray, eps: float=1e-12) -> Tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(X, axis=1) + eps
    return ((X / norms[:, None]).astype(np.float32), norms.astype(np.float32))

def _is_degenerate(X: np.ndarray, atol: float=1e-08) -> bool:
    if X.shape[0] < 2:
        return True
    return bool(np.allclose(X, X[0:1, :], atol=atol))

def _select_hidden_state(out, prefer_penultimate: bool=True) -> torch.Tensor:
    if hasattr(out, 'hidden_states') and out.hidden_states is not None:
        hs = out.hidden_states
        if prefer_penultimate and len(hs) >= 2:
            return hs[-2]
        return hs[-1]
    if isinstance(out, (tuple, list)):
        for elem in reversed(out):
            if torch.is_tensor(elem) and elem.dim() >= 3:
                return elem
            if isinstance(elem, (tuple, list)) and elem:
                cand = elem[-1]
                if torch.is_tensor(cand) and cand.dim() >= 3:
                    return cand
    raise AttributeError('Could not locate hidden states in model output.')

@torch.no_grad()
def _pop_oof(hi: Dict[str, Any], out_root: 'Path', *, disable_io: bool) -> None:
    oof = hi.pop('_oof', None)
    if oof is None or disable_io:
        return
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_root / 'probe_oof.npz', score=np.asarray(oof['score'], dtype=np.float64), label=np.asarray(oof['label'], dtype=np.int64), fold=np.asarray(oof['fold'], dtype=np.int64))
    except Exception as exc:
        logging.warning('[probe-oof] could not write %s: %s', out_root / 'probe_oof.npz', exc)

def _extract_hidden_probe(model: nn.Module, prompts: List[str], labels: List[str], users: List[int], *, tok, device: torch.device, batch_size: int, max_length: int, min_norm: float, global_features: Optional[np.ndarray], prefer_penultimate: bool=True, normalize_output: bool=True) -> Tuple[np.ndarray, List[str], List[int], Dict[str, object]]:
    tgt = model.module if isinstance(model, nn.DataParallel) else model
    if len(prompts) != len(labels) or len(prompts) != len(users):
        raise ValueError('hidden probe: prompts/labels/users must have identical length')
    n = len(prompts)
    out_vecs: List[np.ndarray] = []
    out_labels: List[str] = []
    out_users: List[int] = []
    vec_norms: List[float] = []
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        enc = tok(prompts[start:end], return_tensors='pt', padding=True, truncation=True, max_length=int(max_length))
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        if global_features is not None:
            g_batch = torch.tensor(global_features[start:end], dtype=torch.float32, device=device)
            kwargs['global_features'] = g_batch
        out = tgt(**kwargs)
        H = _select_hidden_state(out, prefer_penultimate=prefer_penultimate)
        if H.dim() == 2:
            H = H.unsqueeze(0)
        am = attention_mask.to(H.dtype).unsqueeze(-1)
        denom = am.sum(dim=1).clamp_min(1.0)
        pooled = (H * am).sum(dim=1) / denom
        norms = pooled.norm(dim=1)
        keep = (norms >= float(min_norm)).detach().cpu().numpy().astype(bool)
        if keep.any():
            v_keep_t = pooled[keep]
            n_keep_t = norms[keep].unsqueeze(1)
            if bool(normalize_output):
                v_keep_t = v_keep_t / (n_keep_t + 1e-12)
            v_keep = v_keep_t.detach().cpu().numpy().astype(np.float32)
            out_vecs.append(v_keep)
            out_labels.extend([labels[i] for i in range(start, end) if keep[i - start]])
            out_users.extend([users[i] for i in range(start, end) if keep[i - start]])
            vec_norms.extend(norms[keep].detach().cpu().numpy().astype(np.float32).tolist())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not out_vecs:
        raise RuntimeError('hidden probe: produced zero usable vectors.')
    X = np.concatenate(out_vecs, axis=0).astype(np.float32)
    meta = {'hidden_norm_mean': float(np.mean(vec_norms)) if vec_norms else float('nan'), 'hidden_norm_std': float(np.std(vec_norms)) if vec_norms else float('nan'), 'hidden_norms': vec_norms, 'max_length': int(max_length), 'prefer_penultimate': bool(prefer_penultimate), 'normalize_output': bool(normalize_output)}
    return (X, out_labels, out_users, meta)

def _extract_gvec_probe(G: np.ndarray, labels: List[str], users: List[int], *, min_norm: float) -> Tuple[np.ndarray, List[str], List[int], Dict[str, object]]:
    norms = np.linalg.norm(G, axis=1)
    keep = norms >= float(min_norm)
    if not keep.any():
        raise RuntimeError('gvec: produced zero usable vectors (all norms < min_norm).')
    V_keep = G[keep].astype(np.float32)
    (X, _) = _l2_normalize_rows(V_keep)
    out_labels = [l for (l, k) in zip(labels, keep) if k]
    out_users = [u for (u, k) in zip(users, keep) if k]
    meta = {'raw_norm_mean': float(np.mean(norms[keep])), 'raw_norm_std': float(np.std(norms[keep]))}
    return (X, out_labels, out_users, meta)

def _umap_reduce(X: np.ndarray, *, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    X_in = X.astype(np.float32, copy=False)
    if X_in.shape[0] >= 2:
        try:
            (_, uniq_idx) = np.unique(X_in, axis=0, return_index=True)
            dup = int(X_in.shape[0] - len(uniq_idx))
        except Exception:
            dup = 0
        if dup > 0:
            span = (X_in.max(axis=0) - X_in.min(axis=0)).astype(np.float32)
            scale = float(np.mean(span)) if np.isfinite(span).all() else 1.0
            rng = np.random.default_rng(int(seed))
            X_in = X_in + rng.normal(0.0, 0.0001 * max(scale, 1e-06), size=X_in.shape).astype(np.float32)
    if X_in.shape[1] > 50:
        X_in = PCA(n_components=50, svd_solver='randomized', random_state=int(seed)).fit_transform(X_in)
    if UMAP is None:
        logging.warning('UMAP unavailable (%r); 2-D projection falls back to PCA', _UMAP_IMPORT_ERR)
        return PCA(n_components=2, svd_solver='randomized', random_state=int(seed)).fit_transform(X_in)
    um = UMAP(n_components=2, n_neighbors=int(n_neighbors), min_dist=float(min_dist), metric='cosine', init='spectral', random_state=int(seed))
    return um.fit_transform(X_in)

def _plot_scatter(X2: np.ndarray, labels: List[str], out_png: Path, title: str) -> None:
    uniq = sorted(set(labels))
    n_cls = len(uniq)
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*']
    if n_cls == 2:
        col_map = {uniq[0]: '#1f77b4', uniq[1]: '#ff7f0e'}
        mrk_map = {uniq[0]: 'o', uniq[1]: 'x'}
    else:
        col_map = {lbl: base_colors[i % len(base_colors)] for (i, lbl) in enumerate(uniq)}
        mrk_map = {lbl: markers[i % len(markers)] for (i, lbl) in enumerate(uniq)}
    if X2.shape[0] >= 2:
        (_, uniq_idx) = np.unique(X2, axis=0, return_index=True)
        dup_mask = np.ones(len(X2), dtype=bool)
        dup_mask[uniq_idx] = False
        if dup_mask.any():
            span = X2.max(axis=0) - X2.min(axis=0) + 1e-12
            rng = np.random.default_rng(0)
            X2 = X2.copy()
            X2[dup_mask] += (rng.random((dup_mask.sum(), 2)) - 0.5) * span * 0.0001
    plt.figure(figsize=(8, 6), dpi=160)
    for lbl in uniq:
        idx = [i for (i, l) in enumerate(labels) if l == lbl]
        plt.scatter(X2[idx, 0], X2[idx, 1], s=28, alpha=0.85, marker=mrk_map[lbl], color=col_map[lbl], label=lbl, edgecolors='none', linewidths=0.6)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

def _maybe_pca(X: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    k = min(int(n_components), int(X.shape[0]), int(X.shape[1]))
    if X.shape[1] <= k:
        return X
    return PCA(n_components=k, svd_solver='randomized', random_state=int(seed)).fit_transform(X)

def _compute_highd_metrics(X: np.ndarray, labels: List[str], *, seed: int, max_eval: int=10000) -> Dict[str, object]:
    out: Dict[str, object] = {}
    n = int(len(labels))
    out['n'] = n
    uniq = sorted(set(labels))
    out['n_classes'] = int(len(uniq))
    out['class_counts'] = {u: int(sum((1 for y in labels if y == u))) for u in uniq}
    if len(uniq) < 2 or n < 4:
        out.update({'silhouette_cosine': float('nan'), 'knn_acc_3fold': float('nan'), 'logreg_acc_3fold': float('nan'), 'logreg_auc_3fold': float('nan'), 'logreg_balanced_acc_3fold': float('nan'), 'logreg_f1_3fold': float('nan')})
        return out
    if n > int(max_eval):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, int(max_eval), replace=False)
        X_eval = X[idx]
        y_eval = [labels[i] for i in idx]
    else:
        X_eval = X
        y_eval = labels
    le = LabelEncoder().fit(y_eval)
    y_num = le.transform(y_eval)
    X_m = _maybe_pca(X_eval, n_components=200, seed=seed)
    try:
        out['silhouette_cosine'] = float(silhouette_score(X_m, y_num, metric='cosine'))
    except Exception:
        out['silhouette_cosine'] = float('nan')
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(seed))
    knn_scores: List[float] = []
    log_acc: List[float] = []
    log_auc: List[float] = []
    log_bal: List[float] = []
    log_f1: List[float] = []
    oof_score = np.full(int(X_m.shape[0]), np.nan, dtype=np.float64)
    oof_fold = np.full(int(X_m.shape[0]), -1, dtype=np.int64)
    n_logreg_fold_failures = 0
    n_knn_fold_failures = 0
    logreg_fold_errors: List[str] = []
    for (_fold_i, (tr, te)) in enumerate(skf.split(X_m, y_num)):
        (X_tr, X_te) = (X_m[tr], X_m[te])
        (y_tr, y_te) = (y_num[tr], y_num[te])
        if KNeighborsClassifier is not None and len(np.unique(y_tr)) >= 2 and (len(X_tr) >= 6):
            k = min(5, len(X_tr))
            try:
                knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
                knn.fit(X_tr, y_tr)
                pred = knn.predict(X_te)
                knn_scores.append(float(accuracy_score(y_te, pred)))
            except Exception:
                n_knn_fold_failures += 1
        try:
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', multi_class='auto', solver='lbfgs')
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            log_acc.append(float(accuracy_score(y_te, pred)))
            log_bal.append(float(balanced_accuracy_score(y_te, pred)))
            log_f1.append(float(f1_score(y_te, pred, average='macro')))
            if len(uniq) == 2:
                prob = clf.predict_proba(X_te)[:, 1]
                oof_score[te] = np.asarray(prob, dtype=np.float64)
                oof_fold[te] = int(_fold_i)
                try:
                    log_auc.append(float(roc_auc_score(y_te, prob)))
                except Exception:
                    n_logreg_fold_failures += 1
        except Exception as _exc:
            n_logreg_fold_failures += 1
            logreg_fold_errors.append(repr(_exc)[:200])
    out['knn_acc_3fold'] = float(np.mean(knn_scores)) if knn_scores else float('nan')
    out['logreg_acc_3fold'] = float(np.mean(log_acc)) if log_acc else float('nan')
    out['logreg_balanced_acc_3fold'] = float(np.mean(log_bal)) if log_bal else float('nan')
    out['logreg_f1_3fold'] = float(np.mean(log_f1)) if log_f1 else float('nan')
    out['logreg_auc_3fold'] = float(np.mean(log_auc)) if log_auc else float('nan')
    out['n_logreg_fold_failures'] = int(n_logreg_fold_failures)
    out['n_knn_fold_failures'] = int(n_knn_fold_failures)
    out['n_logreg_folds_used'] = int(len(log_acc))
    if logreg_fold_errors:
        out['logreg_fold_errors'] = logreg_fold_errors
    out['logreg_auc_oof'] = float('nan')
    out['logreg_auc_oof_ci'] = None
    out['oof_coverage'] = float(np.mean(np.isfinite(oof_score))) if oof_score.size else 0.0
    if len(uniq) == 2 and np.isfinite(oof_score).any():
        try:
            import sys as _sys, os as _os
            _sp = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'scripts')
            if _sp not in _sys.path:
                _sys.path.insert(0, _sp)
            from rq1_stats_core import auc_with_ci as _auc_ci
            _ok = np.isfinite(oof_score)
            _res = _auc_ci(y_num[_ok], oof_score[_ok])
            out['logreg_auc_oof'] = _res.get('auc', float('nan'))
            out['logreg_auc_oof_ci'] = [_res.get('ci_lo'), _res.get('ci_hi')]
            out['logreg_auc_oof_p_vs_chance'] = _res.get('p_vs_chance_mannwhitney')
            out['logreg_auc_oof_at_chance'] = _res.get('at_chance')
            out['logreg_auc_oof_n_pos'] = _res.get('n_pos')
            out['logreg_auc_oof_n_neg'] = _res.get('n_neg')
        except Exception as _exc:
            out['logreg_auc_oof_error'] = repr(_exc)[:300]
    out['_oof'] = {'score': oof_score, 'label': np.asarray(y_num, dtype=np.int64), 'fold': oof_fold}
    out['personality_grad_accuracy'] = out['logreg_acc_3fold']
    return out

def _compute_umap_metrics_2d(X2: np.ndarray, labels: List[str]) -> Dict[str, float]:
    if len(set(labels)) < 2 or len(labels) < 4:
        return {'silhouette_2d_euclidean': float('nan')}
    try:
        sil = float(silhouette_score(X2, labels, metric='euclidean'))
    except Exception:
        sil = float('nan')
    return {'silhouette_2d_euclidean': sil}

def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 3:
        return float('nan')
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float('nan')
    try:
        return float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return float('nan')

def _label_feature_correlation(labels: List[str], features: np.ndarray, feat_names: List[str]) -> Optional[Dict[str, float]]:
    uniq = sorted(set(labels))
    if len(uniq) != 2:
        return None
    y01 = np.asarray([1.0 if l == uniq[1] else 0.0 for l in labels], dtype=np.float32)
    out: Dict[str, float] = {}
    for i in range(min(features.shape[1], len(feat_names))):
        out[feat_names[i]] = _pearson_corr(features[:, i], y01)
    return out
_ALLOWED_LABEL_STEMS = ('labels_sentiment', 'labels_politeness', 'labels_curiosity')

def _filter_label_files(label_paths: List[str]) -> List[str]:
    keep: List[str] = []
    for p in label_paths:
        stem = Path(p).stem.lower()
        if stem in _ALLOWED_LABEL_STEMS:
            keep.append(p)
        elif any((s in stem for s in ('sentiment', 'politeness', 'curiosity'))):
            keep.append(p)
    out: List[str] = []
    seen: set[str] = set()
    for p in keep:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def _load_label_cohort(label_csv: Path, gdf: pd.DataFrame, *, max_users: int, seed: int, demo_mode: bool) -> Tuple[str, pd.DataFrame]:
    label_name = Path(label_csv).stem
    df = pd.read_csv(label_csv).copy()
    if 'target_user_id' not in df.columns or 'label' not in df.columns:
        raise ValueError(f'{label_csv} must contain columns: target_user_id, label')
    df['target_user_id'] = df['target_user_id'].astype('int64')
    df['label'] = df['label'].astype(str)
    cohort = df.merge(gdf, on='target_user_id', how='inner').groupby('target_user_id', sort=False).head(1).reset_index(drop=True)
    if label_name == 'labels_sentiment':
        before = int(len(cohort))
        cohort = cohort[cohort['label'].isin(['rage', 'empath'])].reset_index(drop=True)
        after = int(len(cohort))
        logging.info('Sentiment cohort filtered to rage_vs_empath • %d → %d rows', before, after)
    if label_name == 'labels_curiosity':
        before = int(len(cohort))
        cohort = cohort[cohort['label'].isin(['declarative', 'inquisitive'])].reset_index(drop=True)
        after = int(len(cohort))
        if after != before:
            logging.info('Curiosity cohort filtered to declarative_vs_inquisitive • %d → %d rows', before, after)
    if max_users and max_users > 0 and (len(cohort) > max_users):
        cohort = cohort.sample(n=max_users, random_state=int(seed)).reset_index(drop=True)
    if demo_mode:
        cohort = cohort.head(5).reset_index(drop=True)
    return (label_name, cohort)

def main() -> None:
    global _disable_io, _RESULT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=('lora', 'adapter', 'bias', 'prefix'))
    ap.add_argument('--probe', default='hidden,gvec', help='"hidden|gvec|all" or comma-list')
    ap.add_argument('--results_dir', default='./results')
    ap.add_argument('--test_parquet', required=True)
    ap.add_argument('--global_features', required=True)
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--models_dir', default='models')
    ap.add_argument('--labels_csv', required=True, help='Comma-separated label CSVs; only curiosity+sentiment+politeness are processed')
    ap.add_argument('--max_users', type=int, default=0)
    ap.add_argument('--seed', type=int, default=142)
    ap.add_argument('--demo_mode', action='store_true')
    ap.add_argument('--checklist', default='log_files/validation_models_checklist_10000.txt')
    ap.add_argument('--log_dir', default='log_files')
    ap.add_argument('--min_norm', type=float, default=1e-06)
    ap.add_argument('--reply_sep_token', default=DEFAULT_REPLY_SEP_TOKEN)
    ap.add_argument('--reply_end_token', default=DEFAULT_REPLY_END_TOKEN)
    ap.add_argument('--pad_token', default=DEFAULT_PAD_TOKEN)
    ap.add_argument('--hidden_texts_per_user', type=int, default=2)
    ap.add_argument('--hidden_max_length', type=int, default=256)
    ap.add_argument('--hidden_batch_size', type=int, default=16)
    ap.add_argument('--hidden_fallback_prompt', default='Tell me something interesting.')
    ap.add_argument('--hidden_prompt_mode', default='neutral', choices=('from_test', 'neutral', 'neutral_uid', 'neutral_nonce'), help='How to build per-user prompts for the hidden probe. neutral*: uses a neutral prompt (no held-out user text), which avoids conflating label signal in the prompt with conditioning effects (recommended default). from_test: uses held-out user text from --test_parquet (explicit opt-in; conflates prompt content with persona).')
    ap.add_argument('--hidden_group_suffix', default='_target', help='When hidden_prompt_mode=from_test, keep rows where group_label ends with this suffix (default: "_target")')
    ap.add_argument('--hidden_wrap_reply_tokens', dest='hidden_wrap_reply_tokens', action='store_true', help='When building hidden prompts from test parquet, wrap each selected text with <|reply|> ... <|eoreply|>.')
    ap.add_argument('--no_hidden_wrap_reply_tokens', dest='hidden_wrap_reply_tokens', action='store_false', help='Disable wrapping held-out texts with <|reply|> / <|eoreply|> when building hidden prompts.')
    ap.set_defaults(hidden_wrap_reply_tokens=True)
    ap.add_argument('--neutral_prompts', default='', help='Optional: pipe-separated list of neutral prompts to use for robustness under hidden_prompt_mode=neutral. Example: "Tell me something interesting.|Explain a simple concept.|Write a short paragraph." If empty, uses --hidden_fallback_prompt only.')
    ap.add_argument('--neutral_prompts_per_user', type=int, default=1, help='When --neutral_prompts is provided and hidden_prompt_mode=neutral, number of neutral prompts per user (default: 1).')
    ap.add_argument('--neutral_prompt_entropy', default='low', choices=('low', 'high'), help='Neutral-prompt entropy regime for hidden_prompt_mode=neutral. low: use --neutral_prompts + optional multi-prompt aggregation (default; produces prompt basins/islands). high: draw one prompt per user from a large prompt pool (no aggregation), which tends to recover a single manifold/gradient.')
    ap.add_argument('--neutral_prompt_pool_size', type=int, default=100, help='When --neutral_prompt_entropy=high, size of the synthetic neutral prompt pool if --neutral_prompts is empty (default: 100).')
    ap.add_argument('--run_hyper_controls', action='store_true', help='If set, also run HyperPEFT-LoRA with shuffled-g and zero-g controls under hidden probe.')
    ap.add_argument('--hyper_control_modes', default='normal,shuffle,zero', help='Comma-list of hyper g-modes to run when --run_hyper_controls is set. Allowed: "normal,shuffle,zero".')
    ap.add_argument('--umap_neighbors', type=int, default=15)
    ap.add_argument('--umap_min_dist', type=float, default=0.1)
    ap.add_argument('--hyper_ckpt', default='', help='Optional override path to hypernet checkpoint (.safetensors)')
    ap.add_argument('--vanilla_placeholders', default='', help='Optional override path to vanilla peft_placeholders.safetensors')
    ap.add_argument('--baseline_ckpt', default='', help='Optional override path to baseline checkpoint (dir or file)')
    ap.add_argument('--use_umap', action='store_true', default=True)
    args = ap.parse_args()
    if UMAP is None:
        logging.warning('UMAP unavailable (%r); 2-D projections fall back to a seeded PCA and the meta records pca2_fallback. Probe metrics and saved states are unaffected.', _UMAP_IMPORT_ERR)
    if KNeighborsClassifier is None:
        raise RuntimeError(f'scikit-learn KNN is unavailable: {_KNN_IMPORT_ERR!r}')
    _RESULT_ROOT = Path(args.results_dir) / 'embedding_analysis'
    _RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    _configure_logging(Path(args.log_dir) / 'validate_models_10000.log')
    hf_utils.logging.set_verbosity_error()
    set_seed(int(args.seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    _disable_io = bool(args.demo_mode)
    if _disable_io:
        logging.info('[DEMO] I/O disabled; cohorts clipped to 5 rows.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('device=%s', device)
    gdf = pd.read_parquet(args.global_features).fillna(0.0)
    tok = _build_tokenizer(args.base_ckpt, reply_sep_token=str(args.reply_sep_token), reply_end_token=str(args.reply_end_token), pad_token=str(args.pad_token))
    logging.info('tokenizer_vocab=%d pad_token_id=%s', len(tok), str(tok.pad_token_id))
    probes_raw = ['hidden', 'gvec'] if str(args.probe).lower() in ('all', 'sweep') else [p.strip().lower() for p in str(args.probe).split(',') if p.strip()]
    probes = [p for p in probes_raw if p in ('hidden', 'gvec')]
    unknown = [p for p in probes_raw if p not in ('hidden', 'gvec')]
    if unknown:
        logging.warning('Ignoring unsupported probes: %s', ', '.join(sorted(set(unknown))))
    if not probes:
        raise ValueError('No valid probes selected. Supported: hidden,gvec')
    label_paths_raw = [p.strip() for p in str(args.labels_csv).split(',') if p.strip()]
    label_paths = _filter_label_files(label_paths_raw)
    if not label_paths:
        raise ValueError('No curiosity/sentiment/politeness label CSVs were provided.')
    test_df: Optional[pd.DataFrame] = None
    if 'hidden' in probes and str(args.hidden_prompt_mode) == 'from_test':
        logging.warning('hidden_prompt_mode=from_test uses held-out user text to build prompts. Hidden-probe separability may reflect prompt content rather than conditioning effects. Use hidden_prompt_mode=neutral to avoid conflation.')
        try:
            test_df = pd.read_parquet(args.test_parquet).fillna('')
        except Exception as e:
            raise RuntimeError(f'Failed to read --test_parquet {args.test_parquet!r}: {e!r}')
    hyper_model = _load_hyper_model(args.variant, Path(args.models_dir), args.base_ckpt, tok, device, hyper_ckpt_override=args.hyper_ckpt.strip() or None)
    models_to_run: List[LoadedModel] = [hyper_model]
    try:
        vanilla_model = _load_vanilla_model(args.variant, Path(args.models_dir), args.base_ckpt, tok, device, placeholders_override=args.vanilla_placeholders.strip() or None)
        models_to_run.append(vanilla_model)
    except Exception as e:
        logging.warning('Vanilla model unavailable (%r); running hyper model only.', e)
    try:
        baseline_model = _load_baseline_model(Path(args.models_dir), args.base_ckpt, tok, device, baseline_override=args.baseline_ckpt.strip() or None)
        models_to_run.append(baseline_model)
    except Exception as e:
        logging.warning('Baseline model unavailable (%r); running without baseline.', e)
    if hyper_model.feat_names and len(hyper_model.feat_names) > 0:
        gcols_used = list(hyper_model.feat_names)
        gdim = int(hyper_model.gdim)
    else:
        gcols_used = _default_gcols(gdf)
        gdim = int(len(gcols_used))
    logging.info('Feature subset dims=%d', int(gdim))
    gvec_cache: Dict[str, Tuple[np.ndarray, List[str], List[int], Dict[str, object]]] = {}
    feature_matrix_cache: Dict[str, np.ndarray] = {}
    prompt_cache: Dict[str, List[str]] = {}
    summary: Dict[str, object] = {'variant': args.variant, 'seed': int(args.seed), 'labels': {}}
    for lbl_csv in label_paths:
        (label_name, cohort) = _load_label_cohort(Path(lbl_csv), gdf, max_users=int(args.max_users), seed=int(args.seed), demo_mode=bool(args.demo_mode))
        logging.info('Cohort built for %s • rows=%d', label_name, int(len(cohort)))
        if len(cohort) < 10:
            logging.warning('Cohort too small for reliable metrics (%d rows).', int(len(cohort)))
        labels_list = cohort['label'].astype(str).tolist()
        users_list = cohort['target_user_id'].astype(int).tolist()
        F_raw = _build_feature_matrix(cohort, gcols_used, target_dim=int(gdim))
        F_proc = _apply_feature_stats(F_raw, zscore_global=bool(hyper_model.zscore_global), zscore_center_only=bool(hyper_model.zscore_center_only), mean=hyper_model.feat_mean, std=hyper_model.feat_std)
        feature_matrix_cache[label_name] = F_proc
        if 'hidden' in probes:
            if str(args.hidden_prompt_mode) == 'from_test':
                assert test_df is not None
                prompts_map = _build_user_prompts_from_test_parquet(test_df, users_list, texts_per_user=int(args.hidden_texts_per_user), seed=int(args.seed), fallback_prompt=str(args.hidden_fallback_prompt), group_suffix=str(args.hidden_group_suffix), wrap_reply_tokens=bool(args.hidden_wrap_reply_tokens), reply_sep_token=str(args.reply_sep_token), reply_end_token=str(args.reply_end_token))
                prompt_cache[label_name] = [prompts_map.get(int(uid), str(args.hidden_fallback_prompt)) for uid in users_list]
            elif str(args.hidden_prompt_mode) == 'neutral' and str(args.neutral_prompt_entropy).strip().lower() == 'high':
                pool = _make_high_entropy_prompt_pool(base_prompt=str(args.hidden_fallback_prompt), neutral_prompts=str(args.neutral_prompts), pool_size=int(args.neutral_prompt_pool_size))
                prompts_one: List[str] = []
                for uid in users_list:
                    r = np.random.default_rng(int(args.seed) + int(uid))
                    idx = int(r.integers(0, len(pool)))
                    prompts_one.append(str(pool[idx]))
                prompt_cache[label_name] = prompts_one
                prompt_cache[label_name + '.__prompt_entropy__'] = [str('high')]
                prompt_cache[label_name + '.__prompt_pool_size__'] = [str(int(len(pool)))]
            elif str(args.hidden_prompt_mode) == 'neutral' and str(args.neutral_prompts or '').strip():
                pool = _parse_neutral_prompt_pool(base_prompt=str(args.hidden_fallback_prompt), neutral_prompts=str(args.neutral_prompts))
                (prompts_rep, labels_rep, users_rep, P) = _expand_prompts_multi(users_list, labels_list, prompt_pool=pool, prompts_per_user=int(args.neutral_prompts_per_user), seed=int(args.seed))
                prompt_cache[label_name] = prompts_rep
                prompt_cache[label_name + '.__rep_labels__'] = labels_rep
                prompt_cache[label_name + '.__rep_users__'] = [str(u) for u in users_rep]
                prompt_cache[label_name + '.__rep_P__'] = [str(int(P))]
                prompt_cache[label_name + '.__prompt_entropy__'] = [str('low')]
                prompt_cache[label_name + '.__prompt_pool_size__'] = [str(int(len(pool)))]
            else:
                prompts_map = _build_user_prompts_neutral(users_list, mode=str(args.hidden_prompt_mode), seed=int(args.seed), base_prompt=str(args.hidden_fallback_prompt))
                prompt_cache[label_name] = [prompts_map.get(int(uid), str(args.hidden_fallback_prompt)) for uid in users_list]
                prompt_cache[label_name + '.__prompt_entropy__'] = [str('low')]
                prompt_cache[label_name + '.__prompt_pool_size__'] = [str(int(1))]
        summary['labels'][label_name] = {}
        for probe in probes:
            logging.info('=== label=%s probe=%s ===', label_name, probe)
            summary['labels'][label_name][probe] = {}
            if probe == 'gvec':
                if label_name not in gvec_cache:
                    (Xg, yg, ug, mg) = _extract_gvec_probe(F_proc, labels_list, users_list, min_norm=float(args.min_norm))
                    gvec_cache[label_name] = (Xg, yg, ug, mg)
                (X_base, y_base, u_base, meta_base) = gvec_cache[label_name]
            else:
                X_base = y_base = u_base = meta_base = None
            for m in models_to_run:
                model_tag = m.model_key
                out_root = _RESULT_ROOT / model_tag / label_name / probe
                out_umap = out_root / 'umap'
                out_umap.mkdir(parents=True, exist_ok=True)
                if probe == 'gvec':
                    X = X_base
                    y = y_base
                    u = u_base
                    extract_meta = dict(meta_base)
                    extract_meta['feature_dim'] = int(X.shape[1])
                    extract_meta['zscore_global'] = bool(hyper_model.zscore_global)
                    extract_meta['zscore_center_only'] = bool(hyper_model.zscore_center_only)
                    extract_meta['feature_names'] = list(gcols_used)
                elif probe == 'hidden':
                    prompts = prompt_cache.get(label_name, [str(args.hidden_fallback_prompt)] * len(users_list))
                    rep_labels = prompt_cache.get(label_name + '.__rep_labels__', None)
                    rep_users_s = prompt_cache.get(label_name + '.__rep_users__', None)
                    if rep_labels is not None and rep_users_s is not None:
                        labels_in = list(rep_labels)
                        users_in = [int(x) for x in rep_users_s]
                    else:
                        labels_in = labels_list
                        users_in = users_list
                    if m.kind == 'hyper':
                        if bool(args.run_hyper_controls):
                            modes_raw = [s.strip().lower() for s in str(args.hyper_control_modes).split(',') if s.strip()]
                            modes = [mm for mm in modes_raw if mm in ('normal', 'shuffle', 'zero')]
                            if not modes:
                                modes = ['normal', 'shuffle', 'zero']
                        else:
                            modes = ['normal']
                    else:
                        modes = ['normal']
                    best_mode = None
                    best_pack = None
                    for gmode in modes:
                        if m.kind == 'hyper':
                            GF = _apply_hyper_g_mode(F_proc, users_list, mode=str(gmode), seed=int(args.seed))
                        else:
                            GF = None
                        if m.kind == 'hyper' and GF is not None:
                            if rep_labels is not None and rep_users_s is not None:
                                uid_to_idx = {int(uid): i for (i, uid) in enumerate(users_list)}
                                gf_rep = np.zeros((len(users_in), GF.shape[1]), dtype=np.float32)
                                for (j, uid) in enumerate(users_in):
                                    gf_rep[j, :] = GF[uid_to_idx[int(uid)], :]
                                GF_in = gf_rep
                            else:
                                GF_in = GF
                        else:
                            GF_in = None
                        (X_rep, y_rep, u_rep, extract_meta) = _extract_hidden_probe(m.module, prompts, labels_in, users_in, tok=tok, device=device, batch_size=int(args.hidden_batch_size), max_length=int(args.hidden_max_length), min_norm=float(args.min_norm), global_features=GF_in if m.kind == 'hyper' else None, prefer_penultimate=True, normalize_output=not (rep_labels is not None and rep_users_s is not None))
                        if rep_labels is not None and rep_users_s is not None:
                            (X_user, users_agg) = _aggregate_user_vectors_mean(X_rep, u_rep, min_norm=float(args.min_norm))
                            uid_to_label = {int(uid): str(lab) for (uid, lab) in zip(users_in, labels_in)}
                            y_user = [uid_to_label[int(uid)] for uid in users_agg]
                            X = X_user
                            y = y_user
                            u = users_agg
                            extract_meta['neutral_prompt_pool'] = _parse_neutral_prompt_pool(base_prompt=str(args.hidden_fallback_prompt), neutral_prompts=str(args.neutral_prompts))
                            extract_meta['neutral_prompts_per_user'] = int(args.neutral_prompts_per_user)
                            extract_meta['aggregated_multi_prompts'] = True
                        else:
                            (X, y, u) = (X_rep, y_rep, u_rep)
                            extract_meta['aggregated_multi_prompts'] = False
                        extract_meta.update({'prompt_mode': str(args.hidden_prompt_mode), 'neutral_prompt_entropy': str(args.neutral_prompt_entropy), 'neutral_prompt_pool_size': int(args.neutral_prompt_pool_size), 'texts_per_user': int(args.hidden_texts_per_user), 'fallback_prompt': str(args.hidden_fallback_prompt), 'neutral_prompts': str(args.neutral_prompts), 'neutral_prompts_per_user': int(args.neutral_prompts_per_user), 'hyper_g_mode': str(gmode), 'test_parquet': str(args.test_parquet), 'group_suffix': str(args.hidden_group_suffix), 'wrap_reply_tokens': bool(args.hidden_wrap_reply_tokens), 'reply_sep_token': str(args.reply_sep_token), 'reply_end_token': str(args.reply_end_token)})
                        if len(modes) == 1:
                            best_mode = str(gmode)
                            best_pack = (X, y, u, extract_meta)
                            break
                        mode_suffix = {'normal': '', 'shuffle': '_gshuffle', 'zero': '_gzero'}.get(str(gmode), f'_{str(gmode)}')
                        model_tag_saved = model_tag
                        model_display_saved = m.model_display
                        model_tag = f'{model_tag_saved}{mode_suffix}'
                        if str(gmode) != 'normal':
                            m.model_display = f'{model_display_saved} ({str(gmode)})'
                        out_root = _RESULT_ROOT / model_tag / label_name / probe
                        out_umap = out_root / 'umap'
                        out_umap.mkdir(parents=True, exist_ok=True)
                        hi = _compute_highd_metrics(X, y, seed=int(args.seed))
                        hi.update({'model_key': model_tag, 'model_display': m.model_display, 'variant': m.variant, 'kind': m.kind, 'probe': probe, 'label_set': label_name, 'min_norm': float(args.min_norm), 'tokenizer_vocab': int(len(tok))})
                        try:
                            uid_to_idx = {uid: i for (i, uid) in enumerate(users_list)}
                            sel = np.array([uid_to_idx[uid] for uid in u], dtype=int)
                            F_sel = F_proc[sel]
                            hi['label_feature_corr'] = _label_feature_correlation(y, F_sel, gcols_used)
                        except Exception:
                            hi['label_feature_corr'] = None
                        _pop_oof(hi, out_root, disable_io=_disable_io)
                        if not _disable_io:
                            with (out_root / 'metrics_highd.json').open('w') as fh:
                                json.dump(hi, fh, indent=2)
                        if _is_degenerate(X):
                            X2 = np.zeros((X.shape[0], 2), dtype=np.float32)
                            umap_meta = {'degenerate_input': True}
                        else:
                            X2 = _umap_reduce(X, n_neighbors=int(args.umap_neighbors), min_dist=float(args.umap_min_dist), seed=int(args.seed)).astype(np.float32)
                            umap_meta = {'degenerate_input': False}
                        m2 = _compute_umap_metrics_2d(X2, y)
                        meta = {'model_key': model_tag, 'model_display': m.model_display, 'variant': m.variant, 'kind': m.kind, 'label_set': label_name, 'probe': probe, 'method': _umap_method(), 'seed': int(args.seed), 'umap_neighbors': int(args.umap_neighbors), 'umap_min_dist': float(args.umap_min_dist), 'extract': extract_meta, 'metrics_highd': hi, 'metrics_2d': m2, **umap_meta}
                        if not _disable_io:
                            if os.environ.get('H2_DUMP_HIGHD', '0') == '1':
                                np.save(out_umap / 'X_highd.npy', np.asarray(X, dtype=np.float32))
                            np.save(out_umap / 'X2.npy', X2.astype(np.float32))
                            (out_umap / 'labels.json').write_text(json.dumps(y))
                            (out_umap / 'users.json').write_text(json.dumps(u))
                            with (out_umap / 'meta.json').open('w') as fh:
                                json.dump(meta, fh, indent=2)
                            plot_title = f'{m.model_display} • {label_name} • {probe} • UMAP'
                            out_png = out_umap / f'umap_{model_tag}_{probe}.png'
                            _plot_scatter(X2, y, out_png, plot_title)
                        summary['labels'][label_name][probe][model_tag] = {'metrics_highd': hi, 'metrics_2d': m2, 'degenerate_input': bool(umap_meta.get('degenerate_input', False))}
                        logging.info('%s | %s | %s • sil=%.4f knn=%.3f logacc=%.3f auc=%.3f', model_tag, label_name, probe, float(hi.get('silhouette_cosine', float('nan'))), float(hi.get('knn_acc_3fold', float('nan'))), float(hi.get('logreg_acc_3fold', float('nan'))), float(hi.get('logreg_auc_3fold', float('nan'))))
                        model_tag = model_tag_saved
                        m.model_display = model_display_saved
                    if m.kind == 'hyper' and bool(args.run_hyper_controls):
                        continue
                    if best_pack is None:
                        raise RuntimeError('hidden probe: no extraction results.')
                    (X, y, u, extract_meta) = best_pack
                else:
                    raise ValueError(f'Unknown probe {probe!r}')
                assert isinstance(X, np.ndarray)
                assert isinstance(y, list)
                assert isinstance(u, list)
                hi = _compute_highd_metrics(X, y, seed=int(args.seed))
                hi.update({'model_key': model_tag, 'model_display': m.model_display, 'variant': m.variant, 'kind': m.kind, 'probe': probe, 'label_set': label_name, 'min_norm': float(args.min_norm), 'tokenizer_vocab': int(len(tok))})
                try:
                    uid_to_idx = {uid: i for (i, uid) in enumerate(users_list)}
                    sel = np.array([uid_to_idx[uid] for uid in u], dtype=int)
                    F_sel = F_proc[sel]
                    hi['label_feature_corr'] = _label_feature_correlation(y, F_sel, gcols_used)
                except Exception:
                    hi['label_feature_corr'] = None
                _pop_oof(hi, out_root, disable_io=_disable_io)
                if not _disable_io:
                    with (out_root / 'metrics_highd.json').open('w') as fh:
                        json.dump(hi, fh, indent=2)
                if _is_degenerate(X):
                    X2 = np.zeros((X.shape[0], 2), dtype=np.float32)
                    umap_meta = {'degenerate_input': True}
                else:
                    X2 = _umap_reduce(X, n_neighbors=int(args.umap_neighbors), min_dist=float(args.umap_min_dist), seed=int(args.seed)).astype(np.float32)
                    umap_meta = {'degenerate_input': False}
                m2 = _compute_umap_metrics_2d(X2, y)
                meta = {'model_key': model_tag, 'model_display': m.model_display, 'variant': m.variant, 'kind': m.kind, 'label_set': label_name, 'probe': probe, 'method': _umap_method(), 'seed': int(args.seed), 'umap_neighbors': int(args.umap_neighbors), 'umap_min_dist': float(args.umap_min_dist), 'extract': extract_meta, 'metrics_highd': hi, 'metrics_2d': m2, **umap_meta}
                if not _disable_io:
                    if os.environ.get('H2_DUMP_HIGHD', '0') == '1':
                        np.save(out_umap / 'X_highd.npy', np.asarray(X, dtype=np.float32))
                    np.save(out_umap / 'X2.npy', X2.astype(np.float32))
                    (out_umap / 'labels.json').write_text(json.dumps(y))
                    (out_umap / 'users.json').write_text(json.dumps(u))
                    with (out_umap / 'meta.json').open('w') as fh:
                        json.dump(meta, fh, indent=2)
                    plot_title = f'{m.model_display} • {label_name} • {probe} • UMAP'
                    out_png = out_umap / f'umap_{model_tag}_{probe}.png'
                    _plot_scatter(X2, y, out_png, plot_title)
                summary['labels'][label_name][probe][model_tag] = {'metrics_highd': hi, 'metrics_2d': m2, 'degenerate_input': bool(umap_meta.get('degenerate_input', False))}
                logging.info('%s | %s | %s • sil=%.4f knn=%.3f logacc=%.3f auc=%.3f', model_tag, label_name, probe, float(hi.get('silhouette_cosine', float('nan'))), float(hi.get('knn_acc_3fold', float('nan'))), float(hi.get('logreg_acc_3fold', float('nan'))), float(hi.get('logreg_auc_3fold', float('nan'))))
    if not _disable_io:
        with (_RESULT_ROOT / f'summary_{args.variant}.json').open('w') as fh:
            json.dump(summary, fh, indent=2, allow_nan=False)
    if not _disable_io:
        ck = Path(args.checklist)
        ck.parent.mkdir(parents=True, exist_ok=True)
        done = set(ck.read_text().splitlines()) if ck.exists() else set()
        if args.variant not in done:
            with ck.open('a') as fh:
                fh.write(f'{args.variant}\n')
            logging.info('Checklist updated → %s', ck)
if __name__ == '__main__':
    main()
    logging.shutdown()
