from __future__ import annotations
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
import hashlib
import json
import math
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import single_meteor_score as _meteor
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed, utils as hf_utils
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
sys.path.extend([str(ROOT), str(ROOT / 'data_scripts'), str(ROOT / 'PEFT_scripts')])
from hypernetwork_structure_160M import RedditConversationDataset10000 as BaseDataset10000, HypernetGlobalOnlyDataset10000, FlatHypernetwork, PEFTHypernet, count_trainable_params, ModelLoRAQKV as HyperLoRAWrap, ModelAdapter as HyperAdapterWrap, BiasTuningModel as HyperBiasWrap, ModelPrefixOnly as HyperPrefixWrap
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
DEFAULT_BASE_CKPT = './base_models/pythia_160m_clean'
DEFAULT_TEST_PARQUET = './data/test_full_10000.parquet'
DEFAULT_AUTHOR_PARQUET = './data/author_static_10000.parquet'
DEFAULT_RESULTS_DIR = './results'
DEFAULT_SBERT_PATH = './sentence_transformers/all-MiniLM-L6-v2'
DEFAULT_BERT_MODEL_PATH = './bert_models/roberta-large'
DEFAULT_HYPERLORA_CKPT = './models/pythia125m_flat/lora_k6_final_w32d6hr4/feat_k6_hyp_base_w32_d6_hr4/final_hypernet.safetensors'
DEFAULT_HYPERADAPTER_CKPT = './models/pythia125m_flat/hyper_adapter/final_hypernet.safetensors'
DEFAULT_HYPERBIAS_CKPT = './models/pythia125m_flat/hyper_bias/final_hypernet.safetensors'
DEFAULT_HYPERPREFIX_CKPT = './models/pythia125m_flat/hyper_prefix/final_hypernet.safetensors'
DEFAULT_NONHYPER_LORA_PH = './models/pythia125m_flat_nonhyper/nonhypernet_lora_model_10000/peft_placeholders.safetensors'
DEFAULT_NONHYPER_ADAPTER_PH = './models/pythia125m_flat_nonhyper/nonhypernet_adapter_model_10000/peft_placeholders.safetensors'
DEFAULT_NONHYPER_BIAS_PH = './models/pythia125m_flat_nonhyper/nonhypernet_bias_model_10000/peft_placeholders.safetensors'
DEFAULT_NONHYPER_PREFIX_PH = './models/pythia125m_flat_nonhyper/nonhypernet_prefix_model_10000/peft_placeholders.safetensors'
DEFAULT_NONHYPER_BASE_DIR = './models/pythia125m_flat_nonhyper/nonhypernet_base_model_10000/checkpoints/step_0017000/model.safetensors'
REPLY_SEP_TOKEN = '<|reply|>'
REPLY_END_TOKEN = '<|eoreply|>'
BOS_ID: Optional[int] = None
EOS_ID: Optional[int] = None
_PACK_SANITY_PRINTED = False

def _load_tensor_dict_any(path: Path) -> Dict[str, torch.Tensor]:
    p = Path(path)
    if not p.exists():
        tried = [p]
        parts = p.parts
        if 'pythia125m_flat' in parts:
            i = parts.index('pythia125m_flat')
            alt = Path(*parts[:i], 'pythia125m_flat_nonhyper', *parts[i + 1:])
            tried.append(alt)
            if alt.exists():
                p = alt
            else:
                raise FileNotFoundError('Checkpoint not found. Tried:\n  ' + '\n  '.join((str(x) for x in tried)))
        else:
            raise FileNotFoundError(str(p))
    if p.suffix == '.safetensors':
        try:
            from safetensors.torch import load_file
            return load_file(p.as_posix())
        except Exception:
            pass
    obj = torch.load(p.as_posix(), map_location='cpu')
    if isinstance(obj, dict):
        return obj
    raise TypeError(f'Unsupported checkpoint object type: {type(obj)}')

def _read_safetensors_metadata(path: Path) -> Dict[str, str]:
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework='pt', device='cpu') as f:
            md = f.metadata()
            return dict(md) if md else {}
    except Exception:
        return {}

def _resolve_pretrained_dir(path: Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p.parent
    if not p.is_dir():
        raise FileNotFoundError(p.as_posix())
    if (p / 'model.safetensors').is_file() or (p / 'pytorch_model.bin').is_file():
        return p
    latest = p / 'LATEST'
    if latest.is_file():
        try:
            rel = latest.read_text().strip()
            if rel:
                cand = Path(rel)
                if not cand.is_absolute():
                    cand = p / cand
                if cand.is_file():
                    return cand.parent
                if cand.is_dir():
                    if (cand / 'model.safetensors').is_file() or (cand / 'pytorch_model.bin').is_file():
                        return cand
        except Exception:
            pass
    ckroot = p / 'checkpoints'
    if ckroot.is_dir():
        steps: List[Tuple[int, Path]] = []
        for d in ckroot.iterdir():
            if not d.is_dir():
                continue
            nm = d.name
            if not nm.startswith('step_'):
                continue
            try:
                steps.append((int(nm.split('_')[1]), d))
            except Exception:
                continue
        if steps:
            steps.sort(key=lambda x: x[0], reverse=True)
            for (_, d) in steps:
                if (d / 'model.safetensors').is_file() or (d / 'pytorch_model.bin').is_file():
                    return d
            return steps[0][1]
    raise FileNotFoundError(f'No loadable pretrained directory found at: {p.as_posix()}')

def _read_json_first(existing: List[Path]) -> Dict[str, Any]:
    for p in existing:
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return {}

def _infer_clamp_value(ckpt_file: Path, fallback: float=0.1) -> float:
    try:
        if str(ckpt_file).endswith('.safetensors') and ckpt_file.exists():
            md = _read_safetensors_metadata(Path(ckpt_file))
            if isinstance(md, dict):
                v = md.get('hyper_clamp', None)
                if v is not None:
                    return float(v)
    except Exception:
        pass
    try:
        args_path = ckpt_file.parent / 'args.json'
        if args_path.exists():
            d = json.loads(args_path.read_text())
            if 'clamp_max' in d and 'clamp_min' in d:
                cmin = float(d['clamp_min'])
                cmax = float(d['clamp_max'])
                return 0.5 * (cmin + cmax)
            if 'clamp' in d:
                return float(d['clamp'])
    except Exception:
        pass
    try:
        best_path = ckpt_file.parent / 'best_metrics.json'
        if best_path.exists():
            d = json.loads(best_path.read_text())
            if 'clamp' in d:
                return float(d['clamp'])
    except Exception:
        pass
    return float(fallback)

def _infer_feature_spec_from_ckpt(ckpt_file: Path) -> Dict[str, Any]:
    ck_dir = ckpt_file.parent
    spec: Dict[str, Any] = {'feature_include': None, 'feature_exclude': None, 'include_lists': True, 'zscore': False, 'zscore_center_only': False, 'feature_profile': None, 'source': 'defaults'}

    def _try_load_json(p: Path) -> Dict[str, Any]:
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    def _apply_blob(blob: Dict[str, Any]) -> None:
        if isinstance(blob.get('feature_include'), list):
            spec['feature_include'] = blob['feature_include']
        if isinstance(blob.get('feature_exclude'), list):
            spec['feature_exclude'] = blob['feature_exclude']
        if isinstance(blob.get('include_lists'), bool):
            spec['include_lists'] = bool(blob['include_lists'])
        if isinstance(blob.get('no_lists'), bool):
            spec['include_lists'] = not bool(blob['no_lists'])
        if isinstance(blob.get('zscore'), bool):
            spec['zscore'] = bool(blob['zscore'])
        if isinstance(blob.get('zscore_global'), bool):
            spec['zscore'] = bool(blob['zscore_global'])
        if isinstance(blob.get('zscore_center_only'), bool):
            spec['zscore_center_only'] = bool(blob['zscore_center_only'])
        if isinstance(blob.get('feature_profile'), str):
            spec['feature_profile'] = blob['feature_profile']
    for fn in ['args.json', 'train_args.json', 'hyper_cfg.json']:
        p = ck_dir / fn
        if not p.is_file():
            continue
        blob = _try_load_json(p)
        if blob:
            _apply_blob(blob)
            spec['source'] = fn
            break
    man_path = ck_dir / 'ablation_manifest.json'
    if man_path.is_file():
        man = _try_load_json(man_path)
        if man:
            if isinstance(man.get('feature_include'), list):
                spec['feature_include'] = man['feature_include']
            if isinstance(man.get('feature_exclude'), list):
                spec['feature_exclude'] = man['feature_exclude']
            if 'include_lists' in man:
                spec['include_lists'] = bool(man['include_lists'])
            if 'zscore' in man:
                spec['zscore'] = bool(man['zscore'])
            if 'zscore_center_only' in man:
                spec['zscore_center_only'] = bool(man['zscore_center_only'])
            if 'feature_profile' in man and isinstance(man.get('feature_profile'), str):
                spec['feature_profile'] = man['feature_profile']
            if spec['feature_include'] is None:
                for k in ['selected_features', 'features', 'feature_list', 'include_features']:
                    v = man.get(k, None)
                    if isinstance(v, list) and len(v) > 0:
                        spec['feature_include'] = v
                        break
            spec['source'] = 'ablation_manifest.json'
    if not spec['feature_include']:
        spec['feature_include'] = ['gstat_*']
    if spec['feature_exclude'] is None:
        spec['feature_exclude'] = []
    return spec

def _infer_nonhyper_peft_cfg_from_placeholders(variant: str, placeholder_sd: Dict[str, torch.Tensor], placeholder_file: Optional[Path]=None, *, lora_alpha_fallback: Optional[float]=None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if variant == 'lora':
        rank: Optional[int] = None
        for suf in ['.A_q', '.A_k', '.A_v', '.B_q', '.B_k', '.B_v']:
            for (k, v) in placeholder_sd.items():
                if not (k.endswith(suf) and torch.is_tensor(v) and (v.ndim == 2)):
                    continue
                rank = int(min(int(v.shape[0]), int(v.shape[1])))
                break
            if rank is not None:
                break
        if rank is not None:
            cfg['rank'] = int(rank)
        alpha: Optional[float] = None
        if placeholder_file is not None:
            ck_dir = placeholder_file.parent
            blob = _read_json_first([ck_dir / 'train_args.json', ck_dir / 'args.json', ck_dir / 'config.json', ck_dir / 'peft_args.json'])

            def _find_alpha(d: Dict[str, Any]) -> Optional[float]:
                for key in ['lora_alpha', 'alpha', 'vanilla_lora_alpha']:
                    if isinstance(d.get(key), (int, float)):
                        return float(d[key])
                for subk in ['lora', 'peft', 'model', 'train', 'args', 'hparams']:
                    sub = d.get(subk)
                    if not isinstance(sub, dict):
                        continue
                    for key in ['lora_alpha', 'alpha', 'vanilla_lora_alpha']:
                        if isinstance(sub.get(key), (int, float)):
                            return float(sub[key])
                return None
            alpha = _find_alpha(blob)
        if alpha is not None:
            cfg['alpha'] = float(alpha)
        elif lora_alpha_fallback is not None:
            cfg['alpha'] = float(lora_alpha_fallback)
        return cfg
    if variant == 'adapter':
        bottleneck: Optional[int] = None
        for (k, v) in placeholder_sd.items():
            if not (k.endswith('.down_proj.weight') and torch.is_tensor(v) and (v.ndim == 2)):
                continue
            bottleneck = int(v.shape[0])
            break
        if bottleneck is not None:
            cfg['bottleneck'] = int(bottleneck)
        cfg['use_layer_norm'] = any(('.ln.weight' in k or '.ln.bias' in k for k in placeholder_sd.keys()))
        return cfg
    if variant == 'prefix':
        prefix_len: Optional[int] = None
        for (k, v) in placeholder_sd.items():
            if not (k.endswith('.prefix_key') and torch.is_tensor(v) and (v.ndim == 3)):
                continue
            prefix_len = int(v.shape[1])
            break
        if prefix_len is not None:
            cfg['prefix_length'] = int(prefix_len)
        return cfg
    return cfg

def _align_state_dict_to_model(state: Dict[str, torch.Tensor], model: torch.nn.Module) -> Tuple[Dict[str, torch.Tensor], str, int]:
    model_keys = set(model.state_dict().keys())

    def _overlap(sd: Dict[str, torch.Tensor]) -> int:
        return int(len(set(sd.keys()) & model_keys))
    cands: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    cands.append(('as_is', state))
    cands.append(('strip_backbone', {k[len('backbone.'):] if k.startswith('backbone.') else k: v for (k, v) in state.items()}))
    cands.append(('add_backbone', {k if k.startswith('backbone.') else f'backbone.{k}': v for (k, v) in state.items()}))
    cands.append(('strip_model', {k[len('model.'):] if k.startswith('model.') else k: v for (k, v) in state.items()}))
    cands.append(('add_model', {k if k.startswith('model.') else f'model.{k}': v for (k, v) in state.items()}))
    (best_name, best_sd) = max(cands, key=lambda x: _overlap(x[1]))
    return (best_sd, best_name, _overlap(best_sd))

def _maybe_swap_lora_ab(state: Dict[str, torch.Tensor], model_sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], bool]:
    suf_pairs = [('A_q', 'B_q'), ('A_k', 'B_k'), ('A_v', 'B_v')]
    bases: set[str] = set()
    for k in state.keys():
        for (a, b) in suf_pairs:
            if k.endswith(f'.{a}'):
                bases.add(k[:-len(f'.{a}')])
            elif k.endswith(f'.{b}'):
                bases.add(k[:-len(f'.{b}')])
    if not bases:
        return (state, False)
    out = dict(state)
    swapped = False
    for base in bases:
        for (a, b) in suf_pairs:
            ka = f'{base}.{a}'
            kb = f'{base}.{b}'
            if ka not in state or kb not in state:
                continue
            if ka not in model_sd or kb not in model_sd:
                continue
            va = state[ka]
            vb = state[kb]
            ma = model_sd[ka]
            mb = model_sd[kb]
            if tuple(va.shape) == tuple(ma.shape) and tuple(vb.shape) == tuple(mb.shape):
                continue
            if tuple(vb.shape) == tuple(ma.shape) and tuple(va.shape) == tuple(mb.shape):
                out[ka] = vb
                out[kb] = va
                swapped = True
    return (out, swapped)

def _fmt_metric(x: Any, spec: str='.4f') -> str:
    return 'n/a' if x is None else format(float(x), spec)

def _distinct_n(token_ids: List[List[int]], n: int) -> float:
    if not token_ids:
        return 0.0
    (grams, total) = (set(), 0)
    for seq in token_ids:
        total += max(0, len(seq) - n + 1)
        for i in range(max(0, len(seq) - n + 1)):
            grams.add(tuple(seq[i:i + n]))
    return len(grams) / max(1, total)

def _extract_loss(outputs: Any, labels: torch.Tensor) -> torch.Tensor:
    if getattr(outputs, 'loss', None) is not None:
        return outputs.loss
    if isinstance(outputs, dict) and 'loss' in outputs:
        return outputs['loss']
    logits = getattr(outputs, 'logits', None)
    if logits is None and isinstance(outputs, dict):
        logits = outputs.get('logits', None)
    if logits is None:
        raise ValueError('No loss/logits found in model outputs.')
    lg = logits[..., :-1, :].contiguous()
    y = labels[..., 1:].contiguous()
    m = min(lg.size(1), y.size(1))
    lg = lg[:, :m, :]
    y = y[:, :m]
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1), ignore_index=-100)

def _meteor_safe(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    if not ref_tokens or not hyp_tokens:
        return 0.0
    try:
        return float(_meteor(ref_tokens, hyp_tokens))
    except Exception:
        return 0.0

def _select_eval_subset(df: pd.DataFrame, n_users: int, threads_per_user: int, seed: int, max_examples: int=0) -> pd.DataFrame:
    if n_users <= 0 or threads_per_user <= 0:
        out = df.copy()
    else:
        if 'target_user_id' not in df.columns:
            raise KeyError('test parquet missing required column: target_user_id')
        users = df['target_user_id'].dropna().unique()
        if len(users) == 0:
            return df.head(0).copy()
        rng = np.random.default_rng(seed)
        n_users_eff = min(int(n_users), len(users))
        chosen = rng.choice(users, size=n_users_eff, replace=False)
        tmp = df[df['target_user_id'].isin(chosen)].copy()
        if 'gid' in tmp.columns:
            sort_cols = ['target_user_id', 'gid']
            if 'group_label' in tmp.columns:
                sort_cols.append('group_label')
            tmp = tmp.sort_values(sort_cols, kind='mergesort')
            if 'group_label' in tmp.columns:
                gl = tmp['group_label'].astype(str)
                is_tgt = gl.str.endswith('_target')
                tmp = tmp.assign(_is_tgt=is_tgt)
                grp = tmp.groupby(['target_user_id', 'gid'], sort=False)['_is_tgt'].agg(['any', 'all'])
                valid = grp['any'] & ~grp['all']
                pairs = valid[valid].reset_index()[['target_user_id', 'gid']]
            else:
                pairs = tmp.drop_duplicates(['target_user_id', 'gid'], keep='first')[['target_user_id', 'gid']]
            pairs = pairs.sort_values(['target_user_id', 'gid'], kind='mergesort')
            pairs = pairs.groupby('target_user_id', sort=False).head(int(threads_per_user)).reset_index(drop=True)
            if max_examples and max_examples > 0 and (len(pairs) > int(max_examples)):
                pairs = pairs.head(int(max_examples)).copy()
            out = tmp.merge(pairs, on=['target_user_id', 'gid'], how='inner').copy()
            if '_is_tgt' in out.columns:
                out = out.drop(columns=['_is_tgt'])
        else:
            tmp = tmp.sort_values(['target_user_id'], kind='mergesort')
            out = tmp.groupby('target_user_id', sort=False).head(int(threads_per_user)).copy()
    out = out.reset_index(drop=True)
    if max_examples and max_examples > 0 and ('gid' not in out.columns) and (len(out) > int(max_examples)):
        out = out.head(int(max_examples)).copy()
    return out

def _ensure_special_tokens(tok: AutoTokenizer) -> Tuple[int, int]:
    if REPLY_SEP_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({'additional_special_tokens': [REPLY_SEP_TOKEN]})
    if REPLY_END_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({'additional_special_tokens': [REPLY_END_TOKEN]})
    reply_sep_id = int(tok.convert_tokens_to_ids(REPLY_SEP_TOKEN))
    reply_end_id = int(tok.convert_tokens_to_ids(REPLY_END_TOKEN))
    if tok.pad_token_id is None or tok.pad_token_id == tok.eos_token_id:
        tok.add_special_tokens({'pad_token': '[PAD]'})
        tok.pad_token_id = int(tok.convert_tokens_to_ids('[PAD]'))
    return (reply_sep_id, reply_end_id)

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

def _resize_token_embeddings_like_training(backbone: nn.Module, tok: AutoTokenizer, *, log_prefix: str='', force_special_init: bool=False) -> None:
    try:
        if hasattr(backbone, 'config') and backbone.config is not None:
            backbone.config.use_cache = False
    except Exception:
        pass
    try:
        in_emb = backbone.get_input_embeddings()
        old_n = int(getattr(in_emb, 'weight').size(0))
    except Exception:
        old_n = -1
    new_n = int(len(tok))
    created_rows = False
    if old_n > 0 and new_n != old_n:
        if new_n > old_n:
            backbone.resize_token_embeddings(new_n)
            created_rows = True
            eos_id = int(tok.eos_token_id) if tok.eos_token_id is not None else 0
            new_token_ids = list(range(old_n, new_n))
            _init_token_rows_from_id(backbone, new_token_ids, src_id=eos_id)
            if log_prefix:
                logging.info('[%s] resized token embeddings %d→%d (init %d new rows from EOS=%d)', log_prefix, old_n, new_n, len(new_token_ids), eos_id)
        elif log_prefix:
            logging.info('[%s] tokenizer len %d < model embeddings %d (padded vocab); keeping embeddings as-is', log_prefix, new_n, old_n)
    if created_rows or force_special_init:
        eos_id = int(tok.eos_token_id) if tok.eos_token_id is not None else 0
        init_ids = [int(tok.convert_tokens_to_ids(REPLY_SEP_TOKEN)), int(tok.convert_tokens_to_ids(REPLY_END_TOKEN)), int(tok.pad_token_id) if tok.pad_token_id is not None else -1]
        init_ids = [tid for tid in init_ids if tid >= 0 and tid != eos_id]
        _init_token_rows_from_id(backbone, init_ids, src_id=eos_id)
        if log_prefix:
            logging.info('[%s] EOS-initialized special rows %s from EOS=%d (created_rows=%s forced=%s)', log_prefix, init_ids, eos_id, created_rows, force_special_init)
    elif log_prefix:
        logging.info('[%s] special rows kept from the checkpoint (no resize happened, so they are trained values, not placeholders)', log_prefix)
REQUIRE_SPECIAL_ROWS: bool = False
EXCLUDE_REPLY_END_FROM_SPAN: bool = False
DENY_BOUNDARY_ROW: str = 'none'
SPECIAL_ROW_STATUS: Dict[str, str] = {}

def _special_rows_outcome(variant: str, status: str, detail: str='') -> bool:
    SPECIAL_ROW_STATUS[variant] = status if not detail else f'{status}: {detail}'
    if status == 'loaded':
        logging.info('[special-rows] %s | loaded | %s', variant, detail)
        return True
    msg = f'[special-rows] {variant} | NOT LOADED ({status}) | {detail} | scoring proceeds with UNTRAINED boundary-token rows'
    if REQUIRE_SPECIAL_ROWS:
        raise RuntimeError(msg.replace('scoring proceeds with', 'refusing to score with'))
    logging.warning(msg)
    return False

def _maybe_load_special_rows_from_train_checkpoint(peft_model: nn.Module, hyper_ckpt_file: Path, device: torch.device, variant: str='hyper') -> bool:
    if not (hasattr(peft_model, 'special_row_params') and callable(getattr(peft_model, 'special_row_params'))):
        SPECIAL_ROW_STATUS[variant] = 'n/a (head has no special rows)'
        return False
    candidates = [Path(hyper_ckpt_file).parent / 'train_checkpoints' / 'checkpoint_last.pt', Path(hyper_ckpt_file).parent.parent / 'train_checkpoints' / 'checkpoint_last.pt']
    ckpt_path: Optional[Path] = None
    for p in candidates:
        if p.is_file():
            ckpt_path = p
            break
    if ckpt_path is None:
        return _special_rows_outcome(variant, 'no sibling train_checkpoints/checkpoint_last.pt', ' or '.join((str(c) for c in candidates)))
    payload = None
    load_err: Optional[Exception] = None
    try:
        payload = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
    except TypeError:
        try:
            payload = torch.load(str(ckpt_path), map_location='cpu')
        except Exception as e:
            load_err = e
    except Exception as e:
        load_err = e
    if payload is None:
        try:
            payload = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
            load_err = None
        except TypeError:
            try:
                payload = torch.load(str(ckpt_path), map_location='cpu')
                load_err = None
            except Exception as e:
                load_err = e
        except Exception as e:
            load_err = e
    if payload is None:
        return _special_rows_outcome(variant, 'checkpoint unreadable', f'{ckpt_path} ({load_err})')
    if not isinstance(payload, dict):
        return _special_rows_outcome(variant, 'checkpoint is not a dict', f'{ckpt_path} holds {type(payload).__name__}')
    tensors = payload.get('special_rows', None)
    if not isinstance(tensors, list) or len(tensors) == 0:
        return _special_rows_outcome(variant, "no 'special_rows' key", str(ckpt_path))
    try:
        params = list(peft_model.special_row_params())
    except Exception as exc:
        return _special_rows_outcome(variant, 'special_row_params() raised', str(exc))
    if len(params) != len(tensors):
        return _special_rows_outcome(variant, 'row-count mismatch', f'checkpoint has {len(tensors)}, model exposes {len(params)}, from {ckpt_path}')
    with torch.no_grad():
        for (i, (p, t)) in enumerate(zip(params, tensors)):
            if not torch.is_tensor(t):
                return _special_rows_outcome(variant, 'non-tensor row', f'row {i} is {type(t).__name__}')
            if tuple(t.shape) != tuple(p.shape):
                return _special_rows_outcome(variant, 'row-shape mismatch', f'row {i}: checkpoint {tuple(t.shape)} vs model {tuple(p.shape)}')
            p.copy_(t.to(device=device, dtype=p.dtype))
    return _special_rows_outcome(variant, 'loaded', f'{len(tensors)} rows from {ckpt_path}')

def _pack_reply_layout(batch: Dict[str, torch.Tensor], *, pad_id: int, seq_len: int, reply_sep_id: int, reply_end_id: Optional[int], reply_budget: int, keep_ctx: str) -> Dict[str, torch.Tensor]:
    if keep_ctx not in {'prefix', 'suffix'}:
        raise ValueError(f"keep_ctx must be 'prefix' or 'suffix'; got {keep_ctx!r}")
    ctx_ids = batch['input_ids']
    ctx_attn = batch['attention_mask']
    tgt_lbl = batch['labels']
    (B, _) = ctx_ids.shape
    dev = ctx_ids.device
    out_ids = torch.full((B, seq_len), int(pad_id), dtype=torch.long, device=dev)
    out_attn = torch.zeros((B, seq_len), dtype=torch.long, device=dev)
    out_lbl = torch.full((B, seq_len), -100, dtype=torch.long, device=dev)
    use_eoreply = reply_end_id is not None
    extra = 1 + (1 if use_eoreply else 0)
    rb = max(1, int(reply_budget))
    stripped_ctx_eos = 0
    stripped_tgt_bos = 0
    stripped_tgt_eos = 0
    global _PACK_SANITY_PRINTED, BOS_ID, EOS_ID
    for i in range(B):
        ctx_len_true = int(ctx_attn[i].sum().item())
        ctx_tokens = ctx_ids[i, :ctx_len_true]
        if EOS_ID is not None:
            eos = int(EOS_ID)
            while ctx_tokens.numel() > 0 and int(ctx_tokens[-1].item()) == eos:
                ctx_tokens = ctx_tokens[:-1]
                stripped_ctx_eos += 1
        tgt_tokens = tgt_lbl[i][tgt_lbl[i] != -100]
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
        ctx_len = int(ctx_tokens.numel())
        tgt_len = int(tgt_tokens.numel())
        ctx_cap = min(ctx_len, max(0, seq_len - rb - extra))
        room = max(0, seq_len - ctx_cap - extra)
        tlen = int(min(room, tgt_len))
        if tlen <= 0:
            ctx_cap = min(ctx_cap, max(0, seq_len - (extra + 1)))
            room = max(0, seq_len - ctx_cap - extra)
            tlen = int(min(room, tgt_len))
        if ctx_cap > 0:
            if keep_ctx == 'suffix' and ctx_len > ctx_cap:
                ctx_keep = ctx_tokens[-ctx_cap:]
            else:
                ctx_keep = ctx_tokens[:ctx_cap]
        else:
            ctx_keep = ctx_tokens[:0]
        pos = 0
        if ctx_cap > 0:
            out_ids[i, :ctx_cap] = ctx_keep
            pos += ctx_cap
        out_ids[i, pos] = int(reply_sep_id)
        pos += 1
        if tlen > 0:
            out_ids[i, pos:pos + tlen] = tgt_tokens[:tlen]
            out_lbl[i, pos:pos + tlen] = tgt_tokens[:tlen]
            pos += tlen
        if use_eoreply:
            out_ids[i, pos] = int(reply_end_id)
            out_lbl[i, pos] = -100 if EXCLUDE_REPLY_END_FROM_SPAN else int(reply_end_id)
            pos += 1
        out_attn[i, :pos] = 1
    if not _PACK_SANITY_PRINTED:
        logging.info('[pack] strip_special_tokens: ctx_eos=%d tgt_bos=%d tgt_eos=%d (BOS_ID=%s EOS_ID=%s)', stripped_ctx_eos, stripped_tgt_bos, stripped_tgt_eos, str(BOS_ID), str(EOS_ID))
        _PACK_SANITY_PRINTED = True
    return {'input_ids': out_ids, 'attention_mask': out_attn, 'labels': out_lbl}

def _extract_prompt_from_packed(packed_ids: torch.Tensor, packed_labels: torch.Tensor) -> List[torch.Tensor]:
    prompts: List[torch.Tensor] = []
    for (ids_1, lbl_1) in zip(packed_ids, packed_labels):
        pos = (lbl_1 != -100).nonzero(as_tuple=True)[0]
        cut = int(pos[0].item()) if len(pos) else int(ids_1.size(0))
        prompts.append(ids_1[:cut])
    return prompts

def _pad_2d(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    lens = [int(s.numel()) for s in seqs]
    if not lens:
        raise ValueError('No sequences to pad.')
    T = max(lens)
    B = len(seqs)
    out = torch.full((B, T), pad_id, dtype=seqs[0].dtype, device=seqs[0].device)
    attn = torch.zeros((B, T), dtype=torch.long, device=seqs[0].device)
    for (i, s) in enumerate(seqs):
        out[i, :s.numel()] = s
        attn[i, :s.numel()] = 1
    return (out, attn, lens)

def _infer_hyper_arch_from_state(hyper_sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    lin_idx = []
    for (k, v) in hyper_sd.items():
        if k.startswith('hidden.') and k.endswith('.weight') and (v.ndim == 2):
            try:
                idx = int(k.split('.')[1])
                lin_idx.append(idx)
            except Exception:
                continue
    if not lin_idx:
        raise ValueError('Could not infer hypernet depth/width (no hidden.*.weight keys).')
    lin_idx = sorted(set(lin_idx))
    first_w = hyper_sd[f'hidden.{lin_idx[0]}.weight']
    width = int(first_w.shape[0])
    diffs = [b - a for (a, b) in zip(lin_idx[:-1], lin_idx[1:])]
    stride = min(diffs) if diffs else 2
    dropout = 0.1 if stride >= 3 else 0.0
    depth = int(len(lin_idx) + 1)
    group_splits = None
    if any((k.startswith('head_V_list.') and k.endswith('.weight') for k in hyper_sd)):
        ordered = []
        for i in range(9999):
            kk = f'head_V_list.{i}.weight'
            if kk in hyper_sd and hyper_sd[kk].ndim == 2:
                ordered.append(int(hyper_sd[kk].shape[0]))
            elif ordered:
                break
        if ordered:
            group_splits = ordered
    if group_splits is None and any((k.startswith('head_list.') and k.endswith('.weight') for k in hyper_sd)):
        ordered = []
        for i in range(9999):
            kk = f'head_list.{i}.weight'
            if kk in hyper_sd and hyper_sd[kk].ndim == 2:
                ordered.append(int(hyper_sd[kk].shape[0]))
            elif ordered:
                break
        if ordered:
            group_splits = ordered
    head_rank = 0
    if 'head_U.weight' in hyper_sd and hyper_sd['head_U.weight'].ndim == 2:
        head_rank = int(hyper_sd['head_U.weight'].shape[0])
    if head_rank <= 0 and group_splits is not None:
        for i in range(9999):
            ku = f'head_U_list.{i}.weight'
            if ku in hyper_sd and hyper_sd[ku].ndim == 2:
                head_rank = int(hyper_sd[ku].shape[0])
                break
        if head_rank <= 0:
            for i in range(9999):
                kv = f'head_V_list.{i}.weight'
                if kv in hyper_sd and hyper_sd[kv].ndim == 2:
                    head_rank = int(hyper_sd[kv].shape[1])
                    break
    return {'width': width, 'depth': depth, 'dropout': float(dropout), 'head_rank': int(head_rank), 'group_splits': group_splits}

def _infer_peft_cfg_from_theta(theta_sd: Dict[str, torch.Tensor], variant: str, ckpt_file: Optional[Path]=None) -> Dict[str, Any]:
    if variant == 'hyperlora':
        Aq = None
        Bq = None
        for (k, v) in theta_sd.items():
            if k.endswith('.A_q') and torch.is_tensor(v) and (v.ndim == 2):
                Aq = v
                break
        for (k, v) in theta_sd.items():
            if k.endswith('.B_q') and torch.is_tensor(v) and (v.ndim == 2):
                Bq = v
                break
        ref = Aq if Aq is not None else Bq
        if ref is None:
            raise ValueError('Could not infer LoRA rank (no *.A_q or *.B_q in theta state).')
        rank = int(min(int(ref.shape[0]), int(ref.shape[1])))
        alpha = float(rank)
        if ckpt_file is not None:
            ck_dir = ckpt_file.parent
            blob = _read_json_first([ck_dir / 'hyper_cfg.json', ck_dir / 'ablation_manifest.json', ck_dir / 'args.json', ck_dir / 'train_args.json', ck_dir / 'config.json'])
            if isinstance(blob, dict) and blob:
                found = False
                for key in ['lora_alpha', 'alpha', 'vanilla_lora_alpha']:
                    if isinstance(blob.get(key), (int, float)):
                        alpha = float(blob[key])
                        found = True
                        break
                if not found:
                    for subk in ['lora', 'peft', 'model', 'train', 'args', 'hyper', 'hparams']:
                        sub = blob.get(subk)
                        if not isinstance(sub, dict):
                            continue
                        for key in ['lora_alpha', 'alpha', 'vanilla_lora_alpha']:
                            if isinstance(sub.get(key), (int, float)):
                                alpha = float(sub[key])
                                found = True
                                break
                        if found:
                            break
        return {'rank': rank, 'alpha': alpha}
    if variant == 'hyperadapter':
        dw = None
        for (k, v) in theta_sd.items():
            if '.down_proj.weight' in k and torch.is_tensor(v) and (v.ndim == 2):
                dw = v
                break
        if dw is None:
            raise ValueError('Could not infer adapter bottleneck (no *.down_proj.weight in theta state).')
        bottleneck = int(dw.shape[0])
        use_layer_norm = any(('.ln.weight' in k or '.ln.bias' in k for k in theta_sd.keys()))
        return {'bottleneck': bottleneck, 'dropout': 0.0, 'use_layer_norm': bool(use_layer_norm)}
    if variant == 'hyperprefix':
        pk = None
        for (k, v) in theta_sd.items():
            if k.endswith('.prefix_key') and torch.is_tensor(v) and (v.ndim == 3):
                pk = v
                break
        if pk is None:
            raise ValueError('Could not infer prefix length (no *.prefix_key in theta state).')
        prefix_len = int(pk.shape[1])
        return {'prefix_length': prefix_len}
    if variant == 'hyperbias':
        return {}
    raise ValueError(f'Unknown hyper variant for PEFT inference: {variant}')
HYPER_G_MODE: str = 'normal'
_G_MODE_RNG = torch.Generator().manual_seed(142)

def _pad_global_features(g: torch.Tensor, gdim_expected: int) -> torch.Tensor:
    if g.dim() != 2:
        raise ValueError('global_features must be 2D (B, gdim)')
    (B, K) = g.shape
    if K == gdim_expected:
        out = g
    elif K < gdim_expected:
        pad = torch.zeros((B, gdim_expected - K), dtype=g.dtype, device=g.device)
        out = torch.cat([g, pad], dim=1)
    else:
        out = g[:, :gdim_expected]
    mode = str(HYPER_G_MODE).strip().lower()
    if mode == 'zero':
        out = torch.zeros_like(out)
    elif mode == 'shuffle':
        if out.size(0) > 1:
            perm = torch.randperm(out.size(0), generator=_G_MODE_RNG).to(out.device)
            out = out.index_select(0, perm)
    elif mode != 'normal':
        raise ValueError(f'HYPER_G_MODE must be normal|shuffle|zero, got {mode!r}')
    return out

def _load_hyper_ckpt_split(ckpt_file: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    raw = _load_tensor_dict_any(ckpt_file)
    hyper_sd = {k[len('hyper.'):]: v for (k, v) in raw.items() if k.startswith('hyper.')}
    theta_sd = {k[len('theta.'):]: v for (k, v) in raw.items() if k.startswith('theta.')}
    if not hyper_sd or not theta_sd:
        raise ValueError(f'Expected both hyper.* and theta.* tensors in {ckpt_file}')
    return (hyper_sd, theta_sd)

def _apply_theta_bar_to_placeholders(model: PEFTHypernet) -> None:
    for (pname, p) in model._placeholders.items():
        key = pname
        if key not in model._theta_bar:
            continue
        tb = model._theta_bar[key]
        if tb.shape != p.data.shape:
            continue
        if tb.device != p.data.device:
            tb = tb.to(p.data.device)
        p.data.copy_(tb)

def _load_hyper_weights_into_model(model: PEFTHypernet, hyper_sd: Dict[str, torch.Tensor], theta_sd: Dict[str, torch.Tensor], device: torch.device) -> None:
    model.hypernet.load_state_dict(hyper_sd, strict=True)
    for k in list(model._theta_bar.keys()):
        if k in theta_sd:
            src = theta_sd[k]
            if src.shape != model._theta_bar[k].shape:
                raise ValueError(f'Theta shape mismatch for {k}: ckpt {tuple(src.shape)} vs model {tuple(model._theta_bar[k].shape)}')
            model._theta_bar[k] = src.to(device)
    _apply_theta_bar_to_placeholders(model)

def _build_nonhyper_model(variant: str, *, base_ckpt: str, tok: AutoTokenizer, device: torch.device, lora_rank: int, lora_alpha: float, adapter_bottleneck: int, adapter_dropout: float, prefix_length: int, nonhyper_placeholders: Optional[Path], nonhyper_base_dir: Optional[Path], force_special_init: bool=False, nonhyper_train_token_rows: Optional[List[int]]=None, require_all_keys: bool=False) -> torch.nn.Module:
    placeholder_sd: Optional[Dict[str, torch.Tensor]] = None
    if variant != 'base':
        if nonhyper_placeholders is None:
            raise ValueError('nonhyper_placeholders is required for PEFT non-base variants')
        placeholder_sd = _load_tensor_dict_any(nonhyper_placeholders)
        placeholder_sd = {k[7:] if k.startswith('module.') else k: v for (k, v) in placeholder_sd.items()}
        inferred = _infer_nonhyper_peft_cfg_from_placeholders(variant, placeholder_sd, placeholder_file=nonhyper_placeholders, lora_alpha_fallback=float(lora_alpha))
        if variant == 'lora':
            if isinstance(inferred.get('rank'), int) and inferred['rank'] > 0:
                lora_rank = int(inferred['rank'])
            if isinstance(inferred.get('alpha'), (int, float)):
                lora_alpha = float(inferred['alpha'])
            logging.info('[%s] inferred cfg from placeholders: rank=%s alpha=%s', variant, inferred.get('rank', None), inferred.get('alpha', None))
        if variant == 'adapter':
            if isinstance(inferred.get('bottleneck'), int) and inferred['bottleneck'] > 0:
                adapter_bottleneck = int(inferred['bottleneck'])
            use_layer_norm = bool(inferred.get('use_layer_norm', True))
            logging.info('[%s] inferred cfg from placeholders: bottleneck=%s use_layer_norm=%s', variant, inferred.get('bottleneck', None), bool(inferred.get('use_layer_norm', True)))
        else:
            use_layer_norm = True
        if variant == 'prefix':
            if isinstance(inferred.get('prefix_length'), int) and inferred['prefix_length'] > 0:
                prefix_length = int(inferred['prefix_length'])
            logging.info('[%s] inferred cfg from placeholders: prefix_length=%s', variant, inferred.get('prefix_length', None))
    if variant == 'base':
        if nonhyper_base_dir is None:
            raise ValueError('nonhyper_base_dir is required for base variant')
        resolved = _resolve_pretrained_dir(nonhyper_base_dir)
        logging.info('[base] loading from %s (input=%s)', resolved.as_posix(), nonhyper_base_dir.as_posix())
        model = AutoModelForCausalLM.from_pretrained(resolved.as_posix(), local_files_only=True)
        _resize_token_embeddings_like_training(model, tok, log_prefix='nonhyper')
        model.to(device)
        model.eval()
        return model
    cfg = AutoConfig.from_pretrained(base_ckpt, local_files_only=True)
    cfg.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(base_ckpt, config=cfg, local_files_only=True)
    _resize_token_embeddings_like_training(model, tok, log_prefix='nonhyper', force_special_init=bool(force_special_init))
    if variant == 'lora':
        model = HyperLoRAWrap(model, rank=int(lora_rank), alpha=float(lora_alpha), train_token_rows=list(nonhyper_train_token_rows) if nonhyper_train_token_rows else None)
    elif variant == 'adapter':
        model = HyperAdapterWrap(model, bottleneck_dim=int(adapter_bottleneck), adapter_dropout=float(adapter_dropout), use_layer_norm=bool(use_layer_norm))
    elif variant == 'bias':
        model = HyperBiasWrap(model)
    elif variant == 'prefix':
        model = HyperPrefixWrap(model, prefix_length=int(prefix_length), dropout_prob=0.0)
    else:
        raise ValueError(f'Unknown nonhyper variant: {variant}')
    if placeholder_sd is None:
        raise ValueError('nonhyper_placeholders is required for PEFT non-base variants')
    (aligned, how, overlap) = _align_state_dict_to_model(placeholder_sd, model)
    if overlap <= 0:
        example = next(iter(aligned.keys())) if aligned else '<empty>'
        raise RuntimeError(f'[{variant}] no placeholder keys matched model.state_dict(); example key={example}')
    model_sd = model.state_dict()
    if variant == 'lora':
        (aligned, swapped) = _maybe_swap_lora_ab(aligned, model_sd)
        if swapped:
            logging.info('[%s] swapped LoRA A/B tensors to match model shapes', variant)
    filtered: Dict[str, torch.Tensor] = {}
    bad: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    dropped: List[str] = []
    for (k, v) in aligned.items():
        if k not in model_sd:
            dropped.append(k)
            continue
        if tuple(model_sd[k].shape) != tuple(v.shape):
            bad.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        filtered[k] = v
    if bad:
        preview = '\n'.join((f'  {k}: ckpt{a} vs model{b}' for (k, a, b) in bad[:8]))
        raise RuntimeError(f'[{variant}] placeholder tensor shape mismatches:\n{preview}')
    if dropped:
        logging.warning('[%s] placeholder keys the model has no slot for, so NOT loaded: %s', variant, ', '.join(sorted(dropped)[:8]))
        if require_all_keys:
            raise RuntimeError(f'[{variant}] {len(dropped)} placeholder tensors were not loaded because the model has no parameter of that name: {sorted(dropped)}. The caller required every key, so this run is refused rather than scoring a different arm.')
    (missing, unexpected) = model.load_state_dict(filtered, strict=False)
    if DENY_BOUNDARY_ROW != 'none':
        import torch as _t
        _rows = {'separator': [0], 'end': [1], 'both': [0, 1]}.get(DENY_BOUNDARY_ROW)
        if _rows is None:
            raise ValueError(f'DENY_BOUNDARY_ROW={DENY_BOUNDARY_ROW!r} is not a known choice')
        _hit = []
        with _t.no_grad():
            for _nm in ('special_in_delta', 'special_out_delta'):
                _p = getattr(model, _nm, None)
                if _p is None:
                    continue
                if _p.shape[0] <= max(_rows):
                    raise ValueError(f'{_nm} has {_p.shape[0]} rows, cannot deny {_rows}')
                for _r in _rows:
                    _p[_r].zero_()
                _hit.append(_nm)
        if not _hit:
            raise ValueError(f'DENY_BOUNDARY_ROW={DENY_BOUNDARY_ROW!r} was asked for but this model carries no learned boundary-row tensors. Pass --nonhyper_learn_boundary_rows, or this run would measure nothing.')
        logging.info('[%s] DENIED boundary row(s) %s in %s', variant, _rows, ', '.join(_hit))
    trainable = {n: p for (n, p) in model.named_parameters() if p.requires_grad}
    total_trainable = int(sum((p.numel() for p in trainable.values())))
    loaded_trainable = int(sum((filtered[n].numel() for n in trainable.keys() if n in filtered)))
    cov = float(loaded_trainable) / max(1.0, float(total_trainable))
    logging.info('[%s] placeholder load: keys=%d matched=%d remap=%s trainable_cov=%.4f', variant, len(placeholder_sd), len(filtered), how, cov)
    if total_trainable > 0 and cov < 0.98:
        miss = [n for n in list(trainable.keys()) if n not in filtered]
        raise RuntimeError(f'[{variant}] trainable placeholder coverage too low ({cov:.3f}); example missing={miss[:5]}')
    if unexpected:
        logging.warning('[%s] unexpected keys ignored (e.g. %s)', variant, unexpected[:3])
    if missing:
        pass
    model.to(device)
    model.eval()
    return model

def _build_hyper_model(variant: str, *, base_ckpt: str, tok: AutoTokenizer, device: torch.device, ckpt_file: Path, force_special_init: bool=True) -> Tuple[PEFTHypernet, int]:
    (hyper_sd, theta_sd) = _load_hyper_ckpt_split(ckpt_file)
    arch = _infer_hyper_arch_from_state(hyper_sd)
    hidden_lin_keys = [k for k in hyper_sd.keys() if k.startswith('hidden.') and k.endswith('.weight')]
    first_key = sorted(hidden_lin_keys, key=lambda x: int(x.split('.')[1]))[0]
    gdim_expected = int(hyper_sd[first_key].shape[1])
    peft_cfg = _infer_peft_cfg_from_theta(theta_sd, variant, ckpt_file=ckpt_file)
    cfg = AutoConfig.from_pretrained(base_ckpt, local_files_only=True)
    cfg.use_cache = False
    backbone = AutoModelForCausalLM.from_pretrained(base_ckpt, config=cfg, local_files_only=True)
    _resize_token_embeddings_like_training(backbone, tok, log_prefix='hyper', force_special_init=bool(force_special_init))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    if variant == 'hyperlora':
        reply_sep_id = int(tok.convert_tokens_to_ids(REPLY_SEP_TOKEN))
        reply_end_id = int(tok.convert_tokens_to_ids(REPLY_END_TOKEN))
        peft_model = HyperLoRAWrap(backbone, rank=int(peft_cfg['rank']), alpha=float(peft_cfg['alpha']), train_token_rows=[reply_sep_id, reply_end_id])
    elif variant == 'hyperadapter':
        peft_model = HyperAdapterWrap(backbone, bottleneck_dim=int(peft_cfg['bottleneck']), adapter_dropout=float(peft_cfg.get('dropout', 0.0)), use_layer_norm=bool(peft_cfg.get('use_layer_norm', True)))
    elif variant == 'hyperbias':
        peft_model = HyperBiasWrap(backbone)
    elif variant == 'hyperprefix':
        peft_model = HyperPrefixWrap(backbone, prefix_length=int(peft_cfg['prefix_length']), dropout_prob=0.0)
    else:
        raise ValueError(f'Unknown hyper variant: {variant}')
    out_dim = int(sum((v.numel() for v in theta_sd.values())))
    exclude_ids: List[int] = []
    if hasattr(peft_model, 'special_row_params') and callable(getattr(peft_model, 'special_row_params')):
        try:
            exclude_ids = [id(p) for p in list(peft_model.special_row_params())]
        except Exception:
            exclude_ids = []
    out_dim_expected = int(count_trainable_params(peft_model, exclude_param_ids=exclude_ids))
    if out_dim != out_dim_expected:
        raise ValueError(f'Theta dimensionality mismatch for {variant}: ckpt theta has {out_dim} elements, but wrapper exposes {out_dim_expected} trainable elements.')
    logging.info('[%s] inferred hyper arch: gdim=%d out_dim=%d width=%d depth=%d dropout=%.3f head_rank=%d groups=%s', variant, int(gdim_expected), int(out_dim), int(arch['width']), int(arch['depth']), float(arch['dropout']), int(arch['head_rank']), len(arch['group_splits']) if arch['group_splits'] is not None else 0)
    hyper = FlatHypernetwork(input_dim=gdim_expected, output_dim=out_dim, depth=int(arch['depth']), width=int(arch['width']), dropout=float(arch['dropout']), group_splits=arch['group_splits'], head_rank=int(arch['head_rank']))
    clamp_val = _infer_clamp_value(ckpt_file, fallback=0.2)
    model = PEFTHypernet(peft_model, hyper, clamp=float(clamp_val)).to(device)
    _load_hyper_weights_into_model(model, hyper_sd, theta_sd, device=device)
    _maybe_load_special_rows_from_train_checkpoint(peft_model, ckpt_file, device, variant=variant)
    model.eval()
    return (model, gdim_expected)

def _refuse_best_checkpoint(path, *, flag: str) -> None:
    if path is None:
        return
    s = str(path)
    if not s:
        return
    import os
    if os.path.basename(s).startswith('best_'):
        raise SystemExit('REFUSED: %s points at %s.\n  best_* is a MID-TRAINING snapshot selected on 4.27%% of the users (427 of 10,000).\n  Score final_hypernet.safetensors, which saw all 10,000 users for the full schedule.\n  If the arm has no final_ checkpoint yet, it is not finished. Wait for it.' % (flag, s))

def _describe_model_source(src: Optional[Path]) -> Optional[Dict[str, Any]]:
    import datetime as _dt
    if src is None:
        return None
    p = Path(src)
    d: Dict[str, Any] = {'path': p.as_posix(), 'exists': p.exists()}
    try:
        if p.is_dir():
            d['kind'] = 'dir'
            d['mtime'] = _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds')
        elif p.is_file():
            st = p.stat()
            d['kind'] = 'file'
            d['bytes'] = int(st.st_size)
            d['mtime'] = _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')
            if st.st_size <= 256 * 1024 * 1024:
                hh = hashlib.sha256()
                with p.open('rb') as fh:
                    for chunk in iter(lambda : fh.read(1 << 20), b''):
                        hh.update(chunk)
                d['sha256'] = hh.hexdigest()
        for root in ('models_matched_solo', 'models_matched_ddp', 'models_matched_gh200', 'models_matched_fast', 'models_matched'):
            if '/%s/' % root in p.as_posix():
                d['root'] = root
                break
    except OSError as e:
        d['error'] = str(e)
    return d

def _dump_per_example(out: Dict[str, Any], *, variant: str, dump_dir: Optional[str], model_source: Optional[Path]=None) -> Optional[Dict[str, Any]]:
    pe = out.pop('per_example', None)
    if pe is None:
        return None
    gid = np.asarray(pe['gid'], dtype=np.int64)
    uid = np.asarray(pe['target_user_id'], dtype=np.int64)
    ntk = np.asarray(pe['n_tok'], dtype=np.int64)
    nll = np.asarray(pe['nll_sum'], dtype=np.float64)
    h = hashlib.sha256()
    h.update(gid.tobytes())
    h.update(uid.tobytes())
    h.update(ntk.tobytes())
    fingerprint = h.hexdigest()
    info: Dict[str, Any] = {'n_examples': int(nll.size), 'n_users': int(np.unique(uid[uid >= 0]).size), 'slice_fingerprint_sha256': fingerprint, 'n_nonfinite': int(np.sum(~np.isfinite(nll))), 'n_zero_token_examples': int(np.sum(ntk <= 0)), 'model_source': _describe_model_source(model_source)}
    if dump_dir:
        d = Path(dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / ('per_example_%s.parquet' % variant)
        pd.DataFrame({'row': np.arange(nll.size, dtype=np.int64), 'gid': gid, 'target_user_id': uid, 'nll_sum': nll, 'n_tok': ntk, 'nll_mean': np.where(ntk > 0, nll / np.maximum(ntk, 1), np.nan)}).to_parquet(path, index=False)
        info['path'] = str(path)
        (d / ('per_example_%s.meta.json' % variant)).write_text(json.dumps({'variant': variant, **info}, indent=2))
        logging.info('[%s] per-example NLL dumped: %d examples, %d users, fingerprint %s', variant, info['n_examples'], info['n_users'], fingerprint[:16])
    return info

@torch.inference_mode()
def _evaluate_one_model(*, name: str, model: torch.nn.Module, loader: DataLoader, tok: AutoTokenizer, device: torch.device, reply_sep_id: int, reply_end_id: Optional[int], use_eoreply: bool, keep_ctx: str, max_len: int, reply_budget: int, gen_new_tokens: int, do_sample: bool, temperature: float, top_p: float, topk_samples: int, is_hyper: bool, hyper_gdim_expected: int, skip_generation: bool=False, gen_top_k: Optional[int]=None) -> Dict[str, Any]:
    bleu_sm = SmoothingFunction().method1
    rouge_sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    ce_sum = 0.0
    tok_sum = 0
    ce_sum_modelloss = 0.0
    n_modelloss_batches = 0
    ex_nll_sum: List[float] = []
    ex_n_tok: List[int] = []
    gids: List[int] = []
    uids: List[int] = []
    ctx_texts: List[str] = []
    ref_texts: List[str] = []
    gen_texts: List[str] = []
    gen_ids_all: List[List[int]] = []
    bleu_list: List[float] = []
    meteor_list: List[float] = []
    rouge_list: List[float] = []
    for bt in loader:
        bt = {k: v.to(device) if torch.is_tensor(v) else v for (k, v) in bt.items()}
        packed = _pack_reply_layout(bt, seq_len=max_len, reply_budget=reply_budget, pad_id=int(tok.pad_token_id), reply_sep_id=int(reply_sep_id), reply_end_id=int(reply_end_id) if use_eoreply and reply_end_id is not None else None, keep_ctx=keep_ctx)
        inp = packed['input_ids']
        attn = packed['attention_mask']
        lbl = packed['labels']
        if is_hyper:
            g = bt['global_features']
            g = _pad_global_features(g, hyper_gdim_expected)
            out = model(input_ids=inp, attention_mask=attn, labels=lbl, global_features=g)
        else:
            out = model(input_ids=inp, attention_mask=attn, labels=lbl)
        nt = int((lbl != -100).sum().item())
        _lg = getattr(out, 'logits', None)
        if _lg is None and isinstance(out, dict):
            _lg = out.get('logits', None)
        if _lg is None:
            raise RuntimeError('[%s] model returned no logits, so per-example NLL cannot be computed. The paired battery would lose this arm.' % name)
        _lg = _lg[..., :-1, :].contiguous()
        _y = lbl[..., 1:].contiguous()
        _m = min(_lg.size(1), _y.size(1))
        (_lg, _y) = (_lg[:, :_m, :], _y[:, :_m])
        _tokwise = F.cross_entropy(_lg.reshape(-1, _lg.size(-1)).float(), _y.reshape(-1), ignore_index=-100, reduction='none').view(_y.shape)
        _mask = _y != -100
        _rows = (_tokwise * _mask).sum(dim=1).detach().cpu().numpy()
        _cnts = _mask.sum(dim=1).detach().cpu().numpy()
        ex_nll_sum.extend([float(v) for v in _rows])
        ex_n_tok.extend([int(v) for v in _cnts])
        ce_sum += float(_rows.sum())
        tok_sum += int(_cnts.sum())
        try:
            _ml = float(_extract_loss(out, lbl).item())
            if math.isfinite(_ml):
                ce_sum_modelloss += _ml * nt
                n_modelloss_batches += 1
        except Exception:
            pass
        del _lg, _y, _tokwise, _mask
        prompts = _extract_prompt_from_packed(inp, lbl)
        (prompt_batch, prompt_attn, prompt_lens) = _pad_2d(prompts, int(tok.pad_token_id))
        for p in prompts:
            p2 = p
            if p2.numel() > 0 and int(p2[-1].item()) == int(reply_sep_id):
                p2 = p2[:-1]
            ctx_texts.append(tok.decode(p2, skip_special_tokens=True).strip())
        for i in range(lbl.size(0)):
            ref_ids = lbl[i][lbl[i] != -100]
            ref_texts.append(tok.decode(ref_ids, skip_special_tokens=True).strip())
        if 'gid' in bt:
            gids.extend([int(x) for x in bt['gid'].detach().cpu().tolist()])
        else:
            gids.extend([-1] * inp.size(0))
        if 'target_user_id' in bt:
            uids.extend([int(x) for x in bt['target_user_id'].detach().cpu().tolist()])
        else:
            uids.extend([-1] * inp.size(0))
        if skip_generation:
            for _ in range(inp.size(0)):
                gen_ids_all.append([])
                gen_texts.append('')
        elif is_hyper:
            g = bt['global_features']
            g = _pad_global_features(g, hyper_gdim_expected)
            gen_list = model.generate_ctx(input_ids=prompt_batch, attention_mask=prompt_attn, global_features=g, max_new_tokens=int(gen_new_tokens), do_sample=bool(do_sample), temperature=float(temperature), top_p=float(top_p), eos_token_id=int(reply_end_id) if use_eoreply and reply_end_id is not None else None, **{'top_p_rule': 'hf'} if gen_top_k is not None else {})
            for gen_ids in gen_list:
                keep = torch.tensor(gen_ids, device=prompt_batch.device, dtype=torch.long)
                if use_eoreply and reply_end_id is not None:
                    idx = (keep == int(reply_end_id)).nonzero(as_tuple=True)[0]
                    if len(idx):
                        keep = keep[:int(idx[0].item())]
                if keep.numel() == 0:
                    keep = torch.tensor([tok.encode('.')[0]], device=prompt_batch.device)
                out_ids = keep.detach().cpu().tolist()
                gen_ids_all.append(out_ids)
                gen_txt = tok.decode(keep, skip_special_tokens=True).strip()
                if not gen_txt:
                    gen_txt = '.'
                gen_texts.append(gen_txt)
        else:
            _topk_kw = {} if gen_top_k is None else {'top_k': int(gen_top_k)}
            gen_full = model.generate(prompt_batch, attention_mask=prompt_attn, do_sample=bool(do_sample), temperature=float(temperature), top_p=float(top_p), max_new_tokens=int(gen_new_tokens), eos_token_id=int(reply_end_id) if use_eoreply and reply_end_id is not None else None, pad_token_id=int(tok.pad_token_id), **_topk_kw)
            for i in range(gen_full.size(0)):
                full_seq = gen_full[i]
                cut = int(prompt_lens[i])
                gen_only = full_seq[cut:]
                keep = gen_only[gen_only != int(tok.pad_token_id)]
                if use_eoreply and reply_end_id is not None:
                    idx = (keep == int(reply_end_id)).nonzero(as_tuple=True)[0]
                    if len(idx):
                        keep = keep[:int(idx[0].item())]
                if keep.numel() == 0:
                    keep = torch.tensor([tok.encode('.')[0]], device=full_seq.device)
                gen_ids = keep.detach().cpu().tolist()
                gen_ids_all.append(gen_ids)
                gen_txt = tok.decode(keep, skip_special_tokens=True).strip()
                if not gen_txt:
                    gen_txt = '.'
                gen_texts.append(gen_txt)
    test_ce = ce_sum / max(1, tok_sum)
    test_ppl = float(math.exp(min(float(test_ce), 50.0)))
    (_pe_rebuilt, _pe_absdiff) = (float('nan'), float('nan'))
    if ex_n_tok:
        _tot_tok = float(sum(ex_n_tok))
        _n_bad = int(sum((1 for v in ex_nll_sum if not math.isfinite(float(v)))))
        if _tot_tok > 0:
            _pe_rebuilt = float(sum(ex_nll_sum)) / _tot_tok
            _pe_absdiff = abs(_pe_rebuilt - float(test_ce))
        if _n_bad or int(_tot_tok) != int(tok_sum) or (not _pe_absdiff < 1e-06):
            raise RuntimeError('[%s] per-example NLL does not rebuild the reported CE: rebuilt %.8f vs test_ce %.8f (|diff| %.2e), tokens %d vs %d, non-finite entries %d. Refusing to emit an array that would make every paired test wrong.' % (name, _pe_rebuilt, float(test_ce), _pe_absdiff, int(_tot_tok), int(tok_sum), _n_bad))
    if n_modelloss_batches and tok_sum > 0:
        _ml_ce = ce_sum_modelloss / float(tok_sum)
        logging.info('[%s][estimator] reported CE (token-micro) %.6f | model loss field aggregated the old way %.6f | gap %+.6f nats  (gap near zero = batched path, non-zero = per-sample loop)', name, float(test_ce), _ml_ce, _ml_ce - float(test_ce))
    distinct1 = _distinct_n(gen_ids_all, 1)
    distinct2 = _distinct_n(gen_ids_all, 2)
    for (ref, gen) in () if skip_generation else zip(ref_texts, gen_texts):
        ref_toks = ref.split()
        gen_toks = gen.split()
        bleu = float(sentence_bleu([ref_toks], gen_toks, smoothing_function=bleu_sm)) if ref_toks and gen_toks else 0.0
        meteor = _meteor_safe(ref_toks, gen_toks)
        try:
            rougeL = float(rouge_sc.score(ref, gen)['rougeL'].fmeasure) if ref and gen else 0.0
        except Exception:
            rougeL = 0.0
        bleu_list.append(bleu)
        meteor_list.append(meteor)
        rouge_list.append(rougeL)
    bert_f1_mean = 0.0
    cos_mean = 0.0
    if gen_texts and ref_texts:
        pass
    global _GLOBAL_BERTSCORER, _GLOBAL_COS_MODEL
    if _GLOBAL_BERTSCORER is None or _GLOBAL_COS_MODEL is None:
        raise RuntimeError('Global scorers not initialized.')
    if skip_generation:
        (bert_f1, bert_f1_mean) = ([], 0.0)
        (cos_vals, cos_mean) = ([], 0.0)
    else:
        (_, _, bert_f1) = _GLOBAL_BERTSCORER.score(gen_texts, ref_texts, verbose=False)
        bert_f1 = bert_f1.detach().cpu().numpy().astype('float32').tolist()
        bert_f1_mean = float(np.mean(bert_f1)) if bert_f1 else 0.0
        vec_gen = _GLOBAL_COS_MODEL.encode(gen_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        vec_ref = _GLOBAL_COS_MODEL.encode(ref_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        cos_vals = (vec_gen * vec_ref).sum(dim=1).detach().cpu().numpy().astype('float32').tolist()
        cos_mean = float(np.mean(cos_vals)) if cos_vals else 0.0
    bleu_mean = float(np.mean(bleu_list)) if bleu_list else 0.0
    meteor_mean = float(np.mean(meteor_list)) if meteor_list else 0.0
    rouge_mean = float(np.mean(rouge_list)) if rouge_list else 0.0
    total_params = int(sum((p.numel() for p in model.parameters())))
    trainable_params = int(sum((p.numel() for p in model.parameters() if p.requires_grad)))
    idxs = list(range(len(gen_texts)))
    _rankable = not skip_generation and len(bert_f1) >= len(idxs) and (len(bleu_list) >= len(idxs))
    if _rankable:
        idxs.sort(key=lambda i: (-float(bert_f1[i]), -float(bleu_list[i])))
    else:
        idxs = []
    samples = []
    for i in idxs[:max(0, int(topk_samples))]:
        samples.append({'gid': int(gids[i]) if i < len(gids) else -1, 'target_user_id': int(uids[i]) if i < len(uids) else -1, 'context': ctx_texts[i], 'reference': ref_texts[i], 'generated': gen_texts[i], 'metrics': {'BLEU': float(bleu_list[i]), 'METEOR': float(meteor_list[i]), 'ROUGE_L': float(rouge_list[i]), 'BERTScore': float(bert_f1[i]), 'CosSim': float(cos_vals[i])}})
    summary: Dict[str, Any] = {'n_eval_examples': int(len(gen_texts)), 'test_ce_tf': float(test_ce), 'test_ppl_tf': float(test_ppl), 'bleu_mean': float(bleu_mean), 'meteor_mean': float(meteor_mean), 'rougeL_mean': float(rouge_mean), 'bertscore_mean': float(bert_f1_mean), 'cosim_mean': float(cos_mean), 'distinct1_mean': float(distinct1), 'distinct2_mean': float(distinct2), 'generation_skipped': bool(skip_generation), 'generation_policy': {'do_sample': bool(do_sample), 'temperature': float(temperature), 'top_p': float(top_p), 'max_new_tokens': int(gen_new_tokens), 'top_k': 'none (generate_ctx applies temperature and top-p only)' if is_hyper else '50 (GenerationConfig default, not passed)' if gen_top_k is None else 'none (top_k=0 passed)' if int(gen_top_k) == 0 else int(gen_top_k), 'path': 'PEFTHypernet.generate_ctx' if is_hyper else 'transformers generate()', 'top_p_rule': 'transformers (keeps the crossing token)' if not is_hyper or gen_top_k is not None else 'generate_ctx legacy (drops the crossing token)'}, 'total_params': total_params, 'trainable_params': trainable_params, 'test_nll_example_macro': float(np.mean([s / max(1, n) for (s, n) in zip(ex_nll_sum, ex_n_tok) if n > 0])) if ex_n_tok else float('nan'), 'per_example_rebuilt_ce': float(_pe_rebuilt), 'per_example_identity_absdiff': float(_pe_absdiff)}
    per_example = {'gid': [int(g) for g in gids], 'target_user_id': [int(u) for u in uids], 'nll_sum': [float(v) for v in ex_nll_sum], 'n_tok': [int(v) for v in ex_n_tok]}
    if skip_generation:
        for _k in ('bleu_mean', 'meteor_mean', 'rougeL_mean', 'bertscore_mean', 'cosim_mean', 'distinct1_mean', 'distinct2_mean'):
            summary[_k] = None
        samples = []
    return {'summary': summary, 'samples_topk': samples, 'per_example': per_example}
_GLOBAL_BERTSCORER: Optional[BERTScorer] = None
_GLOBAL_COS_MODEL: Optional[SentenceTransformer] = None

def main() -> None:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--test_parquet', default=DEFAULT_TEST_PARQUET)
    p.add_argument('--author_features_parquet', default=DEFAULT_AUTHOR_PARQUET)
    p.add_argument('--base_ckpt', default=DEFAULT_BASE_CKPT)
    p.add_argument('--results_dir', default=DEFAULT_RESULTS_DIR)
    p.add_argument('--hyper_delta_scale', type=float, default=None, help='Diagnostic. Multiply the generated delta by this at inference. Omit for the shipped behavior, which is 1.0. See the note at the call site: the clamp is one global number and the four heads sit at different scales.')
    p.add_argument('--hyper_theta_source', default='ckpt', choices=['ckpt', 'zero', 'vanilla'], help="Diagnostic. Which base tensors theta_bar holds under the generated delta. 'ckpt' is the shipped behavior and loads theta_bar from the hypernet checkpoint. 'zero' sets it to zeros, the neutral base. 'vanilla' loads it from a trained shared-head peft_placeholders.safetensors given by --hyper_theta_from. Written for the prefix head, whose placeholders are frozen at nn.init.normal_(std=0.02) random noise while the matched shared head trains those same 147,456 numbers for 21,846 updates.")
    p.add_argument('--hyper_theta_from', default='', help='Path to a shared-head peft_placeholders.safetensors. Required by, and only read under, --hyper_theta_source vanilla.')
    p.add_argument('--exclude_reply_end_from_span', action='store_true', help='A1. Feed the reply-end token but do NOT score it. Masks the one scored position whose output row a boundary-row arm learned, for every arm alike. Changes the metric, so it is comparable only against other runs carrying this same flag.')
    p.add_argument('--deny_boundary_row', default='none', choices=['none', 'separator', 'end', 'both'], help='A2. Zero a learned reply-boundary delta row after loading, to ask which of the two carries the margin. Requires an arm that has them and raises rather than measuring nothing.')
    p.add_argument('--nonhyper_learn_boundary_rows', action='store_true', help="Build the shared LoRA wrapper WITH learnable reply-boundary rows, the way train_nonhypernet_10000.py does under its boundary-row flag. REQUIRED to score an arm trained that way: without it the arm's two learned tensors have no slot on the model, are dropped, and the run scores the plain shared head instead. Also makes every placeholder key mandatory.")
    p.add_argument('--nonhyper_special_rows', default='as_trained', choices=['as_trained', 'force'], help="Diagnostic. 'as_trained' is the shipped behavior and leaves the shared heads' <|reply|>, <|eoreply|> and [PAD] rows at the stock padded-vocab values they trained against. 'force' EOS-seeds them, putting the shared heads on the same footing as the hyper arms, whose trainer seeds those rows before step 0. Never applied to the base variant. Use it to measure how much of an H1 margin is boundary-row preparation rather than conditioning.")
    p.add_argument('--hyper_special_rows', default='force', choices=['force', 'skip'], help="Diagnostic. 'force' is the shipped behavior and EOS-seeds the <|reply|>, <|eoreply|> and [PAD] rows on the stock backbone before the hypernet's deltas land, which reproduces what the trainer did for lora, adapter and bias. 'skip' leaves the padded-vocab rows untouched, which is what the shipped prefix arm actually trained against and what the vanilla path already does for every shared head.")
    p.add_argument('--hyper_g_mode', default='normal', choices=['normal', 'shuffle', 'zero'], help='Conditioning ablation for the CE path: shuffle/zero break the (user, g) pairing.')
    p.add_argument('--variants', default='', help='Comma list to score a subset, e.g. "base,lora,hyperlora". Empty = all nine.')
    p.add_argument('--skip_generation', action='store_true', help='Score reply-span CE only: no generation, no BLEU/METEOR/ROUGE/BERTScore. Use for the full-slice statistics pass that feeds the paired battery; run a second small pass without this flag for the surface metrics.')
    p.add_argument('--dump_per_example', default=None, help='Directory to write per_example_<variant>.parquet (reply-span NLL per example). Required for the paired battery in analysis/h1_paired_stats.py; without it only the scalar CE exists.')
    p.add_argument('--require_special_rows', action='store_true', help='Abort if the learned <|reply|>/<|eoreply|> rows cannot be rehydrated from the sibling train_checkpoints/checkpoint_last.pt. They are NOT stored in final_hypernet.safetensors. Without this flag a missing or mismatched sibling checkpoint only warns, and the model scores with untrained boundary rows. Recommended ON for any number that goes into a paper.')
    p.add_argument('--hyperlora_ckpt', default=DEFAULT_HYPERLORA_CKPT)
    p.add_argument('--hyperadapter_ckpt', default=DEFAULT_HYPERADAPTER_CKPT)
    p.add_argument('--hyperbias_ckpt', default=DEFAULT_HYPERBIAS_CKPT)
    p.add_argument('--hyperprefix_ckpt', default=DEFAULT_HYPERPREFIX_CKPT)
    p.add_argument('--nonhyper_lora_placeholders', default=DEFAULT_NONHYPER_LORA_PH)
    p.add_argument('--nonhyper_adapter_placeholders', default=DEFAULT_NONHYPER_ADAPTER_PH)
    p.add_argument('--nonhyper_bias_placeholders', default=DEFAULT_NONHYPER_BIAS_PH)
    p.add_argument('--nonhyper_prefix_placeholders', default=DEFAULT_NONHYPER_PREFIX_PH)
    p.add_argument('--nonhyper_base_dir', default=DEFAULT_NONHYPER_BASE_DIR)
    p.add_argument('--sbert_path', default=DEFAULT_SBERT_PATH)
    p.add_argument('--bert_model_path', default=DEFAULT_BERT_MODEL_PATH)
    p.add_argument('--n_users', type=int, default=100)
    p.add_argument('--threads_per_user', type=int, default=3)
    p.add_argument('--max_examples', type=int, default=0)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--reply_budget', type=int, default=128)
    p.add_argument('--keep_ctx', choices=['prefix', 'suffix'], default='prefix')
    p.add_argument('--use_eoreply', dest='use_eoreply', action='store_true', default=True, help='Use <|eoreply|> as the end-of-reply marker (default: enabled).')
    p.add_argument('--no_eoreply', dest='use_eoreply', action='store_false', help='Disable <|eoreply|> end-of-reply marker (legacy).')
    p.add_argument('--gen_new_tokens', type=int, default=64)
    p.add_argument('--do_sample', action='store_true')
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--gen_top_k', type=int, default=None, help="top_k passed to generate() for the vanilla arms. Unset keeps the old behavior, GenerationConfig's default of 50, which the hypernetwork arms never had. 0 disables it, so every arm samples under temperature and top-p alone (2026-10-12, issue 15). A value above 0 is refused while any hypernetwork arm is scored, since generate_ctx cannot apply it.")
    p.add_argument('--vanilla_lora_rank', type=int, default=32)
    p.add_argument('--vanilla_lora_alpha', type=float, default=32.0)
    p.add_argument('--vanilla_adapter_bottleneck', type=int, default=64)
    p.add_argument('--vanilla_adapter_dropout', type=float, default=0.0)
    p.add_argument('--vanilla_prefix_length', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--bsz', type=int, default=4)
    p.add_argument('--seed', type=int, default=142)
    p.add_argument('--topk_samples', type=int, default=10)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('-f', '--f', help=argparse.SUPPRESS)
    (args, _) = p.parse_known_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    hf_utils.logging.set_verbosity_error()
    global HYPER_G_MODE, REQUIRE_SPECIAL_ROWS
    HYPER_G_MODE = str(args.hyper_g_mode)
    logging.info('[g-mode] hypernet conditioning = %s', HYPER_G_MODE)
    REQUIRE_SPECIAL_ROWS = bool(getattr(args, 'require_special_rows', False))
    global EXCLUDE_REPLY_END_FROM_SPAN, DENY_BOUNDARY_ROW
    EXCLUDE_REPLY_END_FROM_SPAN = bool(getattr(args, 'exclude_reply_end_from_span', False))
    DENY_BOUNDARY_ROW = str(getattr(args, 'deny_boundary_row', 'none'))
    logging.info('[span] reply-end scored = %s | boundary row denied = %s', not EXCLUDE_REPLY_END_FROM_SPAN, DENY_BOUNDARY_ROW)
    logging.info('[special-rows] require_special_rows = %s', REQUIRE_SPECIAL_ROWS)
    set_seed(int(args.seed))
    device = torch.device(args.device)
    df_test = pd.read_parquet(args.test_parquet)
    df_eval = _select_eval_subset(df_test, n_users=int(args.n_users), threads_per_user=int(args.threads_per_user), seed=int(args.seed), max_examples=int(args.max_examples))
    n_gids = int(df_eval['gid'].nunique()) if 'gid' in df_eval.columns else -1
    n_users_eff = int(df_eval['target_user_id'].nunique()) if 'target_user_id' in df_eval.columns else -1
    logging.info('Eval subset rows=%d gids=%d users=%d (from %d)', len(df_eval), n_gids, n_users_eff, len(df_test))
    author_df = pd.read_parquet(args.author_features_parquet)
    tok = AutoTokenizer.from_pretrained(args.base_ckpt, local_files_only=True, trust_remote_code=True)
    tok.padding_side = 'right'
    (reply_sep_id, reply_end_id) = _ensure_special_tokens(tok)
    global BOS_ID, EOS_ID
    BOS_ID = tok.bos_token_id
    EOS_ID = tok.eos_token_id
    global _GLOBAL_BERTSCORER, _GLOBAL_COS_MODEL
    _GLOBAL_BERTSCORER = BERTScorer(model_type=str(args.bert_model_path), num_layers=24, lang='en', rescale_with_baseline=False, batch_size=64, device=device)
    _GLOBAL_COS_MODEL = SentenceTransformer(str(args.sbert_path), device=str(device))
    base_ds = BaseDataset10000(df_eval, tok, max_length=int(args.max_len), add_special_tokens=False, pretokenize=False)
    base_loader = DataLoader(base_ds, batch_size=int(args.bsz), shuffle=False, num_workers=2, pin_memory=True)
    results: Dict[str, Any] = {}
    per_example_info: Dict[str, Any] = {}
    _want = {s.strip() for s in str(getattr(args, 'variants', '') or '').split(',') if s.strip()}
    _all9 = ['lora', 'adapter', 'bias', 'prefix', 'base', 'hyperlora', 'hyperadapter', 'hyperbias', 'hyperprefix']
    if _want:
        _unknown = sorted(_want - set(_all9))
        if _unknown:
            raise SystemExit(f'--variants: unknown {_unknown}; choose from {_all9}')
        logging.info('[variants] scoring subset: %s', sorted(_want))
    else:
        logging.info('[variants] scoring all nine')
    _gtk = getattr(args, 'gen_top_k', None)
    _hyper_wanted = [k for k in ('hyperlora', 'hyperadapter', 'hyperbias', 'hyperprefix') if not _want or k in _want]
    if _gtk is not None and int(_gtk) > 0 and _hyper_wanted and (not getattr(args, 'skip_generation', False)):
        raise SystemExit(f'--gen_top_k {_gtk}: the hypernetwork arms {_hyper_wanted} decode through generate_ctx, which applies no top-k; use 0 so every arm samples under the same rule')
    logging.info('[gen-policy] temperature %s, top_p %s, vanilla top_k %s, hypernetwork arms: no top-k', getattr(args, 'temperature', None), getattr(args, 'top_p', None), '50 (GenerationConfig default)' if _gtk is None else _gtk)
    nonhyper_variants = [v for v in ['lora', 'adapter', 'bias', 'prefix', 'base'] if not _want or v in _want]
    nonhyper_ph = {'lora': Path(args.nonhyper_lora_placeholders), 'adapter': Path(args.nonhyper_adapter_placeholders), 'bias': Path(args.nonhyper_bias_placeholders), 'prefix': Path(args.nonhyper_prefix_placeholders)}
    for v in nonhyper_variants:
        logging.info('=== Evaluating %s ===', v)
        model = _build_nonhyper_model(v, base_ckpt=str(args.base_ckpt), tok=tok, device=device, lora_rank=int(args.vanilla_lora_rank), lora_alpha=float(args.vanilla_lora_alpha), adapter_bottleneck=int(args.vanilla_adapter_bottleneck), adapter_dropout=float(args.vanilla_adapter_dropout), prefix_length=int(args.vanilla_prefix_length), nonhyper_placeholders=nonhyper_ph.get(v, None), nonhyper_base_dir=Path(args.nonhyper_base_dir) if v == 'base' else None, force_special_init=v != 'base' and str(getattr(args, 'nonhyper_special_rows', 'as_trained')) == 'force', nonhyper_train_token_rows=[int(reply_sep_id), int(reply_end_id)] if v == 'lora' and bool(getattr(args, 'nonhyper_learn_boundary_rows', False)) else None, require_all_keys=bool(getattr(args, 'nonhyper_learn_boundary_rows', False)))
        out = _evaluate_one_model(name=v, model=model, loader=base_loader, tok=tok, device=device, reply_sep_id=reply_sep_id, reply_end_id=reply_end_id, use_eoreply=bool(args.use_eoreply), keep_ctx=str(args.keep_ctx), max_len=int(args.max_len), reply_budget=int(args.reply_budget), gen_new_tokens=int(args.gen_new_tokens), do_sample=bool(args.do_sample), temperature=float(args.temperature), top_p=float(args.top_p), topk_samples=int(args.topk_samples), is_hyper=False, hyper_gdim_expected=0, skip_generation=bool(getattr(args, 'skip_generation', False)), gen_top_k=getattr(args, 'gen_top_k', None))
        results[v] = out
        _src = Path(args.nonhyper_base_dir) if v == 'base' else nonhyper_ph.get(v)
        pe_info = _dump_per_example(out, variant=v, dump_dir=getattr(args, 'dump_per_example', None), model_source=_src)
        if pe_info:
            per_example_info[v] = pe_info
        _s = out['summary']
        logging.info('[%s] CE=%s PPL=%s | BLEU=%s MET=%s RL=%s BERT=%s COS=%s D1=%s Prm%%=%s', v, _fmt_metric(_s['test_ce_tf']), _fmt_metric(_s['test_ppl_tf'], '.1f'), _fmt_metric(_s['bleu_mean']), _fmt_metric(_s['meteor_mean']), _fmt_metric(_s['rougeL_mean']), _fmt_metric(_s['bertscore_mean']), _fmt_metric(_s['cosim_mean']), _fmt_metric(_s['distinct1_mean']), _fmt_metric(100.0 * _s['trainable_params'] / max(1, _s['total_params'])))
    hyper_variants = {k: v for (k, v) in {'hyperlora': Path(args.hyperlora_ckpt), 'hyperadapter': Path(args.hyperadapter_ckpt), 'hyperbias': Path(args.hyperbias_ckpt), 'hyperprefix': Path(args.hyperprefix_ckpt)}.items() if not _want or k in _want}
    for (v, ckpt) in hyper_variants.items():
        _refuse_best_checkpoint(ckpt, flag='--%s_ckpt' % v)
        logging.info('=== Evaluating %s === (checkpoint %s)', v, ckpt)
        feat_spec = _infer_feature_spec_from_ckpt(ckpt)
        logging.info('[%s] feature spec: source=%s profile=%s include_lists=%s zscore=%s center_only=%s include=%d exclude=%d', v, feat_spec.get('source', 'unknown'), feat_spec.get('feature_profile', None), feat_spec.get('include_lists', True), feat_spec.get('zscore', False), feat_spec.get('zscore_center_only', False), len(feat_spec.get('feature_include', []) or []), len(feat_spec.get('feature_exclude', []) or []))
        hyper_ds = HypernetGlobalOnlyDataset10000(df_eval, tok, author_features_df=author_df, max_length=int(args.max_len), add_special_tokens=False, pretokenize=False, feature_include=feat_spec['feature_include'], feature_exclude=feat_spec['feature_exclude'], include_lists=bool(feat_spec['include_lists']), zscore=bool(feat_spec['zscore']), zscore_center_only=bool(feat_spec['zscore_center_only']))
        hyper_loader = DataLoader(hyper_ds, batch_size=int(args.bsz), shuffle=False, num_workers=2, pin_memory=True)
        (model_h, gdim_expected) = _build_hyper_model(v, base_ckpt=str(args.base_ckpt), tok=tok, device=device, ckpt_file=ckpt, force_special_init=str(getattr(args, 'hyper_special_rows', 'force')) == 'force')
        if getattr(args, 'hyper_delta_scale', None) is not None:
            s = float(args.hyper_delta_scale)
            model_h.set_delta_scale(s)
            logging.info('[%s][delta-scale] generated delta rescaled to %.4f at inference', v, s)
        _theta_src = str(getattr(args, 'hyper_theta_source', 'ckpt') or 'ckpt')
        if _theta_src != 'ckpt':
            _tb = model_h._theta_bar
            if _theta_src == 'zero':
                for _k in list(_tb.keys()):
                    _tb[_k] = torch.zeros_like(_tb[_k])
                logging.info('[%s][theta-source] theta_bar set to ZERO on %d tensors', v, len(_tb))
            else:
                _src_path = Path(str(getattr(args, 'hyper_theta_from', '') or '').strip())
                if not _src_path.is_file():
                    raise ValueError(f'[{v}] --hyper_theta_source vanilla needs --hyper_theta_from to point at a readable peft_placeholders.safetensors; got {_src_path.as_posix()!r}')
                _van = _load_tensor_dict_any(_src_path)
                _by_suffix: Dict[str, torch.Tensor] = {}
                for (_vk, _vv) in _van.items():
                    _by_suffix.setdefault(_vk.split('backbone.')[-1], _vv)
                (_hit, _miss) = (0, [])
                for _k in list(_tb.keys()):
                    _cand = _by_suffix.get(_k.split('backbone.')[-1], None)
                    if _cand is None or tuple(_cand.shape) != tuple(_tb[_k].shape):
                        _miss.append(_k)
                        continue
                    _tb[_k] = _cand.to(device=_tb[_k].device, dtype=_tb[_k].dtype)
                    _hit += 1
                if _miss:
                    raise ValueError(f'[{v}] theta_bar substitution incomplete: matched {_hit} of {len(_tb)} tensors from {_src_path.as_posix()}; unmatched={_miss[:8]}')
                logging.info('[%s][theta-source] theta_bar replaced by the trained shared head, %d/%d tensors from %s', v, _hit, len(_tb), _src_path.as_posix())
            _apply_theta_bar_to_placeholders(model_h)
            _rms = float(torch.sqrt(sum(((t.float() ** 2).sum() for t in _tb.values())) / max(1, sum((t.numel() for t in _tb.values())))).item())
            logging.info('[%s][theta-source] mode=%s • theta_bar RMS now %.6f', v, _theta_src, _rms)
        if int(hyper_ds.g_dim) != int(gdim_expected):
            raise ValueError(f"[{v}] global feature dim mismatch: dataset g_dim={int(hyper_ds.g_dim)} but hypernet expects {int(gdim_expected)}. ckpt={ckpt.as_posix()} (feature_profile={feat_spec.get('feature_profile', None)} source={feat_spec.get('source', 'unknown')})")
        out = _evaluate_one_model(name=v, model=model_h, loader=hyper_loader, tok=tok, device=device, reply_sep_id=reply_sep_id, reply_end_id=reply_end_id, use_eoreply=bool(args.use_eoreply), keep_ctx=str(args.keep_ctx), max_len=int(args.max_len), reply_budget=int(args.reply_budget), gen_new_tokens=int(args.gen_new_tokens), do_sample=bool(args.do_sample), temperature=float(args.temperature), top_p=float(args.top_p), topk_samples=int(args.topk_samples), is_hyper=True, hyper_gdim_expected=int(gdim_expected), skip_generation=bool(getattr(args, 'skip_generation', False)), gen_top_k=getattr(args, 'gen_top_k', None))
        results[v] = out
        pe_info = _dump_per_example(out, variant=v, dump_dir=getattr(args, 'dump_per_example', None), model_source=ckpt)
        if pe_info:
            per_example_info[v] = pe_info
        _s = out['summary']
        logging.info('[%s] CE=%s PPL=%s | BLEU=%s MET=%s RL=%s BERT=%s COS=%s D1=%s Prm%%=%s', v, _fmt_metric(_s['test_ce_tf']), _fmt_metric(_s['test_ppl_tf'], '.1f'), _fmt_metric(_s['bleu_mean']), _fmt_metric(_s['meteor_mean']), _fmt_metric(_s['rougeL_mean']), _fmt_metric(_s['bertscore_mean']), _fmt_metric(_s['cosim_mean']), _fmt_metric(_s['distinct1_mean']), _fmt_metric(100.0 * _s['trainable_params'] / max(1, _s['total_params'])))
    out_path = Path(args.results_dir) / 'eval_models_results_10000.json'
    payload = {k: v for (k, v) in results.items()}
    payload['_run_provenance'] = {'per_example_dump': dict(per_example_info), 'hyper_g_mode': HYPER_G_MODE, 'require_special_rows': bool(REQUIRE_SPECIAL_ROWS), 'special_row_status': dict(SPECIAL_ROW_STATUS), 'seed': int(getattr(args, 'seed', 142))}
    with out_path.open('w') as f:
        json.dump(payload, f, indent=2)
    for (_v, _s) in SPECIAL_ROW_STATUS.items():
        logging.info('[special-rows] %-14s %s', _v, _s)
    logging.info('Wrote combined evaluation JSON → %s', out_path.as_posix())
if __name__ == '__main__':
    main()
