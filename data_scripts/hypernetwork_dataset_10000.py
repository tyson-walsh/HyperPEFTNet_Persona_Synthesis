from __future__ import annotations
import argparse
import importlib.util
import json
import logging
import math
import numbers
import os
import random
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
_DATA_ROOT = Path('./data')
DEFAULT_SPLIT_PARQUET = _DATA_ROOT / 'train_data_10000.parquet'
DEFAULT_AUTHOR_PARQUET = _DATA_ROOT / 'author_static_10000.parquet'
DEFAULT_GLOBAL_PARQUET = _DATA_ROOT / 'global_features_10000.parquet'
DEFAULT_COLS_JSON = _DATA_ROOT / 'global_features_10000_cols.json'
RNG_SEED = 142
LEAKY_GLOBAL_SENTIMENT: set[str] = {'gstat_user_sent_mean', 'gstat_user_sent_var', 'gstat_gap_sentiment'}
LEAKY_GLOBAL_BEAST_SENT: set[str] = {'gstat_beast_sent_oof', 'gstat_beast_sent_resid', 'gstat_beast_sent_qrank', 'gstat_beast_comp_importance', 'gstat_beast_importance_conc', 'gstat_beast_leaf64'}
ENV_G_INCLUDE = 'HN_G_INCLUDE'
ENV_G_EXCLUDE = 'HN_G_EXCLUDE'
ENV_EXTRA_EXCLUDES = 'HN_EXCLUDE_COLS'
ENV_LEAKAGE_SAFE = 'HN_LEAKAGE_SAFE'
ENV_RETURN_MASKS = 'HN_RETURN_MASKS'
ENV_SANITY_SAMPLES = 'HN_SANITY_SAMPLES'
ENV_SANITY_ZERO_TOL = 'HN_SANITY_ZERO_TOL'
ENV_SANITY_STRICT = 'HN_SANITY_STRICT'

def _load_base_dataset() -> type:
    search_paths: List[Path] = []
    try:
        search_paths.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    search_paths.append(Path.cwd())
    search_paths.append(Path('./data_scripts'))
    for directory in search_paths:
        cand = directory / 'dataset_10000.py'
        if cand.exists():
            spec = importlib.util.spec_from_file_location('dataset_10000', str(cand))
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            sys.modules['dataset_10000'] = mod
            return mod.RedditConversationDataset10000
    raise ModuleNotFoundError('dataset_10000.py not found in expected search paths.')
RedditConversationDataset10000 = _load_base_dataset()

def _truthy_env(name: str, default: bool=False) -> bool:
    raw = os.getenv(name, '')
    if not raw:
        return default
    return raw.strip().lower() in {'1', 'true', 't', 'yes', 'y', 'on'}

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, '')
    if not raw:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, '')
    if not raw:
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default

def _parse_csv_set(raw: Optional[str]) -> set[str]:
    if not raw:
        return set()
    return {c.strip() for c in raw.split(',') if c.strip()}

def _parse_env_set(name: str) -> set[str]:
    return _parse_csv_set(os.getenv(name, ''))

def _ensure_int_series(s: pd.Series) -> pd.Series:
    try:
        return s.astype('int64')
    except Exception:
        return pd.to_numeric(s, errors='coerce').fillna(0).astype('int64')

def _is_scalar(x: Any) -> bool:
    return isinstance(x, numbers.Number) or isinstance(x, np.generic)

def _is_missing_obj(v: Any) -> bool:
    if v is None:
        return True
    try:
        m = pd.isna(v)
    except Exception:
        return False
    if isinstance(m, (bool, np.bool_)):
        return bool(m)
    return False

def _collect_gstat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith('gstat_')]

def _read_cols_json(cols_json: Optional[Path]) -> Optional[List[str]]:
    if cols_json is None or not cols_json.exists():
        return None
    try:
        obj = json.loads(cols_json.read_text())
        if isinstance(obj, list) and all((isinstance(x, str) for x in obj)):
            return list(obj)
        return None
    except Exception:
        return None

def _ordered_gstat_cols(*, df: pd.DataFrame, cols_json: Optional[Path]) -> List[str]:
    gstat = _collect_gstat_cols(df)
    if not gstat:
        return []
    order = _read_cols_json(cols_json)
    if order:
        base = [c for c in order if c.startswith('gstat_') and c in gstat]
        tail = sorted([c for c in gstat if c not in base])
        return base + tail
    return sorted(gstat)

def _select_gstat_cols(*, ordered_cols: List[str], include: Optional[Sequence[str]], exclude: Optional[Sequence[str]], leakage_safe: bool, extra_excludes: Optional[Sequence[str]]) -> List[str]:
    cols = list(ordered_cols)
    if include:
        inc = set(include)
        cols = [c for c in cols if c in inc]
    if exclude:
        exc = set(exclude)
        cols = [c for c in cols if c not in exc]
    env_inc = _parse_env_set(ENV_G_INCLUDE)
    env_exc = _parse_env_set(ENV_G_EXCLUDE)
    env_extra = _parse_env_set(ENV_EXTRA_EXCLUDES)
    if env_inc:
        cols = [c for c in cols if c in env_inc]
    if env_exc:
        cols = [c for c in cols if c not in env_exc]
    if env_extra:
        cols = [c for c in cols if c not in env_extra]
    if extra_excludes:
        ex2 = set(extra_excludes)
        cols = [c for c in cols if c not in ex2]
    if leakage_safe:
        block = LEAKY_GLOBAL_SENTIMENT | LEAKY_GLOBAL_BEAST_SENT
        cols = [c for c in cols if c not in block]
    return cols

@dataclass(frozen=True)
class _ColSpec:
    name: str
    kind: str
    length: int

def _infer_colspecs(gdf: pd.DataFrame, cols: List[str]) -> List[_ColSpec]:
    specs: List[_ColSpec] = []
    for c in cols:
        ser = gdf[c]
        kind = 'scalar'
        length = 1
        if ser.dtype == object:
            v0 = None
            for v in ser.head(64).tolist():
                if not _is_missing_obj(v):
                    v0 = v
                    break
            if isinstance(v0, np.ndarray):
                if v0.ndim == 0:
                    kind = 'scalar'
                    length = 1
                else:
                    kind = 'vector'
                    length = int(np.asarray(v0).size)
            elif isinstance(v0, (list, tuple)):
                kind = 'vector'
                length = int(len(v0))
            elif _is_scalar(v0):
                kind = 'scalar'
                length = 1
        specs.append(_ColSpec(name=c, kind=kind, length=max(int(length), 1)))
    return specs

def _flatten_row(values: Tuple[Any, ...], specs: List[_ColSpec]) -> np.ndarray:
    out: List[float] = []
    for (v, spec) in zip(values, specs):
        if spec.kind == 'vector':
            if _is_missing_obj(v):
                out.extend([0.0] * spec.length)
            else:
                arr = np.asarray(v, dtype=np.float32).ravel()
                if arr.size < spec.length:
                    out.extend(arr.tolist())
                    out.extend([0.0] * (spec.length - int(arr.size)))
                elif arr.size > spec.length:
                    out.extend(arr[:spec.length].tolist())
                else:
                    out.extend(arr.tolist())
        elif _is_missing_obj(v):
            out.append(0.0)
        else:
            try:
                out.append(float(v))
            except Exception:
                out.append(0.0)
    return np.asarray(out, dtype=np.float32)

def _load_parquet_head(path: Path, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0:
        return pd.read_parquet(path)
    pf = pq.ParquetFile(path)
    batches = []
    got = 0
    batch_size = min(200000, max_rows)
    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        batches.append(df)
        got += len(df)
        if got >= max_rows:
            break
    out = pd.concat(batches, ignore_index=True)
    if len(out) > max_rows:
        out = out.iloc[:max_rows].reset_index(drop=True)
    return out

def _collapse_global_to_author_table(global_parquet: Path, *, cols_json: Optional[Path]) -> pd.DataFrame:
    if not global_parquet.exists():
        raise FileNotFoundError(str(global_parquet))
    pf = pq.ParquetFile(global_parquet)
    schema_cols = list(pf.schema_arrow.names)
    gstat_cols = [c for c in schema_cols if c.startswith('gstat_')]
    cols = ['target_user_id'] + gstat_cols
    dset = ds.dataset(str(global_parquet), format='parquet')
    scanner = dset.scanner(columns=cols, batch_size=250000)
    last_rows: Dict[int, Dict[str, Any]] = {}
    for batch in scanner.to_batches():
        pdf = batch.to_pandas()
        if pdf.empty:
            continue
        pdf['target_user_id'] = pd.to_numeric(pdf['target_user_id'], errors='coerce').fillna(-1).astype('int64')
        pdf = pdf[pdf['target_user_id'] > 0]
        if pdf.empty:
            continue
        pdf = pdf.drop_duplicates('target_user_id', keep='last')
        for (_, row) in pdf.iterrows():
            uid = int(row['target_user_id'])
            last_rows[uid] = row.to_dict()
    out = pd.DataFrame(last_rows.values())
    out = out.sort_values('target_user_id').reset_index(drop=True)
    ordered = _ordered_gstat_cols(df=out, cols_json=cols_json)
    keep = ['target_user_id'] + ordered
    keep = [c for c in keep if c in out.columns]
    out = out[keep]
    return out

def _choose_author_table(split_df: pd.DataFrame, *, author_parquet: Optional[Path], global_parquet: Optional[Path], cols_json: Optional[Path]) -> pd.DataFrame:
    if author_parquet is not None and author_parquet.exists():
        gdf = pd.read_parquet(author_parquet)
        if 'target_user_id' not in gdf.columns:
            raise KeyError(f"{author_parquet} missing 'target_user_id'")
        return gdf
    split_g = _collect_gstat_cols(split_df)
    if split_g:
        tmp = split_df[['target_user_id'] + split_g].copy()
        tmp['target_user_id'] = pd.to_numeric(tmp['target_user_id'], errors='coerce').fillna(-1).astype('int64')
        tmp = tmp[tmp['target_user_id'] > 0]
        return tmp.drop_duplicates('target_user_id', keep='last').reset_index(drop=True)
    if global_parquet is not None and global_parquet.exists():
        return _collapse_global_to_author_table(global_parquet, cols_json=cols_json)
    raise RuntimeError('Could not resolve an author feature table.\nProvide --author_parquet, or a split parquet that already contains gstat_*, or --global_parquet.')

def _resolve_hf_token(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    for k in ('HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
        v = os.getenv(k, '').strip()
        if v:
            return v
    return None

def _load_hf_tokenizer(repo_or_path: str, hf_token: Optional[str]=None) -> PreTrainedTokenizer:
    tok = _resolve_hf_token(hf_token)
    try:
        return AutoTokenizer.from_pretrained(repo_or_path, use_fast=True, token=tok)
    except TypeError:
        return AutoTokenizer.from_pretrained(repo_or_path, use_fast=True, use_auth_token=tok)

def _dataset_mode() -> str:
    m = os.environ.get('HN_DATASET_MODE', 'seq2seq').strip().lower()
    return m if m in ('seq2seq', 'causal') else 'seq2seq'

class HypernetGlobalOnlyDataset10000(Dataset):

    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, *, author_parquet: Optional[Path]=None, global_parquet: Optional[Path]=None, cols_json: Optional[Path]=None, max_length: int=512, add_special_tokens: bool=True, pretokenize: bool=False, leakage_safe: Optional[bool]=None, include_gstats: Optional[Sequence[str]]=None, exclude_gstats: Optional[Sequence[str]]=None, disable_gstats: Optional[Sequence[str]]=None, return_masks: Optional[bool]=None, sanity_samples: Optional[int]=None, sanity_zero_tol: Optional[float]=None, sanity_strict: Optional[bool]=None):
        super().__init__()
        if 'gid' not in df.columns or 'target_user_id' not in df.columns:
            raise KeyError("Input dataframe must contain columns: 'gid' and 'target_user_id'.")
        self.df = df
        self._text_ds = RedditConversationDataset10000(dataframe=df, tokenizer=tokenizer, max_length=max_length, add_special_tokens=add_special_tokens, pretokenize=pretokenize, mode=_dataset_mode())
        self._leakage_safe = bool(_truthy_env(ENV_LEAKAGE_SAFE, default=True) if leakage_safe is None else leakage_safe)
        self._return_masks = bool(_truthy_env(ENV_RETURN_MASKS, default=False) if return_masks is None else return_masks)
        self._sanity_samples = int(_env_int(ENV_SANITY_SAMPLES, 1024) if sanity_samples is None else sanity_samples)
        self._sanity_zero_tol = float(_env_float(ENV_SANITY_ZERO_TOL, 0.02) if sanity_zero_tol is None else sanity_zero_tol)
        self._sanity_strict = bool(_truthy_env(ENV_SANITY_STRICT, default=True) if sanity_strict is None else sanity_strict)
        gdf = _choose_author_table(df, author_parquet=author_parquet, global_parquet=global_parquet, cols_json=cols_json).copy()
        if 'target_user_id' not in gdf.columns:
            raise KeyError("Resolved author feature table missing 'target_user_id'.")
        gdf['target_user_id'] = pd.to_numeric(gdf['target_user_id'], errors='coerce').fillna(-1).astype('int64')
        gdf = gdf[gdf['target_user_id'] > 0].copy()
        if gdf.empty:
            raise RuntimeError('Author feature table is empty after filtering invalid target_user_id.')
        if gdf.duplicated('target_user_id').any():
            gdf = gdf.drop_duplicates('target_user_id', keep='last').reset_index(drop=True)
        ordered = _ordered_gstat_cols(df=gdf, cols_json=cols_json)
        selected = _select_gstat_cols(ordered_cols=ordered, include=include_gstats, exclude=exclude_gstats, leakage_safe=self._leakage_safe, extra_excludes=disable_gstats)
        if not selected:
            raise RuntimeError('No gstat_* columns selected after include/exclude/leakage filters.')
        self._g_cols = selected
        gdf = gdf[['target_user_id'] + self._g_cols].reset_index(drop=True)
        specs = _infer_colspecs(gdf, self._g_cols)
        self._specs = specs
        self._g_dim = int(sum((s.length for s in specs)))
        if self._g_dim <= 0:
            raise RuntimeError('Global feature dimension is zero after spec inference.')
        uids = gdf['target_user_id'].to_numpy(dtype=np.int64)
        self._uid_to_row: Dict[int, int] = {int(uid): int(i) for (i, uid) in enumerate(uids.tolist())}
        mat = np.zeros((len(gdf), self._g_dim), dtype=np.float32)
        values_df = gdf[self._g_cols]
        for (i, row_vals) in enumerate(values_df.itertuples(index=False, name=None)):
            mat[i, :] = _flatten_row(row_vals, specs)
        self._g_tensor = torch.from_numpy(mat)
        self._zero_vec = torch.zeros((self._g_dim,), dtype=torch.float32)
        if self._return_masks:
            has = (mat != 0.0).any(axis=1).astype(np.int8)
            self._mask_tensor = torch.from_numpy(has).view(-1, 1)
        else:
            self._mask_tensor = None
        self._run_quick_sanity()

    @property
    def global_columns(self) -> List[str]:
        return list(self._g_cols)

    @property
    def g_dim(self) -> int:
        return int(self._g_dim)

    def __len__(self) -> int:
        return len(self._text_ds)

    def __getitem__(self, idx: int) -> dict:
        base = self._text_ds[idx]
        uid = int(base['target_user_id'])
        ridx = self._uid_to_row.get(uid, None)
        if ridx is None:
            g = self._zero_vec
            m = torch.tensor([0], dtype=torch.int8) if self._return_masks else None
        else:
            g = self._g_tensor[ridx]
            if self._return_masks and self._mask_tensor is not None:
                m = self._mask_tensor[ridx]
            else:
                m = None
        out = dict(base)
        out['global_features'] = g
        if self._return_masks:
            out['global_mask'] = m
        return out

    def active_feature_manifest(self) -> Dict[str, Any]:
        return {'global_columns': list(self._g_cols), 'g_dim': int(self._g_dim)}

    def save_feature_manifest(self, path: str | os.PathLike) -> None:
        Path(path).write_text(json.dumps(self.active_feature_manifest(), indent=2))

    def _run_quick_sanity(self) -> None:
        n = len(self)
        if n == 0:
            return
        k = min(max(1, int(self._sanity_samples)), n)
        tol = float(self._sanity_zero_tol)
        rng = random.Random(RNG_SEED)
        idxs = rng.sample(range(n), k) if n > k else list(range(n))
        zero = 0
        for j in idxs:
            s = self._text_ds[j]
            uid = int(s['target_user_id'])
            ridx = self._uid_to_row.get(uid, None)
            if ridx is None:
                zero += 1
                continue
            vec = self._g_tensor[ridx]
            if not torch.any(vec != 0.0):
                zero += 1
        rate = zero / max(1, len(idxs))
        if rate > tol:
            msg = f'[sanity] all-zero global_features rate={rate:.3%} exceeds tol={tol:.1%}'
            if self._sanity_strict:
                raise RuntimeError(msg)
            logging.warning(msg)

def _smoke_report(df: pd.DataFrame, ds_obj: HypernetGlobalOnlyDataset10000) -> None:
    counts = df['target_user_id'].value_counts()
    report = f"\n    Hypernet global-only dataset smoke test\n    --------------------------------------\n    samples loaded         : {len(df):,}\n    dataset length         : {len(ds_obj):,}\n    unique users (sample)  : {df['target_user_id'].nunique():,}\n    samples/user (sample)  : min={(int(counts.min()) if not counts.empty else 0)},\n                             median={(int(median(counts.tolist())) if not counts.empty else 0)},\n                             max={(int(counts.max()) if not counts.empty else 0)}\n\n    global vector dim (G)  : {ds_obj.g_dim}\n    gstat cols selected    : {len(ds_obj.global_columns)}\n    first 12 gstat cols    : {ds_obj.global_columns[:12]}\n    "
    print(textwrap.dedent(report).strip())
    for i in [0, min(1, len(ds_obj) - 1), min(10, len(ds_obj) - 1)]:
        if i < 0:
            continue
        s = ds_obj[i]
        g = s['global_features']
        print(f"[sample idx={i}] uid={int(s['target_user_id'])} gid={int(s['gid'])} g.shape={tuple(g.shape)} g.nonzero={int(torch.count_nonzero(g))}")

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    ap = argparse.ArgumentParser(description='HyperNet global-only dataset (10000) with a smoke test.')
    ap.add_argument('--parquet', type=Path, default=DEFAULT_SPLIT_PARQUET, help='Split parquet to load.')
    ap.add_argument('--author_parquet', type=Path, default=DEFAULT_AUTHOR_PARQUET, help='Preferred author_static parquet.')
    ap.add_argument('--global_parquet', type=Path, default=DEFAULT_GLOBAL_PARQUET, help='Fallback global_features parquet.')
    ap.add_argument('--cols_json', type=Path, default=DEFAULT_COLS_JSON, help='Optional columns-order sidecar JSON.')
    ap.add_argument('--rows', type=int, default=50000, help='How many rows to load for the smoke test (<=0 loads full parquet).')
    ap.add_argument('--tokenizer', type=str, default='EleutherAI/pythia-160m', help='Tokenizer repo id or local path.')
    ap.add_argument('--hf_token', type=str, default=None, help='HF token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN).')
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--pretok', action='store_true', help='Eagerly pre-tokenize the base dataset (RAM heavy).')
    ap.add_argument('--leakage_safe', action=argparse.BooleanOptionalAction, default=None, help='Apply leakage-safe blocklist.')
    ap.add_argument('--include_gstats', type=str, default='', help='CSV gstat_* include list (intersection).')
    ap.add_argument('--exclude_gstats', type=str, default='', help='CSV gstat_* exclude list.')
    ap.add_argument('--disable_gstats', type=str, default='', help='CSV extra gstat_* excludes (e.g., training disable list).')
    ap.add_argument('--return_masks', action=argparse.BooleanOptionalAction, default=None, help='Return global_mask.')
    ap.add_argument('--sanity_samples', type=int, default=None, help='Override sanity probe sample count.')
    ap.add_argument('--sanity_zero_tol', type=float, default=None, help='Override sanity all-zero tolerance.')
    ap.add_argument('--sanity_strict', action=argparse.BooleanOptionalAction, default=None, help='Raise on sanity failure.')
    ap.add_argument('-f', '--f', help=argparse.SUPPRESS)
    (args, _unknown) = ap.parse_known_args()
    if not args.parquet.exists():
        raise SystemExit(f'Missing split parquet: {args.parquet}')
    df = _load_parquet_head(args.parquet, int(args.rows))
    if 'gid' not in df.columns or 'target_user_id' not in df.columns:
        raise SystemExit("Split parquet must contain 'gid' and 'target_user_id' columns.")
    tok = _load_hf_tokenizer(args.tokenizer, hf_token=args.hf_token)
    if tok.pad_token_id is None:
        if getattr(tok, 'eos_token', None) is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({'pad_token': '[PAD]'})
    include = sorted(_parse_csv_set(args.include_gstats))
    exclude = sorted(_parse_csv_set(args.exclude_gstats))
    disable = sorted(_parse_csv_set(args.disable_gstats))
    ds_obj = HypernetGlobalOnlyDataset10000(df=df, tokenizer=tok, author_parquet=args.author_parquet if args.author_parquet and args.author_parquet.exists() else None, global_parquet=args.global_parquet if args.global_parquet and args.global_parquet.exists() else None, cols_json=args.cols_json if args.cols_json and args.cols_json.exists() else None, max_length=int(args.max_len), pretokenize=bool(args.pretok), leakage_safe=args.leakage_safe, include_gstats=include or None, exclude_gstats=exclude or None, disable_gstats=disable or None, return_masks=args.return_masks, sanity_samples=args.sanity_samples, sanity_zero_tol=args.sanity_zero_tol, sanity_strict=args.sanity_strict)
    _smoke_report(df, ds_obj)
if __name__ == '__main__':
    main()
