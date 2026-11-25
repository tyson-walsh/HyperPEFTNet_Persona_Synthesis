from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, replace as _dc_replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
DEFAULT_DATA_DIR = Path('./data')
DEFAULT_AUTHOR_PARQ = 'author_static_10000.parquet'
DEFAULT_GLOBAL_PARQ = 'global_features_10000.parquet'
DEFAULT_NORM_STATS = 'feature_norm_stats_10000.json'
USER_COL = 'target_user_id'
PAPER_K6_COND_FEATURES: List[str] = ['gstat_user_ttr', 'gstat_firstperson_ratio', 'gstat_user_subreddit_entropy', 'gstat_user_sr_max_share', 'gstat_punct_ratio', 'gstat_question_ratio']
SENTIMENT_QUINTILE_LABELS: List[str] = ['rage', 'grumpy', 'mellow', 'calm', 'empath']

def _stamp() -> str:
    return time.strftime('%Y%m%d_%H%M%S')

def _backup(path: Path) -> Path:
    dst = path.with_suffix(f'.bak_{_stamp()}{path.suffix}')
    shutil.copy2(path, dst)
    return dst

def _atomic_write_csv(df: pd.DataFrame, path: Path, *, backup_existing: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup_existing and path.exists():
        _backup(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    if tmp.exists():
        tmp.unlink()
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def _load_norm_stats(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}

def _to_raw(series_z: pd.Series, col: str, norm: Dict[str, Dict[str, float]]) -> pd.Series:
    stats = norm.get(col)
    if not stats:
        return series_z
    mu = float(stats.get('mean', 0.0))
    sd = float(stats.get('std', 1.0))
    if not np.isfinite(sd) or sd <= 0:
        return series_z
    return series_z.astype('float64') * sd + mu

def _ensure_int_uid(df: pd.DataFrame) -> pd.DataFrame:
    df[USER_COL] = pd.to_numeric(df[USER_COL], errors='coerce').fillna(-1).astype('int64')
    df = df[df[USER_COL] > 0].copy()
    return df

def _stable_sort(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return df.sort_values([score_col, USER_COL], ascending=[True, True], kind='mergesort').reset_index(drop=True)

def _tail_counts(n: int, low_q: float, high_q: float) -> Tuple[int, int]:
    n_low = int(math.floor(low_q * n + 1e-12))
    n_high = int(math.floor((1.0 - high_q) * n + 1e-12))
    return (n_low, n_high)

def build_sentiment_quintiles(users: pd.DataFrame, scores: pd.Series, *, labels: Sequence[str]=SENTIMENT_QUINTILE_LABELS) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if len(labels) != 5:
        raise ValueError('Sentiment quintiles expects exactly 5 labels.')
    df = pd.DataFrame({USER_COL: users[USER_COL].to_numpy(), 'score': pd.to_numeric(scores, errors='coerce')})
    df = df.replace([np.inf, -np.inf], np.nan)
    df['score'] = df['score'].fillna(df['score'].median())
    df = _stable_sort(df, 'score')
    n = len(df)
    ranks = np.arange(n, dtype=np.int64)
    bin_idx = ranks * 5 // max(1, n)
    bin_idx = np.clip(bin_idx, 0, 4)
    df['label'] = [labels[i] for i in bin_idx.tolist()]
    out = df[[USER_COL, 'label']].copy()
    out[USER_COL] = out[USER_COL].astype('int64')
    counts = out['label'].value_counts().to_dict()
    boundaries = {}
    for (i, q) in enumerate([0.2, 0.4, 0.6, 0.8]):
        k = int(math.floor(q * n))
        k = min(max(k, 0), n - 1)
        boundaries[str(q)] = float(df.loc[k, 'score'])
    meta = {'mode': 'sentiment_quintiles_rank', 'n_users': int(n), 'counts': {k: int(v) for (k, v) in counts.items()}, 'rank_boundaries_score_at_fraction': boundaries}
    return (out.sort_values(USER_COL).reset_index(drop=True), meta)

def build_binary_tails(users: pd.DataFrame, scores: pd.Series, *, low_q: float, high_q: float, low_label: str, high_label: str, method: str='rank') -> Tuple[pd.DataFrame, Dict[str, object]]:
    method = (method or 'rank').strip().lower()
    if not (0.0 < low_q < 0.5 and 0.5 < high_q < 1.0):
        raise ValueError(f'Bad tail cutpoints low_q={low_q} high_q={high_q}')
    df = pd.DataFrame({USER_COL: users[USER_COL].to_numpy(), 'score': pd.to_numeric(scores, errors='coerce')})
    df = df.replace([np.inf, -np.inf], np.nan)
    valid = df.dropna(subset=['score']).copy()
    valid['score'] = valid['score'].astype('float64')
    n = len(valid)
    if n == 0:
        return (pd.DataFrame(columns=[USER_COL, 'label']), {'mode': 'binary_tails', 'method': method, 'n_valid': 0, 'low_q': low_q, 'high_q': high_q, 'counts': {}})
    (n_low, n_high) = _tail_counts(n, low_q, high_q)
    if n_low <= 0 or n_high <= 0:
        raise ValueError(f'Tail sizes too small for n={n}: n_low={n_low} n_high={n_high}')
    if n_low + n_high > n:
        raise ValueError(f'Tail sizes overlap for n={n}: n_low+n_high={n_low + n_high}')
    valid_sorted = _stable_sort(valid, 'score')
    if method == 'rank':
        low = valid_sorted.iloc[:n_low].copy()
        high = valid_sorted.iloc[-n_high:].copy()
    elif method == 'quantile':
        (lo_thr, hi_thr) = valid_sorted['score'].quantile([low_q, high_q]).tolist()
        low = valid_sorted[valid_sorted['score'] <= lo_thr].copy()
        high = valid_sorted[valid_sorted['score'] >= hi_thr].copy()
        low = _stable_sort(low, 'score').iloc[:n_low].copy()
        high = _stable_sort(high, 'score').iloc[-n_high:].copy()
    else:
        raise ValueError(f'Unknown method: {method}')
    low['label'] = low_label
    high['label'] = high_label
    out = pd.concat([low[[USER_COL, 'label']], high[[USER_COL, 'label']]], ignore_index=True)
    out[USER_COL] = out[USER_COL].astype('int64')
    low_max = float(low['score'].max()) if len(low) else float('nan')
    high_min = float(high['score'].min()) if len(high) else float('nan')
    counts = out['label'].value_counts().to_dict()
    meta = {'mode': 'binary_tails', 'method': method, 'n_valid': int(n), 'low_q': float(low_q), 'high_q': float(high_q), 'n_low': int(n_low), 'n_high': int(n_high), 'boundary_low_max_score': low_max, 'boundary_high_min_score': high_min, 'counts': {k: int(v) for (k, v) in counts.items()}}
    return (out.sort_values(USER_COL).reset_index(drop=True), meta)

def score_col(df: pd.DataFrame, col: str, *, raw: bool, norm: Dict[str, Dict[str, float]]) -> pd.Series:
    s = pd.to_numeric(df[col], errors='coerce')
    if raw:
        s = _to_raw(s, col, norm)
    return s

def score_expressiveness(df: pd.DataFrame, *, norm: Dict[str, Dict[str, float]]) -> pd.Series:
    caps = _to_raw(pd.to_numeric(df['gstat_caps_ratio'], errors='coerce'), 'gstat_caps_ratio', norm)
    punct = _to_raw(pd.to_numeric(df['gstat_punct_ratio'], errors='coerce'), 'gstat_punct_ratio', norm)
    eps = 1e-12
    return 2.0 * (caps * punct) / (caps + punct + eps)

def load_user_frame(*, source: str, author_parq: Path, global_parq: Path, needed_cols: Sequence[str]) -> pd.DataFrame:
    source = (source or 'author').strip().lower()
    cols = [USER_COL] + sorted(set(needed_cols))
    if source == 'author':
        if not author_parq.exists():
            raise FileNotFoundError(f'Missing author parquet: {author_parq}')
        df = pd.read_parquet(author_parq, columns=cols)
        df = _ensure_int_uid(df)
        if df[USER_COL].duplicated().any():
            df = df.drop_duplicates(USER_COL, keep='first').reset_index(drop=True)
        return df.sort_values(USER_COL).reset_index(drop=True)
    if source == 'global':
        if not global_parq.exists():
            raise FileNotFoundError(f'Missing global parquet: {global_parq}')
        df = pd.read_parquet(global_parq, columns=cols)
        df = _ensure_int_uid(df)
        df = df.drop_duplicates(USER_COL, keep='first').reset_index(drop=True)
        return df.sort_values(USER_COL).reset_index(drop=True)
    raise ValueError(f'Unknown --source {source!r} (expected: author|global)')

@dataclass(frozen=True)
class BinarySpec:
    name: str
    low_q: float
    high_q: float
    low_label: str
    high_label: str
    score_cols: Tuple[str, ...]
    scorer: Callable[[pd.DataFrame], pd.Series]
    intended_band: str
    note: str = ''

def main() -> None:
    ap = argparse.ArgumentParser(description='Create diagnostic label CSVs from hypernet global-static user features.')
    ap.add_argument('--data_dir', type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument('--source', choices=['author', 'global'], default='author', help='Read from author_static (recommended) or global_features (deduped).')
    ap.add_argument('--author_parquet', type=Path, default=None, help=f'Override author parquet (default: <data_dir>/{DEFAULT_AUTHOR_PARQ})')
    ap.add_argument('--global_parquet', type=Path, default=None, help=f'Override global parquet (default: <data_dir>/{DEFAULT_GLOBAL_PARQ})')
    ap.add_argument('--norm_stats', type=Path, default=None, help=f'Override norm stats JSON (default: <data_dir>/{DEFAULT_NORM_STATS})')
    ap.add_argument('--out_dir', type=Path, default=None, help='Output directory (default: --data_dir)')
    ap.add_argument('--preset', choices=['extended', 'paper', 'all'], default='extended', help='Which label set to produce.')
    ap.add_argument('--sentiment_mode', choices=['quintiles', 'extremes', 'both'], default=None, help='Override sentiment output mode (default depends on preset).')
    ap.add_argument('--tail_overrides', type=str, default='', help='Per-family low-quantile override, e.g. "curiosity=0.20,politeness=0.15". High quantile is mirrored (1-low). Omit to use 0.10/0.90 tails. The paper uses curiosity=0.20 (n=4,000) and politeness=0.15 (n=3,000).')
    ap.add_argument('--tail_method', choices=['rank', 'quantile'], default='rank', help='Binary tails selection method.')
    ap.add_argument('--conditioning_features', type=str, default='paper_k6', help="Comma-separated conditioning cols for leakage warnings, or 'paper_k6', or 'none'.")
    ap.add_argument('--backup', action='store_true', help='Backup existing output CSVs before overwrite.')
    ap.add_argument('--dry_run', action='store_true', help='Compute and print summary, but do not write files.')
    ap.add_argument('-f', '--f', help=argparse.SUPPRESS)
    (args, _) = ap.parse_known_args()
    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir if args.out_dir is not None else data_dir
    author_parq = args.author_parquet if args.author_parquet is not None else data_dir / DEFAULT_AUTHOR_PARQ
    global_parq = args.global_parquet if args.global_parquet is not None else data_dir / DEFAULT_GLOBAL_PARQ
    norm_path = args.norm_stats if args.norm_stats is not None else data_dir / DEFAULT_NORM_STATS
    preset = args.preset
    sentiment_mode = args.sentiment_mode
    if sentiment_mode is None:
        sentiment_mode = {'extended': 'quintiles', 'paper': 'extremes', 'all': 'both'}[preset]
    bin_specs_all: List[BinarySpec] = [BinarySpec(name='politeness', low_q=0.1, high_q=0.9, low_label='polite', high_label='vulgar', score_cols=('gstat_profanity_ratio',), scorer=lambda df: score_col(df, 'gstat_profanity_ratio', raw=False, norm=norm), intended_band='out_of_band', note='Politeness extremes: bottom/top 10% tails; n=1000 per cohort.'), BinarySpec(name='curiosity', low_q=0.1, high_q=0.9, low_label='declarative', high_label='inquisitive', score_cols=('gstat_question_ratio',), scorer=lambda df: score_col(df, 'gstat_question_ratio', raw=False, norm=norm), intended_band='in_band', note='Curiosity extremes: bottom/top 10% tails; in-band positive control in K=6.'), BinarySpec(name='tempo', low_q=0.1, high_q=0.9, low_label='reactive', high_label='deliberate', score_cols=('gstat_reply_delay_mean',), scorer=lambda df: score_col(df, 'gstat_reply_delay_mean', raw=False, norm=norm), intended_band='out_of_band', note='Tempo extremes: bottom/top 10% tails; reply delay mean (z-scored).'), BinarySpec(name='self_focus', low_q=0.1, high_q=0.9, low_label='selfless', high_label='egocentric', score_cols=('gstat_firstperson_ratio',), scorer=lambda df: score_col(df, 'gstat_firstperson_ratio', raw=False, norm=norm), intended_band='out_of_band', note='Self-focus extremes: bottom/top 10% tails; first-person pronoun ratio.'), BinarySpec(name='expressiveness', low_q=0.1, high_q=0.9, low_label='reserved', high_label='emphatic', score_cols=('gstat_caps_ratio', 'gstat_punct_ratio'), scorer=lambda df: score_expressiveness(df, norm=norm), intended_band='in_band', note='Expressiveness extremes: bottom/top 10% tails; harmonic mean of caps & punct.')]
    if preset == 'paper':
        bin_specs = [s for s in bin_specs_all if s.name in ('politeness', 'curiosity')]
    else:
        bin_specs = bin_specs_all
    if args.tail_overrides:
        overrides: Dict[str, float] = {}
        for item in str(args.tail_overrides).split(','):
            item = item.strip()
            if not item:
                continue
            if '=' not in item:
                raise ValueError(f'--tail_overrides entry must be name=low_q, got {item!r}')
            (k, v) = item.split('=', 1)
            lo = float(v)
            if not 0.0 < lo < 0.5:
                raise ValueError(f'--tail_overrides {k}: low_q must be in (0, 0.5), got {lo}')
            overrides[k.strip()] = lo
        known = {s.name for s in bin_specs_all}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f'--tail_overrides names not recognized: {sorted(unknown)}; known: {sorted(known)}')
        new_specs = []
        for s in bin_specs:
            if s.name in overrides:
                lo = overrides[s.name]
                hi = 1.0 - lo
                n_each = int(math.floor(lo * 10000 + 1e-12))
                print(f'[tail_override] {s.name}: low_q {s.low_q} -> {lo}, high_q {s.high_q} -> {hi} (~{n_each} per tail at n=10,000)')
                s = _dc_replace(s, low_q=lo, high_q=hi, note=f'{s.note} [tail_override low_q={lo} high_q={hi}]'.strip())
            new_specs.append(s)
        bin_specs = new_specs
    needed_cols: Set[str] = {'gstat_user_sent_mean'}
    for s in bin_specs:
        needed_cols |= set(s.score_cols)
    norm = _load_norm_stats(norm_path)
    users = load_user_frame(source=args.source, author_parq=author_parq, global_parq=global_parq, needed_cols=sorted(needed_cols))
    n_users = int(users[USER_COL].nunique())
    if n_users == 0:
        raise RuntimeError('No users found after filtering target_user_id > 0.')
    cond_arg = (args.conditioning_features or '').strip().lower()
    if cond_arg in ('', 'none', 'null'):
        cond_features: Set[str] = set()
    elif cond_arg == 'paper_k6':
        cond_features = set(PAPER_K6_COND_FEATURES)
    else:
        cond_features = {c.strip() for c in args.conditioning_features.split(',') if c.strip()}
    manifest: Dict[str, object] = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'preset': preset, 'sentiment_mode': sentiment_mode, 'tail_method': args.tail_method, 'source': args.source, 'author_parquet': str(author_parq), 'global_parquet': str(global_parq), 'norm_stats': str(norm_path), 'n_users': n_users, 'conditioning_features_used_for_warnings': sorted(cond_features), 'outputs': {}, 'warnings': []}

    def warn_overlap(label_name: str, cols: Sequence[str]) -> None:
        if not cond_features:
            return
        overlap = sorted(set(cols) & cond_features)
        if overlap:
            manifest['warnings'].append(f"[leakage] label '{label_name}' uses columns that overlap conditioning features: {overlap}")
    sent_score = score_col(users, 'gstat_user_sent_mean', raw=False, norm=norm)
    if sentiment_mode in ('quintiles', 'both'):
        (sent_df, sent_meta) = build_sentiment_quintiles(users, sent_score)
        out_path = out_dir / 'labels_sentiment.csv' if sentiment_mode == 'quintiles' else out_dir / 'labels_sentiment_quintiles.csv'
        warn_overlap('sentiment_quintiles', ['gstat_user_sent_mean'])
        manifest['outputs'][out_path.name] = {'family': 'sentiment_quintiles', **sent_meta}
        if not args.dry_run:
            _atomic_write_csv(sent_df, out_path, backup_existing=bool(args.backup))
        print(f"[sentiment quintiles] wrote={out_path.name}  counts={sent_df['label'].value_counts().to_dict()}")
    if sentiment_mode in ('extremes', 'both'):
        (sent_ext, sent_ext_meta) = build_binary_tails(users, sent_score, low_q=0.1, high_q=0.9, low_label='rage', high_label='empath', method=args.tail_method)
        out_path = out_dir / 'labels_sentiment.csv' if sentiment_mode == 'extremes' else out_dir / 'labels_sentiment_extremes.csv'
        warn_overlap('sentiment_extremes', ['gstat_user_sent_mean'])
        manifest['outputs'][out_path.name] = {'family': 'sentiment_extremes', **sent_ext_meta}
        if not args.dry_run:
            _atomic_write_csv(sent_ext, out_path, backup_existing=bool(args.backup))
        print(f"[sentiment extremes ] wrote={out_path.name}  counts={sent_ext['label'].value_counts().to_dict()}")
    out_name_map = {'politeness': 'labels_politeness.csv', 'curiosity': 'labels_curiosity.csv', 'tempo': 'labels_tempo.csv', 'self_focus': 'labels_self_focus.csv', 'expressiveness': 'labels_expressiveness.csv'}
    for spec in bin_specs:
        scores = spec.scorer(users)
        (out_df, meta) = build_binary_tails(users, scores, low_q=spec.low_q, high_q=spec.high_q, low_label=spec.low_label, high_label=spec.high_label, method=args.tail_method)
        out_path = out_dir / out_name_map[spec.name]
        warn_overlap(spec.name, list(spec.score_cols))
        meta2 = {'family': spec.name, 'intended_band_relative_to_paper_k6': spec.intended_band, 'score_cols': list(spec.score_cols), 'note': spec.note, **meta}
        manifest['outputs'][out_path.name] = meta2
        if not args.dry_run:
            _atomic_write_csv(out_df, out_path, backup_existing=bool(args.backup))
        print(f"[{spec.name:<14}] wrote={out_path.name}  rows={len(out_df):,} counts={out_df['label'].value_counts().to_dict()}")
    manifest_path = out_dir / 'labels_manifest.json'
    if not args.dry_run:
        tmp = manifest_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        os.replace(tmp, manifest_path)
    if manifest['warnings']:
        print('\nWARNINGS:')
        for w in manifest['warnings']:
            print(' - ' + str(w))
    print(f'\nDONE. manifest={manifest_path.name} (dry_run={args.dry_run})')
if __name__ == '__main__':
    main()
