from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pyarrow.parquet as pq
from scipy import stats
DATA = Path('./data')
DUMPS = Path('./results/iui_rq1_per_example_v2_s142')
PAIRS = [('LoRA', 'lora', 'hyperlora'), ('Adapter', 'adapter', 'hyperadapter'), ('Bias', 'bias', 'hyperbias'), ('Prefix', 'prefix', 'hyperprefix')]
LONG_CHARS = 120

def h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode('utf-8', 'ignore'), digest_size=8).digest(), 'big')

def train_text_hashes() -> Set[int]:
    out: Set[int] = set()
    pf = pq.ParquetFile(DATA / 'train_data_10000.parquet')
    for batch in pf.iter_batches(columns=['text'], batch_size=200000):
        for s in batch.column('text').to_pylist():
            if s:
                out.add(h(s))
    return out

def contaminated_test_gids(train: Set[int]) -> Tuple[Set[int], Set[int], int, int]:
    exact: Set[int] = set()
    long_hit: Set[int] = set()
    n_gids = 0
    n_long = 0
    pf = pq.ParquetFile(DATA / 'test_full_10000.parquet')
    for batch in pf.iter_batches(columns=['group_label', 'text', 'gid'], batch_size=200000):
        lab = batch.column('group_label').to_pylist()
        txt = batch.column('text').to_pylist()
        gid = batch.column('gid').to_pylist()
        for (lb, s, g) in zip(lab, txt, gid):
            if not lb or not str(lb).endswith('_target') or (not s):
                continue
            n_gids += 1
            is_long = len(s) > LONG_CHARS
            n_long += int(is_long)
            if h(s) in train:
                exact.add(int(g))
                if is_long:
                    long_hit.add(int(g))
    return (exact, long_hit, n_gids, n_long)

def per_user_nll(stem: str, drop: Set[int]) -> Dict[int, float]:
    t = pq.read_table(DUMPS / f'per_example_{stem}.parquet', columns=['gid', 'target_user_id', 'nll_sum', 'n_tok'])
    gid = np.asarray(t['gid'])
    uid = np.asarray(t['target_user_id'])
    ex = np.asarray(t['nll_sum'], dtype=np.float64) / np.asarray(t['n_tok'], dtype=np.float64)
    if drop:
        keep = ~np.isin(gid, np.fromiter(drop, dtype=gid.dtype, count=len(drop)))
        (uid, ex) = (uid[keep], ex[keep])
    order = np.argsort(uid, kind='stable')
    (uid, ex) = (uid[order], ex[order])
    edges = np.flatnonzero(np.diff(uid)) + 1
    return {int(uid[lo]): float(ex[lo:hi].mean()) for (lo, hi) in zip(np.r_[0, edges], np.r_[edges, len(uid)])}

def compare(a_stem: str, b_stem: str, drop: Set[int]) -> Dict[str, float]:
    (A, B) = (per_user_nll(a_stem, drop), per_user_nll(b_stem, drop))
    keys = sorted(set(A) & set(B))
    a = np.array([A[k] for k in keys])
    b = np.array([B[k] for k in keys])
    d = b - a
    p1 = float(stats.wilcoxon(b, a, alternative='less', zero_method='wilcox').pvalue)
    return {'n_users': len(keys), 'mean_a': float(a.mean()), 'mean_b': float(b.mean()), 'mean_delta': float(d.mean()), 'p_one_sided': p1, 'users_improved': int((d < 0).sum())}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='')
    ap.add_argument('--dumps', default='', help='override the per-example dump directory')
    args = ap.parse_args()
    if args.dumps:
        global DUMPS
        DUMPS = Path(args.dumps)
    print('  hashing the training split ...')
    train = train_text_hashes()
    print(f'  {len(train):,} distinct training texts')
    (exact, long_hit, n_gids, n_long) = contaminated_test_gids(train)
    print(f'  test target replies: {n_gids:,}  ({n_long:,} over {LONG_CHARS} chars)')
    print(f'  reply text also seen in training: {len(exact):,} ({100 * len(exact) / n_gids:.2f}% of test groups)')
    print(f'  the same, replies over {LONG_CHARS} chars: {len(long_hit):,} ({100 * len(long_hit) / n_gids:.2f}% of test groups)')
    out: Dict[str, object] = {'n_test_groups': n_gids, 'n_train_texts': len(train), 'n_contaminated_exact': len(exact), 'n_contaminated_long': len(long_hit), 'pct_contaminated_exact': 100 * len(exact) / n_gids, 'pct_contaminated_long': 100 * len(long_hit) / n_gids, 'long_chars': LONG_CHARS, 'comparisons': {}}
    for (name, a_stem, b_stem) in PAIRS:
        full = compare(a_stem, b_stem, set())
        clean = compare(a_stem, b_stem, exact)
        out['comparisons'][name] = {'full': full, 'isolated': clean, 'delta_shift': clean['mean_delta'] - full['mean_delta']}
        print(f'\n  {name}')
        print(f"    full slice      delta {full['mean_delta']:+.4f}  {full['users_improved']:,}/{full['n_users']:,} users improve  p {full['p_one_sided']:.3g}")
        print(f"    isolated slice  delta {clean['mean_delta']:+.4f}  {clean['users_improved']:,}/{clean['n_users']:,} users improve  p {clean['p_one_sided']:.3g}")
        print(f"    shift in delta  {clean['mean_delta'] - full['mean_delta']:+.4f} nats")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f'\n  wrote {args.json}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
