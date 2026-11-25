from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pyarrow.parquet as pq
from scipy import stats
RES = Path('./results')
DUMPS = RES / 'iui_rq1_per_example_v2_s142'
FAMILY: List[Tuple[str, str, str, str]] = [('LoRA vs HyperPEFT-LoRA', 'lora', 'hyperlora', 'superiority'), ('Adapter vs HyperPEFT-Adapter', 'adapter', 'hyperadapter', 'superiority'), ('Bias vs HyperPEFT-Bias', 'bias', 'hyperbias', 'superiority'), ('Prefix vs HyperPEFT-Prefix', 'prefix', 'hyperprefix', 'superiority'), ('Full FT vs HyperPEFT-LoRA', 'base', 'hyperlora', 'equivalence')]
MARGIN = 0.05

def per_user_nll(stem: str) -> Dict[int, float]:
    p = DUMPS / f'per_example_{stem}.parquet'
    if not p.is_file():
        raise SystemExit(f'missing dump: {p}')
    t = pq.read_table(p, columns=['target_user_id', 'nll_sum', 'n_tok'])
    uid = np.asarray(t['target_user_id'])
    ex = np.asarray(t['nll_sum'], dtype=np.float64) / np.asarray(t['n_tok'], dtype=np.float64)
    order = np.argsort(uid, kind='stable')
    (uid, ex) = (uid[order], ex[order])
    edges = np.flatnonzero(np.diff(uid)) + 1
    out: Dict[int, float] = {}
    for (lo, hi) in zip(np.r_[0, edges], np.r_[edges, len(uid)]):
        out[int(uid[lo])] = float(ex[lo:hi].mean())
    return out

def paired(a: Dict[int, float], b: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(a) & set(b))
    return (np.array([a[k] for k in keys]), np.array([b[k] for k in keys]))

def fmt_p(p: Optional[float]) -> str:
    if p is None:
        return 'n/a'
    if p == 0.0:
        return '<1e-300'
    if p >= 0.9995:
        return f'{p:.4f}'
    return f'{p:.3g}'

def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for (rank, i) in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='')
    ap.add_argument('--dumps', default='', help='override the per-example dump directory')
    args = ap.parse_args()
    if args.dumps:
        global DUMPS
        DUMPS = Path(args.dumps)
    rows = []
    for (label, a_stem, b_stem, kind) in FAMILY:
        (A, B) = paired(per_user_nll(a_stem), per_user_nll(b_stem))
        d = B - A
        n = len(d)
        mean_d = float(d.mean())
        two_sided = float(stats.wilcoxon(B, A, alternative='two-sided', zero_method='wilcox').pvalue)
        one_sided = float(stats.wilcoxon(B, A, alternative='less', zero_method='wilcox').pvalue)
        if kind == 'superiority':
            claim_p = one_sided
        else:
            se = float(d.std(ddof=1) / math.sqrt(n))
            t_lo = (mean_d + MARGIN) / se
            t_hi = (mean_d - MARGIN) / se
            p_lo = float(stats.t.sf(t_lo, df=n - 1))
            p_hi = float(stats.t.cdf(t_hi, df=n - 1))
            claim_p = max(p_lo, p_hi)
        rows.append({'comparison': label, 'claim': kind, 'n_users': n, 'mean_a': float(A.mean()), 'mean_b': float(B.mean()), 'mean_delta': mean_d, 'p_two_sided': two_sided, 'p_one_sided_superiority': one_sided, 'p_claim': claim_p})
    adj = holm([r['p_claim'] for r in rows])
    for (r, a) in zip(rows, adj):
        r['p_claim_holm'] = a
        if r['claim'] == 'superiority':
            r['verdict'] = 'superiority supported' if a < 0.05 else 'superiority not supported; effect favors the shared head' if r['mean_delta'] > 0 else 'superiority not supported'
        else:
            r['verdict'] = 'equivalence established' if a < 0.05 else 'equivalence not established'
    w = max((len(r['comparison']) for r in rows))
    print(f"\n  {'comparison':<{w}}  {'claim':<12} {'NLL A':>8} {'NLL B':>8} {'delta':>9} {'p 2-sided':>10} {'p claim':>10} {'p Holm':>10}  verdict")
    print('  ' + '-' * (w + 84))
    for r in rows:
        print(f"  {r['comparison']:<{w}}  {r['claim']:<12} {r['mean_a']:>8.4f} {r['mean_b']:>8.4f} {r['mean_delta']:>+9.4f} {fmt_p(r['p_two_sided']):>10} {fmt_p(r['p_claim']):>10} {fmt_p(r['p_claim_holm']):>10}  {r['verdict']}")
    print(f'\n  Holm-Bonferroni applied over the five claim-level p-values, not the difference tests.')
    print(f'  Equivalence margin {MARGIN} nats. Lower NLL is better, so superiority means delta < 0.\n')
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({'margin': MARGIN, 'rows': rows}, indent=2))
        print(f'  wrote {args.json}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
