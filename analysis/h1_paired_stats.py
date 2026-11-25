from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq1_stats_core import across_seed_summary, benjamini_hochberg, holm_bonferroni, paired_battery
NON_HYPER = ['base', 'lora', 'adapter', 'bias', 'prefix']
HYPER = ['hyperlora', 'hyperadapter', 'hyperbias', 'hyperprefix']
ALL_ARMS = NON_HYPER + HYPER
PREREGISTERED: List[Tuple[str, str, str, str]] = [('lora', 'hyperlora', 'superiority', 'PRIMARY. The load-bearing H1 claim: per-user conditioning beats one shared LoRA head, holding the PEFT site, the backbone and the training budget fixed.'), ('base', 'hyperlora', 'non_inferiority', 'SECONDARY. Adding a hypernetwork to a frozen backbone does not harm likelihood relative to updating all parameters. This is an equivalence claim and is tested with TOST.'), ('adapter', 'hyperadapter', 'superiority', 'Head-matched replication of the primary claim at the adapter site.'), ('bias', 'hyperbias', 'superiority', 'Head-matched replication at the bias site.'), ('prefix', 'hyperprefix', 'superiority', 'Head-matched replication at the prefix site.')]
EQUIVALENCE_MARGIN_NATS = 0.05

def _load_dumps(dump_dir: Path) -> Dict[str, pd.DataFrame]:
    found: Dict[str, pd.DataFrame] = {}
    for p in sorted(dump_dir.glob('per_example_*.parquet')):
        variant = p.name[len('per_example_'):-len('.parquet')]
        df = pd.read_parquet(p)
        need = {'row', 'gid', 'target_user_id', 'nll_sum', 'n_tok'}
        missing = need - set(df.columns)
        if missing:
            raise SystemExit('%s is missing columns %s' % (p, sorted(missing)))
        found[variant] = df
    if not found:
        raise SystemExit('no per_example_*.parquet under %s. Run the eval with --dump_per_example first.' % dump_dir)
    return found

def _load_model_sources(dump_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in sorted(dump_dir.glob('per_example_*.meta.json')):
        variant = p.name[len('per_example_'):-len('.meta.json')]
        try:
            meta = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            out[variant] = {'error': 'unreadable meta.json: %s' % e}
            continue
        src = meta.get('model_source')
        out[variant] = src if src else 'unrecorded (pre-2026-08-12 dump)'
    return out

def _fingerprint(df: pd.DataFrame) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(df['gid'].to_numpy(dtype=np.int64).tobytes())
    h.update(df['target_user_id'].to_numpy(dtype=np.int64).tobytes())
    h.update(df['n_tok'].to_numpy(dtype=np.int64).tobytes())
    return h.hexdigest()

def _check_pairable(dumps: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    fps = {v: _fingerprint(df) for (v, df) in dumps.items()}
    uniq = sorted(set(fps.values()))
    report = {'arms': sorted(dumps), 'fingerprints': fps, 'n_distinct_fingerprints': len(uniq), 'pairable': len(uniq) == 1, 'n_examples': {v: int(len(df)) for (v, df) in dumps.items()}}
    if len(uniq) != 1:
        groups: Dict[str, List[str]] = {}
        for (v, f) in fps.items():
            groups.setdefault(f, []).append(v)
        report['groups'] = {f[:16]: sorted(vs) for (f, vs) in groups.items()}
    return report

def _example_macro(df: pd.DataFrame) -> np.ndarray:
    n = df['n_tok'].to_numpy(dtype=np.float64)
    s = df['nll_sum'].to_numpy(dtype=np.float64)
    return np.where(n > 0, s / np.maximum(n, 1.0), np.nan)

def _user_level(df: pd.DataFrame, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    uid = df['target_user_id'].to_numpy(dtype=np.int64)
    ok = np.isfinite(vals) & (uid >= 0)
    (uid, v) = (uid[ok], vals[ok])
    order = np.argsort(uid, kind='mergesort')
    (uid, v) = (uid[order], v[order])
    (uniq, start) = np.unique(uid, return_index=True)
    sums = np.add.reduceat(v, start)
    cnts = np.diff(np.append(start, v.size)).astype(np.float64)
    return (uniq, sums / np.maximum(cnts, 1.0))

def analyze_seed(dump_dir: Path, *, seed_label: str, n_boot: int, boot_seed: int, all_pairs: bool) -> Dict[str, Any]:
    dumps = _load_dumps(dump_dir)
    pairable = _check_pairable(dumps)
    pairable['model_sources'] = _load_model_sources(dump_dir)
    macro = {v: _example_macro(df) for (v, df) in dumps.items()}
    users = {}
    for (v, df) in dumps.items():
        (u, m) = _user_level(df, macro[v])
        users[v] = (u, m)
    present = [a for a in ALL_ARMS if a in dumps]
    marginals = {v: {'n_examples': int(len(dumps[v])), 'token_micro_ce': float(dumps[v]['nll_sum'].sum() / max(1.0, float(dumps[v]['n_tok'].sum()))), 'example_macro_nll': float(np.nanmean(macro[v])), 'ppl_from_token_micro': float(math.exp(min(50.0, dumps[v]['nll_sum'].sum() / max(1.0, float(dumps[v]['n_tok'].sum()))))), 'n_users': int(users[v][0].size)} for v in present}

    def _one(a: str, b: str, kind: str, why: str) -> Dict[str, Any]:
        if a not in dumps or b not in dumps:
            return {'comparison': '%s_vs_%s' % (a, b), 'skipped': 'arm missing', 'have': sorted(dumps)}
        margin = EQUIVALENCE_MARGIN_NATS if kind == 'non_inferiority' else None
        ex = paired_battery(macro[a], macro[b], name='%s->%s (example level)' % (a, b), higher_is_better=False, seed=boot_seed, n_boot=n_boot, equivalence_margin=margin, clusters=dumps[a]['target_user_id'].to_numpy(dtype=np.int64))
        (ua, ma) = users[a]
        (ub, mb) = users[b]
        if ua.size != ub.size or not np.array_equal(ua, ub):
            usr: Dict[str, Any] = {'error': 'user id sets differ between arms; not pairable', 'n_a': int(ua.size), 'n_b': int(ub.size)}
        else:
            usr = paired_battery(ma, mb, name='%s->%s (user level)' % (a, b), higher_is_better=False, seed=boot_seed, n_boot=n_boot, equivalence_margin=margin)
        return {'comparison': '%s_vs_%s' % (a, b), 'a': a, 'b': b, 'claim_kind': kind, 'rationale': why, 'example_level': ex, 'user_level': usr}
    prereg = [_one(a, b, k, w) for (a, b, k, w) in PREREGISTERED]
    (fam_p, fam_lbl) = ([], [])
    for r in prereg:
        if r.get('skipped'):
            continue
        p = (r.get('user_level') or {}).get('wilcoxon', {}).get('p_value')
        fam_p.append(p)
        fam_lbl.append(r['comparison'])
    holm = holm_bonferroni(fam_p)
    for (i, lbl) in enumerate(fam_lbl):
        for r in prereg:
            if r.get('comparison') == lbl:
                r['holm_adjusted_p'] = holm['p_adjusted'][i]
                r['holm_reject_at_0.05'] = holm['reject'][i]
    out: Dict[str, Any] = {'seed_label': seed_label, 'dump_dir': str(dump_dir), 'pairability_gate': pairable, 'marginals': marginals, 'equivalence_margin_nats': EQUIVALENCE_MARGIN_NATS, 'preregistered': prereg, 'preregistered_multiplicity': {'family': fam_lbl, 'designated_p': 'user-level Wilcoxon signed-rank', **holm}}
    if all_pairs:
        (expl, ep, elbl) = ([], [], [])
        for (i, a) in enumerate(present):
            for b in present[i + 1:]:
                r = _one(a, b, 'exploratory', 'all-pairs exploratory comparison')
                expl.append(r)
                ep.append((r.get('user_level') or {}).get('wilcoxon', {}).get('p_value'))
                elbl.append(r['comparison'])
        bh = benjamini_hochberg(ep)
        for (i, lbl) in enumerate(elbl):
            for r in expl:
                if r.get('comparison') == lbl:
                    r['bh_adjusted_p'] = bh['p_adjusted'][i]
                    r['bh_reject_at_0.05'] = bh['reject'][i]
        out['exploratory_all_pairs'] = expl
        out['exploratory_multiplicity'] = {'family': elbl, **bh}
    return out

def _tex_table(per_seed: List[Dict[str, Any]]) -> str:
    r0 = per_seed[0]
    lines = ['% generated by analysis/h1_paired_stats.py -- do not hand-edit', '\\begin{table*}[t]', '\\centering', '\\caption{Paired reply-span negative log-likelihood between arms trained under the matched protocol. Deltas are (B$-$A) so negative favors B. Inference is at the user level ($n$ users, examples averaged within user first); the interval is a user-cluster bootstrap. $p$ is the Wilcoxon signed-rank test, Holm-corrected across the pre-registered family. The non-inferiority row is judged by TOST against a pre-registered margin of ' + '%.2f' % EQUIVALENCE_MARGIN_NATS + ' nats, not by a failure to reject.}', '\\label{tab:paired_nll_matched}', '\\begin{tabular}{llrrrrr}', '\\toprule', 'A $\\rightarrow$ B & Claim & $n$ & $\\Delta$ & 95\\% CI & Holm $p$ & Verdict \\\\', '\\midrule']
    for r in r0.get('preregistered', []):
        if r.get('skipped'):
            continue
        ul = r.get('user_level') or {}
        ci = ul.get('bootstrap_ci_mean_delta') or {}
        d = ul.get('mean_delta')
        n = ul.get('n_pairs')
        p = r.get('holm_adjusted_p')
        if r['claim_kind'] == 'non_inferiority':
            eq = (ul.get('tost_equivalence') or {}).get('equivalent')
            verdict = 'equivalent' if eq else 'not established' if eq is not None else 'n/a'
        else:
            rej = r.get('holm_reject_at_0.05')
            better = ul.get('b_is_better')
            verdict = 'B better' if rej and better else 'A better' if rej and (not better) else 'n.s.'
        lines.append('%s $\\rightarrow$ %s & %s & %s & %s & %s & %s & %s \\\\' % (r['a'].replace('_', '\\_'), r['b'].replace('_', '\\_'), 'non-inf.' if r['claim_kind'] == 'non_inferiority' else 'superiority', '%d' % n if n is not None else '--', '%+.4f' % d if d is not None else '--', '[%+.4f, %+.4f]' % (ci['lo'], ci['hi']) if ci.get('lo') is not None else '--', '$<$0.001' if p is not None and p < 0.001 else '%.3f' % p if p is not None else '--', verdict))
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    return '\n'.join(lines)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dumps', action='append', required=True, help='Directory of per_example_*.parquet, or SEED=DIR. Repeatable.')
    ap.add_argument('--seed-label', default='142')
    ap.add_argument('--out', required=True, help='Output JSON path.')
    ap.add_argument('--tex', default=None, help='Optional LaTeX table path.')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--boot-seed', type=int, default=142)
    ap.add_argument('--all-pairs', action='store_true', help='Also run every unordered pair as an exploratory family under BH-FDR.')
    ap.add_argument('--require-pairable', action='store_true', default=True)
    ap.add_argument('--allow-unpairable', dest='require_pairable', action='store_false', help='Proceed even if arms were scored on different slices. Almost never right.')
    a = ap.parse_args()
    specs: List[Tuple[str, Path]] = []
    for d in a.dumps:
        if '=' in d:
            (lbl, path) = d.split('=', 1)
        else:
            (lbl, path) = (a.seed_label, d)
        specs.append((lbl, Path(path)))
    per_seed = []
    for (lbl, path) in specs:
        print('[h1] analyzing seed %s from %s' % (lbl, path))
        res = analyze_seed(path, seed_label=lbl, n_boot=a.n_boot, boot_seed=a.boot_seed, all_pairs=a.all_pairs)
        gate = res['pairability_gate']
        if not gate['pairable']:
            msg = '[h1] FATAL: arms in %s were not scored on the same slice. %d distinct fingerprints: %s' % (path, gate['n_distinct_fingerprints'], gate.get('groups'))
            if a.require_pairable:
                raise SystemExit(msg)
            print(msg + '  (continuing because --allow-unpairable was passed)')
        per_seed.append(res)
    payload: Dict[str, Any] = {'analysis': 'H1 paired reply-span NLL, matched protocol', 'generated_by': 'analysis/h1_paired_stats.py', 'n_boot_requested': int(a.n_boot), 'equivalence_margin_nats': EQUIVALENCE_MARGIN_NATS, 'preregistered_family': [{'a': x[0], 'b': x[1], 'kind': x[2], 'rationale': x[3]} for x in PREREGISTERED], 'per_seed': per_seed}
    if len(per_seed) > 1:
        seeds_summary = {}
        for r in PREREGISTERED:
            key = '%s_vs_%s' % (r[0], r[1])
            vals = {}
            for s in per_seed:
                for c in s.get('preregistered', []):
                    if c.get('comparison') == key and (not c.get('skipped')):
                        d = (c.get('user_level') or {}).get('mean_delta')
                        if d is not None:
                            try:
                                vals[int(s['seed_label'])] = float(d)
                            except ValueError:
                                vals[len(vals)] = float(d)
            if vals:
                seeds_summary[key] = across_seed_summary(vals, label=key)
        payload['across_seeds'] = seeds_summary
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, indent=2))
    print('[h1] wrote %s' % outp)
    if a.tex:
        texp = Path(a.tex)
        texp.parent.mkdir(parents=True, exist_ok=True)
        texp.write_text(_tex_table(per_seed))
        print('[h1] wrote %s' % texp)
    print('\n' + '=' * 78)
    for s in per_seed:
        print('SEED %s   pairable=%s   arms=%s' % (s['seed_label'], s['pairability_gate']['pairable'], ','.join(sorted(s['marginals']))))
        for (v, m) in sorted(s['marginals'].items(), key=lambda kv: kv[1]['token_micro_ce']):
            print('   %-14s CE %.4f  PPL %7.2f  macroNLL %.4f' % (v, m['token_micro_ce'], m['ppl_from_token_micro'], m['example_macro_nll']))
        for c in s.get('preregistered', []):
            if c.get('skipped'):
                print('   SKIP %s (%s)' % (c['comparison'], c['skipped']))
                continue
            ul = c.get('user_level') or {}
            ci = ul.get('bootstrap_ci_mean_delta') or {}
            line = '   %-26s %-14s d=%+.4f' % (c['comparison'], c['claim_kind'], ul.get('mean_delta', float('nan')))
            if ci.get('lo') is not None:
                line += '  CI[%+.4f,%+.4f]' % (ci['lo'], ci['hi'])
            if c.get('holm_adjusted_p') is not None:
                line += '  holm_p=%.3g' % c['holm_adjusted_p']
            if c['claim_kind'] == 'non_inferiority':
                line += '  TOST_equivalent=%s' % (ul.get('tost_equivalence') or {}).get('equivalent')
            print(line)
    if 'across_seeds' in payload:
        print('\nACROSS SEEDS')
        for (k, v) in payload['across_seeds'].items():
            print('   %-26s mean %+.4f  sd %s  same_sign=%s' % (k, v['mean'], '%.4f' % v['sd'] if v.get('sd') is not None else 'n/a', v['all_same_sign']))
    print('=' * 78)
if __name__ == '__main__':
    main()
