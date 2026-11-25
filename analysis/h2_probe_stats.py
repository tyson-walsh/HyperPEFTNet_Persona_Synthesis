from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq1_stats_core import across_seed_summary, auc_difference_test, auc_with_ci, benjamini_hochberg, holm_bonferroni, permutation_test_auc
CONTROL_SUFFIXES = {'_gshuffle': 'descriptor permuted across users: the pairing is broken, the marginal kept', '_gzero': 'descriptor zeroed: no conditioning signal at all'}
CHANCE_MARGIN = 0.02
IN_BAND_COHORTS = {'curiosity': 'in-band positive control: create_labels.py defines this cohort from gstat_question_ratio, one of the K=6 descriptor entries, so the gvec probe reads AUC 1.000 by construction. Reported as a sanity check, never as evidence.'}
REPRESENTATIONS = ('hidden', 'gvec')

def _find_probes(root: Path, representation: str='hidden') -> Dict[Tuple[str, str, str], Path]:
    found: Dict[Tuple[str, str, str], Path] = {}
    unparsed: List[str] = []
    for p in sorted(root.rglob('probe_oof.npz')):
        rel = p.relative_to(root).parts[:-1]
        if len(rel) < 2:
            unparsed.append(str(p))
            continue
        if rel[-1] in REPRESENTATIONS:
            (rep, cohort_dir, model_dir) = (rel[-1], rel[-2], rel[-3] if len(rel) >= 3 else 'unknown')
        else:
            (rep, cohort_dir, model_dir) = ('hidden', rel[-1], rel[-2])
        if rep != representation:
            continue
        cond = 'treatment'
        for suf in CONTROL_SUFFIXES:
            if model_dir.endswith(suf):
                cond = suf.lstrip('_')
                model_dir = model_dir[:-len(suf)]
                break
        cohort = re.sub('^labels_', '', cohort_dir)
        found[model_dir, cohort, cond] = p
    if unparsed:
        print('[h2] WARNING: %d probe files at an unexpected depth: %s' % (len(unparsed), unparsed[:3]))
    return found

def _load(p: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(p)
    s = np.asarray(z['score'], dtype=np.float64)
    y = np.asarray(z['label'], dtype=np.int64)
    return (y, s)

def analyze(root: Path, *, seed_label: str, n_perm: int, representation: str='hidden', min_models: int=0, min_cohorts: int=0) -> Dict[str, Any]:
    probes = _find_probes(root, representation=representation)
    if not probes:
        raise SystemExit('no probe_oof.npz under %s. Run validate_models_10000.py after the 2026-08-09 patch, which writes out-of-fold probe scores beside each metrics_highd.json.' % root)
    models = sorted({k[0] for k in probes})
    cohorts = sorted({k[1] for k in probes})
    print('[h2] found %d probe files: %d models x %d cohorts (representation=%s)' % (len(probes), len(models), len(cohorts), representation))
    print('[h2] models: %s' % ', '.join(models))
    print('[h2] cohorts: %s' % ', '.join(cohorts))
    if min_models and len(models) < min_models:
        raise SystemExit('[h2] FATAL: %d models discovered under %s, expected at least %d. Check the layout before trusting any number from this root.' % (len(models), root, min_models))
    if min_cohorts and len(cohorts) < min_cohorts:
        raise SystemExit('[h2] FATAL: %d cohorts discovered under %s, expected at least %d.' % (len(cohorts), root, min_cohorts))
    rows: List[Dict[str, Any]] = []
    for model in models:
        for cohort in cohorts:
            key = (model, cohort, 'treatment')
            if key not in probes:
                continue
            (y, s) = _load(probes[key])
            ok = np.isfinite(s)
            if ok.sum() < 10 or np.unique(y[ok]).size < 2:
                rows.append({'model': model, 'cohort': cohort, 'skipped': 'too few finite out-of-fold scores or one class only', 'n_finite': int(ok.sum())})
                continue
            entry: Dict[str, Any] = {'model': model, 'cohort': cohort, 'oof_coverage': float(np.mean(ok)), 'treatment': auc_with_ci(y[ok], s[ok]), 'permutation_vs_chance': permutation_test_auc(y[ok], s[ok], n_perm=n_perm), 'controls': {}}
            if cohort in IN_BAND_COHORTS:
                entry['in_band_note'] = IN_BAND_COHORTS[cohort]
            for (cond, why) in ((c.lstrip('_'), w) for (c, w) in CONTROL_SUFFIXES.items()):
                ck = (model, cohort, cond)
                if ck not in probes:
                    continue
                (yc, sc) = _load(probes[ck])
                if yc.shape != y.shape or not np.array_equal(yc, y):
                    entry['controls'][cond] = {'error': 'control labels differ from treatment labels; the two AUCs are not computed on the same rows and cannot be paired', 'n_treatment': int(y.size), 'n_control': int(yc.size)}
                    continue
                both = ok & np.isfinite(sc)
                ctrl_auc = auc_with_ci(yc[both], sc[both])
                entry['controls'][cond] = {'meaning': why, 'auc': ctrl_auc, 'at_chance_by_ci': ctrl_auc.get('at_chance'), 'equivalent_to_chance_tost': _auc_chance_equivalence(ctrl_auc, margin=CHANCE_MARGIN), 'treatment_vs_control': auc_difference_test(y[both], sc[both], s[both])}
            rows.append(entry)
    (fam_lbl, fam_p) = ([], [])
    for r in rows:
        if r.get('skipped'):
            continue
        d = ((r.get('controls') or {}).get('gshuffle') or {}).get('treatment_vs_control') or {}
        if d.get('p_value') is not None:
            fam_lbl.append('%s|%s|treatment_gt_gshuffle' % (r['model'], r['cohort']))
            fam_p.append(d['p_value'])
    holm = holm_bonferroni(fam_p)
    for (i, lbl) in enumerate(fam_lbl):
        (m, c, _) = lbl.split('|')
        for r in rows:
            if r.get('model') == m and r.get('cohort') == c:
                r.setdefault('multiplicity', {})['holm_adjusted_p_vs_gshuffle'] = holm['p_adjusted'][i]
                r['multiplicity']['holm_reject_at_0.05'] = holm['reject'][i]
    (sec_lbl, sec_p) = ([], [])
    for r in rows:
        if r.get('skipped'):
            continue
        p = (r.get('treatment') or {}).get('p_vs_chance_mannwhitney')
        if p is not None:
            sec_lbl.append('%s|%s|treatment_gt_chance' % (r['model'], r['cohort']))
            sec_p.append(p)
    bh = benjamini_hochberg(sec_p)
    for (i, lbl) in enumerate(sec_lbl):
        (m, c, _) = lbl.split('|')
        for r in rows:
            if r.get('model') == m and r.get('cohort') == c:
                r.setdefault('multiplicity', {})['bh_adjusted_p_vs_chance'] = bh['p_adjusted'][i]
    return {'seed_label': seed_label, 'root': str(root), 'chance_margin': CHANCE_MARGIN, 'n_perm': int(n_perm), 'models': models, 'cohorts': cohorts, 'results': rows, 'primary_family': {'claim': 'treatment AUC > gshuffle AUC on the same users', 'labels': fam_lbl, **holm}, 'secondary_family': {'claim': 'treatment AUC > 0.5', 'labels': sec_lbl, **bh}}

def _auc_chance_equivalence(ctrl_auc: Dict[str, Any], *, margin: float) -> Dict[str, Any]:
    auc = ctrl_auc.get('auc')
    se = ctrl_auc.get('se_delong')
    if auc is None or se is None:
        return {'equivalent': None, 'note': 'AUC or standard error unavailable'}
    from scipy import stats as _sps
    z = float(_sps.norm.ppf(0.95))
    (lo, hi) = (auc - z * se, auc + z * se)
    return {'margin': float(margin), 'auc': float(auc), 'se': float(se), 'interval_90': [float(lo), float(hi)], 'equivalent': bool(lo > 0.5 - margin and hi < 0.5 + margin), 'interpretation': 'equivalent=True means the control is demonstrably uninformative, which is stronger than merely failing to reach significance'}

def _tex(res: Dict[str, Any]) -> str:
    lines = ['% generated by analysis/h2_probe_stats.py -- do not hand-edit', '\\begin{table*}[t]', '\\centering', "\\caption{H2 hidden-state probe. AUC is pooled out-of-fold with a DeLong 95\\% interval. The treatment-versus-\\textsc{gshuffle} column is DeLong's test for two CORRELATED AUCs on the same users, Holm-corrected across the family. A control marked equivalent has a 90\\% interval lying entirely inside $0.5\\pm" + '%.2f' % CHANCE_MARGIN + '$, which establishes that it is uninformative rather than merely failing to reach significance.}', '\\label{tab:h2_probe_matched}', '\\begin{tabular}{llrrrr}', '\\toprule', 'Model & Cohort & Treatment AUC & \\textsc{gshuffle} AUC & $\\Delta$ (Holm $p$) & \\textsc{gzero} equiv. \\\\', '\\midrule']
    for r in res.get('results', []):
        if r.get('skipped'):
            continue
        t = r.get('treatment') or {}
        gs = (r.get('controls') or {}).get('gshuffle') or {}
        gz = (r.get('controls') or {}).get('gzero') or {}
        d = gs.get('treatment_vs_control') or {}
        p = (r.get('multiplicity') or {}).get('holm_adjusted_p_vs_gshuffle')
        gsa = gs.get('auc') or {}
        lines.append('%s & %s & %s & %s & %s & %s \\\\' % (str(r['model']).replace('_', '\\_'), str(r['cohort']).replace('_', '\\_'), '%.3f [%.3f, %.3f]' % (t['auc'], t['ci_lo'], t['ci_hi']) if t.get('auc') is not None else '--', '%.3f [%.3f, %.3f]' % (gsa['auc'], gsa['ci_lo'], gsa['ci_hi']) if gsa.get('auc') is not None else '--', '%+.3f (' % d['diff_b_minus_a'] + ('$<$0.001' if p is not None and p < 0.001 else '%.3f' % p if p is not None else '--') + ')' if d.get('diff_b_minus_a') is not None else '--', {True: 'yes', False: 'no', None: '--'}[(gz.get('equivalent_to_chance_tost') or {}).get('equivalent')]))
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    return '\n'.join(lines)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', action='append', required=True, help='Validation results root containing probe_oof.npz files, or SEED=DIR.')
    ap.add_argument('--seed-label', default='142')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tex', default=None)
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--representation', default='hidden', choices=list(REPRESENTATIONS), help='which probed representation to read: the hidden state, or the descriptor itself (gvec)')
    ap.add_argument('--min-models', type=int, default=0, help='refuse to run if fewer models are discovered under a root')
    ap.add_argument('--min-cohorts', type=int, default=0, help='refuse to run if fewer cohorts are discovered under a root')
    a = ap.parse_args()
    specs: List[Tuple[str, Path]] = []
    for d in a.root:
        (lbl, path) = d.split('=', 1) if '=' in d else (a.seed_label, d)
        specs.append((lbl, Path(path)))
    per_seed = [analyze(p, seed_label=l, n_perm=a.n_perm, representation=a.representation, min_models=a.min_models, min_cohorts=a.min_cohorts) for (l, p) in specs]
    payload: Dict[str, Any] = {'analysis': 'H2 probe statistics, matched protocol', 'representation': a.representation, 'generated_by': 'analysis/h2_probe_stats.py', 'per_seed': per_seed}
    if len(per_seed) > 1:
        agg: Dict[str, Any] = {}
        for r0 in per_seed[0]['results']:
            if r0.get('skipped'):
                continue
            key = '%s|%s' % (r0['model'], r0['cohort'])
            vals = {}
            for s in per_seed:
                for r in s['results']:
                    if r.get('skipped'):
                        continue
                    if r['model'] == r0['model'] and r['cohort'] == r0['cohort']:
                        auc = (r.get('treatment') or {}).get('auc')
                        if auc is not None:
                            try:
                                vals[int(s['seed_label'])] = float(auc)
                            except ValueError:
                                vals[len(vals)] = float(auc)
            if vals:
                agg[key] = across_seed_summary(vals, label=key)
        payload['across_seeds_treatment_auc'] = agg
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, indent=2))
    print('[h2] wrote %s' % outp)
    if a.tex:
        texp = Path(a.tex)
        texp.parent.mkdir(parents=True, exist_ok=True)
        texp.write_text(_tex(per_seed[0]))
        print('[h2] wrote %s' % texp)
    print('\n' + '=' * 90)
    for s in per_seed:
        print('SEED %s' % s['seed_label'])
        for r in s['results']:
            if r.get('skipped'):
                print('   SKIP %s/%s: %s' % (r.get('model'), r.get('cohort'), r['skipped']))
                continue
            t = r['treatment']
            line = '   %-16s %-12s AUC %.3f [%.3f,%.3f]' % (r['model'], r['cohort'], t['auc'], t['ci_lo'], t['ci_hi'])
            gs = (r.get('controls') or {}).get('gshuffle') or {}
            if gs.get('auc'):
                line += '  gshuf %.3f%s' % (gs['auc']['auc'], ' (at chance)' if gs.get('at_chance_by_ci') else '')
            p = (r.get('multiplicity') or {}).get('holm_adjusted_p_vs_gshuffle')
            if p is not None:
                line += '  holm_p=%.3g' % p
            if r.get('in_band_note'):
                line += '   [IN-BAND POSITIVE CONTROL]'
            print(line)
    print('=' * 90)
if __name__ == '__main__':
    main()
