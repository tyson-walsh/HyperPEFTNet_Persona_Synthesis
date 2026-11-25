from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
try:
    from scipy import stats as _sps
except Exception as exc:
    raise SystemExit('scipy is required for rq1_stats_core: %s' % exc)

def _finite_pairs(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError('paired arrays must be the same length: %s vs %s' % (a.shape, b.shape))
    ok = np.isfinite(a) & np.isfinite(b)
    return (a[ok], b[ok], int(a.size - int(ok.sum())))

def _fmt(x: Any) -> Any:
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x

def _boot_means(x: np.ndarray, n_boot: int, rng: np.random.Generator, chunk_cells: int=20000000) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    per_chunk = max(1, int(chunk_cells // max(1, n)))
    out = np.empty((int(n_boot),), dtype=np.float64)
    done = 0
    while done < n_boot:
        c = min(per_chunk, int(n_boot) - done)
        idx = rng.integers(0, n, size=(c, n))
        out[done:done + c] = x[idx].mean(axis=1)
        done += c
    return out

def _adaptive_n_boot(n: int, requested: int) -> Tuple[int, Optional[str]]:
    if n <= 100000 or requested <= 2000:
        return (int(requested), None)
    return (2000, 'n=%d is large, so replicates were reduced from %d to 2000; Monte Carlo error on the endpoints is far below the reported precision at this sample size' % (n, requested))

def _bca_ci(x: np.ndarray, *, n_boot: int=10000, alpha: float=0.05, seed: int=142) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return {'method': 'bca_mean', 'n': int(n), 'lo': None, 'hi': None, 'point': _fmt(x.mean()) if n else None, 'n_boot': 0, 'note': 'n<2, interval undefined'}
    n_boot_req = int(n_boot)
    (n_boot, boot_note) = _adaptive_n_boot(n, n_boot_req)
    rng = np.random.default_rng(int(seed))
    theta_hat = float(x.mean())
    boot = _boot_means(x, int(n_boot), rng)
    prop = float(np.mean(boot < theta_hat))
    prop = min(max(prop, 1.0 / (2.0 * n_boot)), 1.0 - 1.0 / (2.0 * n_boot))
    z0 = float(_sps.norm.ppf(prop))
    tot = float(x.sum())
    jack = (tot - x) / (n - 1.0)
    jbar = float(jack.mean())
    d = jbar - jack
    d2 = float(np.sum(d ** 2))
    d3 = float(np.sum(d ** 3))
    if d2 <= 0.0:
        a_hat = 0.0
        method = 'percentile_mean (jackknife variance zero)'
    else:
        a_hat = d3 / (6.0 * d2 ** 1.5)
        method = 'bca_mean'
    zl = float(_sps.norm.ppf(alpha / 2.0))
    zu = float(_sps.norm.ppf(1.0 - alpha / 2.0))

    def _adj(z: float) -> float:
        denom = 1.0 - a_hat * (z0 + z)
        if denom == 0.0:
            return float('nan')
        return float(_sps.norm.cdf(z0 + (z0 + z) / denom))
    (a1, a2) = (_adj(zl), _adj(zu))
    if not (math.isfinite(a1) and math.isfinite(a2)):
        (a1, a2) = (alpha / 2.0, 1.0 - alpha / 2.0)
        method = 'percentile_mean (bca adjustment undefined)'
    lo = float(np.quantile(boot, min(max(a1, 0.0), 1.0)))
    hi = float(np.quantile(boot, min(max(a2, 0.0), 1.0)))
    res = {'method': method, 'n': int(n), 'n_boot': int(n_boot), 'n_boot_requested': int(n_boot_req), 'alpha': float(alpha), 'point': _fmt(theta_hat), 'lo': _fmt(lo), 'hi': _fmt(hi), 'z0': _fmt(z0), 'a': _fmt(a_hat), 'excludes_zero': bool(lo > 0.0 or hi < 0.0)}
    if boot_note:
        res['n_boot_note'] = boot_note
    return res

def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray, *, n_boot: int=10000, alpha: float=0.05, seed: int=142) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).ravel()
    clusters = np.asarray(clusters).ravel()
    ok = np.isfinite(values)
    (values, clusters) = (values[ok], clusters[ok])
    if values.size == 0:
        return {'method': 'cluster_bootstrap', 'n_clusters': 0, 'lo': None, 'hi': None, 'point': None}
    (uniq, inv) = np.unique(clusters, return_inverse=True)
    k = uniq.size
    sums = np.bincount(inv, weights=values, minlength=k)
    cnts = np.bincount(inv, minlength=k).astype(np.float64)
    per_cluster_mean = sums / np.maximum(cnts, 1.0)
    res = _bca_ci(per_cluster_mean, n_boot=n_boot, alpha=alpha, seed=seed)
    res['method'] = 'cluster_bootstrap_bca(' + res['method'] + ')'
    res['n_clusters'] = int(k)
    res['n_rows'] = int(values.size)
    return res

def sign_test(d: np.ndarray, *, higher_is_better: bool) -> Dict[str, Any]:
    d = np.asarray(d, dtype=np.float64).ravel()
    d = d[np.isfinite(d)]
    imp = d if higher_is_better else -d
    wins = int(np.sum(imp > 0))
    losses = int(np.sum(imp < 0))
    ties = int(np.sum(imp == 0))
    n = wins + losses
    if n == 0:
        return {'wins': wins, 'losses': losses, 'ties': ties, 'n_effective': 0, 'p_value': None}
    bt = _sps.binomtest(wins, n, 0.5, alternative='two-sided')
    return {'wins': wins, 'losses': losses, 'ties': ties, 'n_effective': int(n), 'win_rate': _fmt(wins / n), 'p_value': _fmt(bt.pvalue)}

def tost_equivalence(d: np.ndarray, *, margin: float, alpha: float=0.05) -> Dict[str, Any]:
    d = np.asarray(d, dtype=np.float64).ravel()
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2 or margin <= 0:
        return {'n': int(n), 'margin': _fmt(margin), 'equivalent': None, 'note': 'n<2 or non-positive margin'}
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    if se == 0.0:
        eq = bool(abs(mean) < margin)
        return {'n': int(n), 'margin': _fmt(margin), 'mean_delta': _fmt(mean), 'p_lower': 0.0 if eq else 1.0, 'p_upper': 0.0 if eq else 1.0, 'p_tost': 0.0 if eq else 1.0, 'equivalent': eq, 'note': 'zero variance, decided by the point estimate'}
    df = n - 1
    t_lo = (mean + margin) / se
    t_hi = (mean - margin) / se
    p_lo = float(_sps.t.sf(t_lo, df))
    p_hi = float(_sps.t.cdf(t_hi, df))
    p_tost = max(p_lo, p_hi)
    tcrit = float(_sps.t.ppf(1.0 - alpha, df))
    return {'n': int(n), 'margin': _fmt(margin), 'alpha': _fmt(alpha), 'mean_delta': _fmt(mean), 'se': _fmt(se), 'ci_1_minus_2alpha': [_fmt(mean - tcrit * se), _fmt(mean + tcrit * se)], 'p_lower': _fmt(p_lo), 'p_upper': _fmt(p_hi), 'p_tost': _fmt(p_tost), 'equivalent': bool(p_tost < alpha)}

def paired_battery(a: np.ndarray, b: np.ndarray, *, name: str, higher_is_better: bool=False, seed: int=142, n_boot: int=10000, shapiro_max_n: int=5000, equivalence_margin: Optional[float]=None, clusters: Optional[np.ndarray]=None) -> Dict[str, Any]:
    (a_f, b_f, dropped) = _finite_pairs(a, b)
    d = b_f - a_f
    n = int(d.size)
    imp = d if higher_is_better else -d
    out: Dict[str, Any] = {'comparison': name, 'n_pairs': n, 'n_dropped_nonfinite': int(dropped), 'higher_is_better': bool(higher_is_better), 'mean_a': _fmt(a_f.mean()) if n else None, 'mean_b': _fmt(b_f.mean()) if n else None, 'mean_delta': _fmt(d.mean()) if n else None, 'median_delta': _fmt(np.median(d)) if n else None, 'pct_change': _fmt(100.0 * d.mean() / a_f.mean()) if n and a_f.mean() != 0 else None, 'b_is_better': bool(imp.mean() > 0) if n else None}
    if n < 2:
        out['note'] = 'fewer than two finite pairs; no tests run'
        return out
    sd = float(d.std(ddof=1))
    dz = float(d.mean() / sd) if sd > 0 else float('inf')
    J = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0) if n > 2 else 1.0
    out['effect_size'] = {'cohens_dz': _fmt(dz), 'hedges_gz': _fmt(dz * J), 'sd_delta': _fmt(sd), 'interpretation': 'd_z is the standardized MEAN PAIRED DIFFERENCE, not a between-group d'}
    out['sign_test'] = sign_test(d, higher_is_better=higher_is_better)
    try:
        t = _sps.ttest_rel(b_f, a_f)
        out['paired_t'] = {'t': _fmt(t.statistic), 'p_value': _fmt(t.pvalue), 'df': int(n - 1)}
    except Exception as exc:
        out['paired_t'] = {'error': str(exc)}
    try:
        nz = d[d != 0]
        if nz.size == 0:
            out['wilcoxon'] = {'note': 'all deltas exactly zero', 'p_value': 1.0, 'n_used': 0}
        else:
            method = 'approx' if nz.size > 1000 else 'auto'
            w = _sps.wilcoxon(nz, alternative='two-sided', zero_method='wilcox', method=method)
            out['wilcoxon'] = {'statistic': _fmt(w.statistic), 'p_value': _fmt(w.pvalue), 'n_used': int(nz.size), 'method': method, 'subsampled': False}
    except Exception as exc:
        out['wilcoxon'] = {'error': str(exc)}
    try:
        rng = np.random.default_rng(int(seed))
        d_sh = d if n <= shapiro_max_n else d[rng.choice(n, size=int(shapiro_max_n), replace=False)]
        sw = _sps.shapiro(d_sh)
        out['shapiro_wilk'] = {'W': _fmt(sw.statistic), 'p_value': _fmt(sw.pvalue), 'n_used': int(d_sh.size), 'subsampled': bool(n > shapiro_max_n), 'normality_rejected': bool(sw.pvalue < 0.05), 'consequence': 'if rejected, emphasize Wilcoxon and the bootstrap over the t-test'}
    except Exception as exc:
        out['shapiro_wilk'] = {'error': str(exc)}
    out['bootstrap_ci_mean_delta'] = _bca_ci(d, n_boot=n_boot, seed=seed)
    if clusters is not None:
        cl = np.asarray(clusters).ravel()
        a_raw = np.asarray(a, dtype=np.float64).ravel()
        b_raw = np.asarray(b, dtype=np.float64).ravel()
        ok = np.isfinite(a_raw) & np.isfinite(b_raw)
        if cl.size == ok.size:
            out['cluster_bootstrap_ci_mean_delta'] = cluster_bootstrap_ci(d, cl[ok], n_boot=n_boot, seed=seed)
        else:
            out['cluster_bootstrap_ci_mean_delta'] = {'error': 'cluster length mismatch'}
    if equivalence_margin is not None:
        out['tost_equivalence'] = tost_equivalence(d, margin=float(equivalence_margin))
    return out

def _delong_components(s_pos: np.ndarray, s_neg: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    sp = np.sort(s_pos)
    sn = np.sort(s_neg)
    (n_p, n_n) = (sp.size, sn.size)
    lo = np.searchsorted(sn, s_pos, side='left')
    hi = np.searchsorted(sn, s_pos, side='right')
    v10 = (lo + 0.5 * (hi - lo)) / float(n_n)
    lo2 = np.searchsorted(sp, s_neg, side='left')
    hi2 = np.searchsorted(sp, s_neg, side='right')
    v01 = (n_p - hi2 + 0.5 * (hi2 - lo2)) / float(n_p)
    auc = float(v10.mean())
    return (auc, v10, v01)

def auc_with_ci(y_true: np.ndarray, score: np.ndarray, *, alpha: float=0.05, n_boot: int=0, seed: int=142) -> Dict[str, Any]:
    y = np.asarray(y_true).ravel().astype(int)
    s = np.asarray(score, dtype=np.float64).ravel()
    ok = np.isfinite(s)
    (y, s) = (y[ok], s[ok])
    (s_pos, s_neg) = (s[y == 1], s[y == 0])
    (n_p, n_n) = (s_pos.size, s_neg.size)
    if n_p == 0 or n_n == 0:
        return {'auc': None, 'note': 'one class missing; AUC undefined', 'n_pos': int(n_p), 'n_neg': int(n_n)}
    (auc, v10, v01) = _delong_components(s_pos, s_neg)
    s10 = float(v10.var(ddof=1)) if n_p > 1 else 0.0
    s01 = float(v01.var(ddof=1)) if n_n > 1 else 0.0
    var = s10 / n_p + s01 / n_n
    se = math.sqrt(max(var, 0.0))
    z = float(_sps.norm.ppf(1.0 - alpha / 2.0))
    if se > 0.0 and 0.0 < auc < 1.0:
        lg = math.log(auc / (1.0 - auc))
        se_lg = se / (auc * (1.0 - auc))
        lo = 1.0 / (1.0 + math.exp(-(lg - z * se_lg)))
        hi = 1.0 / (1.0 + math.exp(-(lg + z * se_lg)))
        ci_method = 'delong_logit'
    else:
        (lo, hi) = (max(0.0, auc - z * se), min(1.0, auc + z * se))
        ci_method = 'delong_normal (degenerate AUC or zero variance)'
    u = _sps.mannwhitneyu(s_pos, s_neg, alternative='two-sided')
    out = {'auc': _fmt(auc), 'ci_lo': _fmt(lo), 'ci_hi': _fmt(hi), 'ci_method': ci_method, 'se_delong': _fmt(se), 'alpha': float(alpha), 'n_pos': int(n_p), 'n_neg': int(n_n), 'p_vs_chance_mannwhitney': _fmt(u.pvalue), 'ci_contains_half': bool(lo <= 0.5 <= hi), 'at_chance': bool(lo <= 0.5 <= hi)}
    if int(n_boot) > 0:
        rng = np.random.default_rng(int(seed))
        boots = np.empty(int(n_boot), dtype=np.float64)
        sn_sorted = np.sort(s_neg)
        for i in range(int(n_boot)):
            sp_b = s_pos[rng.integers(0, n_p, size=n_p)]
            sn_b = np.sort(s_neg[rng.integers(0, n_n, size=n_n)])
            l = np.searchsorted(sn_b, sp_b, side='left')
            h = np.searchsorted(sn_b, sp_b, side='right')
            boots[i] = float(((l + 0.5 * (h - l)) / float(n_n)).mean())
        out['bootstrap_cross_check'] = {'n_boot': int(n_boot), 'ci_lo': _fmt(np.quantile(boots, alpha / 2.0)), 'ci_hi': _fmt(np.quantile(boots, 1.0 - alpha / 2.0))}
    return out

def auc_difference_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, *, alpha: float=0.05) -> Dict[str, Any]:
    y = np.asarray(y_true).ravel().astype(int)
    a = np.asarray(score_a, dtype=np.float64).ravel()
    b = np.asarray(score_b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    (y, a, b) = (y[ok], a[ok], b[ok])
    (pos, neg) = (y == 1, y == 0)
    (n_p, n_n) = (int(pos.sum()), int(neg.sum()))
    if n_p < 2 or n_n < 2:
        return {'note': 'need at least two of each class', 'n_pos': n_p, 'n_neg': n_n}
    (auc_a, v10a, v01a) = _delong_components(a[pos], a[neg])
    (auc_b, v10b, v01b) = _delong_components(b[pos], b[neg])
    s10 = np.cov(np.vstack([v10a, v10b]), ddof=1)
    s01 = np.cov(np.vstack([v01a, v01b]), ddof=1)
    cov = s10 / n_p + s01 / n_n
    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    se = math.sqrt(max(var_diff, 0.0))
    diff = auc_b - auc_a
    if se <= 0.0:
        return {'method': 'delong_correlated_auc_difference', 'auc_a': _fmt(auc_a), 'auc_b': _fmt(auc_b), 'diff_b_minus_a': _fmt(diff), 'se': 0.0, 'z': None, 'p_value': None, 'ci_lo': _fmt(diff), 'ci_hi': _fmt(diff), 'significant': False, 'n_pos': n_p, 'n_neg': n_n, 'note': 'variance of the AUC difference is zero; the two scores rank identically'}
    z = diff / se
    p = float(2.0 * _sps.norm.sf(abs(z)))
    zc = float(_sps.norm.ppf(1.0 - alpha / 2.0))
    return {'method': 'delong_correlated_auc_difference', 'auc_a': _fmt(auc_a), 'auc_b': _fmt(auc_b), 'diff_b_minus_a': _fmt(diff), 'se': _fmt(se), 'z': _fmt(z), 'p_value': _fmt(p), 'ci_lo': _fmt(diff - zc * se), 'ci_hi': _fmt(diff + zc * se), 'significant': bool(p < alpha), 'n_pos': n_p, 'n_neg': n_n}

def permutation_test_auc(y_true: np.ndarray, score: np.ndarray, *, n_perm: int=10000, seed: int=142) -> Dict[str, Any]:
    y = np.asarray(y_true).ravel().astype(int)
    s = np.asarray(score, dtype=np.float64).ravel()
    ok = np.isfinite(s)
    (y, s) = (y[ok], s[ok])
    if np.unique(y).size < 2:
        return {'auc': None, 'p_value': None, 'note': 'one class missing'}

    def _auc(yy: np.ndarray) -> float:
        (p, n) = (s[yy == 1], s[yy == 0])
        if p.size == 0 or n.size == 0:
            return float('nan')
        return float(_sps.mannwhitneyu(p, n, alternative='two-sided').statistic) / (p.size * n.size)
    obs = _auc(y)
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(n_perm), dtype=np.float64)
    yy = y.copy()
    for i in range(int(n_perm)):
        rng.shuffle(yy)
        null[i] = _auc(yy)
    extreme = int(np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)))
    return {'auc': _fmt(obs), 'n_perm': int(n_perm), 'p_value': _fmt((extreme + 1) / (n_perm + 1)), 'null_mean': _fmt(null.mean()), 'null_sd': _fmt(null.std(ddof=1))}

def holm_bonferroni(pvals: Sequence[float], alpha: float=0.05) -> Dict[str, Any]:
    p = np.asarray([np.nan if v is None else float(v) for v in pvals], dtype=np.float64)
    m = int(np.sum(np.isfinite(p)))
    adj = np.full(p.shape, np.nan)
    if m == 0:
        return {'method': 'holm-bonferroni', 'm': 0, 'alpha': alpha, 'p_adjusted': [None] * len(pvals), 'reject': [None] * len(pvals)}
    idx = np.flatnonzero(np.isfinite(p))
    order = idx[np.argsort(p[idx], kind='mergesort')]
    running = 0.0
    for (rank, i) in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, min(val, 1.0))
        adj[i] = running
    return {'method': 'holm-bonferroni', 'm': m, 'alpha': float(alpha), 'p_adjusted': [_fmt(v) if np.isfinite(v) else None for v in adj], 'reject': [bool(v < alpha) if np.isfinite(v) else None for v in adj]}

def benjamini_hochberg(pvals: Sequence[float], alpha: float=0.05) -> Dict[str, Any]:
    p = np.asarray([np.nan if v is None else float(v) for v in pvals], dtype=np.float64)
    m = int(np.sum(np.isfinite(p)))
    adj = np.full(p.shape, np.nan)
    if m == 0:
        return {'method': 'benjamini-hochberg', 'm': 0, 'alpha': alpha, 'p_adjusted': [None] * len(pvals), 'reject': [None] * len(pvals)}
    idx = np.flatnonzero(np.isfinite(p))
    order = idx[np.argsort(p[idx], kind='mergesort')]
    running = 1.0
    for rank in range(len(order) - 1, -1, -1):
        i = order[rank]
        val = p[i] * m / (rank + 1.0)
        running = min(running, min(val, 1.0))
        adj[i] = running
    return {'method': 'benjamini-hochberg', 'm': m, 'alpha': float(alpha), 'p_adjusted': [_fmt(v) if np.isfinite(v) else None for v in adj], 'reject': [bool(v < alpha) if np.isfinite(v) else None for v in adj]}

def across_seed_summary(values: Dict[int, float], *, label: str='') -> Dict[str, Any]:
    seeds = sorted(values.keys())
    vals = np.asarray([values[s] for s in seeds], dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {'label': label, 'n_seeds': 0, 'note': 'no finite values'}
    signs = np.sign(finite)
    return {'label': label, 'n_seeds': int(finite.size), 'seeds': [int(s) for s in seeds], 'values': [_fmt(v) for v in vals], 'mean': _fmt(finite.mean()), 'sd': _fmt(finite.std(ddof=1)) if finite.size > 1 else None, 'min': _fmt(finite.min()), 'max': _fmt(finite.max()), 'range': _fmt(finite.max() - finite.min()), 'all_same_sign': bool(np.all(signs == signs[0])), 'direction': 'negative' if signs[0] < 0 else 'positive' if signs[0] > 0 else 'zero', 'interpretation': 'seeds are a robustness check. all_same_sign is the claim; no p-value is computed across three seeds because that would not be a meaningful test.'}
__all__ = ['paired_battery', 'sign_test', 'tost_equivalence', 'cluster_bootstrap_ci', 'auc_with_ci', 'permutation_test_auc', 'holm_bonferroni', 'benjamini_hochberg', 'across_seed_summary']
