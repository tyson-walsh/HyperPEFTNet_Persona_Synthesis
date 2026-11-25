from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple
import pyarrow.parquet as pq
REPO = Path('.')
TRAIN = REPO / 'data' / 'train_data_10000.parquet'
GEN = Path('./results/iui_gen_battery_matched')
NS = (10, 15, 20, 25, 30)
HEADLINE_N = 15
MOD = (1 << 61) - 1
BASE = 1000003
MAX_RECORDS_PER_HASH = 3
TOKRE = re.compile("[a-z0-9']+")
_TOKCACHE: Dict[str, int] = {}
TEST = REPO / 'data' / 'test_full_10000.parquet'
N_GOLD_LARGE = 20000
MIN_ALPHA_FRAC = 0.6
MIN_DISTINCT_FRAC = 0.7
SENSITIVITY = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8))

def content_bearing_at(span: Sequence[str], alpha: float, dist: float) -> bool:
    n = len(span)
    return sum((1 for t in span if t.isalpha() and len(t) >= 2)) >= alpha * n and len(set(span)) >= dist * n

def content_bearing(span: Sequence[str]) -> bool:
    n = len(span)
    return sum((1 for t in span if t.isalpha() and len(t) >= 2)) >= MIN_ALPHA_FRAC * n and len(set(span)) >= MIN_DISTINCT_FRAC * n

def toks(s: str) -> List[str]:
    return TOKRE.findall(s.lower()) if s else []

def th(t: str) -> int:
    v = _TOKCACHE.get(t)
    if v is None:
        v = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), 'big') % MOD
        _TOKCACHE[t] = v
    return v

def windows(tok_hashes: Sequence[int], n: int):
    L = len(tok_hashes)
    if L < n:
        return
    pw = pow(BASE, n - 1, MOD)
    h = 0
    for i in range(n):
        h = (h * BASE + tok_hashes[i]) % MOD
    yield (0, h)
    for i in range(n, L):
        h = ((h - tok_hashes[i - n] * pw) * BASE + tok_hashes[i]) % MOD
        yield (i - n + 1, h)
SCORED_ARM = {'ours': 'hyperlora', 'shared_head': 'vanilla_lora', 'full_ft': 'ft_base', 'stranger_card': 'hyperlora_gshuffle', 'zeroed_card': 'hyperlora_gzero', 'boundary_head': 'boundary_lora'}

def _load_scored_samples(samples_parquet: str, fams: str) -> List[dict]:
    import pandas as pd
    want = {f for f in fams.split(',') if f}
    df = pd.read_parquet(samples_parquet, columns=['fam', 'arm', 'gid', 'reply'])
    if want:
        df = df[df.fam.isin(want)]
    if df.empty:
        raise SystemExit(f'FATAL: no samples in {samples_parquet} for families {sorted(want)}')
    ctx = pd.read_parquet(Path(samples_parquet).with_name('contexts.parquet'), columns=['gid', 'reference', 'context'])
    ref_of = dict(zip(ctx.gid.astype(int), ctx.reference))
    ctx_of = dict(zip(ctx.gid.astype(int), ctx.context))
    items: List[dict] = []
    for (arm, gid, text) in zip(df.arm, df.gid.astype(int), df.reply):
        items.append({'kind': 'generated', 'arm': SCORED_ARM.get(arm, arm), 'gid': str(gid), 'text': text or '', 'ctx_key': f's:{gid}', 'ctx': ctx_of[gid]})
    for gid in sorted(set(df.gid.astype(int))):
        items.append({'kind': 'gold', 'arm': 'reference', 'gid': str(gid), 'text': ref_of[gid], 'ctx_key': f's:{gid}', 'ctx': ctx_of[gid]})
    return items

def load_items(gen_dir: Path=GEN, samples_parquet: str='', fams: str='', prompts_dir: str='') -> List[dict]:
    if prompts_dir:
        items = _load_prompt_layout(Path(prompts_dir))
    elif samples_parquet:
        items = _load_scored_samples(samples_parquet, fams)
    else:
        items = _load_august_layout(gen_dir)
    return _add_controls(items)

def _load_prompt_layout(root: Path) -> List[dict]:
    items: List[dict] = []
    for tag in ('football', 'everest'):
        files = sorted(root.glob(f'slice_*/work/{tag}_generations.json'))
        if not files:
            raise SystemExit(f'FATAL: no slice_*/work/{tag}_generations.json under {root}')
        for f in files:
            for row in json.loads(f.read_text()).get('rows', []):
                gid = str(row.get('target_user_id', '?'))
                for rep in row.get('repeats', []) or []:
                    for (arm, payload) in (rep.get('by_model') or {}).items():
                        items.append({'kind': f'prompt_{tag}', 'arm': str(arm), 'gid': gid, 'text': (payload or {}).get('reply') or ''})
    return items

def _load_august_layout(GEN: Path) -> List[dict]:
    items: List[dict] = []
    seen_gold: Set[Tuple[str, str]] = set()
    for f in sorted(GEN.glob('chunk_*/reddit_generations_by_model/*part*.json')):
        d = json.loads(f.read_text())
        for row in d.get('rows', []):
            arm = str(row.get('model', '?'))
            gid = str(row.get('gid', '?'))
            for rep in row.get('repeats', []):
                items.append({'kind': 'generated', 'arm': arm, 'gid': gid, 'text': rep.get('reply') or ''})
            ref = row.get('reference_reply')
            if ref and (gid, ref[:64]) not in seen_gold:
                seen_gold.add((gid, ref[:64]))
                items.append({'kind': 'gold', 'arm': 'reference', 'gid': gid, 'text': ref})
    for tag in ('football', 'everest'):
        for f in sorted(GEN.glob(f'chunk_*/{tag}_generations.json')):
            d = json.loads(f.read_text())
            for row in d.get('rows', []):
                gid = str(row.get('target_user_id', '?'))
                for rep in row.get('repeats', []) or []:
                    for (arm, payload) in (rep.get('by_model') or {}).items():
                        reply = (payload or {}).get('reply')
                        if reply:
                            items.append({'kind': f'prompt_{tag}', 'arm': str(arm), 'gid': gid, 'text': reply})
    return items

def _add_controls(items: List[dict]) -> List[dict]:
    rng0 = random.Random(142)
    pool: List[Tuple[object, str]] = []
    pf = pq.ParquetFile(TEST)
    for batch in pf.iter_batches(columns=['gid', 'group_label', 'text'], batch_size=200000):
        gids = batch.column('gid').to_pylist()
        lab = batch.column('group_label').to_pylist()
        txt = batch.column('text').to_pylist()
        for (g, lb, t) in zip(gids, lab, txt):
            if lb and str(lb).endswith('_target') and t:
                pool.append((g, t))
    rng0.shuffle(pool)
    drawn = pool[:N_GOLD_LARGE]
    want = {g for (g, _) in drawn}
    ctx_parts: Dict[object, List[str]] = defaultdict(list)
    for batch in pf.iter_batches(columns=['gid', 'group_label', 'text'], batch_size=200000):
        for (g, lb, t) in zip(batch.column('gid').to_pylist(), batch.column('group_label').to_pylist(), batch.column('text').to_pylist()):
            if g in want and (not (lb and str(lb).endswith('_target'))) and t:
                ctx_parts[g].append(t)
    for (g, t) in drawn:
        items.append({'kind': 'gold_heldout', 'arm': 'human', 'gid': str(g), 'text': t, 'ctx_key': f'h:{g}', 'ctx': '\n'.join(ctx_parts.get(g, []))})
    rng = random.Random(142)
    for it in [x for x in items if x['kind'] == 'generated']:
        t = toks(it['text'])
        rng.shuffle(t)
        items.append({'kind': 'shuffled', 'arm': it['arm'], 'gid': it['gid'], 'text': ' '.join(t), 'ctx_key': it.get('ctx_key', ''), 'ctx': it.get('ctx', '')})
    return items

def build_index(items: List[dict]):
    idx: Dict[int, Dict[int, List[Tuple[int, int]]]] = {n: defaultdict(list) for n in NS}
    tokenized: List[List[str]] = []
    for (i, it) in enumerate(items):
        t = toks(it['text'])
        tokenized.append(t)
        hs = [th(x) for x in t]
        for n in NS:
            for (start, h) in windows(hs, n):
                idx[n][h].append((i, start))
    return ({n: dict(v) for (n, v) in idx.items()}, tokenized)
_SETS: Dict[int, Set[int]] = {}
_FULL: Set[int] = set()

def _init(sets: Dict[int, Set[int]], full: Set[int]) -> None:
    global _SETS, _FULL
    _SETS = sets
    _FULL = full

def full_hash(tokens: Sequence[str]) -> int:
    h = 0
    for t in tokens:
        h = (h * BASE + th(t)) % MOD
    return h

def _scan(job: Tuple[int, int, int]):
    (first, last, _) = job
    pf = pq.ParquetFile(TRAIN)
    base = sum((pf.metadata.row_group(g).num_rows for g in range(first)))
    hits: Dict[int, List[Tuple[int, int, int]]] = {n: [] for n in NS}
    full_hits: Set[int] = set()
    per_hash: Dict[Tuple[int, int], int] = defaultdict(int)
    n_rows = n_tok = 0
    row = base
    for g in range(first, last):
        tbl = pf.read_row_group(g, columns=['text'])
        for s in tbl.column('text').to_pylist():
            t = toks(s)
            n_rows += 1
            n_tok += len(t)
            if t:
                fh = full_hash(t)
                if fh in _FULL:
                    full_hits.add(fh)
            if len(t) >= NS[0]:
                hs = [th(x) for x in t]
                for n in NS:
                    S = _SETS[n]
                    if not S or len(hs) < n:
                        continue
                    for (start, h) in windows(hs, n):
                        if h in S:
                            k = (n, h)
                            per_hash[k] += 1
                            if per_hash[k] <= MAX_RECORDS_PER_HASH:
                                hits[n].append((h, row, start))
            row += 1
    return (n_rows, n_tok, hits, full_hits)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='')
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument('--limit_rows', type=int, default=0, help='smoke mode: stop after ~N training rows')
    ap.add_argument('--gen_dir', default=str(GEN), help="the August battery's layout (chunk_*/reddit_generations_by_model)")
    ap.add_argument('--samples_parquet', default='', help='issue 15: the pooled samples.parquet written by scripts/iui_rq1_gen_score.py')
    ap.add_argument('--fams', default='', help='with --samples_parquet: families to read, comma separated')
    ap.add_argument('--prompts_dir', default='', help='2026-10-12: read only the fixed-prompt replies under DIR/slice_*/work (a prompts run)')
    args = ap.parse_args()
    print('  loading generated text ...', flush=True)
    items = load_items(Path(args.gen_dir), args.samples_parquet, args.fams, args.prompts_dir)
    kinds = defaultdict(int)
    for it in items:
        kinds[it['kind'], it['arm']] += 1
    for k in sorted(kinds):
        print(f'    {k[0]:14s} {k[1]:14s} {kinds[k]:>7,}')
    empt = defaultdict(int)
    for it in items:
        if not toks(it['text']):
            empt[it['kind'], it['arm']] += 1
    for k in sorted(empt):
        print(f'    empty after normalization: {k[0]}/{k[1]} {empt[k]:,}')
    print(f'    total strings {len(items):,}', flush=True)
    out_empty = {f'{k[0]}/{k[1]}': v for (k, v) in empt.items()}
    (idx, tokenized) = build_index(items)
    sets = {n: set(idx[n]) for n in NS}
    full_idx: Dict[int, List[int]] = defaultdict(list)
    for (i, t) in enumerate(tokenized):
        if t:
            full_idx[full_hash(t)].append(i)
    if not any(sets.values()):
        raise SystemExit('FATAL: no generated text was loaded. Check the artifact paths.')
    wants = ('prompt_football', 'prompt_everest', 'gold_heldout') if args.prompts_dir else ('generated', 'gold_heldout', 'shuffled')
    for want in wants:
        if not any((it['kind'] == want for it in items)):
            raise SystemExit(f'FATAL: the {want} group is empty; the audit would report a false clean.')
    for n in NS:
        print(f'    distinct {n:>2}-grams on the generated side: {len(sets[n]):>9,}')
    pf = pq.ParquetFile(TRAIN)
    ngroups = pf.metadata.num_row_groups
    if args.limit_rows:
        cum = 0
        keep = 0
        for g in range(ngroups):
            cum += pf.metadata.row_group(g).num_rows
            keep = g + 1
            if cum >= args.limit_rows:
                break
        ngroups = keep
    step = max(1, (ngroups + args.workers - 1) // args.workers)
    jobs = [(a, min(a + step, ngroups), 0) for a in range(0, ngroups, step)]
    print(f'\n  streaming {ngroups} row groups of {TRAIN.name} across {len(jobs)} workers ...', flush=True)
    n_rows = n_tok = 0
    merged: Dict[int, List[Tuple[int, int, int]]] = {n: [] for n in NS}
    full_seen: Set[int] = set()
    with mp.Pool(len(jobs), initializer=_init, initargs=(sets, set(full_idx))) as pool:
        for (r, t, hits, fh) in pool.imap_unordered(_scan, jobs):
            n_rows += r
            n_tok += t
            full_seen |= fh
            for n in NS:
                merged[n].extend(hits[n])
    print(f'  scanned {n_rows:,} training rows, {n_tok:,} tokens', flush=True)
    if not args.limit_rows and n_rows != pf.metadata.num_rows:
        raise SystemExit(f'FATAL: scanned {n_rows} of {pf.metadata.num_rows} rows.')
    reproduced: Set[int] = set()
    for h in full_seen:
        reproduced.update(full_idx.get(h, ()))
    print(f'  whole strings reproduced verbatim by a training row: {len(reproduced):,}', flush=True)
    need_rows = sorted({row for n in NS for (_, row, _) in merged[n]})
    print(f'  verifying {sum((len(merged[n]) for n in NS)):,} candidate hits against {len(need_rows):,} training rows ...', flush=True)
    rowtoks: Dict[int, List[str]] = {}
    if need_rows:
        want = set(need_rows)
        row = 0
        for g in range(pf.metadata.num_row_groups):
            ng = pf.metadata.row_group(g).num_rows
            if any((row <= r < row + ng for r in need_rows)):
                tbl = pf.read_row_group(g, columns=['text'])
                for (j, s) in enumerate(tbl.column('text').to_pylist()):
                    if row + j in want:
                        rowtoks[row + j] = toks(s)
            row += ng
    confirmed: Dict[int, Set[int]] = {n: set() for n in NS}
    confirmed_c: Dict[int, Set[int]] = {n: set() for n in NS}
    win_hit: Dict[int, Dict[int, Set[int]]] = {n: defaultdict(set) for n in NS}
    win_hit_c: Dict[int, Dict[int, Set[int]]] = {n: defaultdict(set) for n in NS}
    sens: Dict[Tuple[float, float], Dict[int, Set[int]]] = {t: defaultdict(set) for t in SENSITIVITY}
    span_set_c: Dict[int, Dict[int, Set[str]]] = {n: defaultdict(set) for n in NS}
    examples: List[dict] = []
    ex_per_group: Dict[str, int] = defaultdict(int)
    for n in NS:
        for (h, row, start) in merged[n]:
            tr = rowtoks.get(row)
            if tr is None or start + n > len(tr):
                continue
            span = tr[start:start + n]
            ok = content_bearing(span)
            for (item_id, gstart) in idx[n].get(h, []):
                gt = tokenized[item_id]
                if gt[gstart:gstart + n] == span:
                    confirmed[n].add(item_id)
                    win_hit[n][item_id].add(gstart)
                    if n == HEADLINE_N:
                        for t in SENSITIVITY:
                            if content_bearing_at(span, *t):
                                sens[t][item_id].add(gstart)
                    if ok:
                        confirmed_c[n].add(item_id)
                        win_hit_c[n][item_id].add(gstart)
                        span_set_c[n][item_id].add(' '.join(span))
                    if n == HEADLINE_N:
                        g = f"{items[item_id]['kind']}/{items[item_id]['arm']}"
                        key = g + ('/content' if ok else '/filler')
                        if ex_per_group[key] < 6:
                            ex_per_group[key] += 1
                            examples.append({'kind': items[item_id]['kind'], 'arm': items[item_id]['arm'], 'gid': items[item_id]['gid'], 'n': n, 'content_bearing': ok, 'span': ' '.join(span)})
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for (i, it) in enumerate(items):
        groups[it['kind'], it['arm']].append(i)
    human = groups['gold_heldout', 'human']
    groups['gold_heldout', 'human_dedup'] = [i for i in human if i not in reproduced]
    out: Dict[str, object] = {'n_training_rows': n_rows, 'n_training_tokens': n_tok, 'ns': list(NS), 'headline_n': HEADLINE_N, 'n_strings': len(items), 'n_empty': out_empty, 'groups': {}, 'examples': examples}
    hdr = f"\n  {'group':26s} {'strings':>7s} {'meanTok':>7s}" + ''.join((f" {'raw n=' + str(n):>11s}" for n in NS))
    print(hdr)
    for key in sorted(groups):
        ids = groups[key]
        name = f'{key[0]}/{key[1]}'
        mean_tok = sum((len(tokenized[i]) for i in ids)) / max(1, len(ids))
        rates = {}
        line = f'  {name:26s} {len(ids):>7,} {mean_tok:>7.1f}'
        for n in NS:
            c = sum((1 for i in ids if i in confirmed[n]))
            cc = sum((1 for i in ids if i in confirmed_c[n]))
            wtot = sum((max(0, len(tokenized[i]) - n + 1) for i in ids))
            wh = sum((len(win_hit[n].get(i, ())) for i in ids))
            whc = sum((len(win_hit_c[n].get(i, ())) for i in ids))
            elig = sum((1 for i in ids if len(tokenized[i]) >= n))
            rates[str(n)] = {'n_eligible': elig, 'n_with_copy': c, 'pct': 100.0 * c / max(1, len(ids)), 'n_with_content_copy': cc, 'pct_content': 100.0 * cc / max(1, len(ids)), 'n_windows': wtot, 'n_windows_matched': wh, 'pct_windows': 100.0 * wh / max(1, wtot), 'n_windows_matched_content': whc, 'pct_windows_content': 100.0 * whc / max(1, wtot), 'pct_spans_unique': 100.0 * len(set().union(*[span_set_c[n].get(i, set()) for i in ids]) if ids else set()) / whc if whc else 0.0, 'n_distinct_spans_content': len(set().union(*[span_set_c[n].get(i, set()) for i in ids]) if ids else set())}
            line += f' {c:>4,}/{100.0 * c / max(1, len(ids)):>5.2f}%'
        nrep = sum((1 for i in ids if i in reproduced))
        out['groups'][name] = {'n_strings': len(ids), 'mean_tokens': mean_tok, 'n_reproduced_whole': nrep, 'pct_reproduced_whole': 100.0 * nrep / max(1, len(ids)), 'by_n': rates}
        print(line, flush=True)
    n = HEADLINE_N
    print(f'\n  content-bearing only, n={n} (filler such as counting sequences excluded)')
    print(f"  {'group':26s} {'replies with a copy':>21s} {'windows matched':>18s} {'distinct spans':>15s} {'whole-reply':>12s}")
    for key in sorted(groups):
        name = f'{key[0]}/{key[1]}'
        g = out['groups'][name]
        r = g['by_n'][str(n)]
        print(f"  {name:26s} {r['n_with_content_copy']:>8,} / {r['pct_content']:>6.2f}%   {r['n_windows_matched_content']:>8,} / {r['pct_windows_content']:>6.3f}%   {r['n_distinct_spans_content']:>13,}   {g['pct_reproduced_whole']:>10.2f}%", flush=True)
    ctx_cache: Dict[str, Set[Tuple[str, ...]]] = {}
    echo_win: Dict[int, int] = {}
    for (item_id, starts) in win_hit_c[n].items():
        it = items[item_id]
        key = it.get('ctx_key') or ''
        if not key:
            continue
        if key not in ctx_cache:
            ct = toks(it.get('ctx') or '')
            ctx_cache[key] = {tuple(ct[i:i + n]) for i in range(len(ct) - n + 1)}
        tk = tokenized[item_id]
        echo_win[item_id] = sum((1 for s in starts if tuple(tk[s:s + n]) in ctx_cache[key]))
    print(f"\n  content-bearing, n={n}, split by whether the matched span sits in the reply's own context")
    print(f"  {'group':26s} {'windows matched':>15s} {'in own context':>15s} {'not in context':>15s} {'rate not in context':>20s} {'replies, not in context':>24s}")
    for key in sorted(groups):
        name = f'{key[0]}/{key[1]}'
        ids = groups[key]
        r = out['groups'][name]['by_n'][str(n)]
        with_ctx = sum((1 for i in ids if items[i].get('ctx_key')))
        if with_ctx == 0:
            r['context_split'] = 'no contexts loaded for this group'
            print(f"  {name:26s} {'no contexts loaded for this group':>60s}")
            continue
        wtot = r['n_windows']
        whc = r['n_windows_matched_content']
        echo = sum((echo_win.get(i, 0) for i in ids))
        not_ctx = whc - echo
        rep_not = sum((1 for i in ids if len(win_hit_c[n].get(i, ())) - echo_win.get(i, 0) > 0))
        r.update({'n_with_context': with_ctx, 'n_windows_matched_content_in_own_context': echo, 'n_windows_matched_content_not_in_context': not_ctx, 'pct_windows_content_not_in_context': 100.0 * not_ctx / max(1, wtot), 'n_with_content_copy_not_in_context': rep_not, 'pct_content_not_in_context': 100.0 * rep_not / max(1, len(ids))})
        print(f'  {name:26s} {whc:>15,} {echo:>15,} {not_ctx:>15,} {100.0 * not_ctx / max(1, wtot):>19.3f}% {rep_not:>9,} / {100.0 * rep_not / max(1, len(ids)):>6.2f}%', flush=True)
    print(f'\n  threshold sensitivity at n={HEADLINE_N}, share of windows matched')
    hdr = f"  {'group':26s}" + ''.join((f' {str(t):>14s}' for t in SENSITIVITY))
    print(hdr)
    out['sensitivity'] = {}
    for key in (('generated', 'hyperlora'), ('generated', 'vanilla_lora'), ('gold_heldout', 'human_dedup'), ('shuffled', 'hyperlora')):
        ids = groups.get(key, [])
        if not ids:
            continue
        name = f'{key[0]}/{key[1]}'
        wtot = sum((max(0, len(tokenized[i]) - HEADLINE_N + 1) for i in ids))
        row = {}
        line = f'  {name:26s}'
        for t in SENSITIVITY:
            wh = sum((len(sens[t].get(i, ())) for i in ids))
            pct = 100.0 * wh / max(1, wtot)
            row[f'{t[0]}/{t[1]}'] = pct
            line += f' {pct:>13.3f}%'
        out['sensitivity'][name] = row
        print(line, flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f'\n  wrote {args.json}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
