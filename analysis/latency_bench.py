from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
REPO = Path('.')
sys.path.insert(0, str(REPO / 'training_scripts'))
sys.path.insert(0, str(REPO / 'data_scripts'))
sys.path.insert(0, str(REPO))
import evaluate_models_10000 as EV

def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _evict(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)

def _summary(samples: List[float]) -> Dict[str, float]:
    if not samples:
        raise ValueError('no samples')
    s = sorted(samples)
    n = len(s)
    return {'n': n, 'mean_ms': statistics.fmean(s), 'median_ms': statistics.median(s), 'stdev_ms': statistics.stdev(s) if n > 1 else 0.0, 'p05_ms': s[max(0, int(0.05 * n) - 1)], 'p95_ms': s[min(n - 1, int(0.95 * n))], 'min_ms': s[0], 'max_ms': s[-1]}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hyper_ckpt', required=True)
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--lora_placeholders', required=True, help='shared LoRA head; its 72 tensors define the per-user adapter file shape')
    ap.add_argument('--author_features_parquet', required=True)
    ap.add_argument('--adapter_dir_local', required=True, help='node-local scratch where the synthetic per-user adapter files are written')
    ap.add_argument('--adapter_dir_nfs', default='', help='optional NFS directory for the secondary, slower storage measurement')
    ap.add_argument('--n_files', type=int, default=128)
    ap.add_argument('--n_switch', type=int, default=1000)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--ctx_len', type=int, default=512)
    ap.add_argument('--gen_tokens', type=int, default=64)
    ap.add_argument('--out_json', required=True)
    args = ap.parse_args()
    EV._refuse_best_checkpoint(args.hyper_ckpt, flag='--hyper_ckpt')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type != 'cuda':
        raise SystemExit('REFUSED: no CUDA device. A CPU number would not describe the deployment.')
    print(f'[lat] device {torch.cuda.get_device_name(0)}')
    print(f'[lat] torch {torch.__version__}')
    tok = AutoTokenizer.from_pretrained(args.base_ckpt, local_files_only=True)
    (model, gdim) = EV._build_hyper_model('hyperlora', base_ckpt=args.base_ckpt, tok=tok, device=dev, ckpt_file=Path(args.hyper_ckpt), force_special_init=True)
    model.eval()
    ph = model._placeholders
    theta_bar = model._theta_bar
    n_par = int(sum((p.numel() for p in ph.values())))
    print(f'[lat] gdim {gdim}  placeholders {len(ph)}  parameters {n_par:,}')
    import pyarrow.parquet as pq
    names = json.loads((Path(args.hyper_ckpt).parent / 'feature_names.json').read_text())
    if len(names) != gdim:
        raise SystemExit(f'REFUSED: feature_names.json has {len(names)} names, generator wants {gdim}')
    tbl = pq.read_table(args.author_features_parquet, columns=names)
    G = torch.tensor([[float(tbl[c][i].as_py()) for c in names] for i in range(tbl.num_rows)], dtype=torch.float32, device=dev)
    if G.shape[0] < args.n_switch:
        raise SystemExit(f'REFUSED: only {G.shape[0]} users available, {args.n_switch} requested')
    print(f'[lat] descriptors {tuple(G.shape)} from {args.author_features_parquet}')
    src = load_file(args.lora_placeholders)
    keys_needed = set(ph.keys())
    keys_have = set(src.keys())
    if not keys_needed.issubset(keys_have):
        missing = sorted(keys_needed - keys_have)[:4]
        raise SystemExit(f'REFUSED: shared LoRA file lacks {len(keys_needed - keys_have)} placeholder keys, e.g. {missing}')
    base_sd = {k: src[k].contiguous() for k in ph.keys()}
    bytes_per_file = sum((v.numel() * v.element_size() for v in base_sd.values()))

    def _materialize(d: Path, n: int) -> List[Path]:
        d.mkdir(parents=True, exist_ok=True)
        out: List[Path] = []
        gen = torch.Generator().manual_seed(142)
        for i in range(n):
            p = d / f'user_{i:05d}_lora.safetensors'
            if not p.is_file():
                jit = {k: (v + 0.001 * torch.randn(v.shape, generator=gen)).contiguous() for (k, v) in base_sd.items()}
                save_file(jit, str(p))
            out.append(p)
        return out
    local_files = _materialize(Path(args.adapter_dir_local), args.n_files)
    print(f'[lat] wrote/found {len(local_files)} adapter files in {args.adapter_dir_local}, {bytes_per_file / 1000000.0:.2f} MB each')
    nfs_files: List[Path] = []
    if args.adapter_dir_nfs:
        nfs_files = _materialize(Path(args.adapter_dir_nfs), args.n_files)
        print(f'[lat] wrote/found {len(nfs_files)} adapter files in {args.adapter_dir_nfs}')
    results: Dict[str, object] = {'device': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'hyper_ckpt': args.hyper_ckpt, 'peft_parameters': n_par, 'adapter_file_bytes': int(bytes_per_file), 'n_placeholder_tensors': len(ph), 'dtype': str(next(iter(ph.values())).dtype)}

    def _switch_hyper(i: int) -> None:
        g = G[i % G.shape[0]].unsqueeze(0)
        delta = model.hypernet(g)
        ptr = 0
        for (name, p) in ph.items():
            n = p.numel()
            p.data.copy_(theta_bar[name] + delta[0, ptr:ptr + n].view_as(p))
            ptr += n
    with torch.no_grad():
        for i in range(args.warmup):
            _switch_hyper(i)
        _sync()
        a_full: List[float] = []
        for i in range(args.n_switch):
            t0 = time.perf_counter()
            _switch_hyper(i)
            _sync()
            a_full.append((time.perf_counter() - t0) * 1000.0)
        for i in range(args.warmup):
            model.hypernet(G[i].unsqueeze(0))
        _sync()
        a_gen: List[float] = []
        for i in range(args.n_switch):
            g = G[i % G.shape[0]].unsqueeze(0)
            t0 = time.perf_counter()
            model.hypernet(g)
            _sync()
            a_gen.append((time.perf_counter() - t0) * 1000.0)
    results['A_descriptor_switch_total'] = _summary(a_full)
    results['A_generator_forward_only'] = _summary(a_gen)
    print(f"[lat] A total     median {results['A_descriptor_switch_total']['median_ms']:.3f} ms")
    print(f"[lat] A generator median {results['A_generator_forward_only']['median_ms']:.3f} ms")

    def _switch_ckpt(path: Path) -> None:
        sd = load_file(str(path))
        for (name, p) in ph.items():
            p.data.copy_(sd[name])

    def _bench_ckpt(files: List[Path], evict: bool) -> Dict[str, float]:
        with torch.no_grad():
            for f in files[:min(8, len(files))]:
                _switch_ckpt(f)
            _sync()
            out: List[float] = []
            reps = len(files) if evict else max(len(files), args.n_switch // 4)
            for i in range(reps):
                f = files[i % len(files)]
                if evict:
                    _evict(f)
                t0 = time.perf_counter()
                _switch_ckpt(f)
                _sync()
                out.append((time.perf_counter() - t0) * 1000.0)
        return _summary(out)
    results['B_local_cold'] = _bench_ckpt(local_files, evict=True)
    results['B_local_warm'] = _bench_ckpt(local_files, evict=False)
    print(f"[lat] B local cold median {results['B_local_cold']['median_ms']:.3f} ms")
    print(f"[lat] B local warm median {results['B_local_warm']['median_ms']:.3f} ms")
    if nfs_files:
        results['B_nfs_cold'] = _bench_ckpt(nfs_files, evict=True)
        results['B_nfs_warm'] = _bench_ckpt(nfs_files, evict=False)
        print(f"[lat] B nfs   cold median {results['B_nfs_cold']['median_ms']:.3f} ms")
    ids = torch.randint(low=0, high=int(tok.vocab_size), size=(1, args.ctx_len), device=dev)
    with torch.no_grad():
        for _ in range(8):
            model.backbone(input_ids=ids)
        _sync()
        c_fwd: List[float] = []
        for _ in range(100):
            t0 = time.perf_counter()
            model.backbone(input_ids=ids)
            _sync()
            c_fwd.append((time.perf_counter() - t0) * 1000.0)
    results['C_forward_pass_ctx'] = _summary(c_fwd)
    results['C_forward_pass_ctx']['ctx_len'] = args.ctx_len
    print(f"[lat] C forward  median {results['C_forward_pass_ctx']['median_ms']:.3f} ms at ctx {args.ctx_len}")
    gen_ids = ids[:, :args.ctx_len // 2]
    with torch.no_grad():
        for _ in range(3):
            model.backbone.generate(input_ids=gen_ids, max_new_tokens=args.gen_tokens, do_sample=False, pad_token_id=int(tok.eos_token_id))
        _sync()
        c_gen: List[float] = []
        for _ in range(20):
            t0 = time.perf_counter()
            model.backbone.generate(input_ids=gen_ids, max_new_tokens=args.gen_tokens, do_sample=False, pad_token_id=int(tok.eos_token_id))
            _sync()
            c_gen.append((time.perf_counter() - t0) * 1000.0)
    results['C_generate'] = _summary(c_gen)
    results['C_generate']['new_tokens'] = args.gen_tokens
    results['C_generate']['prompt_len'] = int(gen_ids.shape[1])
    print(f"[lat] C generate median {results['C_generate']['median_ms']:.3f} ms for {args.gen_tokens} new tokens")
    a = results['A_descriptor_switch_total']['median_ms']
    results['derived'] = {'speedup_vs_local_cold': results['B_local_cold']['median_ms'] / a, 'speedup_vs_local_warm': results['B_local_warm']['median_ms'] / a, 'switch_as_frac_of_forward': a / results['C_forward_pass_ctx']['median_ms'], 'switch_as_frac_of_generate': a / results['C_generate']['median_ms']}
    if nfs_files:
        results['derived']['speedup_vs_nfs_cold'] = results['B_nfs_cold']['median_ms'] / a
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f'[lat] wrote {out}')
    print(json.dumps(results['derived'], indent=2))
    return 0
if __name__ == '__main__':
    sys.exit(main())
