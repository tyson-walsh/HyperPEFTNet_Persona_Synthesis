from __future__ import annotations
import argparse
import hashlib
import logging
import os
import pickle
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
DEFAULT_PARQUET = './data/train_data_10000.parquet'
DEFAULT_REMOTE_TOKENIZER_ID = 'EleutherAI/pythia-160m'
DEFAULT_FALLBACK_TOKENIZER = './base_models/pythia_160m_clean'
DEFAULT_HF_TOKEN_FILE = './HF_TOKEN.txt'

def _read_token_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        tok = p.read_text(encoding='utf-8').strip()
        return tok or None
    except Exception:
        return None

def _resolve_hf_token(hf_token: Optional[str], hf_token_file: Optional[str]) -> Optional[str]:
    if hf_token and hf_token.strip():
        return hf_token.strip()
    env_tok = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
    if env_tok and env_tok.strip():
        return env_tok.strip()
    file_tok = _read_token_file(hf_token_file)
    if file_tok and file_tok.strip():
        return file_tok.strip()
    return None

def _maybe_enable_online(online: bool) -> None:
    if online:
        os.environ['HF_HUB_OFFLINE'] = '0'
        os.environ['TRANSFORMERS_OFFLINE'] = '0'

def _ensure_pad_token(tok: PreTrainedTokenizer) -> PreTrainedTokenizer:
    if tok.pad_token_id is None:
        tok.add_special_tokens({'pad_token': '[PAD]'})
    return tok

def _load_tokenizer_once(model_or_path: str, *, online: bool, hf_token: Optional[str]) -> PreTrainedTokenizer:
    local_only = not online
    try:
        tok = AutoTokenizer.from_pretrained(model_or_path, local_files_only=local_only, token=hf_token, trust_remote_code=True, use_fast=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(model_or_path, local_files_only=local_only, use_auth_token=hf_token, trust_remote_code=True, use_fast=True)
    return _ensure_pad_token(tok)

def _load_tokenizer_robust(primary: str, *, online: bool, hf_token: Optional[str], fallback: str) -> PreTrainedTokenizer:
    try:
        return _load_tokenizer_once(primary, online=online, hf_token=hf_token)
    except Exception as e1:
        if online:
            try:
                return _load_tokenizer_once(primary, online=False, hf_token=hf_token)
            except Exception:
                pass
        try:
            tok_fb = _load_tokenizer_once(fallback, online=False, hf_token=None)
        except Exception as e2:
            raise RuntimeError(f'Tokenizer load failed.\n\nPrimary:  {primary}\nFallback: {fallback}\n\nPrimary error:\n{type(e1).__name__}: {e1}\n\nFallback error:\n{type(e2).__name__}: {e2}\n\nFixes:\n• For gated repos: ensure you have access and provide a token via --hf_token or HF_TOKEN env var.\n• Or provide a local tokenizer path via --tokenizer.\n• For k8s: mount a Secret as env (HF_TOKEN) or as a file and pass --hf_token_file.\n')
        print(textwrap.dedent(f'\n                [warn] Primary tokenizer failed; falling back to local tokenizer.\n                       Primary : {primary}\n                       Fallback: {fallback}\n                       Error   : {type(e1).__name__}: {e1}\n                ').strip())
        return tok_fb
_DEFAULT_PRETOK_CACHE_DIR = './data/pretok_cache'
_PRETOK_CACHE_VERSION = 'v1'
_ITEMS_CACHE_VERSION = 'v1'

def _items_fingerprint(df, max_length, mode, prompt_eos, add_special_tokens, tokenizer) -> str:
    h = hashlib.sha256()
    h.update(f'iver={_ITEMS_CACHE_VERSION}|rows={len(df)}|ml={int(max_length)}|mode={mode}|peos={int(bool(prompt_eos))}|ast={int(bool(add_special_tokens))}'.encode())
    tok_name = getattr(tokenizer, 'name_or_path', '') or ''
    tok_vocab = int(getattr(tokenizer, 'vocab_size', 0) or 0)
    h.update(f'tok={tok_name}|vocab={tok_vocab}'.encode())
    try:
        h.update(f"gid_sum={int(df['gid'].sum())}|uid_sum={int(df['target_user_id'].sum())}".encode())
    except Exception:
        pass
    return h.hexdigest()[:16]

def _load_or_build_items_cache(*, df, max_length, mode, prompt_eos, add_special_tokens, tokenizer, build_fn):
    cache_dir = Path(os.environ.get('HN_PRETOK_CACHE_DIR', _DEFAULT_PRETOK_CACHE_DIR))
    fp = _items_fingerprint(df, max_length, mode, prompt_eos, add_special_tokens, tokenizer)
    cache_file = cache_dir / f'items_{fp}_rows{len(df)}_ml{int(max_length)}_{mode}.pkl'
    if cache_file.exists():
        t0 = time.time()
        logging.info('[items-cache] HIT  %s', cache_file)
        try:
            with open(cache_file, 'rb') as f:
                items = pickle.load(f)
            logging.info('[items-cache] loaded %d items in %.2fs', len(items), time.time() - t0)
            return items
        except Exception as e:
            logging.warning('[items-cache] load failed (%s); rebuilding', e)
    logging.info('[items-cache] MISS %s', cache_file)
    t0 = time.time()
    items = build_fn()
    logging.info('[items-cache] built %d items in %.2fs', len(items), time.time() - t0)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix('.pkl.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_file)
        size_mb = cache_file.stat().st_size / 1000000.0
        logging.info('[items-cache] saved -> %s (%.1f MB)', cache_file, size_mb)
    except Exception as e:
        logging.warning('[items-cache] save failed (%s); continuing without cache', e)
    return items

def _pretok_fingerprint(items, tokenizer, max_length, mode, prompt_eos, add_special_tokens) -> str:
    h = hashlib.sha256()
    h.update(f'ver={_PRETOK_CACHE_VERSION}|n={len(items)}|ml={int(max_length)}|mode={mode}|peos={int(bool(prompt_eos))}|ast={int(bool(add_special_tokens))}'.encode())
    tok_name = getattr(tokenizer, 'name_or_path', '') or ''
    tok_vocab = int(getattr(tokenizer, 'vocab_size', 0) or 0)
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    h.update(f'tok={tok_name}|vocab={tok_vocab}|pad={pad_id}|eos={eos_id}'.encode())
    if items:
        gids_sum = sum((int(it['gid']) for it in items))
        uids_sum = sum((int(it['target_user_id']) for it in items))
        h.update(f'gids_sum={gids_sum}|uids_sum={uids_sum}'.encode())
        for j in (0, len(items) // 2, len(items) - 1):
            it = items[j]
            h.update(f"s{j}:gid={it['gid']},uid={it['target_user_id']},ctx_len={len(it['context_text'])},tgt_len={len(it['target_text'])}".encode())
    return h.hexdigest()[:16]

def _load_or_build_pretok_cache(*, encode_fn, items, tokenizer, max_length, mode, prompt_eos, add_special_tokens):
    cache_dir = Path(os.environ.get('HN_PRETOK_CACHE_DIR', _DEFAULT_PRETOK_CACHE_DIR))
    fp = _pretok_fingerprint(items, tokenizer, max_length, mode, prompt_eos, add_special_tokens)
    cache_file = cache_dir / f'pretok_{fp}_n{len(items)}_ml{int(max_length)}_{mode}.pt'
    if cache_file.exists():
        t0 = time.time()
        logging.info('[pretok-cache] HIT  %s', cache_file)
        try:
            data = torch.load(cache_file, map_location='cpu', weights_only=False)
            if isinstance(data, list) and len(data) == len(items):
                logging.info('[pretok-cache] loaded %d samples in %.2fs', len(data), time.time() - t0)
                return data
            logging.warning('[pretok-cache] length mismatch (cache=%d items=%d); rebuilding', len(data), len(items))
        except Exception as e:
            logging.warning('[pretok-cache] load failed (%s); rebuilding', e)
    logging.info('[pretok-cache] MISS %s', cache_file)
    t0 = time.time()
    cached = [encode_fn(it) for it in items]
    logging.info('[pretok-cache] tokenized %d samples in %.2fs', len(cached), time.time() - t0)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix('.pt.tmp')
        torch.save(cached, tmp)
        os.replace(tmp, cache_file)
        size_mb = cache_file.stat().st_size / 1000000.0
        logging.info('[pretok-cache] saved -> %s (%.1f MB)', cache_file, size_mb)
    except Exception as e:
        logging.warning('[pretok-cache] save failed (%s); continuing without cache', e)
    return cached

class RedditConversationDataset10000(Dataset):

    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer, *, max_length: int=512, add_special_tokens: bool=True, pretokenize: bool=False, mode: str='causal', prompt_eos: bool=True):
        required = {'gid', 'group_label', 'text', 'target_user_id'}
        if not required.issubset(dataframe.columns):
            raise ValueError(f'DataFrame must contain {required} - found {dataframe.columns.tolist()}')
        mode = str(mode).strip().lower()
        if mode not in {'causal', 'seq2seq'}:
            raise ValueError("mode must be 'causal' or 'seq2seq'")
        df = dataframe.copy()

        def _turn_key(gl: str) -> int:
            gl = str(gl)
            if gl.endswith('_target'):
                return 10 ** 9
            try:
                return int(gl.rsplit('_', 1)[-1])
            except Exception:
                return 10 ** 8

        def _build_items() -> List[dict]:
            _df = df.copy()
            _df['_turn'] = _df['group_label'].map(_turn_key)
            _df.sort_values(['gid', '_turn'], inplace=True, ignore_index=True)
            _items: List[dict] = []
            for (gid, sub) in _df.groupby('gid', sort=False):
                tgt = sub[sub['group_label'].astype(str).str.endswith('_target')]
                if len(tgt) != 1:
                    continue
                final_row = tgt.iloc[0]
                tgt_text = str(final_row['text']).strip()
                if not tgt_text:
                    continue
                ctx_rows = sub[~sub['group_label'].astype(str).str.endswith('_target')]
                if ctx_rows.empty:
                    continue
                ctx_text = '\n'.join(ctx_rows['text'].astype(str).tolist()).strip()
                if not ctx_text:
                    continue
                _items.append(dict(gid=int(gid), target_user_id=int(final_row['target_user_id']), context_text=ctx_text, target_text=tgt_text))
            return _items
        items = _load_or_build_items_cache(df=df, max_length=int(max_length), mode=mode, prompt_eos=prompt_eos, add_special_tokens=add_special_tokens, tokenizer=tokenizer, build_fn=_build_items)
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.add_special_tokens = bool(add_special_tokens)
        self.mode = mode
        self.prompt_eos = bool(prompt_eos)
        if pretokenize:
            self.cached_data = _load_or_build_pretok_cache(encode_fn=self._encode, items=items, tokenizer=tokenizer, max_length=self.max_length, mode=self.mode, prompt_eos=self.prompt_eos, add_special_tokens=self.add_special_tokens)
        else:
            self.cached_data = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.cached_data[idx] if self.cached_data is not None else self._encode(self.items[idx])

    def _encode(self, item: dict) -> Dict[str, torch.Tensor | int]:
        ctx = str(item['context_text'])
        tgt = str(item['target_text'])
        if self.mode == 'seq2seq':
            enc_ctx = self.tokenizer(ctx, truncation=True, max_length=self.max_length, padding='max_length', add_special_tokens=self.add_special_tokens, return_tensors='pt')
            enc_tgt = self.tokenizer(tgt, truncation=True, max_length=self.max_length, padding='max_length', add_special_tokens=self.add_special_tokens, return_tensors='pt')
            labels = enc_tgt['input_ids'].squeeze(0)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100
            return {'input_ids': enc_ctx['input_ids'].squeeze(0), 'attention_mask': enc_ctx['attention_mask'].squeeze(0), 'labels': labels, 'target_user_id': int(item['target_user_id']), 'gid': int(item['gid'])}
        eos = self.tokenizer.eos_token or ''
        prompt = ctx.strip()
        if self.prompt_eos and eos:
            prompt = prompt + '\n' + eos + '\n'
        else:
            prompt = prompt + '\n\n'
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=self.add_special_tokens)
        target_ids = self.tokenizer.encode(tgt, add_special_tokens=False)
        if len(prompt_ids) + len(target_ids) > self.max_length:
            if len(target_ids) >= self.max_length:
                target_ids = target_ids[:self.max_length]
                prompt_ids = []
            else:
                max_prompt = self.max_length - len(target_ids)
                if max_prompt <= 0:
                    prompt_ids = []
                else:
                    prompt_ids = prompt_ids[-max_prompt:]
        input_ids = prompt_ids + target_ids
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
            if pad_id is None:
                raise RuntimeError('Tokenizer has neither pad_token_id nor eos_token_id; cannot pad.')
        labels = [-100] * len(prompt_ids) + target_ids
        attn = [1] * len(input_ids)
        if len(input_ids) < self.max_length:
            pad_n = self.max_length - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_n
            attn = attn + [0] * pad_n
            labels = labels + [-100] * pad_n
        else:
            input_ids = input_ids[:self.max_length]
            attn = attn[:self.max_length]
            labels = labels[:self.max_length]
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long), 'attention_mask': torch.tensor(attn, dtype=torch.long), 'labels': torch.tensor(labels, dtype=torch.long), 'target_user_id': int(item['target_user_id']), 'gid': int(item['gid'])}

def _stats(arr: List[int]) -> Dict[str, float | int]:
    return dict(mean=float(sum(arr) / len(arr)), median=int(median(arr)), max=int(max(arr)))

def _smoke_test(df: pd.DataFrame, ds: RedditConversationDataset10000, tok: PreTrainedTokenizer) -> None:
    n_samples = len(ds)
    n_users = df['target_user_id'].nunique()
    per_user = Counter(map(int, df['target_user_id']))
    counts = list(per_user.values())
    (lens_ctx, lens_tgt) = ([], [])
    step = max(1, n_samples // 20000)
    for i in range(0, n_samples, step):
        item = ds[i]
        lens_ctx.append(int(item['attention_mask'].sum()))
        lens_tgt.append(int((item['labels'] != -100).sum()))
    buckets = defaultdict(int)
    for L in lens_ctx:
        if L <= 64:
            buckets['<=64'] += 1
        elif L <= 128:
            buckets['<=128'] += 1
        elif L <= 256:
            buckets['<=256'] += 1
        else:
            buckets['>256'] += 1
    print(textwrap.dedent(f"\n            ------------------------ Smoke-test summary ------------------------\n            Samples                      : {n_samples:,}\n            Users                        : {n_users:,}\n            Samples / user               : min={(min(counts) if counts else 0)}, median={(median(counts) if counts else 0)}, max={(max(counts) if counts else 0)}\n            Context token lengths        : {_stats(lens_ctx)}\n            Target  token lengths        : {_stats(lens_tgt)}\n            Context length buckets       : {dict(buckets)}\n            Empty text rows (should be 0): {int(df['text'].astype(str).str.strip().eq('').sum())}\n            NaN text rows   (should be 0): {int(df['text'].isna().sum())}\n            Tokenizer pad id             : {tok.pad_token_id}\n            Sequence length (max_length) : {ds.max_length}\n            -------------------------------------------------------------------\n            ").strip())

def main() -> None:
    tic = time.time()
    ap = argparse.ArgumentParser(description='Run a quick integrity smoke-test on a 10,000-user Reddit split.')
    ap.add_argument('--parquet', type=str, default=DEFAULT_PARQUET)
    ap.add_argument('--tokenizer', type=str, default=DEFAULT_REMOTE_TOKENIZER_ID, help='Tokenizer path or HF model ID (can be gated).')
    ap.add_argument('--fallback_tokenizer', type=str, default=DEFAULT_FALLBACK_TOKENIZER, help='Local tokenizer path used if --tokenizer fails.')
    ap.add_argument('--hf_token', type=str, default=None, help='HF token (or set HF_TOKEN env). Required for gated repos unless cached.')
    ap.add_argument('--hf_token_file', type=str, default=DEFAULT_HF_TOKEN_FILE, help='Path to a token file inside the runtime (works if mounted/baked into image).')
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--pretok', action='store_true')
    ap.add_argument('--online', action=argparse.BooleanOptionalAction, default=True, help='Enable/disable fetching from HF Hub (default: --online).')
    ap.add_argument('-f', '--f', help=argparse.SUPPRESS)
    args = ap.parse_args()
    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f'Parquet file not found: {parquet_path}')
    _maybe_enable_online(bool(args.online))
    hf_token = _resolve_hf_token(args.hf_token, args.hf_token_file)
    print(f'Loading DataFrame → {parquet_path}')
    df_raw = pd.read_parquet(parquet_path)
    print(f'Rows: {len(df_raw):,} | Columns: {df_raw.columns.tolist()}')
    tok = _load_tokenizer_robust(args.tokenizer, online=bool(args.online), hf_token=hf_token, fallback=str(args.fallback_tokenizer))
    ds = RedditConversationDataset10000(dataframe=df_raw, tokenizer=tok, max_length=args.max_len, pretokenize=args.pretok)
    _smoke_test(df_raw, ds, tok)
    print(f'Smoke-test finished in {time.time() - tic:.1f} s')
if __name__ == '__main__':
    main()
