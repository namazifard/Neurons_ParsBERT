import os
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# 21 PUD languages
LANGS_DEFAULT = ["ar","zh","cs","en","fi","fr","de","hi","id","it",
         "ja","ko","pl","pt","ru","es","sv","th","tr","is","gl"]

# PUD TXT splits (already created by split_pud_txt_by_sentences.py)
TXT_TRAIN_DIR = Path("data/pud/splits/txt/train")
TXT_EVAL_DIR  = Path("data/pud/splits/txt/eval")

# Output dirs that LAPE expects
OUT_VALID = Path("data/pud/valid")
OUT_TRAIN = Path("data/pud/train")

# Common short tags → HF model ids. Extend as you like.
MODELS = {
    # LLaMA family
    "Llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "Llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "Llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "Llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B",

    "PersianLlama-13b": "ViraIntelligentDataMining/PersianLLaMA-13B-Instruct",
    "Maral-7b": "MaralGPT/Maral-7B-alpha-1",
    "PersianMind-7b": "universitytehran/PersianMind-v1.0",
    "Dorna2-8b": "PartAI/Dorna2-Llama3.1-8B-Instruct",

    "ParsBERT": "HooshvareLab/bert-base-parsbert-uncased",

    # Mistral family
    "Mistral-7B": "mistralai/Mistral-7B-v0.1",
    "Mixtral-8x7B": "mistralai/Mixtral-8x7B-v0.1",

    "Bloom-7b1": "bigscience/bloom-7b1",

    # Gemma family (adjust if you use different variants)
    "Gemma-2-9b": "google/gemma-2-9b",
    "Gemma-2-27b": "google/gemma-2-27b",

    # XLM-R / mBERT tokenizers can also be useful
    "XLM-R-base": "xlm-roberta-base",
    "XLM-R-large": "xlm-roberta-large",
    "mBERT": "bert-base-multilingual-cased",
}

def resolve_model(tag_or_id: str) -> tuple[str, str]:
    """
    Returns (tag, model_id).
    - If input contains '/', treat it as a full HF id and invent a safe tag.
    - Else, look up in MODELS dict.
    """
    if "/" in tag_or_id:
        model_id = tag_or_id
        # make a filesystem/tag-safe name from the tail of the repo id
        tail = model_id.split("/")[-1]
        tag = tail.replace(".", "-")
        return tag, model_id
    if tag_or_id in MODELS:
        return tag_or_id, MODELS[tag_or_id]
    raise ValueError(
        f"Unknown tag '{tag_or_id}'. Use one of {list(MODELS.keys())} "
        f"or pass a full HuggingFace model id like 'meta-llama/Meta-Llama-3-8B'."
    )

def tokenize_lines_to_ids(lines, tok, cap_tokens=None, add_bos=False, add_eos=False):
    """
    Tokenize an iterable of strings to a flat list of ids, up to cap_tokens if set.
    Optional BOS/EOS if your tokenizer defines them.
    """
    ids_out, total = [], 0
    bos_id = tok.bos_token_id if add_bos and tok.bos_token_id is not None else None
    eos_id = tok.eos_token_id if add_eos and tok.eos_token_id is not None else None

    for s in lines:
        s = s.strip()
        if not s:
            continue
        ids = tok(s, add_special_tokens=False).input_ids
        if bos_id is not None:
            ids = [bos_id] + ids
        if eos_id is not None:
            ids = ids + [eos_id]

        if cap_tokens is None:
            ids_out.extend(ids)
        else:
            room = cap_tokens - total
            if room <= 0:
                break
            if len(ids) <= room:
                ids_out.extend(ids)
                total += len(ids)
            else:
                ids_out.extend(ids[:room])
                total += room
                break
    return ids_out

def main():
    p = argparse.ArgumentParser(description="Prepare PUD train/eval token tensors for LAPE.")
    p.add_argument("--tag", required=True,
                   help="Model tag (e.g., Llama-2-7b) OR a full HF id (e.g., meta-llama/Meta-Llama-3-8B).")
    p.add_argument("--langs", nargs="+", default=LANGS_DEFAULT,
                   help=f"Languages to process (default: {LANGS_DEFAULT}).")
    p.add_argument("--cap-tokens", type=int, default=None,
                   help="Optional cap on total tokens per split per lang (useful for quick smoke tests).")
    p.add_argument("--add-bos", action="store_true", help="Prepend BOS if tokenizer defines it.")
    p.add_argument("--add-eos", action="store_true", help="Append EOS if tokenizer defines it.")
    p.add_argument("--hf-cache", default=os.path.expanduser("~/hf_cache"),
                   help="HF cache dir (default: ~/hf_cache).")
    args = p.parse_args()

    # Resolve tag → model id or accept full id
    tag, model_id = resolve_model(args.tag)
    print(f"Using model_id='{model_id}' (tag='{tag}')")

    # Set HF caches
    os.environ["HF_HOME"] = args.hf_cache
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Create outputs
    OUT_VALID.mkdir(parents=True, exist_ok=True)
    OUT_TRAIN.mkdir(parents=True, exist_ok=True)

    # Process languages
    for lang in args.langs:
        train_txt = TXT_TRAIN_DIR / f"{lang}.txt"
        eval_txt  = TXT_EVAL_DIR  / f"{lang}.txt"
        if not train_txt.exists() or not eval_txt.exists():
            print(f"⚠️  Missing PUD TXT for '{lang}' ({train_txt} / {eval_txt}), skipping.")
            continue

        train_lines = train_txt.read_text(encoding="utf-8").splitlines()
        eval_lines  = eval_txt.read_text(encoding="utf-8").splitlines()
        print(f"→ {lang}: {len(train_lines)} train + {len(eval_lines)} eval sentences")

        eval_ids  = tokenize_lines_to_ids(
            tqdm(eval_lines,  desc=f"eval  {lang}"),
            tok, cap_tokens=args.cap_tokens, add_bos=args.add_bos, add_eos=args.add_eos
        )
        train_ids = tokenize_lines_to_ids(
            tqdm(train_lines, desc=f"train {lang}"),
            tok, cap_tokens=args.cap_tokens, add_bos=args.add_bos, add_eos=args.add_eos
        )

        eval_out  = OUT_VALID / f"id.{lang}.eval.{tag}.pud"
        train_out = OUT_TRAIN / f"id.{lang}.train.{tag}.pud"

        torch.save(torch.tensor(eval_ids,  dtype=torch.long), eval_out)
        torch.save(torch.tensor(train_ids, dtype=torch.long), train_out)

        print(f"✅ {lang}: saved {len(eval_ids)} eval + {len(train_ids)} train tokens")
        print(f"   ↳ {eval_out}")
        print(f"   ↳ {train_out}")

if __name__ == "__main__":
    main()