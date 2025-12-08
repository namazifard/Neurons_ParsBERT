#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer


LANGS21 = [
    "ar", "zh", "cs", "en", "fi", "fr", "de", "hi", "id", "it",
    "ja", "ko", "pl", "pt", "ru", "es", "sv", "th", "tr", "is", "gl"
]

LANGS8 = ["ar", "en", "fr", "de", "hi",  "ja", "ru", "tr"]

def parse_feats(feats_str: str):
    """Parse FEATS column from UD CoNLL-U into a dict."""
    feats = {}
    feats_str = feats_str.strip()
    if feats_str == "_" or feats_str == "":
        return feats
    for item in feats_str.split("|"):
        if "=" in item:
            k, v = item.split("=", 1)
            feats[k] = v
    return feats


def read_conllu_tokens(path: Path):
    """
    Read a CoNLL-U file and yield sentences as lists of
    (form, upos, feats_dict).
    Skips multiword tokens (3-4) and empty nodes (1.1).
    """
    sentences = []
    current = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) != 10:
                continue

            tid, form, lemma, upos, xpos, feats_str, head, deprel, deps, misc = cols

            # skip multi-word tokens and empty nodes
            if "-" in tid or "." in tid:
                continue

            feats = parse_feats(feats_str)
            current.append((form, upos, feats))

    if current:
        sentences.append(current)

    return sentences


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build category-based ID tensors for ParsBERT: "
            "UPOS-*, Case-*, Gender-* across all PUD languages."
        )
    )
    ap.add_argument(
        "--model-id",
        default="HooshvareLab/bert-base-parsbert-uncased",
        help="HF model id for ParsBERT tokenizer.",
    )
    ap.add_argument(
        "--tag",
        default="ParsBERT",
        help="Short tag used in output filenames (e.g. ParsBERT).",
    )
    ap.add_argument(
        "--pud-dir",
        default="data/pud/conllu",
        help="Directory containing {lang}.conllu PUD files.",
    )
    ap.add_argument(
        "--out-dir",
        default="data/pud/train",
        help="Output directory for id.*.train.{tag}.pud files.",
    )
    ap.add_argument(
        "--langs",
        nargs="+",
        default=LANGS8,
        help=f"Languages to process (default: {LANGS8})",
    )
    args = ap.parse_args()

    pud_dir = Path(args.pud_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    # These dicts will collect token ids for each category across *all* languages
    upos_to_ids = defaultdict(list)     # e.g. "NOUN" -> [id1, id2, ...]
    case_to_ids = defaultdict(list)     # e.g. "Nom"  -> [...]
    gender_to_ids = defaultdict(list)   # e.g. "Masc" -> [...]

    # To see how many tokens per category we have
    upos_counts = defaultdict(int)
    case_counts = defaultdict(int)
    gender_counts = defaultdict(int)

    for lang in args.langs:
        conllu_path = pud_dir / f"{lang}.conllu"
        if not conllu_path.exists():
            print(f"⚠️ Missing CoNLL-U for {lang}: {conllu_path} (skipping)")
            continue

        print(f"→ Processing {lang} from {conllu_path} ...")
        sentences = read_conllu_tokens(conllu_path)

        for sent in sentences:
            # sent is a list of (form, upos, feats)
            for form, upos, feats in sent:
                # Tokenize with ParsBERT tokenizer, *no* special tokens
                sub = tok(
                    form,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                ids = sub["input_ids"]
                if len(ids) == 0:
                    continue

                # UPOS category (always available in PUD)
                if upos != "_" and upos is not None:
                    upos_to_ids[upos].extend(ids)
                    upos_counts[upos] += len(ids)

                # Case feature
                case = feats.get("Case", None)
                if case is not None:
                    case_to_ids[case].extend(ids)
                    case_counts[case] += len(ids)

                # Gender feature
                gender = feats.get("Gender", None)
                if gender is not None:
                    gender_to_ids[gender].extend(ids)
                    gender_counts[gender] += len(ids)

    # ---- Save everything as .pud tensors ----
    def save_category_tensors(cat_dict, prefix: str):
        """
        cat_dict: mapping category_name -> list of token ids
        prefix: e.g. "UPOS" / "Case" / "Gender"
        """
        for cat, ids in cat_dict.items():
            if not ids:
                continue
            tensor = torch.tensor(ids, dtype=torch.long)
            fname = f"id.{prefix}-{cat}.train.{args.tag}.pud"
            out_path = out_dir / fname
            torch.save(tensor, out_path)
            print(f"   Saved {prefix}-{cat}: {len(ids)} tokens -> {out_path}")

    print("\nUPOS counts (token pieces):")
    for k in sorted(upos_counts.keys()):
        print(f"  {k:10s}: {upos_counts[k]}")

    print("\nCase counts (token pieces):")
    for k in sorted(case_counts.keys()):
        print(f"  {k:10s}: {case_counts[k]}")

    print("\nGender counts (token pieces):")
    for k in sorted(gender_counts.keys()):
        print(f"  {k:10s}: {gender_counts[k]}")

    print("\nSaving tensors...")
    save_category_tensors(upos_to_ids, "UPOS")
    save_category_tensors(case_to_ids, "Case")
    save_category_tensors(gender_to_ids, "Gender")

    print("\n✅ Done. Category-based ID tensors written to:", out_dir.resolve())

if __name__ == "__main__":
    main()
