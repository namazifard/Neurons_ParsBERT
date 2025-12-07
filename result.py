#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# HF imports (only used in bert-mlm backend)
from transformers import AutoModelForMaskedLM, AutoTokenizer

# vLLM imports (only used in llama-vllm backend)
try:
    from vllm import LLM as VLLM, SamplingParams
    import torch.nn.functional as F
except ImportError:
    VLLM = None
    SamplingParams = None
    F = None


# ----------------------- small helpers -----------------------
def rprint(*args):
    print(*args, flush=True)


def label_bars(ax, fmt="{:d}", fontsize=9, dy=0.01):
    """Write value labels above each bar in a bar plot."""
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h):
            continue
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            h + dy * span,
            fmt.format(int(round(h))),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def heat_no_cbar(mat, row_labels, col_labels, title, outpath, fmt="0.2f"):
    plt.figure(figsize=(8, 5))
    _ = plt.imshow(mat, aspect="auto")
    # no colorbar
    plt.xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ----------------------- neuron helpers (shared) -----------------------
def load_neurons(neuron_path: Path):
    """
    Expect a combined *.neuron.pth with structure:
      neurons[lang_idx][layer_idx] = 1D LongTensor of neuron indices
    (Same format as your identify_hf.py / identify.py)
    """
    raw = torch.load(str(neuron_path))
    norm = []
    for lang_blocks in raw:
        lang_list = []
        for t in lang_blocks:
            if t is None:
                lang_list.append(np.array([], dtype=np.int64))
            else:
                if hasattr(t, "cpu"):
                    arr = t.cpu().numpy()
                else:
                    arr = np.array(t)
                lang_list.append(np.array(arr, dtype=np.int64))
        norm.append(lang_list)
    return norm  # [num_langs][num_layers] -> np.array of indices


def per_language_per_layer_counts(neurons):
    L = len(neurons)
    num_layers = len(neurons[0])
    counts = np.zeros((L, num_layers), dtype=int)
    for i in range(L):
        for j in range(num_layers):
            counts[i, j] = int(neurons[i][j].size)
    return counts


def support_histograms(neurons):
    """
    For each layer: map neuron index -> number of languages containing it.
    Returns:
      per_layer_hist: list of dicts: layer -> {k: count_with_support_k}
      overall_counts: dict k -> count across all layers
      pair_intersections & jaccard aggregated across layers
    """
    L = len(neurons)
    num_layers = len(neurons[0])

    per_layer_hist = []
    overall_counts = {}
    pair_intersections = np.zeros((L, L), dtype=int)
    pair_unions = np.zeros((L, L), dtype=int)

    for j in range(num_layers):
        idx2langs = {}
        layer_sets = [set(neurons[i][j].tolist()) for i in range(L)]
        for i in range(L):
            for idx in layer_sets[i]:
                s = idx2langs.get(idx)
                if s is None:
                    idx2langs[idx] = {i}
                else:
                    s.add(i)
        hist = {}
        for _, langs_set in idx2langs.items():
            k = len(langs_set)
            hist[k] = hist.get(k, 0) + 1
            overall_counts[k] = overall_counts.get(k, 0) + 1
        per_layer_hist.append(hist)

        for a in range(L):
            for b in range(L):
                inter = len(layer_sets[a].intersection(layer_sets[b]))
                union = len(layer_sets[a].union(layer_sets[b]))
                pair_intersections[a, b] += inter
                pair_unions[a, b] += union

    pair_jaccard = np.zeros_like(pair_intersections, dtype=float)
    nz = pair_unions != 0
    pair_jaccard[nz] = pair_intersections[nz] / pair_unions[nz]
    return per_layer_hist, overall_counts, pair_intersections, pair_jaccard


def to_full_hist_row(hist_dict, max_k):
    """Convert sparse dict {k:count} to dense list length max_k (1..max_k)."""
    return [int(hist_dict.get(k, 0)) for k in range(1, max_k + 1)]


def make_neuron_plots(langs, neurons, outdir: Path):
    """
    Make all the neuron distribution/support plots, given:
      langs: list of language codes
      neurons: [num_langs][num_layers] arrays of indices
    """
    L = len(langs)
    num_layers = len(neurons[0])
    layer_labels_csv = [f"layer{j+1}" for j in range(num_layers)]
    xpos = np.arange(1, num_layers + 1)
    layer_ticks = [str(j) for j in xpos]

    # 1) per-language per-layer counts
    counts = per_language_per_layer_counts(neurons[:L])  # (L, num_layers)
    counts_df = pd.DataFrame(counts, index=langs, columns=layer_labels_csv)
    counts_df.to_csv(outdir / "language_layer_counts.csv")

    layer_totals = counts.sum(axis=0)
    lang_totals = counts.sum(axis=1)
    pd.Series(layer_totals, index=layer_labels_csv).to_csv(outdir / "layer_totals.csv")
    pd.Series(lang_totals, index=langs).to_csv(outdir / "language_totals.csv")

    # totals per layer
    plt.figure(figsize=(12, 4))
    plt.bar(xpos, layer_totals)
    plt.xticks(xpos, layer_ticks, rotation=0)
    plt.xlabel("Layer"); plt.ylabel("# Neurons (sum across languages)")
    plt.title("Language-specific neurons per layer (total)")
    plt.tight_layout(); plt.savefig(outdir / "layer_totals_bar.png", dpi=200); plt.close()

    # totals per language
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.bar(np.arange(L), lang_totals)
    ax.set_xticks(np.arange(L))
    ax.set_xticklabels(langs, rotation=0)
    ax.set_xlabel("Language"); ax.set_ylabel("# Neurons")
    ax.set_title("Distribution of neurons across languages")
    label_bars(ax, fmt="{:d}")
    plt.tight_layout(); plt.savefig(outdir / "language_totals_bar.png", dpi=200); plt.close()

    # stacked per layer
    plt.figure(figsize=(12, 5))
    bottom = np.zeros(num_layers)
    for i, lg in enumerate(langs):
        plt.bar(xpos, counts[i], bottom=bottom, label=lg)
        bottom += counts[i]
    plt.xticks(xpos, layer_ticks, rotation=0)
    plt.xlabel("Layer"); plt.ylabel("# Neurons")
    plt.title("Per-language neurons per layer (stacked)")
    plt.legend(ncol=min(6, L), fontsize=8)
    plt.tight_layout(); plt.savefig(outdir / "per_language_stacked_bar.png", dpi=200); plt.close()

    # per-language bars per layer
    for i, lg in enumerate(langs):
        plt.figure(figsize=(12, 4))
        plt.bar(xpos, counts[i])
        plt.xticks(xpos, layer_ticks, rotation=0)
        plt.xlabel("Layer"); plt.ylabel("# Neurons")
        plt.title(f"{lg}: neurons per layer")
        plt.tight_layout(); plt.savefig(outdir / f"per_layer_{lg}_bar.png", dpi=200); plt.close()

    # 2) support histograms
    per_layer_hist, overall_counts, pair_intersections, pair_jaccard = support_histograms(neurons[:L])
    max_k = L

    overall_row = to_full_hist_row(overall_counts, max_k)
    overall_df = pd.DataFrame({"k": list(range(1, max_k + 1)), "count": overall_row})
    overall_df.to_csv(outdir / "support_hist_overall.csv", index=False)

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.bar(np.arange(1, max_k + 1), overall_row)
    ax.set_xlabel("k (languages neuron appears in)"); ax.set_ylabel("# Neurons")
    ax.set_title("Neuron support size (overall)")
    label_bars(ax, fmt="{:d}")
    plt.tight_layout(); plt.savefig(outdir / "support_hist_overall_bar.png", dpi=200); plt.close()

    per_layer_dense = np.vstack([to_full_hist_row(h, max_k) for h in per_layer_hist])  # (num_layers, max_k)
    per_layer_df = pd.DataFrame(per_layer_dense, columns=[f"k={k}" for k in range(1, max_k + 1)])
    per_layer_df.insert(0, "layer", layer_labels_csv)
    per_layer_df.to_csv(outdir / "support_hist_per_layer.csv", index=False)

    for j in range(num_layers):
        row = per_layer_dense[j]
        plt.figure(figsize=(8, 3.2))
        plt.bar(np.arange(1, max_k + 1), row)
        plt.xlabel("k (languages neuron appears in)"); plt.ylabel("# Neurons")
        plt.title(f"Neuron support size per layer — {j+1}")
        plt.tight_layout(); plt.savefig(outdir / f"per_layer_support_hist_layer{j+1}_bar.png", dpi=200); plt.close()

    # multilingual neurons (k > 2, k > 3)
    k_gt_2_per_layer = per_layer_dense[:, 2:].sum(axis=1) if max_k >= 3 else np.zeros(num_layers, dtype=int)
    k_gt_3_per_layer = per_layer_dense[:, 3:].sum(axis=1) if max_k >= 4 else np.zeros(num_layers, dtype=int)

    pd.DataFrame({"layer": layer_labels_csv, "count": k_gt_2_per_layer}
                 ).to_csv(outdir / "multilingual_k_gt_2_per_layer.csv", index=False)
    pd.DataFrame({"layer": layer_labels_csv, "count": k_gt_3_per_layer}
                 ).to_csv(outdir / "multilingual_k_gt_3_per_layer.csv", index=False)

    plt.figure(figsize=(12, 4))
    plt.bar(xpos, k_gt_2_per_layer)
    plt.xticks(xpos, layer_ticks, rotation=0)
    plt.xlabel("Layer"); plt.ylabel("# Neurons")
    plt.title("Multilingual neurons per layer (k > 2 languages)")
    plt.tight_layout(); plt.savefig(outdir / "multilingual_k_gt_2_per_layer_bar.png", dpi=200); plt.close()

    plt.figure(figsize=(12, 4))
    plt.bar(xpos, k_gt_3_per_layer)
    plt.xticks(xpos, layer_ticks, rotation=0)
    plt.xlabel("Layer"); plt.ylabel("# Neurons")
    plt.title("Highly multilingual neurons per layer (k > 3 languages)")
    plt.tight_layout(); plt.savefig(outdir / "multilingual_k_gt_3_per_layer_bar.png", dpi=200); plt.close()

    # 3) pairwise intersections & Jaccard
    inter_df = pd.DataFrame(pair_intersections[:L, :L], index=langs, columns=langs)
    jac_df = pd.DataFrame(pair_jaccard[:L, :L], index=langs, columns=langs)
    inter_df.to_csv(outdir / "pairwise_intersection_counts.csv")
    jac_df.to_csv(outdir / "pairwise_jaccard.csv")

    heat_no_cbar(inter_df.values.astype(int), langs, langs,
                 "Pairwise intersections (aggregated)", outdir / "pairwise_intersections_heatmap.png", fmt="d")
    heat_no_cbar(jac_df.values.astype(float), langs, langs,
                 "Pairwise Jaccard (aggregated)", outdir / "pairwise_jaccard_heatmap.png", fmt="0.2f")

    # 4) cumulative "k or more"
    k_or_more = np.cumsum(overall_row[::-1])[::-1]
    k_or_more_df = pd.DataFrame({"k_or_more": list(range(1, max_k + 1)), "count": k_or_more})
    k_or_more_df.to_csv(outdir / "shared_neurons_k_or_more.csv", index=False)

    print(f"Multilingual totals (overall): k>2 = {int(k_gt_2_per_layer.sum())}, "
          f"k>3 = {int(k_gt_3_per_layer.sum())}")


# ----------------------- BERT MLM backend -----------------------
def compute_pseudo_ppl_mlm(
    model,
    mask_token_id: int,
    ids: torch.Tensor,
    max_len: int,
    batch_size_mask: int,
    device: torch.device,
):
    """
    Pseudo-PPL for an MLM:
    For each token position t, mask it, get p(x_t | x_{≠t}), accumulate -log p.
    We batch multiple masked positions per forward pass with batch_size_mask.
    """
    model.eval()

    L = ids.size(0)
    max_len = min(max_len, model.config.max_position_embeddings)
    L_trim = (L // max_len) * max_len
    if L_trim == 0:
        raise ValueError("Not enough tokens to form a single sequence of length max_len.")
    ids = ids[:L_trim]
    seqs = ids.view(-1, max_len)  # [num_seqs, max_len]

    total_log_prob = 0.0
    total_tokens = 0

    with torch.no_grad():
        for seq_idx in range(seqs.size(0)):
            seq = seqs[seq_idx].to(device)  # [max_len]
            positions = list(range(max_len))
            for start in range(0, max_len, batch_size_mask):
                pos_batch = positions[start : start + batch_size_mask]
                if not pos_batch:
                    continue

                bsz = len(pos_batch)
                batch_input = seq.unsqueeze(0).repeat(bsz, 1)  # [B, max_len]
                for i, pos in enumerate(pos_batch):
                    batch_input[i, pos] = mask_token_id

                attention_mask = torch.ones_like(batch_input, device=device)

                outputs = model(input_ids=batch_input, attention_mask=attention_mask)
                logits = outputs.logits  # [B, max_len, vocab]
                log_probs = logits.log_softmax(dim=-1)

                for i, pos in enumerate(pos_batch):
                    true_id = seq[pos].item()
                    lp = log_probs[i, pos, true_id].item()
                    total_log_prob += lp
                    total_tokens += 1

    avg_nats = -total_log_prob / total_tokens
    avg_bits = avg_nats / math.log(2.0)
    ppl = math.exp(avg_nats)
    return avg_nats, avg_bits, ppl, total_tokens


def register_ablation_hooks_bert(model, lang_masks, device):
    """
    lang_masks: list of length num_layers, where each element is a 1D tensor
                of neuron indices to ablate in that layer's intermediate output.
    Returns: list of hook handles.
    """
    hooks = []
    encoder_layers = model.bert.encoder.layer
    num_layers = len(encoder_layers)
    assert len(lang_masks) == num_layers, "Mask length must match number of layers."

    for layer_idx, layer in enumerate(encoder_layers):
        neuron_idx = lang_masks[layer_idx]
        if neuron_idx is None or len(neuron_idx) == 0:
            continue
        idx_tensor = neuron_idx.to(device)

        def make_hook(idx_tensor_inner):
            def hook(module, inputs, output):
                # output: [B, T, intermediate_size], after GELU
                if idx_tensor_inner.numel() == 0:
                    return output
                out = output.clone()
                out[..., idx_tensor_inner] = 0
                return out
            return hook

        h = layer.intermediate.register_forward_hook(make_hook(idx_tensor))
        hooks.append(h)

    return hooks


def run_bert_mlm_backend(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    neuron_path = Path(args.mask)

    # ---- model + tokenizer ----
    rprint(f"[BERT-MLM] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer has no [MASK] token (mask_token_id is None).")

    model = AutoModelForMaskedLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # ---- masks ----
    neurons = load_neurons(neuron_path)
    L_mask = len(neurons)
    langs = args.langs
    L = len(langs)
    if L_mask != L:
        rprint(f"⚠️ Mask has {L_mask} languages, --langs has {L}. Using first {min(L_mask, L)}.")
    L = min(L_mask, L)
    langs = langs[:L]
    neurons = neurons[:L]

    # ---- load PUD ids ----
    ids_by_lang = {}
    for lang in langs:
        if args.split == "train":
            pud_path = Path("data/pud/train") / f"id.{lang}.train.{args.tag}.pud"
        else:
            pud_path = Path("data/pud/valid") / f"id.{lang}.eval.{args.tag}.pud"
        if not pud_path.exists():
            rprint(f"⚠️ Skipping {lang}: missing {pud_path}")
            continue
        ids = torch.load(pud_path)
        ids_by_lang[lang] = ids
        rprint(f"Loaded {ids.numel()} tokens for lang={lang} from {pud_path}")

    langs = [lg for lg in langs if lg in ids_by_lang]
    L = len(langs)
    rprint(f"Languages with data: {langs}")

    # ---- baseline pseudo-PPL ----
    baseline_nats = np.zeros(L, dtype=float)
    baseline_bits = np.zeros(L, dtype=float)
    baseline_ppl = np.zeros(L, dtype=float)
    num_tokens = np.zeros(L, dtype=int)

    for i, lang in enumerate(langs):
        rprint(f"[BERT-MLM] Baseline pseudo-PPL: {lang} ({args.split})")
        ids = ids_by_lang[lang]
        avg_nats, avg_bits, ppl, n_tok = compute_pseudo_ppl_mlm(
            model=model,
            mask_token_id=mask_token_id,
            ids=ids,
            max_len=args.max_length,
            batch_size_mask=args.mask_batch_size,
            device=device,
        )
        baseline_nats[i] = avg_nats
        baseline_bits[i] = avg_bits
        baseline_ppl[i] = ppl
        num_tokens[i] = n_tok
        rprint(f"  avg_surprisal (nats): {avg_nats:.4f}  bits: {avg_bits:.4f}  PPL: {ppl:.2f}  (tokens={n_tok})")

    # ---- ablated pseudo-PPL matrix ----
    ablated_ppl = np.zeros((L, L), dtype=float)

    for mask_idx, mask_lang in enumerate(langs):
        rprint(f"[BERT-MLM] Ablation using neurons of mask_lang={mask_lang}")
        lang_masks = neurons[mask_idx]  # list[num_layers]
        hooks = register_ablation_hooks_bert(model, [torch.tensor(m) for m in lang_masks], device)
        try:
            for eval_idx, eval_lang in enumerate(langs):
                rprint(f"  → eval lang={eval_lang}")
                ids = ids_by_lang[eval_lang]
                avg_nats, avg_bits, ppl, _ = compute_pseudo_ppl_mlm(
                    model=model,
                    mask_token_id=mask_token_id,
                    ids=ids,
                    max_len=args.max_length,
                    batch_size_mask=args.mask_batch_size,
                    device=device,
                )
                ablated_ppl[mask_idx, eval_idx] = ppl
                rprint(f"    PPL: {ppl:.2f}")
        finally:
            for h in hooks:
                h.remove()

    # ---- write PPL CSVs & heatmaps ----
    ppl_df = pd.DataFrame(ablated_ppl, columns=langs)
    ppl_df.insert(0, "mask_lang", langs)
    ppl_df.to_csv(outdir / "ppl_results.csv", index=False)

    base_df = pd.DataFrame({
        "lang": langs,
        "baseline_avg_nats": baseline_nats,
        "baseline_avg_bits": baseline_bits,
        "baseline_ppl": baseline_ppl,
        "num_tokens": num_tokens,
    })
    base_df.to_csv(outdir / "baseline_pseudo_ppl.csv", index=False)

    delta = ablated_ppl - baseline_ppl[None, :]
    delta_df = pd.DataFrame(delta, index=langs, columns=langs)
    delta_df.to_csv(outdir / "delta_ppl.csv")

    heat_no_cbar(ablated_ppl, langs, langs,
                 "Ablated pseudo-PPL (BERT-MLM)", outdir / "ablated_ppl_heatmap.png", fmt="0.2f")
    heat_no_cbar(delta, langs, langs,
                 "ΔPPL (ablated − baseline)", outdir / "delta_ppl_heatmap.png", fmt="0.2f")

    # ---- neuron plots ----
    make_neuron_plots(langs, neurons, outdir)

    print("✅ [BERT-MLM] Finished. Outputs in:", outdir.resolve())


# ----------------------- LLaMA / decoder backend (vLLM) -----------------------
def compute_loss_vllm(model, ids: torch.Tensor, max_length: int, device: torch.device):
    """
    Approximate cross-entropy (negative log-prob per token) using vLLM.
    We follow your old ppl.py logic: use prompt_logprobs on fixed-length chunks.
    Returns avg_nats (per token).
    """
    max_len = min(max_length, model.llm_engine.model_config.max_model_len)
    L = ids.size(0)
    L_trim = (L // max_len) * max_len
    if L_trim == 0:
        raise ValueError("Not enough tokens for one full sequence.")

    ids = ids[:L_trim]
    input_ids = ids.view(-1, max_len)  # [num_seqs, max_len]

    outputs = model.generate(
        prompt_token_ids=input_ids.tolist(),
        sampling_params=SamplingParams(max_tokens=1, prompt_logprobs=0),
    )

    all_logprobs = []
    for out in outputs:
        # out.prompt_logprobs is a list (per token) of dicts {token_id: logprob}
        # we approximate by taking the logprob of the actually chosen token:
        token_logps = []
        for r in out.prompt_logprobs:
            if not r:
                continue
            # r is {token_id: logprob}; we take the single value (as in your code)
            token_logps.append(next(iter(r.values())))
        if token_logps:
            all_logprobs.extend(token_logps)

    if not all_logprobs:
        raise ValueError("No logprobs collected from vLLM outputs.")
    avg_logp = float(np.mean(all_logprobs))
    avg_nats = -avg_logp
    return avg_nats


def apply_llama_ablation(model, lang_mask, device: torch.device):
    """
    Monkey-patch vLLM internal MLP forward to zero out a subset of neurons.
    lang_mask: list[num_layers] of 1D tensors with neuron indices per layer.
    """
    model_name = model.llm_engine.model_config.hf_config.model_type.lower()
    # We'll treat all non-BLOOM as LLaMA/Mistral-style
    is_bloom = ("bloom" in model_name)

    num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers

    for i in range(num_layers):
        layer_mask = lang_mask[i]
        if layer_mask is None or len(layer_mask) == 0:
            continue
        mask = layer_mask.to(device)

        def factory(mask_inner):
            def llama_forward(self, x):
                gate_up, _ = self.gate_up_proj(x)  # [B, T, 2*hidden]
                i2 = gate_up.size(-1)
                activation = F.silu(gate_up[:, :, : i2 // 2])
                activation.index_fill_(2, mask_inner, 0)
                x2 = activation * gate_up[:, :, i2 // 2 :]
                x2, _ = self.down_proj(x2)
                return x2

            def bloom_forward(self, x: torch.Tensor):
                x2, _ = self.dense_h_to_4h(x)
                x2 = self.gelu_impl(x2)
                x2.index_fill_(2, mask_inner, 0)
                x2, _ = self.dense_4h_to_h(x2)
                return x2

            return bloom_forward if is_bloom else llama_forward

        if is_bloom:
            obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
        else:
            obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
        obj.forward = factory(mask)


def run_llama_vllm_backend(args):
    if VLLM is None:
        raise ImportError("vLLM is not installed, but backend=llama-vllm was requested.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    neuron_path = Path(args.mask)

    # ---- load masks ----
    neurons = load_neurons(neuron_path)
    L_mask = len(neurons)
    langs = args.langs
    L = len(langs)
    if L_mask != L:
        rprint(f"⚠️ Mask has {L_mask} languages, --langs has {L}. Using first {min(L_mask, L)}.")
    L = min(L_mask, L)
    langs = langs[:L]
    neurons = neurons[:L]

    # ---- load PUD ids ----
    ids_by_lang = {}
    for lang in langs:
        if args.split == "train":
            pud_path = Path("data/pud/train") / f"id.{lang}.train.{args.tag}.pud"
        else:
            pud_path = Path("data/pud/valid") / f"id.{lang}.eval.{args.tag}.pud"
        if not pud_path.exists():
            rprint(f"⚠️ Skipping {lang}: missing {pud_path}")
            continue
        ids = torch.load(pud_path)
        ids_by_lang[lang] = ids
        rprint(f"Loaded {ids.numel()} tokens for lang={lang} from {pud_path}")

    langs = [lg for lg in langs if lg in ids_by_lang]
    L = len(langs)
    rprint(f"[LLAMA] Languages with data: {langs}")

    # ---- init vLLM model ----
    rprint(f"[LLAMA] Loading vLLM model: {args.model}")
    model = VLLM(
        model=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="float16",
    )

    # ---- baseline loss (avg_nats) & PPL ----
    baseline_nats = np.zeros(L, dtype=float)
    baseline_ppl = np.zeros(L, dtype=float)

    for i, lang in enumerate(langs):
        rprint(f"[LLAMA] Baseline loss: {lang}")
        ids = ids_by_lang[lang]
        avg_nats = compute_loss_vllm(model, ids, args.max_model_len, device)
        baseline_nats[i] = avg_nats
        baseline_ppl[i] = math.exp(avg_nats)
        rprint(f"  avg_nats: {avg_nats:.4f}  PPL: {baseline_ppl[i]:.2f}")

    # ---- ablated loss matrix ----
    ablated_nats = np.zeros((L, L), dtype=float)

    for mask_idx, mask_lang in enumerate(langs):
        rprint(f"[LLAMA] Ablation using neurons of mask_lang={mask_lang}")
        # Re-apply mlp.forward for each layer according to the current mask
        lang_mask = [torch.tensor(m, dtype=torch.long, device=device) for m in neurons[mask_idx]]
        apply_llama_ablation(model, lang_mask, device)

        for eval_idx, eval_lang in enumerate(langs):
            rprint(f"  → eval lang={eval_lang}")
            ids = ids_by_lang[eval_lang]
            avg_nats = compute_loss_vllm(model, ids, args.max_model_len, device)
            ablated_nats[mask_idx, eval_idx] = avg_nats
            rprint(f"    avg_nats: {avg_nats:.4f}  PPL: {math.exp(avg_nats):.2f}")

    # ---- write CSVs & heatmaps ----
    ppl_df = pd.DataFrame(ablated_nats, columns=langs)
    ppl_df.insert(0, "mask_lang", langs)
    ppl_df.to_csv(outdir / "loss_results_nats.csv", index=False)

    base_df = pd.DataFrame({
        "lang": langs,
        "baseline_avg_nats": baseline_nats,
        "baseline_ppl": baseline_ppl,
    })
    base_df.to_csv(outdir / "baseline_loss_ppl.csv", index=False)

    delta = ablated_nats - baseline_nats[None, :]
    delta_df = pd.DataFrame(delta, index=langs, columns=langs)
    delta_df.to_csv(outdir / "delta_loss_nats.csv")

    heat_no_cbar(ablated_nats, langs, langs,
                 "Ablated loss (nats, LLAMA)", outdir / "ablated_loss_heatmap.png", fmt="0.2f")
    heat_no_cbar(delta, langs, langs,
                 "Δloss (ablated − baseline) [nats]", outdir / "delta_loss_heatmap.png", fmt="0.2f")

    # ---- neuron plots ----
    make_neuron_plots(langs, neurons, outdir)

    print("✅ [LLAMA] Finished. Outputs in:", outdir.resolve())


# ----------------------- main CLI -----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Unified LAPE analysis: BERT-MLM (HF) or LLaMA-like (vLLM)."
    )
    ap.add_argument("--backend", choices=["bert-mlm", "llama-vllm"], required=True,
                    help="Which backend to use: 'bert-mlm' for ParsBERT, 'llama-vllm' for PersianLLaMA/Maral/etc.")
    ap.add_argument("--model", "-m", required=True,
                    help="HF model id (for bert-mlm) or vLLM model id (for llama-vllm).")
    ap.add_argument("--tag", required=True,
                    help="Tag used in PUD filenames and mask name, e.g. ParsBERT or PersianLLaMA-13B-Instruct.")
    ap.add_argument("--mask", "-a", required=True,
                    help="Path to combined *.neuron.pth file (from identify_hf.py / identify.py).")
    ap.add_argument("--langs", nargs="+", required=True,
                    help="Language order, must match mask order (e.g., ar en fr hi ja ru tr).")
    ap.add_argument("--split", choices=["train", "valid"], default="valid",
                    help="Which PUD split to use (train or valid).")
    ap.add_argument("--outdir", default="results/LAPE",
                    help="Output directory for CSVs and plots.")

    # BERT-MLM specific
    ap.add_argument("--max-length", type=int, default=512,
                    help="[bert-mlm] Sequence length for chunking the token ids.")
    ap.add_argument("--mask-batch-size", type=int, default=32,
                    help="[bert-mlm] How many token positions to mask per forward pass.")
    ap.add_argument("--device", default="cuda",
                    help="[bert-mlm] Device, e.g. 'cuda' or 'cpu'.")

    # LLaMA/vLLM specific
    ap.add_argument("--max-model-len", type=int, default=1024,
                    help="[llama-vllm] max_model_len for vLLM (adjust if you hit KV cache issues).")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                    help="[llama-vllm] GPU memory utilization for vLLM.")

    args = ap.parse_args()

    if args.backend == "bert-mlm":
        run_bert_mlm_backend(args)
    else:
        run_llama_vllm_backend(args)


if __name__ == "__main__":
    main()
