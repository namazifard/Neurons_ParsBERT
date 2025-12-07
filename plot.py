#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


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
    """Simple heatmap without colorbar, with values printed in each cell."""
    plt.figure(figsize=(6, 5))
    _ = plt.imshow(mat, aspect="auto")
    plt.xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, format(mat[i, j], fmt),
                     ha="center", va="center", fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def load_neurons(neuron_path: Path):
    """
    Expect a *.neuron.pth with structure:
      List[ languages ] where each entry is List[num_layers] of LongTensor indices.
    """
    neurons = torch.load(str(neuron_path))
    norm = []
    for lang_blocks in neurons:
        lang_list = []
        for t in lang_blocks:
            if isinstance(t, torch.Tensor):
                lang_list.append(t.cpu().numpy().astype(np.int64))
            else:
                lang_list.append(np.array(t, dtype=np.int64))
        norm.append(lang_list)
    return norm  # shape: [num_langs][num_layers] => np.array of indices


def per_language_per_layer_counts(neurons):
    L = len(neurons)
    num_layers = len(neurons[0])
    counts = np.zeros((L, num_layers), dtype=int)
    for i in range(L):
        for j in range(num_layers):
            counts[i, j] = int(len(neurons[i][j]))
    return counts


def support_histograms(neurons):
    """
    For each layer: map neuron index -> number of languages containing it.
    Returns:
      per_layer_hist: list of dicts: layer -> {k: count_with_support_k}
      overall_counts: dict k -> count across all layers
      pair_intersections & pair_jaccard aggregated across layers.
    """
    L = len(neurons)
    num_layers = len(neurons[0])

    per_layer_hist = []
    overall_counts = {}
    pair_intersections = np.zeros((L, L), dtype=int)
    pair_unions = np.zeros((L, L), dtype=int)

    for j in range(num_layers):
        layer_sets = [set(neurons[i][j].tolist()) for i in range(L)]
        idx2langs = {}
        for i in range(L):
            for idx in layer_sets[i]:
                s = idx2langs.get(idx)
                if s is None:
                    idx2langs[idx] = {i}
                else:
                    s.add(i)

        # support hist for this layer
        hist = {}
        for _, langs_set in idx2langs.items():
            k = len(langs_set)
            hist[k] = hist.get(k, 0) + 1
            overall_counts[k] = overall_counts.get(k, 0) + 1
        per_layer_hist.append(hist)

        # pairwise intersections / unions in this layer
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
    """Convert sparse dict {k:count} to dense list length max_k (for k=1..max_k)."""
    return [int(hist_dict.get(k, 0)) for k in range(1, max_k + 1)]


def main():
    ap = argparse.ArgumentParser(
        description="Neuron distribution & support plots for ParsBERT (or any encoder) "
                    "given a *.neuron.pth mask and a list of 'langs' (conditions)."
    )
    ap.add_argument("--tag", required=True,
                    help="Model tag (used to locate activation_mask/<tag>.neuron.pth)")
    ap.add_argument("--langs", nargs="+", required=True,
                    help="List of condition labels (e.g., languages or UPOS tags).")
    ap.add_argument("--mask", default=None,
                    help="Path to neuron mask; defaults to activation_mask/<tag>.neuron.pth")
    ap.add_argument("--outdir", required=True,
                    help="Directory to write CSVs and plots.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mask_path = Path(args.mask) if args.mask else Path("activation_mask") / f"{args.tag}.neuron.pth"
    print(f"Loading neuron mask from: {mask_path}")
    neurons = load_neurons(mask_path)

    L = min(len(neurons), len(args.langs))
    neurons = neurons[:L]
    langs = args.langs[:L]

    num_layers = len(neurons[0])
    layer_labels_csv = [f"layer{j+1}" for j in range(num_layers)]
    xpos = np.arange(1, num_layers + 1)
    layer_ticks = [str(j) for j in xpos]

    # 1) Per-language per-layer counts
    counts = per_language_per_layer_counts(neurons)  # (L, num_layers)
    counts_df = pd.DataFrame(counts, index=langs, columns=layer_labels_csv)
    counts_df.to_csv(outdir / "language_layer_counts.csv")

    # Totals
    layer_totals = counts.sum(axis=0)
    lang_totals = counts.sum(axis=1)
    pd.Series(layer_totals, index=layer_labels_csv).to_csv(outdir / "layer_totals.csv")
    pd.Series(lang_totals, index=langs).to_csv(outdir / "language_totals.csv")

    # Totals per layer
    plt.figure(figsize=(10, 3))
    plt.bar(xpos, layer_totals)
    plt.xticks(xpos, layer_ticks)
    plt.xlabel("Layer"); plt.ylabel("# Neurons (sum across conditions)")
    plt.title("Condition-specific neurons per layer (total)")
    plt.tight_layout()
    plt.savefig(outdir / "layer_totals_bar.png", dpi=200)
    plt.close()

    # Totals per language
    plt.figure(figsize=(8, 3))
    ax = plt.gca()
    ax.bar(np.arange(len(langs)), lang_totals)
    ax.set_xticks(np.arange(len(langs)))
    ax.set_xticklabels(langs)
    ax.set_xlabel("Condition"); ax.set_ylabel("# Neurons")
    ax.set_title("Neurons per condition")
    label_bars(ax, fmt="{:d}")
    plt.tight_layout()
    plt.savefig(outdir / "language_totals_bar.png", dpi=200)
    plt.close()

    # Stacked per layer by language
    plt.figure(figsize=(10, 4))
    bottom = np.zeros(num_layers)
    for i, lg in enumerate(langs):
        plt.bar(xpos, counts[i], bottom=bottom, label=lg)
        bottom += counts[i]
    plt.xticks(xpos, layer_ticks)
    plt.xlabel("Layer"); plt.ylabel("# Neurons")
    plt.title("Per-condition neurons per layer (stacked)")
    plt.legend(ncol=min(4, L), fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "per_language_stacked_bar.png", dpi=200)
    plt.close()

    # Per-language per-layer bar
    for i, lg in enumerate(langs):
        plt.figure(figsize=(10, 3))
        plt.bar(xpos, counts[i])
        plt.xticks(xpos, layer_ticks)
        plt.xlabel("Layer"); plt.ylabel("# Neurons")
        plt.title(f"{lg}: neurons per layer")
        plt.tight_layout()
        plt.savefig(outdir / f"per_layer_{lg}_bar.png", dpi=200)
        plt.close()

    # 2) Support histograms and pairwise overlaps
    per_layer_hist, overall_counts, pair_intersections, pair_jaccard = support_histograms(neurons)
    max_k = L

    # Overall support
    overall_row = to_full_hist_row(overall_counts, max_k)
    overall_df = pd.DataFrame({"k": list(range(1, max_k + 1)), "count": overall_row})
    overall_df.to_csv(outdir / "support_hist_overall.csv", index=False)

    plt.figure(figsize=(8, 3))
    ax = plt.gca()
    ax.bar(np.arange(1, max_k + 1), overall_row)
    ax.set_xlabel("k (conditions neuron appears in)")
    ax.set_ylabel("# Neurons")
    ax.set_title("Neuron support size (overall)")
    label_bars(ax, fmt="{:d}")
    plt.tight_layout()
    plt.savefig(outdir / "support_hist_overall_bar.png", dpi=200)
    plt.close()

    # Per-layer support (table + plots)
    per_layer_dense = np.vstack([
        to_full_hist_row(h, max_k) for h in per_layer_hist
    ])  # (num_layers, max_k)
    per_layer_df = pd.DataFrame(
        per_layer_dense,
        columns=[f"k={k}" for k in range(1, max_k + 1)],
    )
    per_layer_df.insert(0, "layer", layer_labels_csv)
    per_layer_df.to_csv(outdir / "support_hist_per_layer.csv", index=False)

    for j in range(num_layers):
        row = per_layer_dense[j]
        plt.figure(figsize=(8, 3))
        plt.bar(np.arange(1, max_k + 1), row)
        plt.xlabel("k (conditions neuron appears in)")
        plt.ylabel("# Neurons")
        plt.title(f"Neuron support size per layer — {j+1}")
        plt.tight_layout()
        plt.savefig(outdir / f"per_layer_support_hist_layer{j+1}_bar.png", dpi=200)
        plt.close()

    # Pairwise intersections / Jaccard
    inter_df = pd.DataFrame(pair_intersections[:L, :L], index=langs, columns=langs)
    jac_df = pd.DataFrame(pair_jaccard[:L, :L], index=langs, columns=langs)
    inter_df.to_csv(outdir / "pairwise_intersection_counts.csv")
    jac_df.to_csv(outdir / "pairwise_jaccard.csv")

    heat_no_cbar(
        inter_df.values.astype(int),
        langs,
        langs,
        "Pairwise intersections (aggregated)",
        outdir / "pairwise_intersections_heatmap.png",
        fmt="d",
    )
    heat_no_cbar(
        jac_df.values.astype(float),
        langs,
        langs,
        "Pairwise Jaccard (aggregated)",
        outdir / "pairwise_jaccard_heatmap.png",
        fmt="0.2f",
    )

    # Cumulative "k or more"
    k_or_more = np.cumsum(overall_row[::-1])[::-1]
    k_or_more_df = pd.DataFrame(
        {"k_or_more": list(range(1, max_k + 1)), "count": k_or_more}
    )
    k_or_more_df.to_csv(outdir / "shared_neurons_k_or_more.csv", index=False)

    print("✅ Wrote neuron analyses to:", outdir.resolve())


if __name__ == "__main__":
    main()
