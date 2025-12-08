#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


# ----------------------- small helpers -----------------------
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


# ----------------------- Neuron helpers -----------------------
def load_neurons(neuron_path: Path):
    """
    Expect a combined *.neuron.pth with structure:
      neurons[cat_idx][layer_idx] = 1D LongTensor of neuron indices
    (what identify.py currently writes)
    """
    neurons_raw = torch.load(str(neuron_path))

    neurons = []
    for cat_blocks in neurons_raw:
        cat_layers = []
        for t in cat_blocks:
            if isinstance(t, torch.Tensor):
                cat_layers.append(np.array(t.cpu().numpy(), dtype=np.int64))
            else:
                cat_layers.append(np.array(t, dtype=np.int64))
        neurons.append(cat_layers)

    return neurons  # list[cat][layer] -> np.array of indices


def per_category_per_layer_counts(neurons):
    C = len(neurons)
    num_layers = len(neurons[0])
    counts = np.zeros((C, num_layers), dtype=int)
    for i in range(C):
        for j in range(num_layers):
            counts[i, j] = int(neurons[i][j].size)
    return counts


def support_histograms(neurons):
    """
    For each layer: map neuron index -> number of categories containing it.
    Returns:
      per_layer_hist: list of dicts: layer -> {k: count_with_support_k}
      overall_counts: dict k -> count across all layers
      pairwise intersections & jaccard aggregated across layers
    """
    C = len(neurons)
    num_layers = len(neurons[0])

    per_layer_hist = []
    overall_counts = {}
    pair_intersections = np.zeros((C, C), dtype=int)
    pair_unions = np.zeros((C, C), dtype=int)

    for j in range(num_layers):
        idx2cats = {}
        layer_sets = [set(neurons[i][j].tolist()) for i in range(C)]

        # build: neuron index -> set of categories it appears in
        for i in range(C):
            for idx in layer_sets[i]:
                if idx not in idx2cats:
                    idx2cats[idx] = {i}
                else:
                    idx2cats[idx].add(i)

        hist = {}
        for _, cat_set in idx2cats.items():
            k = len(cat_set)
            hist[k] = hist.get(k, 0) + 1
            overall_counts[k] = overall_counts.get(k, 0) + 1
        per_layer_hist.append(hist)

        # pairwise intersections / unions for this layer
        for a in range(C):
            for b in range(C):
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


def heat_no_cbar(mat, row_labels, col_labels, title, outpath, fmt="0.2f"):
    plt.figure(figsize=(6, 5))
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


# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Analyse category-specific neuron masks (no PPL): "
                    "per-layer counts, support histograms, intersections, etc."
    )
    ap.add_argument(
        "--mask", "-a", required=True,
        help="Path to combined *.neuron.pth file (from identify.py)",
    )
    ap.add_argument(
        "--cats", nargs="+", required=True,
        help="Category names in the same order used in identify.py "
             "(e.g., UPOS-NOUN UPOS-VERB ...)",
    )
    ap.add_argument(
        "--outdir", default="results/categories",
        help="Output directory for CSVs and plots",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------- 1) Load neuron mask --------
    neuron_path = Path(args.mask)
    neurons = load_neurons(neuron_path)

    C = min(len(neurons), len(args.cats))  # align if neuron file has extra slices
    neurons = neurons[:C]
    cats = args.cats[:C]
    num_layers = len(neurons[0])

    # 1-based layer indices for plots/CSVs
    layer_labels_csv = [f"layer{j+1}" for j in range(num_layers)]
    xpos = np.arange(1, num_layers + 1)
    layer_ticks = [str(j) for j in xpos]

    # -------- 2) Per-category per-layer counts --------
    counts = per_category_per_layer_counts(neurons)  # (C, num_layers)

    counts_df = pd.DataFrame(counts, index=cats, columns=layer_labels_csv)
    counts_df.to_csv(outdir / "category_layer_counts.csv")

    layer_totals = counts.sum(axis=0)
    cat_totals = counts.sum(axis=1)

    pd.Series(layer_totals, index=layer_labels_csv).to_csv(outdir / "layer_totals.csv")
    pd.Series(cat_totals, index=cats).to_csv(outdir / "category_totals.csv")

    # Totals per layer (ticks 1..N)
    plt.figure(figsize=(10, 4))
    plt.bar(xpos, layer_totals)
    plt.xticks(xpos, layer_ticks, rotation=0)
    plt.xlabel("Layer")
    plt.ylabel("# Neurons (sum across categories)")
    plt.title("Category-specific neurons per layer (total)")
    plt.tight_layout()
    plt.savefig(outdir / "layer_totals_bar.png", dpi=200)
    plt.close()

    # Totals per category + labels
    plt.figure(figsize=(max(6, 1.2 * C), 4))
    ax = plt.gca()
    ax.bar(np.arange(C), cat_totals)
    ax.set_xticks(np.arange(C))
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_xlabel("Category")
    ax.set_ylabel("# Neurons")
    ax.set_title("Distribution of neurons across categories")
    label_bars(ax, fmt="{:d}")
    plt.tight_layout()
    plt.savefig(outdir / "category_totals_bar.png", dpi=200)
    plt.close()

    # Stacked per layer by category
    plt.figure(figsize=(10, 5))
    bottom = np.zeros(num_layers)
    for i, cat in enumerate(cats):
        plt.bar(xpos, counts[i], bottom=bottom, label=cat)
        bottom += counts[i]
    plt.xticks(xpos, layer_ticks, rotation=0)
    plt.xlabel("Layer")
    plt.ylabel("# Neurons")
    plt.title("Per-category neurons per layer (stacked)")
    plt.legend(ncol=min(4, C), fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "per_category_stacked_bar.png", dpi=200)
    plt.close()

    # Per-category bars per layer
    for i, cat in enumerate(cats):
        plt.figure(figsize=(10, 4))
        plt.bar(xpos, counts[i])
        plt.xticks(xpos, layer_ticks, rotation=0)
        plt.xlabel("Layer")
        plt.ylabel("# Neurons")
        plt.title(f"{cat}: neurons per layer")
        plt.tight_layout()
        plt.savefig(outdir / f"per_layer_{cat}_bar.png", dpi=200)
        plt.close()

    # -------- 3) Support histograms (how many cats neuron appears in) --------
    per_layer_hist, overall_counts, pair_intersections, pair_jaccard = support_histograms(neurons)
    max_k = C

    # Overall support histogram + labels
    overall_row = to_full_hist_row(overall_counts, max_k)
    overall_df = pd.DataFrame({"k": list(range(1, max_k + 1)), "count": overall_row})
    overall_df.to_csv(outdir / "support_hist_overall.csv", index=False)

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.bar(np.arange(1, max_k + 1), overall_row)
    ax.set_xlabel("k (categories neuron appears in)")
    ax.set_ylabel("# Neurons")
    ax.set_title("Neuron support size (overall)")
    label_bars(ax, fmt="{:d}")
    plt.tight_layout()
    plt.savefig(outdir / "support_hist_overall_bar.png", dpi=200)
    plt.close()

    # Per-layer support histogram table
    per_layer_dense = np.vstack([to_full_hist_row(h, max_k) for h in per_layer_hist])
    per_layer_df = pd.DataFrame(per_layer_dense, columns=[f"k={k}" for k in range(1, max_k + 1)])
    per_layer_df.insert(0, "layer", layer_labels_csv)
    per_layer_df.to_csv(outdir / "support_hist_per_layer.csv", index=False)

    # Per-layer support hist plots
    for j in range(num_layers):
        row = per_layer_dense[j]
        plt.figure(figsize=(8, 3.2))
        plt.bar(np.arange(1, max_k + 1), row)
        plt.xlabel("k (categories neuron appears in)")
        plt.ylabel("# Neurons")
        plt.title(f"Neuron support size per layer — {j+1}")
        plt.tight_layout()
        plt.savefig(outdir / f"per_layer_support_hist_layer{j+1}_bar.png", dpi=200)
        plt.close()

    # Multicategory neurons (k > 2, k > 3)
    k_gt_2_per_layer = per_layer_dense[:, 2:].sum(axis=1) if max_k >= 3 else np.zeros(num_layers, dtype=int)
    k_gt_3_per_layer = per_layer_dense[:, 3:].sum(axis=1) if max_k >= 4 else np.zeros(num_layers, dtype=int)

    pd.DataFrame({"layer": layer_labels_csv, "count": k_gt_2_per_layer}
                 ).to_csv(outdir / "multicategory_k_gt_2_per_layer.csv", index=False)
    pd.DataFrame({"layer": layer_labels_csv, "count": k_gt_3_per_layer}
                 ).to_csv(outdir / "multicategory_k_gt_3_per_layer.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.bar(xpos, k_gt_2_per_layer)
    plt.xticks(xpos, layer_ticks, rotation=0)
    plt.xlabel("Layer")
    plt.ylabel("# Neurons")
    plt.title("Multicategory neurons per layer (k > 2 categories)")
    plt.tight_layout()
    plt.savefig(outdir / "multicategory_k_gt_2_per_layer_bar.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(xpos, k_gt_3_per_layer)
    plt.xticks(xpos, layer_ticks, rotation=0)
    plt.xlabel("Layer")
    plt.ylabel("# Neurons")
    plt.title("Highly multicategory neurons per layer (k > 3 categories)")
    plt.tight_layout()
    plt.savefig(outdir / "multicategory_k_gt_3_per_layer_bar.png", dpi=200)
    plt.close()

    # -------- 4) Pairwise intersections & Jaccard --------
    inter_df = pd.DataFrame(pair_intersections[:C, :C], index=cats, columns=cats)
    jac_df = pd.DataFrame(pair_jaccard[:C, :C], index=cats, columns=cats)
    inter_df.to_csv(outdir / "pairwise_intersection_counts.csv")
    jac_df.to_csv(outdir / "pairwise_jaccard.csv")

    heat_no_cbar(
        inter_df.values.astype(int),
        cats,
        cats,
        "Pairwise intersections (aggregated across layers)",
        outdir / "pairwise_intersections_heatmap.png",
        fmt="d",
    )
    heat_no_cbar(
        jac_df.values.astype(float),
        cats,
        cats,
        "Pairwise Jaccard (aggregated across layers)",
        outdir / "pairwise_jaccard_heatmap.png",
        fmt="0.2f",
    )

    # -------- 5) Cumulative 'k or more' table --------
    k_or_more = np.cumsum(overall_row[::-1])[::-1]
    k_or_more_df = pd.DataFrame({"k_or_more": list(range(1, max_k + 1)), "count": k_or_more})
    k_or_more_df.to_csv(outdir / "shared_neurons_k_or_more.csv", index=False)

    print("✅ Wrote category neuron analyses to:", outdir.resolve())
    print("  - category_layer_counts.csv, layer_totals.csv, category_totals.csv")
    print("  - layer_totals_bar.png, category_totals_bar.png")
    print("  - per_category_stacked_bar.png, per_layer_<CAT>_bar.png")
    print("  - support_hist_overall.csv/.png, support_hist_per_layer.csv + per-layer support plots")
    print("  - multicategory_k_gt_2_per_layer.csv/.png, multicategory_k_gt_3_per_layer.csv/.png")
    print("  - pairwise_intersection_counts.csv, pairwise_jaccard.csv + heatmaps")
    print("  - shared_neurons_k_or_more.csv")


if __name__ == "__main__":
    main()
