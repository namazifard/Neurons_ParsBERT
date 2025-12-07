import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(
        description="Identify language-specific neurons from activation stats (HF-based LAPE)."
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Model tag used in activation filenames, e.g. ParsBERT, PersianMind-7b, Maral-7B-alpha-1.",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["ar","zh","cs","en","fi","fr","de","hi","id","it",
                 "ja","ko","pl","pt","ru","es","sv","th","tr","is","gl"],
        help="Languages to include (must match the ones used for activation).",
    )
    parser.add_argument(
        "--top-rate",
        type=float,
        default=0.01,
        help="Fraction of neurons (by entropy) to keep as candidates (default: 0.01).",
    )
    parser.add_argument(
        "--filter-rate",
        type=float,
        default=0.95,
        help="Percentile threshold on activation probability to filter neurons (default: 0.95).",
    )
    parser.add_argument(
        "--activation-bar-ratio",
        type=float,
        default=0.95,
        help="Percentile threshold on activation probability to decide language-specificity (default: 0.95).",
    )
    args = parser.parse_args()

    TAG = args.tag
    LANGS = args.langs
    top_rate = args.top_rate
    filter_rate = args.filter_rate
    activation_bar_ratio = args.activation_bar_ratio

    n_list, over_zero_list = [], []

    # ---- Load activation stats for each language ----
    for lang in LANGS:
        act_path = Path("activations") / f"activation.{lang}.train.{TAG}"
        if not act_path.exists():
            raise FileNotFoundError(f"Missing activation file: {act_path}")
        data = torch.load(act_path)
        n_list.append(data["n"])
        over_zero_list.append(data["over_zero"])

    n = torch.tensor(n_list)                     # [lang_num]
    over_zero = torch.stack(over_zero_list, -1)  # [num_layers, intermediate_size, lang_num]

    num_layers, intermediate_size, lang_num = over_zero.size()
    print(f"Loaded activation stats for tag={TAG}: "
          f"{lang_num} languages, {num_layers} layers, {intermediate_size} neurons per layer")

    # ---- Compute activation probabilities ----
    activation_probs = over_zero / n            # [L, H, lang]
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0

    # Entropy across languages per neuron (lower entropy → more language-specific)
    log_probs = torch.where(normed_activation_probs > 0,
                            normed_activation_probs.log(), torch.zeros_like(normed_activation_probs))
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)  # [L, H]

    largest = False  # we want *small* entropy → more specific

    if torch.isnan(entropy).any():
        print("Found NaNs in entropy, aborting.")
        raise ValueError("NaNs in entropy")

    # ---- Filter out neurons that are never highly active in any language ----
    flattened_probs = activation_probs.flatten()
    k_filter = round(len(flattened_probs) * filter_rate)
    top_prob_value = flattened_probs.kthvalue(k_filter).values.item()
    print(f"Activation filter threshold (filter_rate={filter_rate}): {top_prob_value:.6f}")

    # dismiss neuron if no language has activation value over this threshold
    top_position = (activation_probs > top_prob_value).sum(dim=-1)  # [L, H]
    entropy[top_position == 0] = torch.inf  # (since largest=False, they'll never be selected)

    # ---- Select top-rate neurons by entropy ----
    flattened_entropy = entropy.flatten()
    k_entropy = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(k_entropy, largest=largest)  # indices into flattened [L*H]

    row_index = index // entropy.size(1)  # layer indices
    col_index = index % entropy.size(1)   # neuron indices within layer

    selected_probs = activation_probs[row_index, col_index]  # [K, lang_num]

    print(f"Selected {selected_probs.size(0)} neuron candidates "
          f"(top_rate={top_rate}), per-language argmax counts:")
    print(torch.bincount(selected_probs.argmax(dim=-1)))

    # ---- Decide language-specific neurons using activation_bar ----
    selected_probs = selected_probs.transpose(0, 1)  # [lang_num, K]
    activation_bar = flattened_probs.kthvalue(
        round(len(flattened_probs) * activation_bar_ratio)
    ).values.item()

    print(f"Activation bar (activation_bar_ratio={activation_bar_ratio}): {activation_bar:.6f}")
    print("Counts of neurons above activation_bar per language:",
          (selected_probs > activation_bar).sum(dim=1).tolist())

    lang_idx, indice = torch.where(selected_probs > activation_bar)

    merged_index = torch.stack((row_index, col_index), dim=-1)  # [K, 2]
    # We will build final_indice[lang_id][layer_id] = tensor of neuron indices

    binc = torch.bincount(lang_idx, minlength=lang_num).tolist()
    final_indice = []
    offset = 0
    for li in range(lang_num):
        count = binc[li]
        if count == 0:
            # no neurons for this language
            layer_index = [[] for _ in range(num_layers)]
            for l in range(num_layers):
                layer_index[l] = torch.tensor([], dtype=torch.long)
            final_indice.append(layer_index)
            continue

        lang_mask_indices = indice[offset : offset + count]
        offset += count

        lang_index = [tuple(row.tolist()) for row in merged_index[lang_mask_indices]]
        lang_index.sort()
        layer_index = [[] for _ in range(num_layers)]
        for l, h in lang_index:
            layer_index[l].append(h)
        for l in range(num_layers):
            layer_index[l] = torch.tensor(layer_index[l], dtype=torch.long)
        final_indice.append(layer_index)

    # ---- Save activation mask ----
    Path("activation_mask").mkdir(parents=True, exist_ok=True)
    out_path = Path("activation_mask") / f"{TAG}.neuron.pth"
    torch.save(final_indice, out_path)
    print(f"Saved neuron mask to {out_path}")


if __name__ == "__main__":
    main()
