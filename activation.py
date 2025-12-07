import argparse
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM


def rprint(*args):
    print(*args, flush=True)


def load_model(model_name: str, task: str, device: torch.device):
    if task == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
    elif task == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
    else:
        raise ValueError(f"Unknown task '{task}', use 'mlm' or 'causal'.")
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="LAPE-style activation counting (HF, BERT + LLaMA/Mistral).")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="HF model id, e.g. HooshvareLab/bert-base-parsbert-uncased or meta-llama/Llama-2-7b-hf")
    parser.add_argument("-l", "--lang", type=str, required=True,
                        help="Language code (used in PUD file names, e.g. en, ar, ...).")
    parser.add_argument("--tag", type=str, required=True,
                        help="Tag used in PUD filenames, e.g. ParsBERT, Llama-2-7b, PersianLLaMA-13B-Instruct, ...")
    parser.add_argument("--task", type=str, choices=["mlm", "causal"], required=True,
                        help="Model type: 'mlm' for encoder-only (BERT/ParsBERT), 'causal' for decoder-only (LLaMA, Mistral, ...).")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Sequence length for chunking tokens.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for forward passes.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on, e.g. 'cuda' or 'cpu'.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rprint(f"Loading model {args.model} (task={args.task}) on {device} ...")
    model = load_model(args.model, args.task, device)
    config = model.config
    model_type = getattr(config, "model_type", None)
    rprint(f"model_type={model_type}")

    # ---- Determine where to hook and intermediate size ----
    hooks = []
    if args.task == "mlm":  # BERT / ParsBERT style
        if model_type not in ["bert", "roberta", "xlm-roberta"]:
            rprint(f"WARNING: task=mlm but model_type={model_type}, assuming BERT-like encoder.")
        # BERT uses encoder.layer[i].intermediate as FFN activation (after GELU)
        encoder_layers = model.bert.encoder.layer
        num_layers = config.num_hidden_layers
        intermediate_size = config.intermediate_size
        rprint(f"BERT-like: num_layers={num_layers}, intermediate_size={intermediate_size}")
        over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int64, device=device)

        def make_hook_bert(layer_idx):
            def hook(module, inputs, output):
                # output: [batch, seq, intermediate_size], already after GELU
                act = output
                over_zero[layer_idx] += (act > 0).sum(dim=(0, 1))
                return None
            return hook

        for i, layer in enumerate(encoder_layers):
            hooks.append(layer.intermediate.register_forward_hook(make_hook_bert(i)))

    elif args.task == "causal":  # LLaMA / Mistral-style decoder
        if model_type in ["llama", "mistral"]:
            # LLaMA / Mistral MLP: gate_proj, up_proj, act_fn, down_proj
            decoder_layers = model.model.layers
            num_layers = config.num_hidden_layers
            intermediate_size = config.intermediate_size
            rprint(f"LLaMA/Mistral-like: num_layers={num_layers}, intermediate_size={intermediate_size}")
            over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int64, device=device)

            def make_hook_llama(layer_idx):
                def hook(module: nn.Module, inputs, output):
                    # module is LlamaMLP / MistralMLP
                    # inputs[0]: [batch, seq, hidden_size]
                    x_in = inputs[0]
                    gate = module.gate_proj(x_in)
                    act = module.act_fn(gate)  # [B, T, intermediate_size]
                    over_zero[layer_idx] += (act > 0).sum(dim=(0, 1))
                    return None
                return hook

            for i, layer in enumerate(decoder_layers):
                hooks.append(layer.mlp.register_forward_hook(make_hook_llama(i)))

        else:
            raise ValueError(
                f"task='causal' currently supports model_type in ['llama', 'mistral'], got {model_type}"
            )
    else:
        raise ValueError(f"Unknown task {args.task}")

    # ---- Load token ids (PUD) ----
    pud_path = Path("data/pud/train") / f"id.{args.lang}.train.{args.tag}.pud"
    if not pud_path.exists():
        raise FileNotFoundError(f"Could not find {pud_path}")
    ids = torch.load(pud_path)  # 1D tensor
    rprint(f"Loaded {ids.numel()} tokens for lang={args.lang} from {pud_path}")

    max_len = min(args.max_length, getattr(config, "max_position_embeddings", args.max_length))
    # Trim to multiple of max_len
    L = ids.size(0)
    L = (L // max_len) * max_len
    if L == 0:
        raise ValueError("Not enough tokens to form a single sequence of length max_len.")
    ids = ids[:L]
    input_ids = ids.view(-1, max_len)  # [num_seqs, max_len]
    rprint(f"Using {L} tokens → {input_ids.size(0)} sequences of length {max_len}")

    # ---- Forward passes to accumulate activation counts ----
    with torch.no_grad():
        for i in range(0, input_ids.size(0), args.batch_size):
            batch = input_ids[i : i + args.batch_size].to(device)
            if args.task == "mlm":
                attention_mask = torch.ones_like(batch, device=device)
                _ = model(input_ids=batch, attention_mask=attention_mask)
            else:  # causal
                _ = model(input_ids=batch)

    # ---- Remove hooks ----
    for h in hooks:
        h.remove()

    # ---- Save in LAPE format ----
    out = {"n": int(L), "over_zero": over_zero.to("cpu")}
    Path("activations").mkdir(parents=True, exist_ok=True)
    out_path = Path("activations") / f"activation.{args.lang}.train.{args.tag}"
    torch.save(out, out_path)
    rprint(f"Saved activation stats to {out_path}")


if __name__ == "__main__":
    main()
