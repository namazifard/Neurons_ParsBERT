# Neurons_ParsBERT

1. Define Languages
   ```bash
   LANGS21=("ar" "zh" "cs" "en" "fi" "fr" "de" "hi" "id" "it" \
         "ja" "ko" "pl" "pt" "ru" "es" "sv" "th" "tr" "is" "gl")
   ```

2. Prepare Data
   ```bash
   python prepare_data.py \
      --tag ParsBERT \
      --langs "${LANGS21[@]}"
   ```

3. Calculate Activation
   ```bash
   for L in "${LANGS21[@]}"; do
      echo "=== ParsBERT activation: $L ==="
      python activation_hf.py \
            -m HooshvareLab/bert-base-parsbert-uncased \
            -l "$L" \
            --tag ParsBERT \
            --task mlm \
            --max-length 512 \
            --batch-size 8
   done
   ```

4. Identify Neurons
   ```bash
   python identify_hf.py \
      --tag ParsBERT \
      --langs "${LANGS21[@]}"
   ```

5. Ablation, Analysis, and Plots
   ```bash
   python result.py \
      --backend bert-mlm \
      --model HooshvareLab/bert-base-parsbert-uncased \
      --tag ParsBERT \
      --mask activation_mask/ParsBERT.neuron.pth \
      --langs "${LANGS21[@]}" \
      --split valid \
      --max-length 512 \
      --mask-batch-size 32 \
      --device cuda \
      --outdir results/ParsBERT_LANGS21
   ```

   ```bash
   python result.py \
      --backend llama-vllm \
      --model ViraIntelligentDataMining/PersianLLaMA-13B \
      --tag PersianLLaMA-13B-Instruct \
      --mask activation_mask/PersianLLaMA-13B-Instruct.neuron.pth \
      --langs "${LANGS21[@]}" \
      --split valid \
      --max-model-len 3152 \
      --gpu-memory-utilization 0.9 \
      --outdir results/PersianLLaMA-13B
   ```