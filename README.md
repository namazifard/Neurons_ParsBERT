# Neurons_ParsBERT

1. Dependencies and Languages
   ```bash
   pip install torch transformers pandas matplotlib
   ```
   ```bash
   LANGS=("ar" "zh" "cs" "en" "fi" "fr" "de" "hi" "id" "it" \
         "ja" "ko" "pl" "pt" "ru" "es" "sv" "th" "tr" "is" "gl")
   ```
   ```bash
   LANGS_CONTACT=("ar" "en" "fr" "de" "hi" "ja" "ru" "tr")
   ```
   ```bash
   UPOS_CATS=("UPOS-NOUN" "UPOS-VERB" "UPOS-ADJ" "UPOS-ADV" "UPOS-PRON" "UPOS-ADP")
   ```
   ```bash
   CASE_CATS=(
  "Case-Nom" "Case-Gen" "Case-Acc" "Case-Loc"
  "Case-Dat" "Case-Ins" "Case-Par" "Case-Ine"
  "Case-Ill" "Case-Abl" "Case-Ade" "Case-Ela"
   )
   ```
   ```bash
   GENDER_CATS=(
  "Gender-Fem" "Gender-Masc" "Gender-Neut" "Gender-Com"
   )
   ```

2. Prepare Data
   ```bash
   python prepare_data.py \
      --tag ParsBERT \
      --langs "${LANGS_CONTACT[@]}"
   ```

3. Calculate Activation
   ```bash
   for L in "${LANGS_CONTACT[@]}"; do
      echo "=== ParsBERT activation: $L ==="
      python activation.py \
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
   python identify.py \
      --tag ParsBERT \
      --langs "${LANGS_CONTACT[@]}"
   ```

5. Ablation, Analysis, and Plots
   ```bash
   python result.py \
      --backend bert-mlm \
      --model HooshvareLab/bert-base-parsbert-uncased \
      --tag ParsBERT \
      --mask activation_mask/ParsBERT.neuron.pth \
      --langs "${LANGS_CONTACT[@]}" \
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
      --langs "${LANGS_CONTACT[@]}" \
      --split valid \
      --max-model-len 3152 \
      --gpu-memory-utilization 0.9 \
      --outdir results/PersianLLaMA-13B
   ```

6. Plots
   ```bash
   python plots_parsbert.py \
      --tag ParsBERT \
      --langs "${LANGS_CONTACT[@]}" \
      --outdir results/ParsBERT
   ```
   