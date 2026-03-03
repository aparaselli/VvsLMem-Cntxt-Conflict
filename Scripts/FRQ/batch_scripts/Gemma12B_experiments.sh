#!/bin/bash

MODEL="google/gemma-3-12b-it"

SCRIPT="/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/FRQ_ctxt_mem_exp.py"
IDENTIFY_SCRIPT="/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/FRQ_identify_entity.py"

OUT_DIR="/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results"

# Activate env
source ~/.bashrc
conda activate patchscopes-llava

modalities=("vision" "text")
benches=("logo" "people")
rag_settings=("True" "False")

# ==============================
# Main FRQ experiment grid
# ==============================

for modality in "${modalities[@]}"; do
  for bench in "${benches[@]}"; do
    for rag in "${rag_settings[@]}"; do

      if [ "$rag" = "True" ]; then
        rag_suffix="RAG"
      else
        rag_suffix="no_RAG"
      fi

      OUT_PATH="${OUT_DIR}/Gemma_FRQ_${rag_suffix}_${modality^^}_Experiment_${bench^}_Results.csv"

      echo "Running ${modality^^}_${bench^^} + ${rag_suffix}"

      python "$SCRIPT" \
        --entity_modality "$modality" \
        --mock_RAG "$rag" \
        --model_name "$MODEL" \
        --bench_type "$bench" \
        --out_path "$OUT_PATH"

    done
  done
done

# ==============================
# Entity Identification Runs
# ==============================

echo "Running identify_entity.py for PEOPLE dataset"

python "$IDENTIFY_SCRIPT" \
  --bench_type people \
  --model_name "$MODEL" \
  --out_csv "${OUT_DIR}/Gemma_EntityID_People.csv"

echo "Running identify_entity.py for LOGO dataset"

python "$IDENTIFY_SCRIPT" \
  --bench_type logo \
  --model_name "$MODEL" \
  --out_csv "${OUT_DIR}/Gemma_EntityID_Logo.csv"

echo "All runs complete."