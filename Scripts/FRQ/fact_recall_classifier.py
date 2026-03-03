import os #SPECIFIC TO ATHU OSCAR
import sys
os.environ["HF_HOME"] = "/oscar/scratch/aparasel/hf_cache"
#os.environ["TRANSFORMERS_CACHE"] = "/oscar/scratch/aparasel/hf_cache/transformers"
#os.environ["DATASETS_CACHE"] = "/oscar/scratch/aparasel/hf_cache/datasets"
os.environ["HF_HUB_CACHE"] = "/oscar/scratch/aparasel/hf_cache/hub"

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (one level up)
parent_dir = os.path.dirname(current_dir)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from classify_response import classify_response_factual_recall
import re
import json


JUDGE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model.eval()

class Llama3Judge:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 5) -> str:
        # Llama 3 Instruct uses chat template
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt", padding=False).to(self.model.device)

        do_sample = (temperature is not None) and (temperature > 0.0)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = out[0, prompt_len:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

judge_model = Llama3Judge(model, tokenizer)



if __name__ == "__main__":

    csvs_list = ["/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_no_RAG_TEXT_Experiment_Logo_Results.csv", 
    "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_no_RAG_TEXT_Experiment_People_Results.csv",
    "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_no_RAG_VISION_Experiment_Logo_Results.csv",
    "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_no_RAG_VISION_Experiment_People_Results.csv"]
    model = "Qwen"
    if model == "Qwen":
        split_wrd = "assistant"
    elif model =="Gemma":
        split_wrd = "model"
    else:
        split_wrd = None
        print("Model type not supported")
    for csv_file in csvs_list:

        UPDATE_CSV = csv_file

        df = pd.read_csv(UPDATE_CSV)


        df['Response_iso'] = df['Response'].apply(lambda x: x.split(split_wrd)[-1])
        df["Pred"] = df.apply(
            lambda x: classify_response_factual_recall(
                query=x["Query"],
                parametric_ans=x["Ground_Truth"],
                response=x['Response_iso'],
                judge_model=judge_model,
            ),
            axis=1,
        )
        df.to_csv(UPDATE_CSV, index=False)