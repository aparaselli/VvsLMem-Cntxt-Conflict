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
from classify_response import classify_response
import re
import json
# ============
# Load Llama 3 8B Instruct as the judge
# ============
JUDGE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model.eval()

class Llama3Judge:
    """
    Provides judge_model.generate(prompt, temperature, max_tokens)
    compatible with your classify_response().
    """
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



def extract_answer(s: str) -> str:
    s = "" if s is None else str(s)

    # Try JSON parse if it looks JSON-ish
    try:
        obj = json.loads(s)
        # common keys
        for k in ["answer", "Answer", "final", "response"]:
            if k in obj:
                return str(obj[k]).strip()
    except Exception:
        pass

    # Otherwise: grab last quoted string or last line-like content
    # remove common wrappers
    s = re.sub(r"(?is)^.*?answer\"\s*:\s*", "", s)  # after answer":
    s = s.strip()

    # strip braces
    s = s.strip(" \n\t\r`\"'{}[]()")

    # if still contains lots of chat template, take last 1–2 lines
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) >= 1:
        s = lines[-1]

    # final strip
    s = s.split(":")[-1]
    return s.strip(" \n\t\r`\"'{}[]()")

def convert_clas(x):
    if (x == "CONTEXTUAL_ANS") or (x=="CONTEXTUAL_ANSWER"):
        return "mis_label"
    elif x =="PARAMETRIC_ANSWER":
        return "gt"
    elif x == "neither":
        return "neither"
    else:
        return x



if __name__ == "__main__":

    csvs_list = ["/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_RAG_TEXT_Experiment_Logo_Results.csv", 
    "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_RAG_TEXT_Experiment_People_Results.csv",
    "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_RAG_VISION_Experiment_Logo_Results.csv",
    "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_RAG_VISION_Experiment_People_Results.csv"]

    for csv_file in csvs_list:
        UPDATE_CSV = csv_file
        df = pd.read_csv(UPDATE_CSV)
        df["Assistant_response"] = df["Response"].apply(extract_answer)

        df["Pred"] = df.apply(
            lambda x: classify_response(
                query=x["Query"],
                parametric_ans=x["Ground_Truth"],
                contextual_ans=x["Mis_Answer_Label"],
                response=x["Assistant_response"],
                judge_model=judge_model,
            ),
            axis=1,
        )
        df['Pred'] = df['Pred'].apply(convert_clas)
        df.to_csv(UPDATE_CSV, index=False)