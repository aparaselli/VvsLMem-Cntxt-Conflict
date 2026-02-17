import os #SPECIFIC TO ATHU OSCAR
os.environ["HF_HOME"] = "/oscar/scratch/aparasel/hf_cache"
#os.environ["TRANSFORMERS_CACHE"] = "/oscar/scratch/aparasel/hf_cache/transformers"
#os.environ["DATASETS_CACHE"] = "/oscar/scratch/aparasel/hf_cache/datasets"
os.environ["HF_HUB_CACHE"] = "/oscar/scratch/aparasel/hf_cache/hub"

from util import *
import argparse
from ast import literal_eval
import functools
import json
import os
import random
import shutil
from PIL import Image

from io import BytesIO


import numpy as np
import pandas as pd
import torch
import datasets
torch.set_grad_enabled(False)

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(context="notebook",
        rc={"font.size":16,
            "axes.titlesize":16,
            "axes.labelsize":16,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "legend.fontsize": 16.0})
palette_ = sns.color_palette("Set1")
palette = palette_[2:5] + palette_[7:]
sns.set_theme(style='whitegrid')

from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor

################################
# Project-root + paths
################################

try:
    # When running as a script
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # When running in a notebook / REPL — assume cwd is repo root
    PROJECT_ROOT = os.path.abspath(os.getcwd())

PATH_TO_QUESTIONS = os.path.join(PROJECT_ROOT, "Questions.csv")
ACT_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "activations")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "Results", "Behavioral_Results.csv")
os.makedirs(ACT_SAVE_DIR, exist_ok=True)


model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_name(0).startswith(("NVIDIA A100","NVIDIA H100")) else torch.float16

print("Loading model")
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    dtype=dtype,               
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa", 
)
processor = AutoProcessor.from_pretrained(model_name)
model.eval()

torch.backends.cuda.matmul.allow_tf32 = True  
torch.set_float32_matmul_precision("high")    
print("Model loaded")


# RAG Simulation 
def rag_model_call(retrieved_context, user_query, choice_ids, image=None, verbose=False):
    if (retrieved_context is None):
        text_before_image = (
            f"Given your knowledge, answer the multiple choice question about the following entity.\n"
            f"Entity: " 
        )
    else:
        text_before_image = (
            f"Context information is below.\n"
            f"---------------------\n"
            f"{retrieved_context}\n"
            f"---------------------\n"
            f"Given the context information and your knowledge, answer the multiple choice question about the following entity.\n"
            f"Entity: " 
        )

    content_list = [{"type": "text", "text": text_before_image}]

    if image is not None:
        content_list.append({"type": "image", "image": image})
        
    content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})

    messages = [
        {
            "role": "user",
            "content": content_list
        }
    ]

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    prefix = "The name of the company logo in the picture is "
    text += prefix

    image_inputs = {}
    if image is not None:
        image_inputs = {"images": [image], "videos": None}

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
        **image_inputs
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        target_logits = next_token_logits[choice_ids]
        probs = torch.softmax(target_logits, dim=0).cpu().numpy()
        prob_dict = {label: float(p) for label, p in zip(["A", "B", "C", "D"], probs)}

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=128,
        do_sample=False 
    )
    response = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    if verbose:
        print("--- RAG Prompt (Snippet) ---")
        print(text_before_image + " [IMAGE] " + user_query)
        print(f"--- Probs: {prob_dict}")
        print(f"--- Model Response: {response}")
        
    return response, prob_dict

def run_experiment_people(file_path, mock_RAG=True,out_path=None):

    with open(file_path, 'r') as file:
        data = json.load(file)

    error_categories = {
        "Temporal_error": "mcq_query_temporal",
        "location_error": "mcq_query_location",
        "Career_error":   "mcq_query_Career"
    }

    results = []

    print("Running people conflict experiment")

    #### PRE PROCESS ANSWER CHOICES SO WE CAN SEARCH THEM UP AT TOKEN GENERATION ####
    target_choices = [" A", " B", " C", " D"]
    choice_ids = []
    for c in target_choices:
        tid = processor.tokenizer.encode(c, add_special_tokens=False)[-1]
        choice_ids.append(tid)
    ###################################################################################

    for item in data:
        instance_name = item['instance']

        pil_image = None
        image_paths = item.get('image_path', [])

        if image_paths:
            first_img_path = image_paths[0]
            full_path = os.path.join("/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data", first_img_path)  
            try:
                if os.path.exists(full_path):
                    pil_image = Image.open(full_path)
                    #original_image = pil_image.copy()
                    print(f"Loaded {first_img_path} | Resolution: {pil_image.size}")
                    if max(pil_image.size) > 1024:
                        pil_image.thumbnail((1024, 1024))
                        print(f"Resized to: {pil_image.size}")
                else:
                    print(f"CRITICAL ERROR: Image not found at {full_path}")
            except Exception as e:
                print(f"Error loading image: {e}")
        
        for err_cat, query_key in error_categories.items():
            
            # Check if this error category exists for this item
            if err_cat in item['mis_knowledge']:
                error_data = item['mis_knowledge'][err_cat]
                
                base_query = item[query_key]
                
                specific_query = "Give me the name of the brand in the picture."


                # only use mis_knowledge1
                for k_key in ['mis_knowledge1']:
                    if k_key in error_data:
                        idx = k_key.replace("mis_knowledge", "")
                        
                        ans_key = f"mis_answer_{idx}"
                        mis_text_answer = error_data.get(ans_key, "N/A")

                        cat_suffix = query_key.split('_')[-1] 
                        mcq_label_key = f"mcq_disanswer_{cat_suffix}_{idx}"
                        mis_mcq_label = item.get(mcq_label_key, "N/A")
                        if mock_RAG:
                            retrieved_context = error_data[k_key]
                        else:
                            retrieved_context = None
                        
                        print(f"Processing {instance_name} | {err_cat} | {k_key}")
                        
                        model_response, probs_map = rag_model_call(
                            retrieved_context=retrieved_context,
                            user_query=specific_query,
                            choice_ids=choice_ids,
                            image=pil_image,
                            verbose=True
                        )

                        #### SEARCH UP ANSWER CHOICES AT FRIST GEN TOKEN AND STORE THEM IN VARIABLES ####
                        P_A = probs_map.get("A", 0.0)
                        P_B = probs_map.get("B", 0.0)
                        P_C = probs_map.get("C", 0.0)
                        P_D = probs_map.get("D", 0.0)
                        #################################

                        
                        # Store result
                        results.append({
                            "ID": item['ID'],
                            "Instance": instance_name,
                            "Category": err_cat,
                            "Mis_Knowledge_Key": k_key,
                            "Context": retrieved_context,
                            "Query": specific_query,
                            "Response": model_response,
                            "Prob_A": P_A,
                            "Prob_B": P_B,
                            "Prob_C": P_C,
                            "Prob_D": P_D,
                            "Ground_Truth": item.get(query_key.replace("query", "groundtruth"), "N/A"),
                            "Mis_Answer_Label": mis_mcq_label
                        })


    df_results = pd.DataFrame(results)
    save_path = out_path
    df_results.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG experiment with optional context.")
    
    parser.add_argument(
        '--mock_RAG', 
        type=str2bool, 
        default=False, 
        help='Set to True to use retrieved context, False to use empty context.'
    )
    
    parser.add_argument(
        '--file_path', 
        type=str, 
        default="/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/logo_knowledge.json",
        help='Path to the input JSON file.'
    )

    parser.add_argument(
        '--out_path', 
        type=str, 
        default="/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/WHO_VISION_Experiment_Results_logo.csv",
        help='Path to the output csv file.'
    )

    args = parser.parse_args()
    run_experiment_people(args.file_path, mock_RAG=args.mock_RAG,out_path=args.out_path)