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
def rag_model_call(retrieved_context, entity, user_query, choice_ids, image=None, verbose=False, ctxt_image = None):
    if (retrieved_context is None):
        if ctxt_image is not None:
            raise ValueError("Error: 'ctxt_image' was provided, but 'retrieved_context' is None. A context image requires context text.")
        prompt_text = (
            f"Given your knowledge, answer the multiple choice question about the following entity.\n"
            f"Entity: {entity}.\n"
            f"Query: {user_query}"
        )
    elif ctxt_image is None:
        # RAG prompt
        prompt_text = (
            f"Context information is below.\n"
            f"---------------------\n"
            f"{retrieved_context}\n"
            f"---------------------\n"
            f"Given the context information and your knowledge, answer the multiple choice question about the following entity.\n"
            f"Entity: {entity}.\n"
            f"Query: {user_query}"
        )
    else: #image context
        text_before_image = (
            f"Context information is below.\n"
            f"---------------------\n"
        )
        content_list = [{"type": "text", "text": text_before_image}]

        if ctxt_image is not None:
            content_list.append({"type": "image", "image": ctxt_image})
            txt_post_ctxt = (
                f"{retrieved_context}\n"
                f"Given the context information and your knowledge, answer the multiple choice question about the following entity.\n"
                f"Entity: {entity}.\n"
                f"Query: {user_query}"
                )
            content_list.append({"type": "text", "text": txt_post_ctxt})

        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]

    if ctxt_image is None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    prefix = "Between A, B, C, and D, the answer is "
    text += prefix

    image_inputs = {} # Empty if text-only
    if ctxt_image is not None:
        image_inputs = {"images": [ctxt_image], "videos": None}


    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
        **image_inputs
    )
    inputs = inputs.to(model.device)


    #Assess what the answer was at the token level 
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        
        target_logits = next_token_logits[choice_ids]
        
        probs = torch.softmax(target_logits, dim=0).cpu().numpy()

        prob_dict = {label: float(p) for label, p in zip(["A", "B", "C", "D"], probs)}


    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=128,
        do_sample=False  # Deterministic for RAG testing
    )
    response = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    if verbose:
        print("\n--- RAG Prompt Start ---")
        
        if ctxt_image is not None:
            print("[INFO] Context Image was included in this prompt.")
            
        print(text) 
        print("--- RAG Prompt End ---\n")
        print(f"--- Model Response: {response}")
    return response, prob_dict

def run_experiment(file_path, mock_RAG=True, out_path=None, bench_type="people", ctxt_type="text"):

    with open(file_path, 'r') as file:
        data = json.load(file)

    if bench_type=="people":
        error_categories = {
            "Temporal_error": "mcq_query_temporal",
            "location_error": "mcq_query_location",
            "Career_error":   "mcq_query_Career"
        }

        results = []

        print("Running people conflict experiment")
    elif bench_type=="logo":
        print("Logo error categories")
        error_categories = {
            "time_error": "mcq_query_time",
            "creator_error": "mcq_query_creator",
            "content_error":   "mcq_query_content"
        }

        results = []

        print("Running logo conflict experiment")

    #### PRE PROCESS ANSWER CHOICES SO WE CAN SEARCH THEM UP AT TOKEN GENERATION ####
    target_choices = [" A", " B", " C", " D"]
    choice_ids = []
    for c in target_choices:
        tid = processor.tokenizer.encode(c, add_special_tokens=False)[-1]
        choice_ids.append(tid)
    ###################################################################################

    for item in data:
        instance_name = item['instance']
        pil_image_ctxt = None
        image_paths = item.get('image_path', [])

        #If context is vision load the second image
        if image_paths and ctxt_type=="vision":
            second_img_path = image_paths[1]
            full_path_2 = os.path.join("/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data", second_img_path)  
            try:
                if os.path.exists(full_path_2):
                    pil_image_ctxt = Image.open(full_path_2)
                    #original_image = pil_image.copy()
                    print(f"Loaded {second_img_path} | Resolution: {pil_image_ctxt.size}")
                    if max(pil_image_ctxt.size) > 1024:
                        pil_image_ctxt.thumbnail((1024, 1024))
                        print(f"Resized to: {pil_image_ctxt.size}")
                else:
                    print(f"CRITICAL ERROR: Image not found at {full_path_2}")
            except Exception as e:
                print(f"Error loading image: {e}")
        
        for err_cat, query_key in error_categories.items():
            
            # Check if this error category exists for this item
            if err_cat in item['mis_knowledge']:
                error_data = item['mis_knowledge'][err_cat]
                
                base_query = item[query_key]
                
                if bench_type=="people":
                    # Text only replacement for people
                    specific_query = base_query.replace("the person in the picture", "the entity")
                    specific_query = specific_query.replace("The person pictured", "the entity") 
                elif bench_type=="logo":
                    specific_query = base_query

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
                            if ctxt_type == "vision":
                                if bench_type == "people":
                                    retrieved_context = retrieved_context.replace(instance_name, "The person pictured")
                                elif bench_type=="logo":
                                    retrieved_context = retrieved_context.replace(instance_name, "The brand pictured")                        
                        else:
                            retrieved_context = None
                        
                        print(f"Processing {instance_name} | {err_cat} | {k_key}")
                        
                        if bench_type == "logo":
                            entity_name = f"The logo of the company known as {instance_name}"
                        else:
                            entity_name = instance_name

                        model_response, probs_map = rag_model_call(
                            retrieved_context=retrieved_context,
                            entity=entity_name,
                            user_query=specific_query,
                            choice_ids=choice_ids,
                            image=None,
                            verbose=True,
                            ctxt_image=pil_image_ctxt
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
        default=True, 
        help='Set to True to use retrieved context, False to use empty context.'
    )
    
    parser.add_argument(
        '--out_path', 
        type=str, 
        default="/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/RAG_Experiment_Results.csv",
        help='Path to the output csv file.'
    )

    parser.add_argument(
        '--bench_type', 
        type=str, 
        default="people",
        choices=["people", "logo"],
        help='Benchmark type (people or logo)'
    )

    parser.add_argument(
        '--ctxt_type', 
        type=str, 
        default="text",
        choices=["text", "vision"],
        help='Benchmark type (text or vision)'
    )

    args = parser.parse_args()
    if args.bench_type == "people":
        file_path = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/people_knowledge.json"
    elif args.bench_type == "logo":
        file_path = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/logo_knowledge.json"
    run_experiment(file_path, mock_RAG=args.mock_RAG,out_path=args.out_path,bench_type=args.bench_type, ctxt_type=args.ctxt_type)