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

# Add parent directory to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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
from FRQ_make_input import make_inputs

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
def rag_model_call(
    retrieved_context,
    user_query,
    entity_name=None,
    image=None,
    verbose=False,
    ctxt_image=None,
    err_cat=None,
    *,
    processor,
    model,
    bench_type="people",
    entity_modality=None,
    mock_RAG=True,
    ctxt_type="text",   # add this so the function knows whether to use ctxt_image
):
    # infer modality
    if entity_modality is None:
        entity_modality = "text" if entity_name is not None else "vision"

    if entity_modality == "text" and entity_name is None:
        raise ValueError("entity_name required for entity_modality='text'")
    if entity_modality == "vision" and image is None:
        raise ValueError("image required for entity_modality='vision'")

    if retrieved_context is None and ctxt_image is not None:
        raise ValueError("ctxt_image provided but retrieved_context is None")



    # Build via make_inputs (must support prefix; ctxt_image optional)
    inputs = make_inputs(
        processor=processor,
        model_device=model.device,
        retrieved_context=retrieved_context,
        user_query=user_query,
        entity_modality=entity_modality,
        mock_RAG=mock_RAG,
        bench_type=bench_type,
        image=image,
        instance_name=entity_name,
        padding=True,
        # if you extend make_inputs:
        # ctxt_image=(ctxt_image if ctxt_type == "vision" else None),
    )
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        # ---- GLOBAL ARGMAX TOKEN (sanity check) ----
        top_token_id = torch.argmax(next_token_logits).item()
        top_token_str = processor.tokenizer.decode([top_token_id])
    if verbose:
        print("--- RAG Prompt (Full) ---")
        print(inputs["_prompt_text"])

    gen_kwargs = {k: v for k, v in inputs.items() if not str(k).startswith("_")}
    generated_ids = model.generate(**gen_kwargs, max_new_tokens=128, do_sample=False)

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    if verbose:
        print(f"--- Model Response: {response}")

    return response, top_token_str



def run_experiment(file_path, entity_modality='vision', mock_RAG=True,out_path=None, bench_type="people", ctxt_type="text"):

    with open(file_path, 'r') as file:
        data = json.load(file)

    if bench_type == "people":
        error_categories = {
            "Temporal_error": "open_query_temporal",
            "location_error": "open_query_location",
            "Career_error":   "open_query_Career"
        }

        results = []

        print("Running people conflict experiment")
    elif bench_type=="logo":
        error_categories = {
            "time_error": "open_query_time",
            "creator_error": "open_query_creator",
            "content_error":   "open_query_content"
        }

        results = []

        print("Running logo conflict experiment")

    # #### PRE PROCESS ANSWER CHOICES SO WE CAN SEARCH THEM UP AT TOKEN GENERATION ####
    # target_choices = [" A", " B", " C", " D"]
    # choice_ids = []
    # for c in target_choices:
    #     tid = processor.tokenizer.encode(c, add_special_tokens=False)[-1]
    #     choice_ids.append(tid)
    # ###################################################################################

    for item in data:
        instance_name = item['instance']
        entity_name=None
        pil_image = None
        pil_image_ctxt = None
        image_paths = item.get('image_path', [])
        item_lc = {k.lower(): v for k, v in item.items()} #fir gettung gt ans later

        if image_paths:
            if entity_modality=="vision":
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
            if ctxt_type=="vision":
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
                    specific_query = base_query.replace("the person in the picture", "the entity")
                    specific_query = specific_query.replace("The person pictured", "the entity") 
                elif bench_type=="logo":
                    specific_query = base_query

                #Get true answer
                gt_ans = item_lc[f'open_groundtruth_{err_cat.split("_")[0].lower()}']

                # use 1 and 2 now 
                for k_key in ['mis_knowledge1', 'mis_knowledge2']:
                    if k_key in error_data:
                        idx = k_key.replace("mis_knowledge", "")
                        
                        ans_key = f"mis_answer_{idx}"
                        mis_text_answer = error_data.get(ans_key, "N/A")

                        cat_suffix = query_key.split('_')[-1] 
                        #mcq_label_key = f"mcq_disanswer_{cat_suffix}_{idx}"
                        #mis_mcq_label = item.get(mcq_label_key, "N/A")
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

                        if entity_modality == "text":
                            if bench_type == "logo":
                                entity_name = f"The company known as {instance_name}"
                            else:
                                entity_name = instance_name


                        model_response, top_tok_str = rag_model_call(
                            retrieved_context=retrieved_context,
                            user_query=specific_query,
                            entity_name=entity_name,
                            entity_modality=entity_modality,
                            image=(pil_image if entity_modality == "vision" else None),
                            verbose=True,
                            #ctxt_image=pil_image_ctxt,
                            err_cat=err_cat,
                            processor=processor,
                            model=model,
                            bench_type=bench_type,
                            mock_RAG=mock_RAG,
                            ctxt_type=ctxt_type,
                        )

                        print(f"Top decoded next token is {top_tok_str}")
                        #### SEARCH UP ANSWER CHOICES AT FRIST GEN TOKEN AND STORE THEM IN VARIABLES ####
                        # P_A = probs_map.get("A", 0.0)
                        # P_B = probs_map.get("B", 0.0)
                        # P_C = probs_map.get("C", 0.0)
                        # P_D = probs_map.get("D", 0.0)
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
                            "Top_token_str": top_tok_str,
                            # "Prob_A": P_A,
                            # "Prob_B": P_B,
                            # "Prob_C": P_C,
                            # "Prob_D": P_D,
                            "Ground_Truth": gt_ans,
                            "Mis_Answer_Label": mis_text_answer
                        })


    df_results = pd.DataFrame(results)
    save_path = out_path
    df_results.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG experiment with optional context.")

    parser.add_argument(
        '--entity_modality', 
        type=str, 
        default="vision",
        choices=["text", "vision"], 
        help='Modality for entity in query (text or vision)'
    )
    
    parser.add_argument(
        '--mock_RAG', 
        type=str2bool, 
        default=True, 
        help='Set to True to use retrieved context, False to use empty context.'
    )

    parser.add_argument(
        '--out_path', 
        type=str, 
        default="/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/FRQ_RAG_VISION_Experiment_Results.csv",
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
    run_experiment(file_path, entity_modality=args.entity_modality, mock_RAG=args.mock_RAG,out_path=args.out_path,bench_type=args.bench_type, ctxt_type=args.ctxt_type)