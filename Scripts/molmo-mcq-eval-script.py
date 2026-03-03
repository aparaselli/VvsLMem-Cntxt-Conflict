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
from transformers import AutoModelForImageTextToText, AutoProcessor

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


model_name = "allenai/Molmo2-8B" # Changed model path

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

print("Loading Molmo model")
# Molmo uses AutoModelForCausalLM and requires trust_remote_code

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model.eval()

torch.backends.cuda.matmul.allow_tf32 = True  
torch.set_float32_matmul_precision("high")    
print("Model loaded")


# RAG Simulation 
def rag_model_call(
    retrieved_context,
    user_query,
    choice_ids,
    entity_name=None,
    image=None,
    verbose=False,
    ctxt_image=None,
):
    """
    Molmo2-compatible version of rag_model_call with the same control flow as your Qwen version.

    - Supports:
        * retrieved_context = None or string
        * ctxt_image = None or PIL.Image (optional "context image")
        * image = None or PIL.Image (the "query image")
        * entity_name provided for text-entity modality
    - Computes choice probs for next token after the forced prefix.
    - Generates a short completion and returns (response, prob_dict).
    """

    # -----------------------------
    # 1) Build messages (same logic)
    # -----------------------------
    if retrieved_context is None:
        if ctxt_image is not None:
            raise ValueError(
                "Error: 'ctxt_image' was provided, but 'retrieved_context' is None. "
                "A context image requires context text."
            )

        text_before_image = (
            "Given your knowledge, answer the multiple choice question about the following entity.\n"
            "Entity: "
        )
        # If no context and entity_name exists, we can just do a text-only prompt
        if entity_name is not None:
            prompt_text = (
                "Given your knowledge, answer the multiple choice question about the following entity.\n"
                f"Entity: {entity_name}.\n"
                f"Query: {user_query}"
            )
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}],
                }
            ]
        elif image is not None:
            content_list = [{"type": "text", "text": text_before_image}]
            content_list.append({"type": "image", "image": image})
            content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})
            messages = [{"role": "user", "content": content_list}]
        else:
            raise ValueError("Either image or entity_name must be provided when no context is used.")

    elif ctxt_image is None:
        # Context text only
        if entity_name is not None:
            prompt_text = (
                "Context information is below.\n"
                "---------------------\n"
                f"{retrieved_context}\n"
                "---------------------\n"
                "Given the context information and your knowledge, answer the multiple choice question about the following entity.\n"
                f"Entity: {entity_name}.\n"
                f"Query: {user_query}"
            )
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}],
                }
            ]
        elif image is not None:
            text_before_image = (
                "Context information is below.\n"
                "---------------------\n"
                f"{retrieved_context}\n"
                "---------------------\n"
                "Given the context information and your knowledge, answer the multiple choice question about the following entity.\n"
                "Entity: "
            )
            content_list = [{"type": "text", "text": text_before_image}]
            content_list.append({"type": "image", "image": image})
            content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})
            messages = [{"role": "user", "content": content_list}]
        else:
            raise ValueError("Either image or entity_name must be provided.")

    else:
        # Context image + context text
        text_before_image = (
            "Context information is below.\n"
            "---------------------\n"
        )
        content_list = [{"type": "text", "text": text_before_image}]

        if entity_name is not None:
            # (ctxt_image) + (retrieved_context + entity + query)
            content_list.append({"type": "image", "image": ctxt_image})
            txt_post_ctxt = (
                f"{retrieved_context}\n"
                "Given the context information and your knowledge, answer the multiple choice question about the following entity.\n"
                f"Entity: {entity_name}.\n"
                f"Query: {user_query}"
            )
            content_list.append({"type": "text", "text": txt_post_ctxt})

        elif image is not None:
            # (ctxt_image) + (retrieved_context + 'Entity:') + (query image) + (query text)
            content_list.append({"type": "image", "image": ctxt_image})
            txt_post_ctxt = (
                f"{retrieved_context}\n"
                "Given the context information and your knowledge, answer the multiple choice question about the following entity.\n"
                "Entity: "
            )
            content_list.append({"type": "text", "text": txt_post_ctxt})
            content_list.append({"type": "image", "image": image})
            content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})

        else:
            raise ValueError("Either image or entity_name must be provided.")

        messages = [{"role": "user", "content": content_list}]

    # -------------------------------------------------------
    # 2) Molmo2 wants dict(type="text", text=...) etc.
    #    Your current messages already match that, but we
    #    normalize to be safe.
    # -------------------------------------------------------
    def _molmoify_content(content_list):
        out = []
        for c in content_list:
            t = c.get("type")
            if t == "text":
                out.append({"type": "text", "text": c["text"]})
            elif t == "image":
                out.append({"type": "image", "image": c["image"]})
            elif t == "video":
                out.append({"type": "video", "video": c["video"]})
            else:
                raise ValueError(f"Unknown content type: {c}")
        return out

    molmo_messages = []
    for m in messages:
        molmo_messages.append(
            {"role": m["role"], "content": _molmoify_content(m["content"])}
        )

    # -------------------------------------------------------
    # 3) Force the next token to be one of A/B/C/D by appending
    #    your prefix exactly like before
    # -------------------------------------------------------
    prefix = "Between A, B, C, and D, the answer is"
    molmo_messages[-1]["content"].append({"type": "text", "text": prefix})

    # -------------------------------------------------------
    # 4) Build tensors via Molmo2 chat template
    # -------------------------------------------------------
    inputs = processor.apply_chat_template(
        molmo_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move all tensors to model.device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(model.device)

    # -------------------------------------------------------
    # 5) Next-token probs over your 4 answer tokens
    # -------------------------------------------------------
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]            # vocab
        target_logits = next_token_logits[choice_ids]          # 4
        probs = torch.softmax(target_logits, dim=0).detach().cpu().to(torch.float32).numpy()
        prob_dict = {label: float(p) for label, p in zip(["A", "B", "C", "D"], probs)}

    # -------------------------------------------------------
    # 6) Greedy decode response
    # -------------------------------------------------------
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

    # Decode only the new tokens beyond the prompt
    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = generated_ids[0, prompt_len:]
    response = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)

    if verbose:
        # Helpful debug: reconstruct prompt text (tokens -> string)
        # NOTE: this will include special tokens; good for debugging.
        prompt_debug = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        print("--- Prompt (Debug) ---")
        print(prompt_debug)
        print(f"--- Probs: {prob_dict}")
        print(f"--- Model Response: {response}")

    return response, prob_dict

def run_experiment(file_path, entity_modality='vision', mock_RAG=True,out_path=None, bench_type="people", ctxt_type="text"):

    with open(file_path, 'r') as file:
        data = json.load(file)

    if bench_type == "people":
        error_categories = {
            "Temporal_error": "mcq_query_temporal",
            "location_error": "mcq_query_location",
            "Career_error":   "mcq_query_Career"
        }

        results = []

        print("Running people conflict experiment")
    elif bench_type=="logo":
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
        entity_name=None
        pil_image = None
        pil_image_ctxt = None
        image_paths = item.get('image_path', [])

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


                # use 1 and 2 now 
                for k_key in ['mis_knowledge1', 'mis_knowledge2']:
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

                        if entity_modality == "text":
                            if bench_type == "logo":
                                entity_name = f"The logo of the company known as {instance_name}"
                            else:
                                entity_name = instance_name

                        model_response, probs_map = rag_model_call(
                            retrieved_context=retrieved_context,
                            user_query=specific_query,
                            choice_ids=choice_ids,
                            entity_name = entity_name,
                            image=pil_image,
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
        default="/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/MOLMO_Results/MOLMO-RAG_VISION_Experiment_Results.csv",
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