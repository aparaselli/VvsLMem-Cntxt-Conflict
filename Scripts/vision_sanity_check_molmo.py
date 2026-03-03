import os  # SPECIFIC TO ATHU OSCAR
os.environ["HF_HOME"] = "/oscar/scratch/aparasel/hf_cache"
os.environ["HF_HUB_CACHE"] = "/oscar/scratch/aparasel/hf_cache/hub"

from util import *  # assumes str2bool lives here
import argparse
import json
import os
from PIL import Image

import numpy as np
import pandas as pd
import torch

torch.set_grad_enabled(False)

from matplotlib import pyplot as plt
import seaborn as sns

sns.set(
    context="notebook",
    rc={
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16.0,
        "ytick.labelsize": 16.0,
        "legend.fontsize": 16.0,
    },
)
palette_ = sns.color_palette("Set1")
palette = palette_[2:5] + palette_[7:]
sns.set_theme(style="whitegrid")

from transformers import AutoProcessor, AutoModelForImageTextToText

################################
# Project-root + paths
################################

try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.getcwd())

PATH_TO_QUESTIONS = os.path.join(PROJECT_ROOT, "Questions.csv")
ACT_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "activations")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "Results", "Behavioral_Results.csv")
os.makedirs(ACT_SAVE_DIR, exist_ok=True)

################################
# Model setup (Molmo2)
################################

model_name = "allenai/Molmo2-8B"

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

print("Loading Molmo2 model")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto",
)
model.eval()

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
print("Model loaded")

################################
# RAG Simulation (Molmo2)
################################


def rag_model_call(retrieved_context, user_query, choice_ids, image=None, verbose=False):
    # Build the same “messages” structure, but we will tokenize via apply_chat_template
    if retrieved_context is None:
        text_before_image = (
            "Given your knowledge, answer the multiple choice question about the following entity.\n"
            "Entity: "
        )
    else:
        text_before_image = (
            "Context information is below.\n"
            "---------------------\n"
            f"{retrieved_context}\n"
            "---------------------\n"
            "Given the context information and your knowledge, answer the multiple choice question about the following entity.\n"
            "Entity: "
        )

    content_list = [{"type": "text", "text": text_before_image}]

    if image is not None:
        content_list.append({"type": "image", "image": image})

    content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})

    messages = [{"role": "user", "content": content_list}]

    # Force-choice prefix (same idea as your Qwen version)
    prefix = "The name of this company is  "
    messages[-1]["content"].append({"type": "text", "text": prefix})

    # Molmo2: build tensors via chat template (includes image handling)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move to device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(model.device)

    # Next-token probs for A/B/C/D (NOTE: assumes choice_ids are correct single-token IDs)
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        target_logits = next_token_logits[choice_ids].float()  # cast to fp32
        probs = torch.softmax(target_logits, dim=0).detach().cpu().numpy()
        prob_dict = {label: float(p) for label, p in zip(["A", "B", "C", "D"], probs)}

    # Greedy decode response
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = generated_ids[0, prompt_len:]
    response = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)

    if verbose:
        # Debug: decode the prompt tokens (may include specials)
        prompt_debug = processor.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=False
        )
        print("--- Prompt (Debug) ---")
        print(prompt_debug)
        print(f"--- Probs: {prob_dict}")
        print(f"--- Model Response: {response}")

    return response, prob_dict


def run_experiment_people(file_path, mock_RAG=True, out_path=None):
    with open(file_path, "r") as file:
        data = json.load(file)

    error_categories = {
        "Temporal_error": "mcq_query_temporal",
        "location_error": "mcq_query_location",
        "Career_error": "mcq_query_Career",
    }

    results = []
    print("Running people conflict experiment")

    #### PRE PROCESS ANSWER CHOICES SO WE CAN SEARCH THEM UP AT TOKEN GENERATION ####
    target_choices = [" A", " B", " C", " D"]
    choice_ids = []
    for c in target_choices:
        ids = processor.tokenizer.encode(c, add_special_tokens=False)
        # Keep your minimal behavior: take the last token id.
        # IMPORTANT: verify these are single-token on Molmo2; otherwise scoring is off.
        choice_ids.append(ids[-1])
    ###################################################################################

    # Optional sanity check (prints once)
    print("Choice tokenization sanity check:")
    for c in target_choices:
        ids = processor.tokenizer.encode(c, add_special_tokens=False)
        print(f"{repr(c)} -> {ids}")

    for item in data:
        instance_name = item["instance"]

        pil_image = None
        image_paths = item.get("image_path", [])

        if image_paths:
            first_img_path = image_paths[0]
            full_path = os.path.join(
                "/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data", first_img_path
            )
            try:
                if os.path.exists(full_path):
                    with Image.open(full_path) as im:
                        pil_image = im.convert("RGB")
                    print(f"Loaded {first_img_path} | Resolution: {pil_image.size}")
                    if max(pil_image.size) > 1024:
                        pil_image.thumbnail((1024, 1024))
                        print(f"Resized to: {pil_image.size}")
                else:
                    print(f"CRITICAL ERROR: Image not found at {full_path}")
            except Exception as e:
                print(f"Error loading image: {e}")

        for err_cat, query_key in error_categories.items():
            if err_cat in item["mis_knowledge"]:
                error_data = item["mis_knowledge"][err_cat]

                # Your code ignores base_query and uses a fixed prompt:
                specific_query = "Give me the name of the brand in the picture."

                # only use mis_knowledge1
                for k_key in ["mis_knowledge1"]:
                    if k_key in error_data:
                        idx = k_key.replace("mis_knowledge", "")

                        cat_suffix = query_key.split("_")[-1]
                        mcq_label_key = f"mcq_disanswer_{cat_suffix}_{idx}"
                        mis_mcq_label = item.get(mcq_label_key, "N/A")

                        if mock_RAG:
                            retrieved_context = error_data[k_key]
                        else:
                            retrieved_context = None

                        print(f"Processing {instance_name} | {err_cat} | {k_key}")

                        model_response, probs_map = rag_model_call(
                            retrieved_context=retrieved_context,
                            user_query="Who is the person in this image?",
                            choice_ids=choice_ids,
                            image=pil_image,
                            verbose=True,
                        )

                        P_A = probs_map.get("A", 0.0)
                        P_B = probs_map.get("B", 0.0)
                        P_C = probs_map.get("C", 0.0)
                        P_D = probs_map.get("D", 0.0)

                        results.append(
                            {
                                "ID": item["ID"],
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
                                "Ground_Truth": item.get(
                                    query_key.replace("query", "groundtruth"), "N/A"
                                ),
                                "Mis_Answer_Label": mis_mcq_label,
                            }
                        )

    df_results = pd.DataFrame(results)
    save_path = out_path
    df_results.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG experiment with optional context.")

    parser.add_argument(
        "--mock_RAG",
        type=str2bool,
        default=False,
        help="Set to True to use retrieved context, False to use empty context.",
    )

    parser.add_argument(
        "--file_path",
        type=str,
        default="/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/people_knowledge.json",
        help="Path to the input JSON file.",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        default="/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/WHO_VISION_Experiment_Results_molmo2_logo.csv",
        help="Path to the output csv file.",
    )

    args = parser.parse_args()
    run_experiment_people(args.file_path, mock_RAG=args.mock_RAG, out_path=args.out_path)
