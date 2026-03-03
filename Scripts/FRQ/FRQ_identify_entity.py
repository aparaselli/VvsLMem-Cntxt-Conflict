#!/usr/bin/env python3
# identify_entity.py

import os
import sys
import json
import argparse
from typing import Tuple, Optional

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText

# ============
# Oscar cache (adjust if needed)
# ============
os.environ["HF_HOME"] = "/oscar/scratch/aparasel/hf_cache"
os.environ["HF_HUB_CACHE"] = "/oscar/scratch/aparasel/hf_cache/hub"

torch.set_grad_enabled(False)


def load_vlm(model_name: str):
    """
    Minimal loader:
    - Gemma: AutoModelForImageTextToText (+ left padding)
    - Qwen:  AutoModelForVision2Seq
    """
    name_lc = model_name.lower()
    is_gemma = ("gemma-3" in name_lc) or ("gemma3" in name_lc)

    # Use bf16 if supported; else fp16
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if (is_gemma and bf16_ok) else torch.float16

    print(f"Loading model: {model_name} | is_gemma={is_gemma} | dtype={dtype}")

    if is_gemma:
        processor = AutoProcessor.from_pretrained(model_name, padding_side="left")
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    model.eval()
    print("Model loaded.")
    return model, processor


def build_inputs(processor, model_device, prompt_text: str, image: Image.Image):
    """
    Creates model inputs for common VLM processors.
    Works for Qwen2.5-VL and Gemma-3 style processors in most cases.
    """
    # Many VLM processors accept: text=..., images=...
    inputs = processor(
        text=prompt_text,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    # Move tensors to model device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(model_device)
    return inputs


@torch.no_grad()
def identify_one(model, processor, image: Image.Image, *, max_new_tokens: int = 16) -> str:
    prompt = "What is the name of the entity in the image? (Reply in 1-3 words)"
    inputs = build_inputs(processor, model.device, prompt, image)

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # Some VLMs echo the prompt; keep a simple heuristic to return the last line.
    # (Still "basic"—you can tighten later.)
    if "\n" in text:
        text = text.split("\n")[-1].strip()

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench_type",
        type=str,
        default="people",
        choices=["people", "logo"],
        help="Which dataset json to use.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-12b-it",
        help="HF model id (e.g., google/gemma-3-12b-it or Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data",
        help="Root directory containing the json and image files.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/entity_id_results.csv",
        help="Where to save results CSV.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Generation length cap (entity name should be short).",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=-1,
        help="For debugging: max number of items to run (-1 = all).",
    )

    args = parser.parse_args()

    if args.bench_type == "people":
        json_path = os.path.join(args.data_root, "people_knowledge.json")
    else:
        json_path = os.path.join(args.data_root, "logo_knowledge.json")

    print("Loading data:", json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    if args.max_items is not None and args.max_items > 0:
        data = data[: args.max_items]

    model, processor = load_vlm(args.model_name)

    rows = []
    for item in tqdm(data, desc="Identifying entities"):
        item_id = item.get("ID", None)
        instance = item.get("instance", None)
        image_paths = item.get("image_path", []) or []

        if not image_paths:
            rows.append(
                {
                    "ID": item_id,
                    "Instance": instance,
                    "ImageRelPath": None,
                    "ImageFullPath": None,
                    "Pred_Entity": None,
                    "Error": "no_image_path",
                }
            )
            continue

        rel_path = image_paths[0]
        full_path = os.path.join(args.data_root, rel_path)

        if not os.path.exists(full_path):
            rows.append(
                {
                    "ID": item_id,
                    "Instance": instance,
                    "ImageRelPath": rel_path,
                    "ImageFullPath": full_path,
                    "Pred_Entity": None,
                    "Error": "image_not_found",
                }
            )
            continue

        try:
            img = Image.open(full_path).convert("RGB")
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024))

            pred = identify_one(
                model,
                processor,
                img,
                max_new_tokens=args.max_new_tokens,
            )

            rows.append(
                {
                    "ID": item_id,
                    "Instance": instance,
                    "ImageRelPath": rel_path,
                    "ImageFullPath": full_path,
                    "Pred_Entity": pred,
                    "Error": None,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "ID": item_id,
                    "Instance": instance,
                    "ImageRelPath": rel_path,
                    "ImageFullPath": full_path,
                    "Pred_Entity": None,
                    "Error": repr(e),
                }
            )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()