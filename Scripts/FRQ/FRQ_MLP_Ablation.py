#!/usr/bin/env python3
# ============================================================
# Component Ablation via Forward Hooks — Qwen2.5-VL / Gemma-3
#
# What this does:
#  - Runs your FRQ pipeline (same make_inputs + filter_df_for_analysis)
#  - Registers forward hooks that ZERO either:
#       * MLP output update   (--ablate_component mlp)
#       * Attention output    (--ablate_component attn)
#    at either:
#       * token positions only (--ablate_scope positions)  [vision tokens or entity span]
#       * all tokens           (--ablate_scope all)
#  - Sweeps progressively more layers:
#       layers [dst_start .. dst_start + k] for k=0..(num_layers-1)
#  - Writes one CSV per sweep step (suffix: _k{K})
#
# Notes:
#  - This is NOT “patch previous layer in”; it is pure zero-ablation of a component’s
#    residual update.
#  - For head-level ablation you’d need deeper model-specific intervention; this script
#    is module-level (attn vs mlp).
# ============================================================

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple, Union
import re

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForVision2Seq, AutoProcessor
try:
    from transformers import Gemma3ForConditionalGeneration
except Exception:
    Gemma3ForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

from FRQ_data_filtering_utils import filter_df_for_analysis
from FRQ_make_input import make_inputs

# -------------------------
# Paths (yours)
# -------------------------
os.environ["HF_HOME"] = "/oscar/scratch/aparasel/hf_cache"
os.environ["HF_HUB_CACHE"] = "/oscar/scratch/aparasel/hf_cache/hub"

QWEN_CELEB_VIS_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_RAG_VISION_Experiment_People_Results.csv"
QWEN_CELEB_TEXT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_RAG_TEXT_Experiment_People_Results.csv"
GEMMA_CELEB_VIS_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Gemma_FRQ_RAG_VISION_Experiment_People_Results.csv"
GEMMA_CELEB_TEXT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Gemma_FRQ_RAG_TEXT_Experiment_People_Results.csv"

QWEN_CELEB_NO_RAG_TEXT_CSV   = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_no_RAG_TEXT_Experiment_People_Results.csv"
QWEN_CELEB_NO_RAG_VISION_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_no_RAG_VISION_Experiment_People_Results.csv"
GEMMA_CELEB_NO_RAG_TEXT_CSV   = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Gemma_FRQ_no_RAG_TEXT_Experiment_People_Results.csv"
GEMMA_CELEB_NO_RAG_VISION_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Gemma_FRQ_no_RAG_VISION_Experiment_People_Results.csv"

JSON_DATA = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/people_knowledge.json"
MLLMKC_ROOT = "/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data"

# -------------------------
# Model boundary helpers
# -------------------------
def get_language_model(m):
    if hasattr(m, "model") and hasattr(m.model, "language_model"):
        return m.model.language_model
    if hasattr(m, "language_model"):
        return m.language_model
    for _, mod in m.named_modules():
        if hasattr(mod, "layers"):
            return mod
    raise AttributeError("Couldn't find language_model on this model.")

def get_layers(m):
    lm = get_language_model(m)
    if hasattr(lm, "layers"):
        return lm.layers
    raise AttributeError("language_model has no .layers")

# -------------------------
# Token position finders (same as your robust ones)
# -------------------------
def get_visual_positions_from_input_ids_qwenstyle(input_ids_1d, tokenizer):
    vs_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    ve_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    ids = input_ids_1d.tolist()
    if vs_id is None or ve_id is None or vs_id < 0 or ve_id < 0:
        return torch.empty(0, dtype=torch.long, device=input_ids_1d.device)

    try:
        s = ids.index(vs_id)
        e = ids.index(ve_id)
    except ValueError:
        return torch.empty(0, dtype=torch.long, device=input_ids_1d.device)

    if e <= s + 1:
        return torch.empty(0, dtype=torch.long, device=input_ids_1d.device)

    return torch.arange(s + 1, e, device=input_ids_1d.device, dtype=torch.long)

def get_visual_positions(processor, input_ids_1d: torch.Tensor) -> torch.LongTensor:
    tok = processor.tokenizer
    pos = get_visual_positions_from_input_ids_qwenstyle(input_ids_1d, tok)
    if pos.numel() > 0:
        return pos

    candidate_ids = set()
    for attr in ["image_token_id", "img_token_id", "vision_token_id"]:
        if hasattr(tok, attr) and getattr(tok, attr) is not None:
            try:
                candidate_ids.add(int(getattr(tok, attr)))
            except Exception:
                pass

    for s in [
        "<image>", "<IMAGE>", "<img>", "<|image|>", "<|image_pad|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|image_start|>", "<|image_end|>",
        "<|vision_pad|>", "<|visual_pad|>",
    ]:
        try:
            tid = tok.convert_tokens_to_ids(s)
            if tid is not None and tid >= 0:
                candidate_ids.add(int(tid))
        except Exception:
            pass

    if not candidate_ids:
        return torch.empty(0, dtype=torch.long, device=input_ids_1d.device)

    ids = input_ids_1d
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for tid in candidate_ids:
        mask |= (ids == tid)

    return mask.nonzero(as_tuple=False).view(-1).to(dtype=torch.long)

def _find_subseq(haystack: List[int], needle: List[int]):
    if len(needle) == 0:
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i, i + len(needle)
    return None

def get_entity_positions_text_subseq(inputs, processor) -> torch.LongTensor:
    tok = processor.tokenizer
    device = inputs["input_ids"].device
    prompt = inputs.get("_prompt_text", "") or ""
    if not prompt:
        return torch.empty(0, dtype=torch.long, device=device)

    m = re.search(r"Entity\s*:\s*", prompt)
    if not m:
        return torch.empty(0, dtype=torch.long, device=device)
    ent_start = m.end()

    m2 = re.search(r"Query\s*:\s*", prompt[ent_start:])
    if not m2:
        return torch.empty(0, dtype=torch.long, device=device)
    ent_end = ent_start + m2.start()

    entity_str = prompt[ent_start:ent_end].strip()
    if not entity_str:
        return torch.empty(0, dtype=torch.long, device=device)

    entity_str_nodot = entity_str.rstrip(" .\n\t")
    ids_model = inputs["input_ids"][0].detach().cpu().tolist()

    candidates = []
    for base in [entity_str, entity_str_nodot]:
        if not base:
            continue
        candidates.extend([
            base,
            base + ".",
            " " + base,
            "\n" + base,
            "\n\n" + base,
            " " + base + ".",
            "\n" + base + ".",
        ])

    best = None
    best_len = -1
    for s in candidates:
        ent_ids = tok(s, add_special_tokens=False)["input_ids"]
        loc = _find_subseq(ids_model, ent_ids)
        if loc is not None and len(ent_ids) > best_len:
            best = loc
            best_len = len(ent_ids)

    if best is None:
        return torch.empty(0, dtype=torch.long, device=device)

    a, b = best
    return torch.arange(a, b, device=device, dtype=torch.long)

# -------------------------
# Small utilities
# -------------------------
def strip_special(keys_dict: Dict) -> Dict:
    return {k: v for k, v in keys_dict.items() if not str(k).startswith("_")}

def safe_decode(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""

def make_noise_image_like(pil_image: Optional[Image.Image], *, seed: Optional[int] = None) -> Image.Image:
    if pil_image is None:
        W, H = 224, 224
    else:
        W, H = pil_image.size
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")

# -------------------------
# Submodule getters (robust-ish across Qwen/Gemma)
# -------------------------
def get_attn_module(layer):
    # common names: self_attn, attn, attention
    for name in ["self_attn", "attn", "attention"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise AttributeError(f"Layer has no attention module among ['self_attn','attn','attention']: {type(layer)}")

def get_mlp_module(layer):
    # common names: mlp, feed_forward, ff, ffn
    for name in ["mlp", "feed_forward", "ffn", "ff"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    # sometimes: layer.mlp is nested
    raise AttributeError(f"Layer has no MLP module among ['mlp','feed_forward','ffn','ff']: {type(layer)}")

# -------------------------
# Output tuple handling
# -------------------------
TensorOrTuple = Union[torch.Tensor, Tuple, List]

def _first_tensor_index(x: TensorOrTuple) -> Optional[int]:
    if torch.is_tensor(x):
        return 0
    if isinstance(x, (tuple, list)):
        for i, v in enumerate(x):
            if torch.is_tensor(v):
                return i
    return None

def _get_first_tensor(x: TensorOrTuple) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for v in x:
            if torch.is_tensor(v):
                return v
    raise TypeError(f"Cannot find tensor in output type={type(x)}")

def _replace_first_tensor(x: TensorOrTuple, new_t: torch.Tensor) -> TensorOrTuple:
    if torch.is_tensor(x):
        return new_t
    if isinstance(x, tuple):
        lst = list(x)
        idx = _first_tensor_index(x)
        if idx is None:
            return x
        lst[idx] = new_t
        return tuple(lst)
    if isinstance(x, list):
        idx = _first_tensor_index(x)
        if idx is None:
            return x
        x[idx] = new_t
        return x
    return x

# -------------------------
# Core: hook that zeros component output update
# -------------------------
def zero_update_at_positions(
    update: torch.Tensor,  # expected [B,T,H] (or [T,H] sometimes)
    pos_1d: Optional[torch.LongTensor],
    scope: str,
) -> torch.Tensor:
    """
    Returns a tensor with the same shape as `update`, with values zeroed either:
      - scope == "all": all tokens
      - scope == "positions": only tokens at indices in pos_1d
    """
    if update is None or not torch.is_tensor(update):
        return update

    if scope == "all":
        return torch.zeros_like(update)

    # positions scope
    if pos_1d is None or pos_1d.numel() == 0:
        return update

    # Normalize to [B,T,H]
    if update.dim() == 2:
        # [T,H] -> pretend B=1
        upd = update.clone()
        T = upd.shape[0]
        pos = pos_1d.to(device=upd.device)
        pos = pos[pos < T]
        if pos.numel() == 0:
            return upd
        upd.index_fill_(0, pos, 0)
        return upd

    if update.dim() != 3:
        # Unexpected, just no-op
        return update

    upd = update.clone()
    B, T, H = upd.shape
    pos = pos_1d.to(device=upd.device)
    pos = pos[pos < T]
    if pos.numel() == 0:
        return upd

    # zero those token positions for all batch items
    upd[:, pos, :] = 0
    return upd

def register_component_ablation_hooks(
    layers,
    ablate_layers: List[int],
    component: str,              # "mlp" or "attn"
    pos_1d: Optional[torch.LongTensor],
    scope: str,                  # "positions" or "all"
):
    """
    Registers hooks on the component modules inside each selected layer.
    Hook zeros the *component output update* (the tensor returned by that module),
    either at specified token positions or for all tokens.
    """
    hooks = []

    if component not in ("mlp", "attn"):
        raise ValueError("component must be one of: mlp, attn")

    if scope not in ("positions", "all"):
        raise ValueError("scope must be one of: positions, all")

    def make_hook(li: int):
        def hook(module, inputs, output):
            # output may be Tensor or tuple/list; first tensor is the update
            t = _get_first_tensor(output)
            t2 = zero_update_at_positions(t, pos_1d=pos_1d, scope=scope)
            return _replace_first_tensor(output, t2)
        return hook

    for li in ablate_layers:
        layer = layers[li]
        if component == "mlp":
            sub = get_mlp_module(layer)
        else:
            sub = get_attn_module(layer)
        hooks.append(sub.register_forward_hook(make_hook(li)))

    return hooks

# -------------------------
# Loader (Qwen vs Gemma) — same as you had
# -------------------------
def load_model_and_processor(model_name: str):
    name_lc = model_name.lower()

    if "gemma" in name_lc:
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_ok else torch.float16
        processor = AutoProcessor.from_pretrained(model_name, padding_side="left")

        if Gemma3ForConditionalGeneration is not None:
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        elif AutoModelForImageTextToText is not None:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            raise RuntimeError("Gemma requested but no Gemma loader available in this transformers install.")
    else:
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_name(0).startswith(("NVIDIA A100", "NVIDIA H100"))
            else torch.float16
        )
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )

    model.eval()
    return model, processor

# -------------------------
# Data loader for analysis_df (your logic)
# -------------------------
def load_analysis_df(model_name: str, dataset: str) -> pd.DataFrame:
    tag_model = "gemma" if "gemma" in model_name.lower() else "qwen"
    if dataset != "celeb":
        raise NotImplementedError("This script is wired to people_knowledge.json + celeb CSVs like your example.")

    if tag_model == "gemma":
        df_text = pd.read_csv(GEMMA_CELEB_TEXT_CSV)
        df_vis  = pd.read_csv(GEMMA_CELEB_VIS_CSV)
        df_txt_corr = pd.read_csv(GEMMA_CELEB_NO_RAG_TEXT_CSV)
        df_vis_corr = pd.read_csv(GEMMA_CELEB_NO_RAG_VISION_CSV)
    else:
        df_text = pd.read_csv(QWEN_CELEB_TEXT_CSV)
        df_vis  = pd.read_csv(QWEN_CELEB_VIS_CSV)
        df_txt_corr = pd.read_csv(QWEN_CELEB_NO_RAG_TEXT_CSV)
        df_vis_corr = pd.read_csv(QWEN_CELEB_NO_RAG_VISION_CSV)

    return filter_df_for_analysis(df_vis, df_text, df_vis_corr, df_txt_corr)

# -------------------------
# Main experiment: run one sweep step k (ablating dst_start..dst_start+k)
# -------------------------
@torch.no_grad()
def run_component_ablation(
    *,
    model,
    processor,
    analysis_df: pd.DataFrame,
    dataset: str,
    modality: str,                 # "vision" or "text"
    mock_RAG: bool,
    ablate_component: str,         # "mlp" or "attn"
    ablate_scope: str,             # "positions" or "all"
    dst_start: int,
    k: int,                        # ablate layers dst_start..dst_start+k
    max_new_tokens: int,
    out_csv: str,
):
    layers = get_layers(model)

    # choose ablation layers this run
    ablate_layers = list(range(dst_start, dst_start + k + 1))
    if max(ablate_layers) >= len(layers):
        raise ValueError(f"Ablate range {ablate_layers[0]}..{ablate_layers[-1]} exceeds n_layers={len(layers)}")

    with open(JSON_DATA, "r") as f:
        data_lookup = {item["ID"]: item for item in json.load(f)}

    root, ext = os.path.splitext(out_csv)
    out_step = f"{root}_k{k}{ext}"
    if os.path.exists(out_step):
        os.remove(out_step)

    results = []

    for _, row in tqdm(analysis_df.iterrows(), total=len(analysis_df), desc=f"ablate {ablate_component} k={k}"):
        item = data_lookup.get(row["ID"])
        if not item:
            continue

        # Load image if needed
        pil_image = None
        if modality == "vision":
            if item.get("image_path"):
                full_path = os.path.join(MLLMKC_ROOT, item["image_path"][0])
                if os.path.exists(full_path):
                    pil_image = Image.open(full_path).convert("RGB")
                    if max(pil_image.size) > 1024:
                        pil_image.thumbnail((1024, 1024))

        # Build inputs (mirrors your current logic)
        model_device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if modality == "vision":
            inputs = make_inputs(
                processor=processor,
                model_device=model_device,
                retrieved_context=(row["Context_vis"] if mock_RAG else None),
                user_query=row["Query_vis"],
                entity_modality="vision",
                mock_RAG=mock_RAG,
                bench_type=("people" if dataset == "celeb" else "logo"),
                image=pil_image,
                padding=True,
            )
        elif modality == "text":
            inputs = make_inputs(
                processor=processor,
                model_device=model_device,
                retrieved_context=(row["Context_vis"] if mock_RAG else None),
                user_query=row["Query_vis"],
                entity_modality="text",
                instance_name=row["Instance_vis"],
                mock_RAG=mock_RAG,
                bench_type=("people" if dataset == "celeb" else "logo"),
                image=None,
                padding=True,
            )
        else:
            raise ValueError("modality must be 'vision' or 'text'")

        # Compute token positions BEFORE stripping helper keys
        if ablate_scope == "positions":
            if modality == "vision":
                pos = get_visual_positions(processor, inputs["input_ids"][0])
            else:
                pos = get_entity_positions_text_subseq(inputs, processor)
        else:
            pos = None  # not needed

        # Strip helper keys after positions computed
        inputs = strip_special(inputs)

        # Register ablation hooks
        hooks = register_component_ablation_hooks(
            layers=layers,
            ablate_layers=ablate_layers,
            component=ablate_component,
            pos_1d=pos,
            scope=ablate_scope,
        )

        # Generate
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Remove hooks
        for h in hooks:
            h.remove()

        seq = gen_out.sequences[0]
        decoded_full = processor.tokenizer.decode(seq, skip_special_tokens=True)
        new_answer = decoded_full.strip().split("\n")[-1].strip()

        # First-step top token
        top_token_id = -1
        top_token_str = ""
        if getattr(gen_out, "scores", None) and len(gen_out.scores) > 0:
            first_step_logits = gen_out.scores[0][0]
            top_token_id = int(first_step_logits.argmax().item())
            top_token_str = safe_decode(processor.tokenizer, top_token_id)

        results.append(
            {
                "ID": row["ID"],
                "Entity": row["Instance_vis"],
                "Category": row["Category"],
                "Query": row["Query_vis"],
                "Group": row["Group"],
                "Original_output": row["Assistant_response_vis"],
                "Original_top_token": row["Top_token_str_vis"],
                "Mis_Knowledge_Key": row["Mis_Knowledge_Key"],
                "Parametric_ans": row["Ground_Truth_vis"],
                "Contextual_ans": row["Mis_Answer_Label_vis"],
                "dst_start": dst_start,
                "k": k,
                "ablated_layers": f"{ablate_layers[0]}..{ablate_layers[-1]}",
                "ablate_component": ablate_component,
                "ablate_scope": ablate_scope,
                "New_Answer": new_answer,
                "TopDecodedTokenID": top_token_id,
                "TopDecodedToken": top_token_str,
                "pos_count": int(pos.numel()) if (pos is not None) else -1,
            }
        )

        if len(results) >= 500:
            pd.DataFrame(results).to_csv(out_step, mode="a", header=not os.path.exists(out_step), index=False)
            results = []

    if results:
        pd.DataFrame(results).to_csv(out_step, mode="a", header=not os.path.exists(out_step), index=False)

    print(f"Done. Wrote: {out_step}")

# -------------------------
# CLI
# -------------------------
def default_out_path(model_name: str, dataset: str, entity_modality: str, ablate_component: str, ablate_scope: str, experiment_type: str) -> str:
    tag_model = "gemma" if "gemma" in model_name.lower() else "qwen"
    tag_data  = "celeb" if dataset == "celeb" else "logo"
    tag_mod   = "vision" if entity_modality == "vision" else "text"
    return (
        f"/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/Corruption_Results/"
        f"FRQ_ablate_{experiment_type}_{ablate_component}_{ablate_scope}_{tag_model}_{tag_data}_{tag_mod}.csv"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="celeb", choices=["celeb"])  # wired for your people CSVs
    parser.add_argument("--entity_modality", type=str, default="vision", choices=["vision", "text"])
    parser.add_argument("--dst_start", type=int, default=0, help="Start layer index for the ablation range.")
    parser.add_argument("--num_layers", type=int, default=8, help="How many layers to include in the sweep window.")
    parser.add_argument("--max_new_tokens", type=int, default=6)

    parser.add_argument("--ablate_component", type=str, required=True, choices=["mlp", "attn"])
    parser.add_argument("--ablate_scope", type=str, default="positions", choices=["positions", "all"])

    parser.add_argument("--exp_type", type=str, default="fact_recall", choices=["fact_recall", "ctxt_mem_confl"])
    parser.add_argument("--out_csv", type=str, default=None)

    # sweep controls
    parser.add_argument("--k_min", type=int, default=0, help="Smallest k in sweep (ablate dst_start..dst_start+k).")
    parser.add_argument("--k_max", type=int, default=None, help="Largest k in sweep (inclusive). Default=num_layers-1.")
    args = parser.parse_args()

    mock_RAG = (args.exp_type == "ctxt_mem_confl")

    print(f"Loading model: {args.model_name}")
    model, processor = load_model_and_processor(args.model_name)

    analysis_df = load_analysis_df(args.model_name, args.dataset)
    print(f"[{args.dataset}] Processing {len(analysis_df)} items...")

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = default_out_path(
            args.model_name, args.dataset, args.entity_modality, args.ablate_component, args.ablate_scope, args.exp_type
        )

    k_max = args.k_max
    if k_max is None:
        k_max = args.num_layers - 1

    # Sweep: ablate layers dst_start..dst_start+k
    for k in range(args.k_min, k_max + 1):
        run_component_ablation(
            model=model,
            processor=processor,
            analysis_df=analysis_df,
            dataset=args.dataset,
            modality=args.entity_modality,
            mock_RAG=mock_RAG,
            ablate_component=args.ablate_component,
            ablate_scope=args.ablate_scope,
            dst_start=args.dst_start,
            k=k,
            max_new_tokens=args.max_new_tokens,
            out_csv=out_csv,
        )