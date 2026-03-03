#!/usr/bin/env python3
# ============================================================
# Attention Output Patching (pre-o_proj) — Qwen2.5-VL / Gemma-3
#
# Supports TWO modes:
#   1) patch_mode="head" (isolated):
#        For each (layer ℓ, head j):
#          - SOURCE: mock_RAG=True capture pre-o_proj head slice at entity tokens
#          - DEST:   mock_RAG=False baseline logit, then patch ONLY (ℓ, j) and re-logit
#
#   2) patch_mode="full" (cumulative):
#        For each end layer i in [dst_start .. dst_start+num_layers-1]:
#          - Patch ALL layers dst_start..i (multi-head full pre-o_proj) at entity tokens
#            from SOURCE (mock_RAG=True) into DEST (mock_RAG=False)
#          - Record baseline logit vs patched logit
#
# Patch direction: mock_RAG -> no_mock_RAG
#
# Patching location:
#   We patch the input to attention output projection (o_proj/out_proj), typically [B,T,H].
# ============================================================

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
import re

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

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

def get_input_device(model) -> torch.device:
    for p in model.parameters():
        if p is not None and hasattr(p, "device") and str(p.device) != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Token position finders (your robust ones)
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

def get_attn_module(layer):
    for name in ["self_attn", "attn", "attention"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise AttributeError(f"Layer has no attention module among ['self_attn','attn','attention']: {type(layer)}")

def get_attn_out_proj(attn):
    for name in ["o_proj", "out_proj", "output_proj", "proj"]:
        if hasattr(attn, name):
            mod = getattr(attn, name)
            if isinstance(mod, torch.nn.Module):
                return mod
    if hasattr(attn, "output") and isinstance(attn.output, torch.nn.Module) and hasattr(attn.output, "dense"):
        return attn.output.dense
    raise AttributeError(f"Attention has no output projection among common names. type(attn)={type(attn)}")

def get_num_heads_and_head_dim(attn) -> Tuple[int, int]:
    for nh_name in ["num_heads", "n_heads", "num_attention_heads", "heads"]:
        if hasattr(attn, nh_name):
            nh = int(getattr(attn, nh_name))
            if nh > 0:
                break
    else:
        nh = None

    for hd_name in ["head_dim", "attention_head_size"]:
        if hasattr(attn, hd_name):
            hd = int(getattr(attn, hd_name))
            if hd > 0:
                break
    else:
        hd = None

    if nh is None or hd is None:
        hidden = None
        for hs_name in ["hidden_size", "embed_dim", "dim", "d_model"]:
            if hasattr(attn, hs_name):
                hidden = int(getattr(attn, hs_name))
                break
        if hidden is None:
            if hasattr(attn, "o_proj") and hasattr(attn.o_proj, "in_features"):
                hidden = int(attn.o_proj.in_features)
        if nh is None and hasattr(attn, "num_key_value_heads"):
            nh = int(getattr(attn, "num_key_value_heads"))
        if nh is None or hidden is None:
            raise AttributeError("Could not infer num_heads/hidden_size for attention.")
        if hd is None:
            if hidden % nh != 0:
                raise ValueError(f"hidden_size {hidden} not divisible by num_heads {nh}")
            hd = hidden // nh

    return nh, hd

def first_token_id_of_contextual_answer(tokenizer, contextual_answer: str) -> Optional[int]:
    s = (contextual_answer or "").strip()
    if not s:
        return None
    ids = tokenizer(s, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = tokenizer(" " + s, add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    return int(ids[0])


# -------------------------
# Loader (Qwen vs Gemma)
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
# Data loader (your logic)
# -------------------------
def load_analysis_df(model_name: str, dataset: str) -> pd.DataFrame:
    tag_model = "gemma" if "gemma" in model_name.lower() else "qwen"
    if dataset != "celeb":
        raise NotImplementedError("Wired to people_knowledge.json + celeb CSVs.")

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
# Hooks: head-specific pre-o_proj capture/patch
# -------------------------
def make_capture_hook(
    *,
    cache: Dict,
    layer_idx: int,
    pos_src: torch.LongTensor,
    n_heads: int,
    head_dim: int,
):
    def _hook(module, inputs):
        x = inputs[0]
        if not torch.is_tensor(x) or x.dim() != 3:
            return

        B, T, H = x.shape
        if H != n_heads * head_dim:
            raise ValueError(f"H={H} != n_heads*head_dim={n_heads*head_dim} at layer {layer_idx}")

        if pos_src is None or pos_src.numel() == 0:
            cache[layer_idx] = None
            return

        pos = pos_src.to(device=x.device)
        pos = pos[pos < T]
        if pos.numel() == 0:
            cache[layer_idx] = None
            return

        x4 = x.view(B, T, n_heads, head_dim)
        cache[layer_idx] = x4[:, pos, :, :].detach()  # [B,P,n_heads,head_dim]
    return _hook

def make_patch_hook(
    *,
    cache: Dict,
    layer_idx: int,
    head_idx: int,
    pos_dst: torch.LongTensor,
    n_heads: int,
    head_dim: int,
):
    def _hook(module, inputs):
        x = inputs[0]
        if not torch.is_tensor(x) or x.dim() != 3:
            return None

        src_all = cache.get(layer_idx, None)
        if src_all is None:
            return None

        B, T, H = x.shape
        if H != n_heads * head_dim:
            raise ValueError(f"H={H} != n_heads*head_dim={n_heads*head_dim} at layer {layer_idx}")

        if pos_dst is None or pos_dst.numel() == 0:
            return None

        pos = pos_dst.to(device=x.device)
        pos = pos[pos < T]
        if pos.numel() == 0:
            return None

        src_all = src_all.to(device=x.device, dtype=x.dtype)  # [B,Psrc,n_heads,head_dim]
        P = int(pos.numel())
        Psrc = int(src_all.shape[1])
        Puse = min(P, Psrc)
        if Puse <= 0:
            return None

        x_new = x.clone()
        x4 = x_new.view(B, T, n_heads, head_dim)
        pos_use = pos[:Puse]
        x4[:, pos_use, head_idx, :] = src_all[:, :Puse, head_idx, :]
        return (x_new,) + tuple(inputs[1:])
    return _hook


# -------------------------
# Hooks: FULL (multi-head) pre-o_proj capture/patch
# -------------------------
def make_capture_hook_full(
    *,
    cache: Dict,
    layer_idx: int,
    pos_src: torch.LongTensor,
):
    def _hook(module, inputs):
        x = inputs[0]
        if not torch.is_tensor(x) or x.dim() != 3:
            return
        if pos_src is None or pos_src.numel() == 0:
            cache[layer_idx] = None
            return
        B, T, H = x.shape
        pos = pos_src.to(device=x.device)
        pos = pos[pos < T]
        if pos.numel() == 0:
            cache[layer_idx] = None
            return
        cache[layer_idx] = x[:, pos, :].detach()  # [B,P,H]
    return _hook

def make_patch_hook_full(
    *,
    cache: Dict,
    layer_idx: int,
    pos_dst: torch.LongTensor,
):
    def _hook(module, inputs):
        x = inputs[0]
        if not torch.is_tensor(x) or x.dim() != 3:
            return None

        src = cache.get(layer_idx, None)
        if src is None:
            return None

        if pos_dst is None or pos_dst.numel() == 0:
            return None

        B, T, H = x.shape
        pos = pos_dst.to(device=x.device)
        pos = pos[pos < T]
        if pos.numel() == 0:
            return None

        src2 = src.to(device=x.device, dtype=x.dtype)  # [B,Psrc,H]
        P = int(pos.numel())
        Psrc = int(src2.shape[1])
        Puse = min(P, Psrc)
        if Puse <= 0:
            return None

        x_new = x.clone()
        x_new[:, pos[:Puse], :] = src2[:, :Puse, :]
        return (x_new,) + tuple(inputs[1:])
    return _hook


def remove_hooks(hooks):
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

def register_many_capture_hooks_full(layers, layer_indices, pos_src, cache):
    hooks = []
    for li in layer_indices:
        attn = get_attn_module(layers[li])
        out_proj = get_attn_out_proj(attn)
        cap_hook = make_capture_hook_full(cache=cache, layer_idx=li, pos_src=pos_src)
        hooks.append(out_proj.register_forward_pre_hook(cap_hook))
    return hooks

def register_many_patch_hooks_full(layers, layer_indices, pos_dst, cache):
    hooks = []
    for li in layer_indices:
        attn = get_attn_module(layers[li])
        out_proj = get_attn_out_proj(attn)
        patch_hook = make_patch_hook_full(cache=cache, layer_idx=li, pos_dst=pos_dst)
        hooks.append(out_proj.register_forward_pre_hook(patch_hook))
    return hooks


# -------------------------
# Compute "next token logit" for contextual first token
# -------------------------
@torch.no_grad()
def compute_next_token_logit(model, inputs_stripped: Dict, token_id: int) -> float:
    out = model(**inputs_stripped, use_cache=False)
    logits = out.logits  # [B,T,V]
    last = logits[0, -1, :]  # next-token distribution
    return float(last[token_id].item())


# -------------------------
# Main experiment
# -------------------------
@torch.no_grad()
def run_head_patching_sweep(
    *,
    model,
    processor,
    analysis_df: pd.DataFrame,
    dataset: str,
    modality: str,          # "vision" or "text"
    dst_start: int,
    num_layers: int,
    heads: Optional[List[int]],
    patch_mode: str,        # "head" or "full"
    max_items: Optional[int],
    out_csv: str,
):
    layers = get_layers(model)
    L_total = len(layers)

    layer_list = list(range(dst_start, min(dst_start + num_layers, L_total)))
    if not layer_list:
        raise ValueError("Empty layer_list; check dst_start/num_layers vs model layers.")

    with open(JSON_DATA, "r") as f:
        data_lookup = {item["ID"]: item for item in json.load(f)}

    if patch_mode == "head":
        attn0 = get_attn_module(layers[layer_list[0]])
        n_heads, head_dim = get_num_heads_and_head_dim(attn0)
        if heads is None:
            heads = list(range(n_heads))
        else:
            heads = [h for h in heads if 0 <= h < n_heads]
            if not heads:
                raise ValueError("No valid heads after filtering. Check --heads.")
    else:
        n_heads, head_dim = None, None
        heads = None

    if os.path.exists(out_csv):
        os.remove(out_csv)

    device_for_inputs = get_input_device(model)
    buf = []

    it = analysis_df.iterrows()
    total = len(analysis_df) if max_items is None else min(len(analysis_df), max_items)

    for idx, row in tqdm(it, total=total, desc="examples"):
        # (per your request: don't worry about max_items logic)

        item = data_lookup.get(row["ID"])
        if not item:
            continue

        ctx_first_id = first_token_id_of_contextual_answer(processor.tokenizer, row.get("Mis_Answer_Label_vis", ""))
        if ctx_first_id is None:
            continue

        pil_image = None
        if modality == "vision":
            if item.get("image_path"):
                full_path = os.path.join(MLLMKC_ROOT, item["image_path"][0])
                if os.path.exists(full_path):
                    pil_image = Image.open(full_path).convert("RGB")
                    if max(pil_image.size) > 1024:
                        pil_image.thumbnail((1024, 1024))

        # -------------------------
        # DEST inputs (no mock_RAG) ONCE per example
        # -------------------------
        if modality == "vision":
            inputs_dst = make_inputs(
                processor=processor,
                model_device=device_for_inputs,
                retrieved_context=None,
                user_query=row["Query_vis"],
                entity_modality="vision",
                mock_RAG=False,
                bench_type=("people" if dataset == "celeb" else "logo"),
                image=pil_image,
                padding=True,
            )
        else:
            inputs_dst = make_inputs(
                processor=processor,
                model_device=device_for_inputs,
                retrieved_context=None,
                user_query=row["Query_vis"],
                entity_modality="text",
                instance_name=row["Instance_vis"],
                mock_RAG=False,
                bench_type=("people" if dataset == "celeb" else "logo"),
                image=None,
                padding=True,
            )

        if modality == "vision":
            pos_dst = get_visual_positions(processor, inputs_dst["input_ids"][0])
        else:
            pos_dst = get_entity_positions_text_subseq(inputs_dst, processor)

        inputs_dst_stripped = strip_special(inputs_dst)
        baseline_logit = compute_next_token_logit(model, inputs_dst_stripped, ctx_first_id)

        # -------------------------
        # SOURCE inputs (mock_RAG=True) ONCE per example
        # -------------------------
        if modality == "vision":
            inputs_src = make_inputs(
                processor=processor,
                model_device=device_for_inputs,
                retrieved_context=row["Context_vis"],
                user_query=row["Query_vis"],
                entity_modality="vision",
                mock_RAG=True,
                bench_type=("people" if dataset == "celeb" else "logo"),
                image=pil_image,
                padding=True,
            )
        else:
            inputs_src = make_inputs(
                processor=processor,
                model_device=device_for_inputs,
                retrieved_context=row["Context_vis"],
                user_query=row["Query_vis"],
                entity_modality="text",
                instance_name=row["Instance_vis"],
                mock_RAG=True,
                bench_type=("people" if dataset == "celeb" else "logo"),
                image=None,
                padding=True,
            )

        if modality == "vision":
            pos_src = get_visual_positions(processor, inputs_src["input_ids"][0])
        else:
            pos_src = get_entity_positions_text_subseq(inputs_src, processor)

        inputs_src_stripped = strip_special(inputs_src)
        logit_mockRAG = compute_next_token_logit(
            model,
            inputs_src_stripped,
            ctx_first_id
        )
        # ============================================================
        # FULL mode: CUMULATIVE patching of layers dst_start..end_li
        # ============================================================
        if patch_mode == "full":
            for end_li in layer_list[20:28]: # FOR DEBGGGGG
                patch_layers = list(range(layer_list[0], end_li + 1))

                cache = {}

                # SOURCE forward: capture for ALL patch layers at once
                cap_hooks = register_many_capture_hooks_full(
                    layers=layers,
                    layer_indices=patch_layers,
                    pos_src=pos_src,
                    cache=cache,
                )
                _ = model(**inputs_src_stripped, use_cache=False)
                remove_hooks(cap_hooks)

                # DEST forward: patch ALL those layers at once
                patch_hooks = register_many_patch_hooks_full(
                    layers=layers,
                    layer_indices=patch_layers,
                    pos_dst=pos_dst,
                    cache=cache,
                )
                patched_logit = compute_next_token_logit(model, inputs_dst_stripped, ctx_first_id)
                remove_hooks(patch_hooks)

                buf.append(
                    {
                        "ID": row["ID"],
                        "Entity": row["Instance_vis"],
                        "Category": row["Category"],
                        "Group": row["Group"],
                        "Query": row["Query_vis"],
                        "Mis_Knowledge_Key": row["Mis_Knowledge_Key"],
                        "Parametric_ans": row["Ground_Truth_vis"],
                        "Contextual_ans": row["Mis_Answer_Label_vis"],
                        "ctx_first_token_id": int(ctx_first_id),
                        "dst_start": int(layer_list[0]),
                        "end_layer": int(end_li),
                        "patched_layers": f"{layer_list[0]}..{end_li}",
                        "head": -1,
                        "pos_src_count": int(pos_src.numel()) if pos_src is not None else -1,
                        "pos_dst_count": int(pos_dst.numel()) if pos_dst is not None else -1,
                        "baseline_logit_noRAG": float(baseline_logit),
                        "patched_logit_noRAG": float(patched_logit),
                        "logit_mockRAG": float(logit_mockRAG),
                        "delta": float(patched_logit - baseline_logit),
                    }
                )

                if len(buf) >= 200:
                    pd.DataFrame(buf).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
                    buf = []

        # ============================================================
        # HEAD mode: ISOLATED (layer ℓ, head j)
        # ============================================================
        else:
            for li in layer_list:
                attn = get_attn_module(layers[li])
                out_proj = get_attn_out_proj(attn)

                cache = {}

                cap_hook = make_capture_hook(
                    cache=cache,
                    layer_idx=li,
                    pos_src=pos_src,
                    n_heads=n_heads,
                    head_dim=head_dim,
                )
                h_cap = out_proj.register_forward_pre_hook(cap_hook)
                _ = model(**inputs_src_stripped, use_cache=False)
                h_cap.remove()

                for hj in heads:
                    patch_hook = make_patch_hook(
                        cache=cache,
                        layer_idx=li,
                        head_idx=hj,
                        pos_dst=pos_dst,
                        n_heads=n_heads,
                        head_dim=head_dim,
                    )
                    h_patch = out_proj.register_forward_pre_hook(patch_hook)

                    patched_logit = compute_next_token_logit(model, inputs_dst_stripped, ctx_first_id)

                    h_patch.remove()

                    buf.append(
                        {
                            "ID": row["ID"],
                            "Entity": row["Instance_vis"],
                            "Category": row["Category"],
                            "Group": row["Group"],
                            "Query": row["Query_vis"],
                            "Mis_Knowledge_Key": row["Mis_Knowledge_Key"],
                            "Parametric_ans": row["Ground_Truth_vis"],
                            "Contextual_ans": row["Mis_Answer_Label_vis"],
                            "ctx_first_token_id": int(ctx_first_id),
                            "layer": int(li),
                            "head": int(hj),
                            "pos_src_count": int(pos_src.numel()) if pos_src is not None else -1,
                            "pos_dst_count": int(pos_dst.numel()) if pos_dst is not None else -1,
                            "baseline_logit_noRAG": float(baseline_logit),
                            "patched_logit_noRAG": float(patched_logit),
                            "logit_mockRAG": float(logit_mockRAG),
                            "delta": float(patched_logit - baseline_logit),
                        }
                    )

                    if len(buf) >= 200:
                        pd.DataFrame(buf).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
                        buf = []

        if len(buf) >= 200:
            pd.DataFrame(buf).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            buf = []

    if buf:
        pd.DataFrame(buf).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"Done. Wrote: {out_csv}")


# -------------------------
# CLI
# -------------------------
def default_out_path(model_name: str, dataset: str, entity_modality: str, patch_mode: str) -> str:
    tag_model = "gemma" if "gemma" in model_name.lower() else "qwen"
    tag_data  = "celeb" if dataset == "celeb" else "logo"
    tag_mod   = "vision" if entity_modality == "vision" else "text"
    return (
        f"/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/Corruption_Results/"
        f"FRQ_patch_{patch_mode}_mockRAG_to_noRAG_{tag_model}_{tag_data}_{tag_mod}.csv"
    )

def parse_heads(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(list(range(a, b + step, step)))
        else:
            out.append(int(part))
    return sorted(set(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="celeb", choices=["celeb"])
    parser.add_argument("--entity_modality", type=str, default="vision", choices=["vision", "text"])
    parser.add_argument("--dst_start", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=8)

    parser.add_argument("--patch_mode", type=str, default="head", choices=["head", "full"])
    parser.add_argument("--heads", type=str, default=None, help='Head indices (e.g. "0-7" or "0,3,5"). Only used if patch_mode=head.')

    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--out_csv", type=str, default=None)

    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model, processor = load_model_and_processor(args.model_name)

    analysis_df = load_analysis_df(args.model_name, args.dataset)
    #analysis_df = analysis_df.iloc[:20] #DEBUG
    analysis_df = analysis_df[analysis_df['Category'] == 'Career_error']
    print(f"[{args.dataset}] Processing {len(analysis_df)} items...")

    out_csv = args.out_csv or default_out_path(args.model_name, args.dataset, args.entity_modality, args.patch_mode)

    heads = None
    if args.patch_mode == "head":
        if args.heads is not None:
            heads = parse_heads(args.heads)
        else:
            heads = None  # means "all"

    run_head_patching_sweep(
        model=model,
        processor=processor,
        analysis_df=analysis_df,
        dataset=args.dataset,
        modality=args.entity_modality,
        dst_start=args.dst_start,
        num_layers=args.num_layers,
        heads=heads,
        patch_mode=args.patch_mode,
        max_items=args.max_items,
        out_csv=out_csv,
    )