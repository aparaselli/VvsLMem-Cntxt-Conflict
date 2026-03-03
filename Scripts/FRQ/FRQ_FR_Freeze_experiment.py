#!/usr/bin/env python3
# ============================================================
# Freeze / ForwardPatch (Backpatch) — Qwen2.5-VL / Gemma-3
# Minimal changes from your script:
#  - add --model_name flag + unified loader (qwen vs gemma)
#  - unify access to language_model.layers (get_layers)
#  - make vision/text position finders robust across Qwen/Gemma
#  - make cache tensor device-safe for sharded Gemma (move inside hook)
#  - fix from_pretrained arg: torch_dtype (not dtype)
# ============================================================

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional
import re
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
from FRQ_data_filtering_utils import filter_df_for_analysis

from transformers import AutoModelForVision2Seq, AutoProcessor
try:
    from transformers import Gemma3ForConditionalGeneration
except Exception:
    Gemma3ForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

from FRQ_make_input import make_inputs

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

DEFAULT_OUT = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/Results/FRQ_patch_corr_outputs.csv"
DEFAULT_OUT_LOGO = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/Results/FRQ_patch_corr_outputs_logo.csv"

MLLMKC_ROOT = "/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data"

def default_out_path(model_name: str, dataset: str, entity_modality: str) -> str:
    tag_model = "gemma" if "gemma" in model_name.lower() else "qwen"
    tag_data  = "celeb" if dataset == "celeb" else "logo"
    tag_mod   = "vision" if entity_modality == "vision" else "text"
    return f"/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/Corruption_Results/" \
           f"FRQ_frontpatch_freeze_{tag_model}_{tag_data}_{tag_mod}.csv"

# -------------------------
# Model boundary helpers (minimal + robust)
# -------------------------
def get_language_model(m):
    if hasattr(m, "model") and hasattr(m.model, "language_model"):
        return m.model.language_model
    if hasattr(m, "language_model"):
        return m.language_model
    # fallback: find something that looks like the LM
    for _, mod in m.named_modules():
        if hasattr(mod, "layers"):
            return mod
    raise AttributeError("Couldn't find language_model on this model.")

def get_layers(m):
    lm = get_language_model(m)
    if hasattr(lm, "layers"):
        return lm.layers
    raise AttributeError("language_model has no .layers")

def extract_hs(output):
    # layer hook output can be tensor or tuple/list; first tensor is usually hs
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x):
                return x
    raise TypeError(f"Cannot extract hidden states from type {type(output)}")


# -------------------------
# sanity check function (your code)
# -------------------------
def print_frozen_edges_only(inputs, processor, frozen_pos, window=12):
    tok = processor.tokenizer
    ids = inputs["input_ids"][0]
    T = ids.numel()

    if frozen_pos.numel() == 0:
        print("No frozen positions.")
        return

    frozen = frozen_pos.detach().cpu().long().tolist()
    frozen_set = set(frozen)
    s = min(frozen)
    e = max(frozen)

    def dump_range(lo, hi, label):
        lo = max(0, lo)
        hi = min(T, hi)
        print(f"\n--- {label} (showing tokens {lo}..{hi-1}) ---")
        for i in range(lo, hi):
            token_id = int(ids[i].item())
            token_str = tok.decode([token_id])

            marks = []
            if i == s: marks.append("FROZEN_START")
            if i == e: marks.append("FROZEN_END")
            if i in frozen_set: marks.append("FROZEN")

            mark = (" [" + ", ".join(marks) + "]") if marks else ""
            print(f"{i:5d} | {repr(token_str):25s}{mark}")

    print("\n" + "=" * 90)
    print(f"Frozen span: [{s}, {e}]  count={len(frozen)}   seq_len={T}")
    print("=" * 90)

    dump_range(s - window, s + window + 1, "AROUND START EDGE")

    if e - window > s + window:
        dump_range(e - window, e + window + 1, "AROUND END EDGE")
    else:
        print("\n(End edge overlaps start window; not printing twice.)")

    print("=" * 90)


def make_noise_image_like(pil_image: Optional[Image.Image], *, seed: Optional[int] = None) -> Image.Image:
    if pil_image is None:
        W, H = 224, 224
    else:
        W, H = pil_image.size
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# -------------------------
# Position finders (minimal change, but robust for Qwen/Gemma)
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
    """
    Try Qwen-style <|vision_start|>..<|vision_end|> first.
    Otherwise fall back to "any image/vision placeholder tokens" mask.
    """
    tok = processor.tokenizer

    # 1) try Qwen delimiters
    pos = get_visual_positions_from_input_ids_qwenstyle(input_ids_1d, tok)
    if pos.numel() > 0:
        return pos

    # 2) fallback: placeholder token IDs
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
    """
    Robust text entity span finder for BOTH Gemma + Qwen.

    - Extract entity string between "Entity:" and "Query:"
      (handles "Entity:\n", "Entity: ", etc.)
    - Tokenize several variants of the entity string
    - Subsequence match in inputs["input_ids"][0]
    """
    tok = processor.tokenizer
    device = inputs["input_ids"].device
    prompt = inputs.get("_prompt_text", "") or ""
    if not prompt:
        return torch.empty(0, dtype=torch.long, device=device)

    # Find "Entity:" (flexible whitespace)
    m = re.search(r"Entity\s*:\s*", prompt)
    if not m:
        return torch.empty(0, dtype=torch.long, device=device)
    ent_start = m.end()

    # Find the FIRST "Query:" after the entity start (flexible whitespace)
    m2 = re.search(r"Query\s*:\s*", prompt[ent_start:])
    if not m2:
        return torch.empty(0, dtype=torch.long, device=device)
    ent_end = ent_start + m2.start()

    entity_str = prompt[ent_start:ent_end].strip()
    if not entity_str:
        return torch.empty(0, dtype=torch.long, device=device)

    # Normalize common punctuation mismatches
    entity_str_nodot = entity_str.rstrip(" .\n\t")

    ids_model = inputs["input_ids"][0].detach().cpu().tolist()

    # Try a few boundary-sensitive variants (SentencePiece/BPE)
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
# helpers (your code)
# -------------------------
def strip_special(keys_dict: Dict) -> Dict:
    return {k: v for k, v in keys_dict.items() if not str(k).startswith("_")}

def safe_decode(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""





# -------------------------
# Hooks (your code, with minimal robustness changes)
# -------------------------
def register_source_cache_hooks(
    layers,
    src_layers: List[int],
    vis_pos: torch.LongTensor,
    cache_dict: Dict[int, torch.Tensor],
):
    hooks = []

    def make_hook(li: int):
        def hook(module, inputs, output):
            hs = extract_hs(output)              # [B,T,H]
            hs0 = hs[0]                          # [T,H]
            if vis_pos.numel() == 0:
                cache_dict[li] = None
                return
            cache_dict[li] = hs0.index_select(0, vis_pos).detach().cpu()  # keep on CPU
        return hook

    for li in src_layers:
        hooks.append(layers[li].register_forward_hook(make_hook(li)))

    return hooks


def register_destination_backpatch_hooks(
    layers,
    dst_layers,
    mapping_src_for_dst,
    vis_pos,
    cache_dict,
    expected_T: int,
):
    hooks = []
    vis_pos = vis_pos.detach().cpu()

    def make_hook(dst_li: int):
        src_li = mapping_src_for_dst[dst_li]

        def hook(module, inputs, output):
            cached = cache_dict.get(src_li, None)
            if cached is None or vis_pos.numel() == 0:
                return output

            # unpack
            if isinstance(output, (tuple, list)):
                hs = extract_hs(output)          # [B,T,H]
                rest = list(output[1:])
            else:
                hs = extract_hs(output)
                rest = None

            T = hs.shape[1]
            if T != expected_T:
                return output

            max_ok = T - 1
            vp = vis_pos[vis_pos <= max_ok]
            if vp.numel() == 0:
                return output

            # align lengths if mismatch
            if cached.shape[0] != vp.numel():
                m = min(cached.shape[0], vp.numel())
                cached_use = cached[:m]
                vp_use = vp[:m]
            else:
                cached_use = cached
                vp_use = vp

            # move cached to the SAME device as hs (important for sharded gemma)
            cached_use = cached_use.to(device=hs.device, non_blocking=True)
            vp_use = vp_use.to(device=hs.device)

            hs = hs.clone()
            if hs.shape[0] != 1:
                hs[:, vp_use, :] = cached_use.unsqueeze(0)
            else:
                hs0 = hs[0]
                hs0.index_copy_(0, vp_use, cached_use)
                hs[0] = hs0

            if rest is None:
                return hs
            else:
                return tuple([hs] + rest)

        return hook

    for dst_li in dst_layers:
        hooks.append(layers[dst_li].register_forward_hook(make_hook(dst_li)))

    return hooks


# -------------------------
# Main runner (your code, minimal edits: layers getter + position getters)
# -------------------------
def run_backpatch(
    model,
    processor,
    dataset: str,
    src_start: int,
    dst_start: int,
    num_layers: int,
    out_csv: str,
    max_new_tokens: int = 6,
    split: int = 0,
    source_type="same",
    modality="vision",
    model_name= None,
    mock_Rag=False,
):
    assert src_start >= dst_start, "Need src_start > dst_start for back-patching (copy later -> earlier)."
    tag_model = "gemma" if "gemma" in model_name.lower() else "qwen"
    if tag_model == "gemma":
        df_text = pd.read_csv(GEMMA_CELEB_TEXT_CSV)
        df_vis  = pd.read_csv(GEMMA_CELEB_VIS_CSV)
        df_txt_corr = pd.read_csv(GEMMA_CELEB_NO_RAG_TEXT_CSV)
        df_vis_corr = pd.read_csv(GEMMA_CELEB_NO_RAG_VISION_CSV)
        analysis_df = filter_df_for_analysis(df_vis, df_text, df_vis_corr, df_txt_corr)
    elif tag_model == "qwen":
        df_text = pd.read_csv(QWEN_CELEB_TEXT_CSV)
        df_vis  = pd.read_csv(QWEN_CELEB_VIS_CSV)
        df_txt_corr = pd.read_csv(QWEN_CELEB_NO_RAG_TEXT_CSV)
        df_vis_corr = pd.read_csv(QWEN_CELEB_NO_RAG_VISION_CSV)
        analysis_df = filter_df_for_analysis(df_vis, df_text, df_vis_corr, df_txt_corr)
    print(f"[{dataset}] Processing {len(analysis_df)} items...")
    with open(JSON_DATA, "r") as f:
        data_lookup = {item["ID"]: item for item in json.load(f)}
    root, ext = os.path.splitext(out_csv)
    out_csv = f"{root}_layer{src_start}_split{split}{ext}"
    if mock_RAG:
        out_csv = f"{root}_ctxt_mem_conf_layer{src_start}_split{split}{ext}"

    layers = get_layers(model)

    if source_type == "same":
        src_layers = [src_start]
        dst_layers = list(range(dst_start, dst_start + num_layers))
        mapping = {dst_layers[i]: src_start for i in range(num_layers)}
    else:
        src_layers = list(range(src_start, src_start + num_layers))
        dst_layers = list(range(dst_start, dst_start + num_layers))
        mapping = {dst_layers[i]: src_layers[i] for i in range(num_layers)}

    results = []
    if os.path.exists(out_csv):
        os.remove(out_csv)

    for _, row in tqdm(analysis_df.iterrows(), total=len(analysis_df)):
        item = data_lookup.get(row["ID"])
        if not item:
            continue

        # Load image (vision)
        pil_image = None
        if modality == "vision":
            if item.get("image_path"):
                full_path = os.path.join(MLLMKC_ROOT, item["image_path"][0])
                if os.path.exists(full_path):
                    pil_image = Image.open(full_path).convert("RGB")
                    if max(pil_image.size) > 1024:
                        pil_image.thumbnail((1024, 1024))

        # Build inputs
        if modality == "vision":
            if mock_RAG:
                inputs = make_inputs(
                    processor=processor,
                    model_device=getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                    retrieved_context=row["Context_vis"],
                    user_query=row["Query_vis"],
                    entity_modality="vision",
                    mock_RAG=mock_RAG,
                    bench_type=("people" if dataset == "celeb" else "logo"),
                    image=pil_image,
                    padding=True,
                )
            else:
                inputs = make_inputs(
                    processor=processor,
                    model_device=getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                    retrieved_context=None,
                    user_query=row["Query_vis"],
                    entity_modality="vision",
                    mock_RAG=mock_RAG,
                    bench_type=("people" if dataset == "celeb" else "logo"),
                    image=pil_image,
                    padding=True,
                )
        elif modality == "text":
            if mock_RAG:
                inputs = make_inputs(
                    processor=processor,
                    model_device=getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                    retrieved_context=row["Context_vis"],
                    user_query=row["Query_vis"],
                    entity_modality="text",
                    instance_name=row["Instance_vis"],
                    mock_RAG=mock_RAG,
                    bench_type=("people" if dataset == "celeb" else "logo"),
                    image=None,
                    padding=True,
                )
            else:
                inputs = make_inputs(
                    processor=processor,
                    model_device=getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                    retrieved_context=None,
                    user_query=row["Query_vis"],
                    entity_modality="text",
                    instance_name=row["Instance_vis"],
                    mock_RAG=mock_RAG,
                    bench_type=("people" if dataset == "celeb" else "logo"),
                    image=None,
                    padding=True,
                )
        else:
            raise ValueError("Modality not supported")

        # compute positions while helper keys still exist
        if modality == "vision":
            vis_pos = get_visual_positions(processor, inputs["input_ids"][0])
        else:
            vis_pos = get_entity_positions_text_subseq(inputs, processor)

        # DEBUG ONLY — comment out later
        #print_frozen_edges_only(inputs, processor, vis_pos, window=12)

        # only strip AFTER positions
        inputs = strip_special(inputs)

        # Pass 1: cache source
        cache_dict: Dict[int, Optional[torch.Tensor]] = {}

        inputs_src = inputs
        vis_pos_src = vis_pos

        hooks1 = register_source_cache_hooks(layers, src_layers, vis_pos_src, cache_dict)
        with torch.inference_mode():
            _ = model(**inputs_src)
        for h in hooks1:
            h.remove()

        # Pass 2: backpatch into destination + generate
        expected_T = inputs["input_ids"].shape[1]
        hooks2 = register_destination_backpatch_hooks(
            layers, dst_layers, mapping, vis_pos, cache_dict, expected_T=expected_T
        )

        with torch.inference_mode():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        for h in hooks2:
            h.remove()

        seq = gen_out.sequences[0]
        decoded_full = processor.tokenizer.decode(seq, skip_special_tokens=True)
        new_answer = decoded_full.strip().split("\n")[-1].strip()

        # Top decoded token from first generation step
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
                "Query": row['Query_vis'],
                "Group": row["Group"],
                "Original_output": row["Assistant_response_vis"],
                "Original_top_token": row["Top_token_str_vis"],
                "Mis_Knowledge_Key": row["Mis_Knowledge_Key"],
                "Parametric_ans": row["Ground_Truth_vis"],
                "Contextual_ans": row["Mis_Answer_Label_vis"],
                "src_start": src_start,
                "dst_start": dst_start,
                "num_layers": num_layers,
                "New_Answer": new_answer,
                "TopDecodedTokenID": top_token_id,
                "TopDecodedToken": top_token_str,
            }
        )

        if len(results) >= 500:
            pd.DataFrame(results).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            results = []

    if results:
        pd.DataFrame(results).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"Done. Wrote: {out_csv}")


# -------------------------
# Loader (Qwen vs Gemma)
# -------------------------
def load_model_and_processor(model_name: str):
    name_lc = model_name.lower()

    if "gemma" in name_lc:
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_ok else torch.float16

        # Gemma: left padding helps decoder-only behavior; harmless for bs=1
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
        # Qwen path
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
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="e.g. Qwen/Qwen2.5-VL-7B-Instruct or google/gemma-3-12b-it")
    parser.add_argument("--dataset", type=str, default="celeb", choices=["celeb", "logo"])
    parser.add_argument("--entity_modality", type=str, default="vision", choices=["vision", "text"])
    parser.add_argument("--src_start", type=int, default=0, help="Start layer for source S (later layers).")
    parser.add_argument("--dst_start", type=int, default=0, help="Start layer for destination D (earlier layers).")
    parser.add_argument("--num_layers", type=int, default=7, help="How many consecutive layers to backpatch.")
    parser.add_argument("--max_new_tokens", type=int, default=6)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--source_type", type=str, default="same", choices=["same"])
    parser.add_argument("--exp_type", type=str, default="fact_recall", choices=["fact_recall", "ctxt_mem_confl"])
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model, processor = load_model_and_processor(args.model_name)
    if args.exp_type == "fact_recall":
        mock_RAG = False
    elif args.exp_type == "ctxt_mem_confl":
        mock_RAG = True
    out_csv = args.out_csv
    if out_csv is None:
        out_csv = default_out_path(args.model_name, args.dataset, args.entity_modality)

    for i in range(5,args.num_layers):
        run_backpatch(
            model=model,
            processor=processor,
            dataset=args.dataset,
            src_start=args.src_start,
            dst_start=args.dst_start,
            num_layers=i + 1,
            out_csv=out_csv,
            max_new_tokens=args.max_new_tokens,
            source_type=args.source_type,
            modality=args.entity_modality,
            split=i,
            model_name=args.model_name,
            mock_Rag=mock_RAG,
        )