# ============================================================
# Grad·(Δh) Attribution Patching — Qwen2.5-VL / Gemma-3 (BINNED)
# Supports entity_modality = {"vision","text"}
#
# Fixes:
#  - Gemma text embed capture/override fallback uses ACTIVE embed_tokens module
#  - ACTIVE_EMB is found ONCE (not per-row)
#  - compute_dataset_sigma_lm_entity uses the same capture logic
# ============================================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/oscar/scratch/aparasel/hf_cache"
os.environ["HF_HUB_CACHE"] = "/oscar/scratch/aparasel/hf_cache/hub"

import json
import contextlib
import re
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import argparse

from FRQ_data_filtering_utils import filter_df_for_analysis
from FRQ_make_input import make_inputs

from transformers import AutoProcessor
from transformers import AutoModelForVision2Seq

try:
    from transformers import Gemma3ForConditionalGeneration
except Exception:
    Gemma3ForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None


# -------------------------
# Paths
# -------------------------
QWEN_CELEB_VIS_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_RAG_VISION_Experiment_People_Results.csv"
QWEN_CELEB_TEXT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_RAG_TEXT_Experiment_People_Results.csv"
GEMMA_CELEB_VIS_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Gemma_FRQ_RAG_VISION_Experiment_People_Results.csv"
GEMMA_CELEB_TEXT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Gemma_FRQ_RAG_TEXT_Experiment_People_Results.csv"
JSON_DATA = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/people_knowledge.json"

QWEN_CELEB_NO_RAG_TEXT_CSV   = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_no_RAG_TEXT_Experiment_People_Results.csv"
QWEN_CELEB_NO_RAG_VISION_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Qwen_FRQ_no_RAG_VISION_Experiment_People_Results.csv"
GEMMA_CELEB_NO_RAG_TEXT_CSV   = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Gemma_FRQ_no_RAG_TEXT_Experiment_People_Results.csv"
GEMMA_CELEB_NO_RAG_VISION_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/Gemma_FRQ_no_RAG_VISION_Experiment_People_Results.csv"

MLLMKC_ROOT = "/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data"
OUT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/Attr_Results/Attribution_Patching_Binned.csv"

# -------------------------
# Hyperparams
# -------------------------
ALPHA = 6.0
MEAN_OVER_ENTITY_TOKENS_ONLY = False
LAYER_TYPE_DEFAULT = "mlp"

SIGMA_CACHE_PATH_VIS  = "sigma_lm_visual_embed.pt"
SIGMA_CACHE_PATH_TEXT = "sigma_lm_text_entity_embed.pt"

SUBBINS_FULLCOVER = {"Context": 3, "Entity": 1, "Query": 20, "Final": 1}


# -------------------------
# Utils
# -------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes","true","t","y","1"): return True
    if v in ("no","false","f","n","0"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def _t(x):
    try:
        return float(x.detach().cpu())
    except Exception:
        return float(x)

def extract_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x):
                return x
    raise TypeError(f"Hook output not tensor/tuple(list) of tensors: {type(output)}")

def masked_tokens(h: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
    m = mask_bt.to(device=h.device)
    if m.dtype != torch.bool:
        m = m.bool()
    return h[m]

def debug_dump_inputs(inputs, input_ids, entity_mask, attn_mask, token_mask, tag="DBG1"):
    print(f"[{tag}] keys:", [k for k in inputs.keys()])
    print(
        f"[{tag}] seq_len={input_ids.shape[1]}  "
        f"attn_tokens={int(attn_mask.sum())}  "
        f"entity_tokens={int(entity_mask.sum())}  "
        f"token_mask_tokens={int(token_mask.sum())}"
    )

def debug_dump_lm_embed_delta(clean_ie, corr_ie, entity_mask, noise, tag="DBG2-LM"):
    noise_abs = noise.abs().mean()
    delta_abs = (corr_ie.float() - clean_ie.float())[entity_mask].abs().mean()
    print(f"[{tag}] mean|noise|={_t(noise_abs)}  mean|Δlm_embed|(entity)={_t(delta_abs)}")

def debug_dump_logits(clean_logits, corrupted_logits, clean_probs, corrupted_probs, tag="DBG3"):
    max_logit_diff = (corrupted_logits - clean_logits).abs().max()
    max_prob_diff  = (corrupted_probs - clean_probs).abs().max()
    print(f"[{tag}] max|Δlogit|={_t(max_logit_diff)}  max|Δprob|={_t(max_prob_diff)}")


# -------------------------
# Model boundary helpers
# -------------------------
def get_language_model(m):
    if hasattr(m, "model") and hasattr(m.model, "language_model"):
        return m.model.language_model
    if hasattr(m, "language_model"):
        return m.language_model
    for _, mod in m.named_modules():
        if hasattr(mod, "layers") and hasattr(mod, "embed_tokens"):
            return mod
    raise AttributeError("Couldn't find language_model on this model.")

@contextlib.contextmanager
def capture_lm_inputs_embeds(model, store_dict, key="lm_inputs_embeds"):
    lm = get_language_model(model)

    def pre_hook(module, args, kwargs):
        ie = kwargs.get("inputs_embeds", None)
        if ie is None and len(args) > 0 and torch.is_tensor(args[0]):
            ie = args[0]
        if ie is not None:
            store_dict[key] = ie.detach()
        return (args, kwargs)

    h = lm.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()

@contextlib.contextmanager
def override_lm_inputs_embeds(model, new_inputs_embeds):
    lm = get_language_model(model)

    def pre_hook(module, args, kwargs):
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            kwargs["inputs_embeds"] = new_inputs_embeds
        return (args, kwargs)

    h = lm.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()

@contextlib.contextmanager
def capture_embed_tokens_output(emb_module, store_dict, key="tok_embeds"):
    def hook(module, module_inputs, module_output):
        store_dict[key] = module_output.detach()
        return module_output
    h = emb_module.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()

@contextlib.contextmanager
def override_embed_tokens_output(emb_module, override_embeds):
    def hook(module, module_inputs, module_output):
        return override_embeds
    h = emb_module.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()

def find_active_embed_tokens_module(model, sample_inputs):
    """
    Finds which embed_tokens module is actually hit during forward.
    Returns module or None.
    """
    hits = []
    hooks = []

    for name, mod in model.named_modules():
        if name.endswith("embed_tokens"):
            def _mk(name_):
                def _hook(m, inp, out):
                    hits.append((name_, m))
                return _hook
            hooks.append(mod.register_forward_hook(_mk(name)))

    with torch.no_grad():
        _ = model(**sample_inputs, use_cache=False)

    for h in hooks:
        h.remove()

    return hits[0][1] if hits else None


# -------------------------
# Entity mask builders
# -------------------------
def get_visual_token_mask(processor, input_ids: torch.Tensor) -> torch.Tensor:
    tok = processor.tokenizer
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

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for tid in candidate_ids:
        mask |= (input_ids == tid)

    if not mask.any():
        try:
            vstart = tok.convert_tokens_to_ids("<|vision_start|>")
            vend   = tok.convert_tokens_to_ids("<|vision_end|>")
            if vstart is not None and vend is not None and vstart >= 0 and vend >= 0:
                B, T = input_ids.shape
                for b in range(B):
                    s = (input_ids[b] == vstart).nonzero(as_tuple=False)
                    e = (input_ids[b] == vend).nonzero(as_tuple=False)
                    if len(s) and len(e):
                        s, e = int(s[0]), int(e[0])
                        if e > s:
                            mask[b, s:e+1] = True
        except Exception:
            pass

    return mask.to(input_ids.device)

def _find_subseq(haystack, needle):
    if len(needle) == 0:
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i, i+len(needle)
    return None

def get_text_entity_token_mask_subseq(input_ids_bt, prompt_text: str, tok) -> torch.Tensor:
    m = re.search(r"Entity:\s*", prompt_text)
    if not m:
        return torch.zeros_like(input_ids_bt, dtype=torch.bool, device=input_ids_bt.device)

    ent_start = m.end()

    m2 = re.search(r"\.\s*(?:\n\s*)?Query:", prompt_text[ent_start:])
    if m2:
        ent_end = ent_start + m2.start()
    else:
        m3 = re.search(r"(?:\n\s*)Query:", prompt_text[ent_start:])
        ent_end = ent_start + (m3.start() if m3 else len(prompt_text))

    entity_str = prompt_text[ent_start:ent_end].strip()

    ids_model = input_ids_bt[0].detach().cpu().tolist()

    # SentencePiece boundary sensitivity: try with leading space/newline too
    candidates = [
        entity_str,
        " " + entity_str,
        "\n" + entity_str,
    ]

    best = None
    best_len = -1
    for s in candidates:
        ent_ids = tok(s, add_special_tokens=False)["input_ids"]
        loc = _find_subseq(ids_model, ent_ids)
        if loc is not None and len(ent_ids) > best_len:
            best = loc
            best_len = len(ent_ids)

    if best is None:
        return torch.zeros_like(input_ids_bt, dtype=torch.bool, device=input_ids_bt.device)

    a, b = best
    mask = torch.zeros_like(input_ids_bt, dtype=torch.bool)
    mask[0, a:b] = True
    mask[0, -1] = False
    return mask


# -------------------------
# Bin + Subbin config
# -------------------------
def make_subbin_ids_for_bin(positions: np.ndarray, n_subbins: int) -> np.ndarray:
    L = len(positions)
    if L == 0:
        return np.array([], dtype=int)
    if n_subbins <= 1:
        return np.zeros(L, dtype=int)

    cuts = np.linspace(0, L, num=n_subbins + 1).round().astype(int)
    sub = np.zeros(L, dtype=int)
    for b in range(n_subbins):
        a, c = cuts[b], cuts[b + 1]
        if c > a:
            sub[a:c] = b
    return sub

def build_subbin_map(token_positions: np.ndarray,
                     token_bin_names: np.ndarray,
                     subbins_cfg: dict = SUBBINS_FULLCOVER) -> np.ndarray:
    K = len(token_positions)
    subbin_names = np.array(["Query_0/10"] * K, dtype=object)

    for bin_name in ["Context", "Entity", "Query", "Final"]:
        n_sub = int(subbins_cfg.get(bin_name, 1))
        idxs = np.where(token_bin_names == bin_name)[0]
        if len(idxs) == 0:
            continue
        order = np.argsort(token_positions[idxs])
        idxs_sorted = idxs[order]
        sub_ids = make_subbin_ids_for_bin(token_positions[idxs_sorted], n_sub)
        for j, k_idx in enumerate(idxs_sorted):
            subbin_names[k_idx] = f"{bin_name}_{sub_ids[j]}/{n_sub}"
    return subbin_names

def build_bins_bt(input_ids_bt: torch.Tensor, prompt_text: str, entity_mask_bt: torch.Tensor, processor):
    tok = processor.tokenizer
    ids = input_ids_bt[0].detach().cpu().tolist()
    T = len(ids)

    labels = np.array(["Query"] * T, dtype=object)
    labels[T - 1] = "Final"

    emask = entity_mask_bt[0].detach().cpu().numpy().astype(bool)
    emask[T - 1] = False
    labels[emask] = "Entity"

    enc = tok(prompt_text, add_special_tokens=True, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    ids_tok = enc["input_ids"]

    if len(ids_tok) == T and ids_tok == ids:
        m = re.search(r"Entity:\s*", prompt_text)
        if m:
            ctx_end_char = m.end()
            for i, (s, e) in enumerate(offsets):
                if i == T - 1:
                    continue
                if labels[i] in ("Entity", "Final"):
                    continue
                if e <= s:
                    continue
                if e <= ctx_end_char:
                    labels[i] = "Context"
    else:
        ent_pos = np.where(emask)[0]
        if len(ent_pos) > 0:
            first_ent = int(ent_pos.min())
            for i in range(first_ent):
                if labels[i] not in ("Entity", "Final"):
                    labels[i] = "Context"

    bin_text = {}

    def recover_span(bin_name: str):
        idxs = np.where(labels == bin_name)[0]
        if len(idxs) == 0:
            return ""
        spans = [offsets[i] for i in idxs if i < len(offsets) and offsets[i][1] > offsets[i][0]]
        if not spans:
            return ""
        a = min(s for s, _ in spans)
        b = max(e for _, e in spans)
        return prompt_text[a:b].strip()

    ctx_txt = recover_span("Context")
    if ctx_txt:
        bin_text["Context"] = ctx_txt

    qry_txt = recover_span("Query")
    if qry_txt:
        bin_text["Query"] = qry_txt

    if emask.any():
        ent_txt = recover_span("Entity")
        bin_text["Entity"] = ent_txt if ent_txt else "<ENTITY TOKENS>"

    try:
        bin_text["Final"] = tok.decode([ids[-1]], skip_special_tokens=False)
    except Exception:
        bin_text["Final"] = str(ids[-1])

    return labels, bin_text


# -------------------------
# Embedding capture (single API)
# -------------------------
def get_clean_lm_embeddings(model, inputs, active_emb=None):
    """
    Returns: (lm_ie_clean, mode_str)
      mode_str in {"inputs_embeds","embed_tokens",None}
    """
    lm_store = {}
    with torch.no_grad():
        with capture_lm_inputs_embeds(model, lm_store, key="lm_ie"):
            _ = model(**inputs, use_cache=False)

    lm_ie = lm_store.get("lm_ie", None)
    if lm_ie is not None:
        return lm_ie, "inputs_embeds"

    if active_emb is not None:
        lm_store2 = {}
        with torch.no_grad():
            with capture_embed_tokens_output(active_emb, lm_store2, key="lm_ie"):
                _ = model(**inputs, use_cache=False)
        lm_ie2 = lm_store2.get("lm_ie", None)
        if lm_ie2 is not None:
            return lm_ie2, "embed_tokens"

    return None, None


# -------------------------
# Sigma computation
# -------------------------
def compute_dataset_sigma_lm_entity(
    model,
    processor,
    analysis_df,
    data_lookup,
    entity_modality: str,
    save_path: str,
    active_emb=None,
    max_samples=None,
) -> float:
    if os.path.exists(save_path):
        print(f"Loading cached sigma from {save_path}")
        return float(torch.load(save_path))

    print(f"Computing dataset-level σ_lm_entity ({entity_modality}) (streaming)...")

    count = 0
    mean = 0.0
    M2 = 0.0

    tok = processor.tokenizer

    for i, (_, row) in enumerate(tqdm(analysis_df.iterrows(), total=len(analysis_df))):
        if max_samples and i >= max_samples:
            break

        if entity_modality == "vision":
            item = data_lookup.get(row["ID"])
            if not item or not item.get("image_path"):
                continue
            full_path = os.path.join(MLLMKC_ROOT, item["image_path"][0])
            if not os.path.exists(full_path):
                continue
            pil_image = Image.open(full_path).convert("RGB")

            inputs_full = make_inputs(
                processor=processor,
                model_device=model.device,
                retrieved_context=row["Context_vis"],
                user_query=row["Query_vis"],
                entity_modality="vision",
                mock_RAG=True,
                bench_type="people",
                image=pil_image,
                padding=True,
                add_mcq_prefix=True,
            )
        else:
            inputs_full = make_inputs(
                processor=processor,
                model_device=model.device,
                retrieved_context=row["Context_text"],
                user_query=row["Query_text"],
                entity_modality="text",
                mock_RAG=True,
                bench_type="people",
                image=None,
                instance_name=row["Instance_text"],
                padding=True,
                add_mcq_prefix=True,
            )

        input_ids = inputs_full.get("input_ids", None)
        if input_ids is None:
            continue

        prompt_text = inputs_full.get("_prompt_text", "")

        if entity_modality == "vision":
            entity_mask = get_visual_token_mask(processor, input_ids)
        else:
            entity_mask = get_text_entity_token_mask_subseq(input_ids, prompt_text, tok)

        if not entity_mask.any():
            continue

        model_inputs = {k: v for k, v in inputs_full.items() if not k.startswith("_")}
        lm_ie, mode = get_clean_lm_embeddings(model, model_inputs, active_emb=active_emb)
        if lm_ie is None:
            continue

        # sanity: lengths should match
        if lm_ie.shape[:2] != input_ids.shape[:2]:
            continue

        vals = lm_ie[entity_mask].float().view(-1)
        n = vals.numel()
        if n < 2:
            continue

        batch_mean = vals.mean().item()
        batch_var = vals.var(unbiased=False).item()
        batch_M2 = batch_var * n

        if count == 0:
            mean = batch_mean
            M2 = batch_M2
            count = n
        else:
            delta = batch_mean - mean
            new_count = count + n
            mean = mean + delta * n / new_count
            M2 = M2 + batch_M2 + (delta**2) * count * n / new_count
            count = new_count

    if count < 2:
        sigma = 1.0
    else:
        variance = M2 / (count - 1)
        sigma = float(variance ** 0.5)

    print(f"σ_lm_entity({entity_modality}) = {sigma:.6f}  (from {count} values)")
    torch.save(float(sigma), save_path)
    return float(sigma)


# -------------------------
# Main attribution runner
# -------------------------
def run_attribution(
    model,
    processor,
    model_name: str,
    layer_type=LAYER_TYPE_DEFAULT,
    slice_idx=0,
    num_slices=10,
    bench_type="people",
    entity_modality="vision",
    mock_RAG=True,
    active_emb=None,
):
    assert entity_modality in {"vision", "text"}

    tag_model = "gemma" if "gemma" in model_name.lower() else "qwen"
    if entity_modality == "vision":
        OUT = OUT_CSV.replace(".csv", f"_{tag_model}_{'no_conflict_' if not mock_RAG else ''}slice{slice_idx+1}of{num_slices}_{layer_type}.csv")
    else:
        OUT = OUT_CSV.replace(".csv", f"_{tag_model}_{'no_conflict_' if not mock_RAG else ''}text_slice{slice_idx+1}of{num_slices}_{layer_type}.csv")

    with open(JSON_DATA, "r") as f:
        data_lookup = {item["ID"]: item for item in json.load(f)}

    if tag_model == "qwen":
        ALPHA = 6.0
        print("Loading Data (Celeb - Qwen)...")
        df_vis  = pd.read_csv(QWEN_CELEB_VIS_CSV)
        df_text = pd.read_csv(QWEN_CELEB_TEXT_CSV)
        df_vis_corr = pd.read_csv(QWEN_CELEB_NO_RAG_VISION_CSV)
        df_txt_corr = pd.read_csv(QWEN_CELEB_NO_RAG_TEXT_CSV)
        analysis_df = filter_df_for_analysis(df_vis, df_text, df_vis_corr, df_txt_corr)
    else:
        if entity_modality =="vision":
            ALPHA = 8.0
        else: 
            ALPHA = 2.0
        print("Loading Data (Celeb - Gemma)...")
        df_vis  = pd.read_csv(GEMMA_CELEB_VIS_CSV)
        df_text = pd.read_csv(GEMMA_CELEB_TEXT_CSV)
        df_vis_corr = pd.read_csv(GEMMA_CELEB_NO_RAG_VISION_CSV)
        df_txt_corr = pd.read_csv(GEMMA_CELEB_NO_RAG_TEXT_CSV)
        analysis_df = filter_df_for_analysis(df_vis, df_text, df_vis_corr, df_txt_corr)

    print(f"Alpha is {ALPHA}")
    if entity_modality == "vision":
        sigma_cache_path = f"sigma_{tag_model}_{bench_type}_vision.pt"
    else:
        sigma_cache_path = f"sigma_{tag_model}_{bench_type}_text.pt"
    # sigma cache per modality
    if entity_modality == "vision":
        sigma_lm_entity = compute_dataset_sigma_lm_entity(
            model, processor, analysis_df, data_lookup, "vision", sigma_cache_path, active_emb=active_emb
        )
    else:
        sigma_lm_entity = compute_dataset_sigma_lm_entity(
            model, processor, analysis_df, data_lookup, "text", sigma_cache_path, active_emb=active_emb
        )

    # slice
    assert 0 <= slice_idx < num_slices
    n = len(analysis_df)
    start = (n * slice_idx) // num_slices
    end   = (n * (slice_idx + 1)) // num_slices
    analysis_df = analysis_df.iloc[start:end].reset_index(drop=True)
    print(f"[slice {slice_idx+1}/{num_slices}] rows {start}:{end} (n={len(analysis_df)})")
    print(f"Processing {len(analysis_df)} items...")

    layers = get_language_model(model).layers

    if layer_type == "attention":
        components = ["attn"]
    elif layer_type == "mlp":
        components = ["mlp"]
    else:
        components = ["attn", "mlp"]

    all_rows = []
    debug_prints = 0
    tok = processor.tokenizer

    for _, row in tqdm(analysis_df.iterrows(), total=len(analysis_df)):
        # -------- inputs --------
        if entity_modality == "vision":
            item = data_lookup.get(row["ID"])
            if not item or not item.get("image_path"):
                continue
            full_path = os.path.join(MLLMKC_ROOT, item["image_path"][0])
            if not os.path.exists(full_path):
                continue
            pil_image = Image.open(full_path).convert("RGB")
            if max(pil_image.size) > 1024:
                pil_image.thumbnail((1024, 1024))

            inputs_full = make_inputs(
                processor=processor,
                model_device=model.device,
                retrieved_context=row["Context_vis"],
                user_query=row["Query_vis"],
                entity_modality="vision",
                mock_RAG=mock_RAG,
                bench_type=bench_type,
                image=pil_image,
                padding=True,
                add_mcq_prefix=False,
            )
        else:
            inputs_full = make_inputs(
                processor=processor,
                model_device=model.device,
                retrieved_context=row["Context_text"],
                user_query=row["Query_text"],
                entity_modality="text",
                mock_RAG=mock_RAG,
                bench_type=bench_type,
                image=None,
                instance_name=row["Instance_text"],
                padding=True,
                add_mcq_prefix=False,
            )

        input_ids = inputs_full.get("input_ids", None)
        if input_ids is None:
            print("INPUT IDS IS NONE")
            continue

        prompt_text = inputs_full.get("_prompt_text", "")
        attn_mask = inputs_full.get("attention_mask", torch.ones_like(input_ids, dtype=torch.long))

        # -------- entity mask --------
        if entity_modality == "vision":
            entity_mask = get_visual_token_mask(processor, input_ids)
        else:
            entity_mask = get_text_entity_token_mask_subseq(input_ids, prompt_text, tok)
        # print(f"ENTITY MASK {entity_mask}")#dbg
        # print(prompt_text)#dbg
        # ids = inputs_full["input_ids"][0].tolist()#dbg
        # txt = processor.tokenizer.decode(ids, skip_special_tokens=False)#dbg
        # print("PROMPT DECODE (first 400 chars):", repr(txt[:400]))#dbg
        # print("ENTITY STR:", repr(row["Instance_text"]))#dbg
        # print("ENTITY IN DECODE?", row["Instance_text"] in txt)#dbg
        if not entity_mask.any():
            continue

        token_mask = (entity_mask & attn_mask.bool()) if MEAN_OVER_ENTITY_TOKENS_ONLY else attn_mask.bool()

        if debug_prints < 2:
            debug_dump_inputs(inputs_full, input_ids, entity_mask, attn_mask, token_mask, tag="DBG1")
            debug_prints += 1

        # -------- bins + subbins --------
        bin_labels_full, bin_text = build_bins_bt(input_ids, prompt_text, entity_mask, processor)
        mask_pos = token_mask[0].detach().cpu().numpy().astype(bool)
        token_positions = np.nonzero(mask_pos)[0]
        token_bin_names = bin_labels_full[token_positions]
        subbin_names = build_subbin_map(token_positions, token_bin_names, SUBBINS_FULLCOVER)

        subbin_to_mask = {}
        for sb in np.unique(subbin_names):
            pos_sb = token_positions[subbin_names == sb]
            if len(pos_sb) == 0:
                continue
            m = torch.zeros_like(token_mask, dtype=torch.bool)
            m[0, torch.tensor(pos_sb, device=m.device, dtype=torch.long)] = True
            subbin_to_mask[sb] = (m & token_mask)

        if len(subbin_to_mask) == 0:
            continue

        # -------- clean pass: cache activations --------
        clean_cache = {}
        clean_hooks = []

        def make_clean_hook(layer_idx, comp):
            def hook(module, module_inputs, module_output):
                h = extract_tensor(module_output).detach()
                for sb, sb_mask in subbin_to_mask.items():
                    h_sb = masked_tokens(h, sb_mask)
                    clean_cache[(layer_idx, comp, sb)] = h_sb.to("cpu", non_blocking=True)
                return module_output
            return hook

        for i, layer in enumerate(layers):
            if "attn" in components:
                clean_hooks.append(layer.self_attn.register_forward_hook(make_clean_hook(i, "attn")))
            if "mlp" in components:
                clean_hooks.append(layer.mlp.register_forward_hook(make_clean_hook(i, "mlp")))

        model_inputs = {k: v for k, v in inputs_full.items() if not k.startswith("_")}

        # capture clean embeddings (inputs_embeds preferred, else embed_tokens)
        lm_ie_clean, emb_mode = get_clean_lm_embeddings(model, model_inputs, active_emb=active_emb)
        if lm_ie_clean is None:
            for h in clean_hooks:
                h.remove()
            continue

        with torch.no_grad():
            out = model(**model_inputs, use_cache=False)
            logits = out.logits[:, -1, :]
            clean_logits = logits.detach()
            clean_probs = F.softmax(logits, dim=-1).detach()

        for h in clean_hooks:
            h.remove()

        # sanity: make sure mask aligns
        if lm_ie_clean.shape[:2] != input_ids.shape[:2]:
            continue

        # -------- corruption at embeddings --------
        lm_corr_f32 = lm_ie_clean.float().clone()

        noise = torch.randn(
            lm_corr_f32[entity_mask].shape,
            device=lm_corr_f32.device,
            dtype=torch.float32
        ) * (ALPHA * float(sigma_lm_entity))

        lm_corr_f32[entity_mask] += noise
        lm_corr = lm_corr_f32.to(dtype=lm_ie_clean.dtype).detach().requires_grad_(True)

        if debug_prints < 4:
            debug_dump_lm_embed_delta(lm_ie_clean, lm_corr, entity_mask, noise, tag="DBG2-LM")
            debug_prints += 1

        # -------- corrupted forward w/ grad hooks --------
        scores = {(i, comp, sb): 0.0
                  for i in range(len(layers))
                  for comp in components
                  for sb in subbin_to_mask.keys()}

        corr_hooks = []

        def make_corr_hook(layer_idx, comp):
            def hook(module, module_inputs, module_output):
                h_corr = extract_tensor(module_output)
                corr_cache = {}
                for sb, sb_mask in subbin_to_mask.items():
                    h_corr_sb = masked_tokens(h_corr, sb_mask)
                    if h_corr_sb.numel() == 0:
                        continue
                    corr_cache[sb] = h_corr_sb.detach()

                def bwd_hook(grad_h_corr):
                    for sb, sb_mask in subbin_to_mask.items():
                        h_clean_sb_cpu = clean_cache.get((layer_idx, comp, sb), None)
                        if h_clean_sb_cpu is None or sb not in corr_cache:
                            continue

                        grad_sb = masked_tokens(grad_h_corr, sb_mask)
                        if grad_sb.numel() == 0:
                            continue

                        h_clean_sb = h_clean_sb_cpu.to(device=grad_sb.device, dtype=grad_sb.dtype, non_blocking=True)
                        h_corr_sb  = corr_cache[sb].to(dtype=grad_sb.dtype)

                        delta = (h_clean_sb - h_corr_sb)
                        token_scores = (grad_sb * delta).sum(dim=-1).abs()

                        bin_name = sb.split("_", 1)[0]
                        val = token_scores.max() if bin_name == "Entity" else token_scores.mean()
                        scores[(layer_idx, comp, sb)] += float(val.detach().cpu())

                h_corr.register_hook(bwd_hook)
                return module_output
            return hook

        for i, layer in enumerate(layers):
            if "attn" in components:
                corr_hooks.append(layer.self_attn.register_forward_hook(make_corr_hook(i, "attn")))
            if "mlp" in components:
                corr_hooks.append(layer.mlp.register_forward_hook(make_corr_hook(i, "mlp")))

        model.zero_grad(set_to_none=True)

        # override at the same place we captured from
        if emb_mode == "inputs_embeds":
            with override_lm_inputs_embeds(model, lm_corr):
                corrupted_out = model(**model_inputs, use_cache=False)
        else:
            # embed_tokens override requires active_emb
            if active_emb is None:
                for h in corr_hooks:
                    h.remove()
                continue
            with override_embed_tokens_output(active_emb, lm_corr):
                corrupted_out = model(**model_inputs, use_cache=False)

        corrupted_logits = corrupted_out.logits[:, -1, :]
        corrupted_probs  = F.softmax(corrupted_logits, dim=-1).detach()



        loss = F.kl_div(
            F.log_softmax(corrupted_logits, dim=-1),
            clean_probs,
            reduction="batchmean",
        )

        loss.backward()

        if debug_prints < 6:
            debug_dump_logits(clean_logits, corrupted_logits, clean_probs, corrupted_probs, tag="DBG3")
            debug_prints += 1   
            top_clean = clean_probs[0].argmax().item()
            top_corr  = corrupted_probs[0].argmax().item()
            print("KL", float(loss.detach().cpu()))
            print("top clean", top_clean, float(clean_probs[0, top_clean]))
            print("top corr ", top_corr,  float(corrupted_probs[0, top_corr]))

        for h in corr_hooks:
            h.remove()

        # -------- write rows --------
        for i in range(len(layers)):
            for comp in components:
                for sb in subbin_to_mask.keys():
                    bin_name = sb.split("_", 1)[0] if "_" in sb else sb
                    all_rows.append({
                        "ID": row["ID"],
                        "Entity": row["Instance_vis"] if entity_modality == "vision" else row["Instance_text"],
                        "Category": row["Category"],
                        "Group": row["Group"],
                        "Layer": i,
                        "Component": comp,
                        "Bin": bin_name,
                        "Subbin": sb,
                        "Attribution": scores[(i, comp, sb)],
                    })

        if len(all_rows) >= 20000:
            out_df = pd.DataFrame(all_rows)
            header = not os.path.exists(OUT)
            out_df.to_csv(OUT, mode="a", header=header, index=False)
            all_rows = []

        # cleanup
        del clean_cache, clean_probs, clean_logits, out
        del corrupted_out, corrupted_logits, loss
        del lm_ie_clean, lm_corr, lm_corr_f32, noise
        del scores, subbin_to_mask
        del token_mask, entity_mask, attn_mask
        torch.cuda.empty_cache()

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        header = not os.path.exists(OUT)
        out_df.to_csv(OUT, mode="a", header=header, index=False)

    print(f"Done. Saved to {OUT}")


# -------------------------
# Model loader
# -------------------------
def load_model_and_processor(model_name: str):
    dtype = torch.bfloat16 if (
        torch.cuda.is_available()
        and torch.cuda.get_device_name(0).startswith(("NVIDIA A100", "NVIDIA H100"))
    ) else torch.float16

    print(f"Loading model: {model_name} (dtype={dtype})")

    name_lc = model_name.lower()

    if "qwen" in name_lc:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
    elif "gemma" in name_lc:
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_ok else torch.float16
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
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )

    processor = AutoProcessor.from_pretrained(model_name)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        try:
            model.model.gradient_checkpointing_enable()
        except Exception:
            pass

    return model, processor


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--slice_idx", type=int, default=0)
    parser.add_argument("--num_slices", type=int, default=10)
    parser.add_argument("--layer_type", type=str, default="mlp", choices=["mlp", "attention", "both"])
    parser.add_argument("--bench_type", type=str, default="people", choices=["people", "logo"])
    parser.add_argument("--entity_modality", type=str, default="text", choices=["text", "vision"])
    parser.add_argument("--mock_RAG", type=str2bool, default=True)

    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_name)

    # Find ACTIVE_EMB once (only needed for Gemma text fallback)
    active_emb = None
    if "gemma" in args.model_name.lower():
        # Build one representative text input to discover which embed_tokens is active
        # We'll pull the first row from Gemma text csv and create a sample input.
        try:
            df_text = pd.read_csv(GEMMA_CELEB_TEXT_CSV)
            df_vis  = pd.read_csv(GEMMA_CELEB_VIS_CSV)
            df_txt_corr = pd.read_csv(GEMMA_CELEB_NO_RAG_TEXT_CSV)
            df_vis_corr = pd.read_csv(GEMMA_CELEB_NO_RAG_VISION_CSV)
            analysis_df0 = filter_df_for_analysis(df_vis, df_text, df_vis_corr, df_txt_corr)
            if len(analysis_df0) > 0:
                r0 = analysis_df0.iloc[0]
                inputs0 = make_inputs(
                    processor=processor,
                    model_device=model.device,
                    retrieved_context=r0["Context_text"],
                    user_query=r0["Query_text"],
                    entity_modality="text",
                    mock_RAG=True,
                    bench_type=args.bench_type,
                    image=None,
                    instance_name=r0["Instance_text"],
                    padding=True,
                    add_mcq_prefix=False,
                )
                sample_inputs = {k: v for k, v in inputs0.items() if not k.startswith("_")}
                active_emb = find_active_embed_tokens_module(model, sample_inputs)
                print(f"[INFO] ACTIVE embed_tokens module = {active_emb}")
        except Exception as e:
            print(f"[WARN] Could not find ACTIVE_EMB (will rely on inputs_embeds only). Error: {e}")

    run_attribution(
        model=model,
        processor=processor,
        model_name=args.model_name,
        layer_type=args.layer_type,
        slice_idx=args.slice_idx,
        num_slices=args.num_slices,
        bench_type=args.bench_type,
        entity_modality=args.entity_modality,
        mock_RAG=args.mock_RAG,
        active_emb=active_emb,
    )