# ============================================================
# Grad·(Δh) Attribution Patching — Qwen2.5-VL (BINNED)
# UPDATED END-TO-END SCRIPT: supports entity_modality = {"vision","text"}
#
# Key updates vs your version:
#  - entity_mask is modality-dependent:
#       vision -> visual placeholder tokens
#       text   -> tokens spanning "Entity: {name}.\nQuery:"
#  - binning uses entity_mask (not hard-coded visual mask)
#  - corruption/noise uses entity_mask (not hard-coded visual mask)
#  - sigma cache computed per modality (visual vs text-entity)
#  - text make_inputs call fixes instance_name=row["Instance_text"]
#  - no longer skips items when visual tokens absent in text mode
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
from transformers import AutoModelForVision2Seq, AutoProcessor
import argparse

from FRQ_make_input import make_inputs  # <-- your provided make_inputs

# -------------------------
# Paths
# -------------------------
VIS_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/FRQ_RAG_VISION_Experiment_Results.csv"
TEXT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/FRQ_RAG_Experiment_Results.csv"
JSON_DATA = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/people_knowledge.json"

WHO_VISION_CSV    = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/WHO_VISION_Experiment_Results.csv"
NO_RAG_TEXT_CSV   = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/No_RAG_Experiment_Results.csv"
NO_RAG_VISION_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/No_RAG_VISION_Experiment_Results.csv"

MLLMKC_ROOT = "/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data"

OUT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/Attr_Results/Attribution_Patching_Binned.csv"

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# -------------------------
# Hyperparams
# -------------------------
ALPHA = 6.0
MEAN_OVER_ENTITY_TOKENS_ONLY = False  
LAYER_TYPE_DEFAULT = "mlp"

SIGMA_CACHE_PATH_VIS  = "sigma_lm_visual_embed.pt"
SIGMA_CACHE_PATH_TEXT = "sigma_lm_text_entity_embed.pt"

dtype = torch.bfloat16 if (
    torch.cuda.is_available()
    and torch.cuda.get_device_name(0).startswith(("NVIDIA A100", "NVIDIA H100"))
) else torch.float16


# -------------------------
# Load model/processor
# -------------------------
print(f"Loading model: {MODEL_NAME} (dtype={dtype})")

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# -------------------------
# Debug helpers
# -------------------------
def _t(x):
    try:
        return float(x.detach().cpu())
    except Exception:
        return float(x)

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
    raise AttributeError("Couldn't find language_model on this Qwen2.5-VL model.")

@contextlib.contextmanager
def capture_lm_inputs_embeds(m, store_dict, key="lm_inputs_embeds"):
    lm = get_language_model(m)

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
def override_lm_inputs_embeds(m, new_inputs_embeds):
    lm = get_language_model(m)

    def pre_hook(module, args, kwargs):
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            kwargs["inputs_embeds"] = new_inputs_embeds
        return (args, kwargs)

    h = lm.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


# -------------------------
# Entity mask builders
# -------------------------
def get_visual_token_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """
    Robust visual-token placeholder detection for Qwen2.5-VL.
    Returns BoolTensor [B, T] where True indicates visual placeholder positions.
    """
    tok = processor.tokenizer
    device = input_ids.device
    candidate_ids = set()

    if hasattr(tok, "image_token_id") and tok.image_token_id is not None:
        candidate_ids.add(int(tok.image_token_id))

    for s in ["<|image_pad|>", "<|vision_pad|>", "<|visual_pad|>", "<|image|>", "<image>"]:
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

    return mask.to(device)

def get_text_entity_token_mask_from_prompt(input_ids_bt, prompt_text: str, tok) -> torch.Tensor:
    """
    Bool mask [B,T] marking tokens corresponding to the entity NAME in:
        "Entity: {entity_name}.\nQuery: ..."
    Uses tokenizer offset_mapping on prompt_text, and verifies alignment with model input_ids.
    """
    # char span
    m = re.search(r"Entity:\s*", prompt_text)
    if not m:
        return torch.zeros_like(input_ids_bt, dtype=torch.bool).to(input_ids_bt.device)

    ent_start = m.end()

    # Prefer ".\nQuery:" boundary (your make_inputs always uses period before newline)
    m2 = re.search(r"\.\s*\nQuery:", prompt_text[ent_start:])
    if m2:
        ent_end = ent_start + m2.start()
    else:
        # fallback: stop at "\nQuery:"
        m3 = re.search(r"\nQuery:", prompt_text[ent_start:])
        ent_end = ent_start + (m3.start() if m3 else len(prompt_text))

    enc = tok(prompt_text, add_special_tokens=True, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    ids_tok = enc["input_ids"]

    ids_model = input_ids_bt[0].detach().cpu().tolist()
    if len(ids_tok) != len(ids_model) or ids_tok != ids_model:
        # offsets no longer reliable
        return torch.zeros_like(input_ids_bt, dtype=torch.bool).to(input_ids_bt.device)

    mask = torch.zeros_like(input_ids_bt, dtype=torch.bool)
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        if not (e <= ent_start or s >= ent_end):
            mask[0, i] = True

    mask[0, -1] = False
    return mask.to(input_ids_bt.device)


# -------------------------
# Hook utils
# -------------------------
def extract_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x):
                return x
    raise TypeError(f"Hook output not tensor/tuple(list) of tensors: {type(output)}")

def masked_tokens(h: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
    """
    h: [B,T,D], mask_bt: [B,T] bool -> returns [K,D]
    """
    m = mask_bt.to(device=h.device)
    if m.dtype != torch.bool:
        m = m.bool()
    return h[m]


# -------------------------
# Bin + Subbin config
# -------------------------
SUBBINS_FULLCOVER = {"Context": 3, "Entity": 1, "Query": 20, "Final": 1}

def build_bins_bt(input_ids_bt: torch.Tensor, prompt_text: str, entity_mask_bt: torch.Tensor):
    """
    Full-coverage binning.
    labels[T] in {"Context","Entity","Query","Final"}.
    bin_text is best-effort and may be empty for some bins due to chat template tokens.
    """
    tok = processor.tokenizer
    ids = input_ids_bt[0].detach().cpu().tolist()
    T = len(ids)

    # 1) init -> Query
    labels = np.array(["Query"] * T, dtype=object)
    labels[T - 1] = "Final"

    # 2) Entity from mask
    emask = entity_mask_bt[0].detach().cpu().numpy().astype(bool)
    emask[T - 1] = False
    labels[emask] = "Entity"

    # 3) Offsets for prompt_text
    enc = tok(prompt_text, add_special_tokens=True, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    ids_tok = enc["input_ids"]

    # 4) Context: everything up to and including "Entity:"
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
        # fallback: before first entity token is context
        ent_pos = np.where(emask)[0]
        if len(ent_pos) > 0:
            first_ent = int(ent_pos.min())
            for i in range(first_ent):
                if labels[i] not in ("Entity", "Final"):
                    labels[i] = "Context"

    # 5) Recover text spans (best-effort; chat template may insert non-offset tokens)
    bin_text = {}

    def recover_span(bin_name: str):
        idxs = np.where(labels == bin_name)[0]
        if len(idxs) == 0:
            return ""
        spans = [
            offsets[i]
            for i in idxs
            if i < len(offsets) and offsets[i][1] > offsets[i][0]
        ]
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
        # For vision entity, there may be no clean text span; for text entity it might.
        ent_txt = recover_span("Entity")
        bin_text["Entity"] = ent_txt if ent_txt else "<ENTITY TOKENS>"

    try:
        bin_text["Final"] = tok.decode([ids[-1]], skip_special_tokens=False)
    except Exception:
        bin_text["Final"] = str(ids[-1])

    return labels, bin_text

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


# -------------------------
# Sigma computation (per modality)
# -------------------------
def compute_dataset_sigma_lm_entity(
    analysis_df,
    data_lookup,
    entity_modality: str,
    save_path: str,
    max_samples=None,
) -> float:
    """
    Computes dataset-level sigma of LM inputs_embeds restricted to entity positions:
      - vision: visual placeholder positions
      - text: tokens spanning entity name in prompt
    Uses streaming mean/var. Caches to save_path.
    """
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

        # Build inputs
        if entity_modality == "vision":
            item = data_lookup.get(row["ID"])
            if not item or not item.get("image_path"):
                continue
            full_path = os.path.join(MLLMKC_ROOT, item["image_path"][0])
            if not os.path.exists(full_path):
                continue
            pil_image = Image.open(full_path).convert("RGB")

            inputs = make_inputs(
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
            # text entity: no image required
            inputs = make_inputs(
                processor=processor,
                model_device=model.device,
                retrieved_context=row["Context_text"],
                user_query=row["Query_text"],
                entity_modality="text",
                mock_RAG=True,
                bench_type="people",
                image=None,
                instance_name=row["Instance_text"],  # IMPORTANT
                padding=True,
                add_mcq_prefix=True,
            )

        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            continue

        prompt_text = inputs.get("_prompt_text", "")

        # entity mask
        if entity_modality == "vision":
            entity_mask = get_visual_token_mask(input_ids)
        else:
            entity_mask = get_text_entity_token_mask_from_prompt(input_ids, prompt_text, tok)

        if not entity_mask.any():
            continue

        store = {}
        with torch.no_grad():
            with capture_lm_inputs_embeds(model, store, key="lm_ie"):
                _ = model(**{k: v for k, v in inputs.items() if not k.startswith("_")}, use_cache=False)

        lm_ie = store.get("lm_ie", None)
        if lm_ie is None:
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
def run_attribution(layer_type=LAYER_TYPE_DEFAULT, slice_idx=0, num_slices=10, bench_type="people", entity_modality="vision", mock_RAG=True):
    assert entity_modality in {"vision", "text"}

    if entity_modality == "vision":
        if mock_RAG:
            OUT = OUT_CSV.replace(".csv", f"_slice{slice_idx+1}of{num_slices}_{layer_type}.csv")
        else:
            OUT = OUT_CSV.replace(".csv", f"_no_cnflict_slice{slice_idx+1}of{num_slices}_{layer_type}.csv")
    else:
        if mock_RAG:
            OUT = OUT_CSV.replace(".csv", f"_text_slice{slice_idx+1}of{num_slices}_{layer_type}.csv")
        else:
            OUT = OUT_CSV.replace(".csv", f"_no_cnflict_text_slice{slice_idx+1}of{num_slices}_{layer_type}.csv")
        

    print("Loading Data...")
    df_vis  = pd.read_csv(VIS_CSV)
    df_text = pd.read_csv(TEXT_CSV)

    with open(JSON_DATA, "r") as f:
        data_lookup = {item["ID"]: item for item in json.load(f)}

    # remove unknown celebs (vision-derived filter)
    df_who = pd.read_csv(WHO_VISION_CSV)
    df_who["Knows_celeb"] = df_who.apply(lambda r: r["Instance"] in r["Response"], axis=1)
    unknown_celebs = df_who[df_who["Knows_celeb"] == False]["Instance"].unique()
    unknown_celebs = unknown_celebs[1:]  # keep your "remove the rock" hack

    df_inherent_text   = pd.read_csv(NO_RAG_TEXT_CSV)
    df_inherent_vision = pd.read_csv(NO_RAG_VISION_CSV)

    def get_correct_indices(df_):
        prob_cols = ["Prob_A", "Prob_B", "Prob_C", "Prob_D"]
        predicted_choice = df_[prob_cols].idxmax(axis=1).str.replace("Prob_", "").str.strip()
        return predicted_choice == df_["Ground_Truth"].str.strip()

    correct_indices = df_inherent_text.index[
        get_correct_indices(df_inherent_text) & get_correct_indices(df_inherent_vision)
    ].tolist()

    df_inherent_text_corr = df_inherent_text.iloc[correct_indices]
    corr_inst_cat = df_inherent_text_corr[["Instance", "Category"]].drop_duplicates()

    df_vis = df_vis.drop_duplicates(subset=["Instance", "Category", "Mis_Knowledge_Key"])
    df_vis = df_vis.merge(corr_inst_cat, on=["Instance", "Category"], how="inner")
    df_vis = df_vis[~df_vis["Instance"].isin(unknown_celebs)]

    df_text = df_text.drop_duplicates(subset=["Instance", "Category", "Mis_Knowledge_Key"])
    df_text = df_text.merge(corr_inst_cat, on=["Instance", "Category"], how="inner")
    df_text = df_text[~df_text["Instance"].isin(unknown_celebs)]

    df_merged = pd.merge(df_vis, df_text, on=["ID", "Category", "Mis_Knowledge_Key"], suffixes=("_vis", "_text"))

    # "true" answers already saved as strings in FRQ
    df_merged["Actual_GT"] = df_merged["Ground_Truth_vis"]
    df_merged["Actual_Param"] = df_merged["Mis_Answer_Label_vis"]

    # Grouping logic assumes Pred_vis/Pred_text already exist and are in {gt, mis_label, neither}
    same_pred_mask = (df_merged["Pred_vis"] == df_merged["Pred_text"]) & (df_merged["Pred_vis"] == "mis_label")
    vis_corr_text_wrong_mask = (df_merged["Pred_vis"] == "gt") & (df_merged["Pred_text"] == "mis_label")
    vistxt_param = (df_merged["Pred_vis"] == df_merged["Pred_text"]) & (df_merged["Pred_vis"] == "gt")
    viscont_text_param = (df_merged["Pred_vis"] == "mis_label") & (df_merged["Pred_text"] == "gt")

    df_merged["Group"] = "Exclude"
    df_merged.loc[same_pred_mask, "Group"] = "VisTxtCont"
    df_merged.loc[vis_corr_text_wrong_mask, "Group"] = "VisParam_TxtCont"
    df_merged.loc[vistxt_param, "Group"] = "VisTxtParam"
    df_merged.loc[viscont_text_param, "Group"] = "VisCont_TxtParam"

    analysis_df = df_merged[df_merged["Group"] != "Exclude"].copy()
    analysis_df = analysis_df.drop_duplicates(subset=["ID", "Category", "Mis_Knowledge_Key"])
    print(analysis_df["Group"].value_counts())

    # sigma over FULL analysis_df, per modality
    if entity_modality == "vision":
        sigma_lm_entity = compute_dataset_sigma_lm_entity(
            analysis_df, data_lookup, entity_modality="vision", save_path=SIGMA_CACHE_PATH_VIS
        )
    else:
        sigma_lm_entity = compute_dataset_sigma_lm_entity(
            analysis_df, data_lookup, entity_modality="text", save_path=SIGMA_CACHE_PATH_TEXT
        )

    # slice
    assert 0 <= slice_idx < num_slices
    n = len(analysis_df)
    start = (n * slice_idx) // num_slices
    end   = (n * (slice_idx + 1)) // num_slices
    analysis_df = analysis_df.iloc[start:end].reset_index(drop=True)
    print(f"[slice {slice_idx+1}/{num_slices}] rows {start}:{end} (n={len(analysis_df)})")
    print(f"Processing {len(analysis_df)} items...")

    layers = model.model.language_model.layers
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
        # -------- build inputs --------
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

            inputs = make_inputs(
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
            inputs = make_inputs(
                processor=processor,
                model_device=model.device,
                retrieved_context=row["Context_text"],
                user_query=row["Query_text"],
                entity_modality="text",
                mock_RAG=mock_RAG,
                bench_type=bench_type,
                image=None,
                instance_name=row["Instance_text"],  # IMPORTANT
                padding=True,
                add_mcq_prefix=False,
            )

        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            continue

        prompt_text = inputs.get("_prompt_text", "")
        attn_mask = inputs.get("attention_mask", torch.ones_like(input_ids, dtype=torch.long))

        # -------- entity mask --------
        if entity_modality == "vision":
            entity_mask = get_visual_token_mask(input_ids)
        else:
            entity_mask = get_text_entity_token_mask_from_prompt(input_ids, prompt_text, tok)

        if not entity_mask.any():
            continue

        # token mask for subbinning/aggregation
        if MEAN_OVER_ENTITY_TOKENS_ONLY:
            token_mask = entity_mask & attn_mask.bool()
        else:
            token_mask = attn_mask.bool()

        if debug_prints < 2:
            debug_dump_inputs(inputs, input_ids, entity_mask, attn_mask, token_mask, tag="DBG1")
            debug_prints += 1

        # -------- build bins + subbins --------
        bin_labels_full, bin_text = build_bins_bt(input_ids, prompt_text, entity_mask)
        valid = {"Context", "Entity", "Query", "Final"}
        bad = set(np.unique(bin_labels_full)) - valid
        assert len(bad) == 0, f"Found invalid bins: {bad}"

        mask_pos = token_mask[0].detach().cpu().numpy().astype(bool)
        token_positions = np.nonzero(mask_pos)[0]
        token_bin_names = bin_labels_full[token_positions]
        subbin_names = build_subbin_map(token_positions, token_bin_names, SUBBINS_FULLCOVER)

        if debug_prints < 4:
            print("[DBG bins]", {k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v) for k, v in bin_text.items()})
            debug_prints += 1

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

        # -------- clean pass: cache activations per (layer, comp, subbin) --------
        clean_cache = {}  # (layer_idx, comp, sb) -> [K_sb, D] CPU
        lm_store = {}
        clean_hooks = []

        def make_clean_hook(layer_idx, comp):
            def hook(module, module_inputs, module_output):
                h = extract_tensor(module_output).detach()  # [B,T,D]
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

        with torch.no_grad():
            with capture_lm_inputs_embeds(model, lm_store, key="lm_ie_clean"):
                out = model(**{k: v for k, v in inputs.items() if not k.startswith("_")}, use_cache=False)
            logits = out.logits[:, -1, :]
            clean_logits = logits.detach()
            clean_probs = F.softmax(logits, dim=-1).detach()

        for h in clean_hooks:
            h.remove()

        lm_ie_clean = lm_store.get("lm_ie_clean", None)
        if lm_ie_clean is None:
            continue

        # -------- corruption: add noise to entity positions (modality-aware) --------
        lm_corr_f32 = lm_ie_clean.float().clone()

        noise = torch.randn(
            lm_corr_f32[entity_mask].shape,
            device=lm_corr_f32.device,
            dtype=torch.float32
        ) * (ALPHA * float(sigma_lm_entity))

        lm_corr_f32[entity_mask] += noise
        lm_corr = lm_corr_f32.to(dtype=lm_ie_clean.dtype).detach().requires_grad_(True)

        if debug_prints < 6:
            debug_dump_lm_embed_delta(lm_ie_clean, lm_corr, entity_mask, noise, tag="DBG2-LM")
            debug_prints += 1

        scores = {(i, comp, sb): 0.0
                  for i in range(len(layers))
                  for comp in components
                  for sb in subbin_to_mask.keys()}

        corr_hooks = []

        def make_corr_hook(layer_idx, comp):
            def hook(module, module_inputs, module_output):
                h_corr = extract_tensor(module_output)  # [B,T,D]
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
                        if bin_name == "Entity":
                            val = token_scores.max()
                        else:
                            val = token_scores.mean()

                        scores[(layer_idx, comp, sb)] += float(val.detach().cpu())

                h_corr.register_hook(bwd_hook)
                return module_output
            return hook

        for i, layer in enumerate(layers):
            if "attn" in components:
                corr_hooks.append(layer.self_attn.register_forward_hook(make_corr_hook(i, "attn")))
            if "mlp" in components:
                corr_hooks.append(layer.mlp.register_forward_hook(make_corr_hook(i, "mlp")))

        corrupted_kwargs = {k: v for k, v in inputs.items() if not k.startswith("_")}
        model.zero_grad(set_to_none=True)

        with override_lm_inputs_embeds(model, lm_corr):
            corrupted_out = model(**corrupted_kwargs, use_cache=False)

        corrupted_logits = corrupted_out.logits[:, -1, :]
        corrupted_probs  = F.softmax(corrupted_logits, dim=-1).detach()

        if debug_prints < 8:
            debug_dump_logits(clean_logits, corrupted_logits, clean_probs, corrupted_probs, tag="DBG3")
            debug_prints += 1

        loss = F.kl_div(
            F.log_softmax(corrupted_logits, dim=-1),
            clean_probs,
            reduction="batchmean",
        )

        if debug_prints < 10:
            diff = (corrupted_probs - clean_probs).abs().max()
            print("KL:", float(loss.detach().cpu()),
                  "max prob diff:", float(diff.detach().cpu()),
                  "entity_tokens:", int(entity_mask.sum()))
            debug_prints += 1

        loss.backward()

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

        # periodic flush
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

    # final flush
    if all_rows:
        out_df = pd.DataFrame(all_rows)
        header = not os.path.exists(OUT)
        out_df.to_csv(OUT, mode="a", header=header, index=False)

    print(f"Done. Saved to {OUT}")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice_idx", type=int, default=0)
    parser.add_argument("--num_slices", type=int, default=10)
    parser.add_argument("--layer_type", type=str, default="mlp", choices=["mlp", "attention", "both"])
    parser.add_argument("--bench_type", type=str, default="people", choices=["people", "logo"])
    parser.add_argument(
        "--entity_modality",
        type=str,
        default="text",
        choices=["text", "vision"],
        help="Modality for entity in query (text or vision)."
    )
    parser.add_argument("--mock_RAG", type=str2bool, default=True)
    args = parser.parse_args()

    run_attribution(
        layer_type=args.layer_type,
        slice_idx=args.slice_idx,
        num_slices=args.num_slices,
        bench_type=args.bench_type,
        entity_modality=args.entity_modality,
        mock_RAG = args.mock_RAG,
    )
