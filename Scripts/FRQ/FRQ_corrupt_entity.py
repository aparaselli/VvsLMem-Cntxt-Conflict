import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor
import numpy as np
from FRQ_make_input import make_inputs


os.environ["HF_HOME"] = "/oscar/scratch/aparasel/hf_cache"
os.environ["HF_HUB_CACHE"] = "/oscar/scratch/aparasel/hf_cache/hub"

VIS_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/FRQ_RAG_VISION_Experiment_Results.csv"
TEXT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/FRQ_RAG_Experiment_Results.csv"
VIS_CSV_LOGO = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/FRQ_RAG_VISION_Experiment_Results_logo.csv"
TEXT_CSV_LOGO = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/FRQ_RAG_Experiment_Results_logo.csv"
JSON_DATA = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/people_knowledge.json"

DEFAULT_OUT = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/Results/FRQ_patch_corr_outputs.csv"
DEFAULT_OUT_LOGO = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/Results/FRQ_patch_corr_outputs_logo.csv"



#sanity check function created with chatGPT
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

    # window around start edge
    dump_range(s - window, s + window + 1, "AROUND START EDGE")

    # window around end edge (avoid duplicate printing if edges overlap)
    if e - window > s + window:
        dump_range(e - window, e + window + 1, "AROUND END EDGE")
    else:
        print("\n(End edge overlaps start window; not printing twice.)")

    print("=" * 90)

def make_noise_image_like(pil_image: Optional[Image.Image], *, seed: Optional[int] = None) -> Image.Image:
    """
    Create a random RGB noise image with the same (W,H) as pil_image.
    If pil_image is None, fall back to a reasonable default size.
    """
    if pil_image is None:
        W, H = 224, 224
    else:
        W, H = pil_image.size  # PIL: (W,H)

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")

def get_entity_span_between_entity_and_query(inputs, processor):
    """
    Returns token positions between:
        'Entity: '  and  'Query'
    i.e., freezes exactly 'Taylor Swift.' in your prompt format.
    """

    tokenizer = processor.tokenizer
    device = inputs["input_ids"].device

    prompt = inputs["_prompt_text"]

    # --- 1) Find character span ---
    start_key = "Entity: "
    end_key = "Query"

    start_char = prompt.index(start_key) + len(start_key)
    end_char = prompt.index(end_key)

    # --- 2) Tokenize with offsets ---
    enc = tokenizer(
        prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    offsets = enc["offset_mapping"]
    input_ids = inputs["input_ids"][0]

    patch_positions = []

    for i, (char_start, char_end) in enumerate(offsets):
        if char_start >= start_char and char_end <= end_char:
            patch_positions.append(i)

    return torch.tensor(patch_positions, device=device, dtype=torch.long)

def get_visual_positions_from_input_ids(input_ids_1d, tokenizer):
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


def strip_special(keys_dict: Dict) -> Dict:
    """Remove helper keys like _prompt_text from make_inputs."""
    return {k: v for k, v in keys_dict.items() if not str(k).startswith("_")}


def safe_decode(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""



def build_analysis_df(dataset: str) -> Tuple[pd.DataFrame, Dict]:
    if dataset == "celeb":
        df_vis = pd.read_csv(VIS_CSV)
        df_text = pd.read_csv(TEXT_CSV)
    elif dataset == "logo":
        df_vis = pd.read_csv(VIS_CSV_LOGO)
        df_text = pd.read_csv(TEXT_CSV_LOGO)
    else:
        raise ValueError("dataset must be celeb or logo")

    with open(JSON_DATA, "r") as f:
        data_lookup = {item["ID"]: item for item in json.load(f)}

    def get_correct_indices(df):
        prob_cols = ["Prob_A", "Prob_B", "Prob_C", "Prob_D"]
        predicted_col = df[prob_cols].idxmax(axis=1)
        predicted_choice = predicted_col.str.replace("Prob_", "")
        is_correct = predicted_choice.str.strip() == df["Ground_Truth"].str.strip()
        return is_correct

    if dataset == "celeb":
        df_who = pd.read_csv(
            "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/WHO_VISION_Experiment_Results.csv"
        )
        df_who["Knows_celeb"] = df_who.apply(lambda row: row["Instance"] in row["Response"], axis=1)
        unknown_celebs = df_who[df_who["Knows_celeb"] == False]["Instance"].unique()
        unknown_celebs = unknown_celebs[1:]  # keep your Rock hack

        df_inherent_text = pd.read_csv(
            "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/No_RAG_Experiment_Results.csv"
        )
        df_inherent_vision = pd.read_csv(
            "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/No_RAG_VISION_Experiment_Results.csv"
        )

        text_correct_mask = get_correct_indices(df_inherent_text)
        vision_correct_mask = get_correct_indices(df_inherent_vision)
        both_correct_mask = text_correct_mask & vision_correct_mask
        correct_indices = df_inherent_text.index[both_correct_mask].tolist()

        df_vis = df_vis.drop_duplicates(subset=["Instance", "Category", "Mis_Knowledge_Key"])
        df_text = df_text.drop_duplicates(subset=["Instance", "Category", "Mis_Knowledge_Key"])

        df_inherent_text_corr = df_inherent_text.iloc[correct_indices]
        corr_inst_cat = df_inherent_text_corr[["Instance", "Category"]].drop_duplicates()

        df_vis = df_vis.merge(corr_inst_cat, on=["Instance", "Category"], how="inner")
        df_text = df_text.merge(corr_inst_cat, on=["Instance", "Category"], how="inner")

        df_vis = df_vis[~df_vis["Instance"].isin(unknown_celebs)]
        df_text = df_text[~df_text["Instance"].isin(unknown_celebs)]

    else:  # logo
        df_inherent_text = pd.read_csv(
            "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/No_RAG_Experiment_Results_logo.csv"
        )
        df_inherent_vision = pd.read_csv(
            "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/No_RAG_VISION_Experiment_Results_logo.csv"
        )

        text_correct_mask = get_correct_indices(df_inherent_text)
        vision_correct_mask = get_correct_indices(df_inherent_vision)
        both_correct_mask = text_correct_mask & vision_correct_mask
        correct_indices = df_inherent_text.index[both_correct_mask].tolist()

        df_vis = df_vis.drop_duplicates(subset=["Instance", "Category", "Mis_Knowledge_Key"])
        df_text = df_text.drop_duplicates(subset=["Instance", "Category", "Mis_Knowledge_Key"])

        df_inherent_text_corr = df_inherent_text.iloc[correct_indices]
        corr_inst_cat = df_inherent_text_corr[["Instance", "Category"]].drop_duplicates()

        df_vis = df_vis.merge(corr_inst_cat, on=["Instance", "Category"], how="inner")
        df_text = df_text.merge(corr_inst_cat, on=["Instance", "Category"], how="inner")

    df_merged = pd.merge(
        df_vis, df_text, on=["ID", "Category", "Mis_Knowledge_Key"], suffixes=("_vis", "_text")
    )

    df_merged["Actual_GT"] = df_merged["Ground_Truth_vis"]
    df_merged["Actual_Param"] = df_merged["Mis_Answer_Label_vis"]

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
    analysis_df = analysis_df[analysis_df['Category'] == "Career_error"] #For testing

    return analysis_df, data_lookup



def register_source_cache_hooks(
    layers,
    src_layers: List[int],
    vis_pos: torch.LongTensor,
    cache_dict: Dict[int, torch.Tensor],
):

    hooks = []

    def make_hook(li: int):
        def hook(module, inputs, output):
            # output can be Tensor or tuple; hidden states is first element if tuple
            hs = output[0] if isinstance(output, (tuple, list)) else output  # [B,T,H]
            hs0 = hs[0]  # [T,H]
            if vis_pos.numel() == 0:
                cache_dict[li] = None
                return
            cache_dict[li] = hs0.index_select(0, vis_pos).detach()  # [V,H]
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

    vis_pos = vis_pos.detach()

    def make_hook(dst_li: int):
        src_li = mapping_src_for_dst[dst_li]

        def hook(module, inputs, output):
            cached = cache_dict.get(src_li, None)
            if cached is None or vis_pos.numel() == 0:
                return output

            # unpack
            if isinstance(output, (tuple, list)):
                hs = output[0]  # [B,T,H]
                rest = list(output[1:])
            else:
                hs = output
                rest = None


            T = hs.shape[1]
            if T != expected_T:
                return output

            max_ok = T - 1
            vp = vis_pos[vis_pos <= max_ok]
            if vp.numel() == 0:
                return output

            if cached.shape[0] != vp.numel():
                m = min(cached.shape[0], vp.numel())
                cached_use = cached[:m]
                vp_use = vp[:m]
            else:
                cached_use = cached
                vp_use = vp


            if hs.shape[0] != 1:
                hs = hs.clone()
                hs[:, vp_use, :] = cached_use.unsqueeze(0)
            else:
                hs = hs.clone()
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
):
    assert src_start >= dst_start, "Need src_start > dst_start for back-patching (copy later -> earlier)."

    analysis_df, data_lookup = build_analysis_df(dataset)
    print(f"[{dataset}] Processing {len(analysis_df)} items...")
    if modality=="vision":
        out_csv = f"/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/Corruption_Results/FRQ_frontpatch_freeze_outputs_{split}.csv"
    else:
        out_csv = f"/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/Corruption_Results/FRQ_frontpatch_freeze_outputs_{split}_text.csv"
    layers = model.model.language_model.layers

    if source_type == "same":
        src_layers = [src_start]
        dst_layers = list(range(dst_start, dst_start + num_layers))
        mapping = {dst_layers[i]: src_start for i in range(num_layers)}
    else:
        src_layers = list(range(src_start, src_start + num_layers))
        dst_layers = list(range(dst_start, dst_start + num_layers))
        mapping = {dst_layers[i]: src_layers[i] for i in range(num_layers)}

    L = len(layers)
    # if max(src_layers,dst_layers) >= L or min(src_layers,dst_layers) < 0:
    #     raise ValueError(f"Layer index out of bounds. Model has {L} layers.")

    results = []
    if os.path.exists(out_csv):
        os.remove(out_csv)

    for _, row in tqdm(analysis_df.iterrows(), total=len(analysis_df)):
        item = data_lookup.get(row["ID"])
        if not item:
            continue

        # Load image
        if modality=="vision":
            pil_image = None
            if item.get("image_path"):
                full_path = os.path.join(
                    "/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data",
                    item["image_path"][0],
                )
                if os.path.exists(full_path):
                    pil_image = Image.open(full_path)
                    if max(pil_image.size) > 1024:
                        pil_image.thumbnail((1024, 1024))
        if source_type == "noise":
            seed = int(row["ID"]) if str(row["ID"]).isdigit() else abs(hash(str(row["ID"]))) % (2**32)
            noise_img = make_noise_image_like(pil_image, seed=seed)
            inputs = make_inputs(
                processor=processor,
                model_device=model.device,
                retrieved_context=None,
                user_query=row["Query_vis"],
                entity_modality="vision",
                mock_RAG=False,
                bench_type=("people" if dataset == "celeb" else "logo"),
                image=noise_img,
                padding=True,
            )
        else:
            if modality=="vision":
                inputs = make_inputs(
                    processor=processor,
                    model_device=model.device,
                    retrieved_context=None,
                    user_query=row["Query_vis"],
                    entity_modality="vision",
                    mock_RAG=False,
                    bench_type=("people" if dataset == "celeb" else "logo"),
                    image=pil_image,
                    padding=True,
                )
            elif modality =="text":
                inputs = make_inputs(
                    processor=processor,
                    model_device=model.device,
                    retrieved_context=None,
                    user_query=row["Query_vis"],
                    entity_modality="text",
                    instance_name=row["Instance_vis"],
                    mock_RAG=False,
                    bench_type=("people" if dataset == "celeb" else "logo"),
                    image=None,
                    padding=True,
                )

        # compute positions while helper keys still exist
        if modality == "vision":
            vis_pos = get_visual_positions_from_input_ids(inputs["input_ids"][0], processor.tokenizer)
        elif modality == "text":
            vis_pos = get_entity_span_between_entity_and_query(inputs, processor)
        else:
            raise ValueError("Modality not supported")
        # DEBUG ONLY — comment out later
        print_frozen_edges_only(inputs, processor, vis_pos, window=12)

        # only strip AFTER you’ve computed positions
        inputs = strip_special(inputs)

        # Pass 1: cache source layers can be the pass without the misinformation context, by setting source type ro clean
        cache_dict: Dict[int, Optional[torch.Tensor]] = {}

        if source_type == "clean":
            if modality=="vision":
                inputs_src = make_inputs(
                    processor=processor,
                    model_device=model.device,
                    retrieved_context=None,
                    user_query=row["Query_vis"],
                    entity_modality="vision",
                    mock_RAG=False,
                    bench_type=("people" if dataset == "celeb" else "logo"),
                    image=pil_image,
                    padding=True,
                )
            elif modality == "text":
                inputs_src = make_inputs(
                    processor=processor,
                    model_device=model.device,
                    retrieved_context=None,
                    user_query=row["Query_vis"],
                    entity_modality="text",
                    instance_name=row["Instance_vis"],
                    mock_RAG=False,
                    bench_type=("people" if dataset == "celeb" else "logo"),
                    image=None,
                    padding=True,
                )
            inputs_src = strip_special(inputs_src)
            vis_pos_src = get_visual_positions_from_input_ids(inputs_src["input_ids"][0], processor.tokenizer)

        elif source_type == "noise":
            inputs_src = make_inputs(
                processor=processor,
                model_device=model.device,
                retrieved_context=None,  
                user_query=row["Query_vis"],
                entity_modality="vision",
                mock_RAG=False,
                bench_type=("people" if dataset == "celeb" else "logo"),
                image=pil_image,
                padding=True,
            )
            inputs_src = strip_special(inputs_src)
            vis_pos_src = get_visual_positions_from_input_ids(inputs_src["input_ids"][0], processor.tokenizer)

        else:
            # source_type == "same"
            inputs_src = inputs
            vis_pos_src = vis_pos

        hooks1 = register_source_cache_hooks(layers, src_layers, vis_pos_src, cache_dict)
        with torch.inference_mode():
            _ = model(**inputs_src)
        for h in hooks1:
            h.remove()

        for k, v in list(cache_dict.items()):
            if v is not None:
                cache_dict[k] = v.to("cuda", non_blocking=True)

        # 2) free Pass-1 large tensors
        if source_type in {"clean", "noise"}:
            del inputs_src
            if source_type == "noise":
                del noise_img

        # Pass 2: backpatch into destination layers + generate
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
            first_step_logits = gen_out.scores[0][0]  # [vocab]
            top_token_id = int(first_step_logits.argmax().item())
            top_token_str = safe_decode(processor.tokenizer, top_token_id)

        results.append(
            {
                "ID": row["ID"],
                "Entity": row["Instance_vis"],
                "Category": row["Category"],
                "Group": row["Group"],  # keep original group
                "Original_output": row["Assistant_response_vis"],
                "Original_top_token": row["Top_token_str_vis"],
                "Mis_Knowledge_Key": row["Mis_Knowledge_Key"],
                "Parametric_ans": row["Ground_Truth_vis"],
                "src_start": src_start,
                "dst_start": dst_start,
                "num_layers": num_layers,
                "New_Answer": new_answer,
                "TopDecodedTokenID": top_token_id,
                "TopDecodedToken": top_token_str,
            }
        )

        # periodic flush
        if len(results) >= 500:
            pd.DataFrame(results).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            results = []

    if results:
        pd.DataFrame(results).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"Done. Wrote: {out_csv}")

# Repurposed backpatch script, but this tells you how many of the entity layers are needed for factual recall
# THIS IS FACTUAL RECALL NOT CONTEXT-MEMORY CONFLICT
# THIS IS THE FREEZING EXPERIMENT (Copy acts from mlp 0 to layer 0-i)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="celeb", choices=["celeb", "logo"])
    parser.add_argument("--entity_modality", type=str, default="vision", choices=["vision", "text"])
    parser.add_argument("--src_start", type=int, default=0, help="Start layer for source S (later layers).")
    parser.add_argument("--dst_start", type=int, default=0, help="Start layer for destination D (earlier layers).")
    parser.add_argument("--num_layers", type=int, default=25, help="How many consecutive layers to backpatch.")
    parser.add_argument("--max_new_tokens", type=int, default=6)
    #parser.add_argument("--max_new_tokens", type=int, default=6)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--source_type", type=str, default="same", choices=["same"]) 
    args = parser.parse_args()
    #source type noise automatically calls with not mock_RAG
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_name(0).startswith(("NVIDIA A100", "NVIDIA H100"))
        else torch.float16
    )

    print(f"Loading model: {model_name} ({dtype})")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = DEFAULT_OUT if args.dataset == "celeb" else DEFAULT_OUT_LOGO
        if args.source_type =="clean":
            out_csv = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ/Results/FRQ_frontpatch_outputs_clean.csv"

    for i in range(args.num_layers):
        run_backpatch(
            model=model,
            processor=processor,
            dataset=args.dataset,
            src_start=args.src_start,
            dst_start=args.dst_start,
            num_layers=i+1,
            out_csv=out_csv,
            max_new_tokens=args.max_new_tokens,
            source_type=args.source_type,
            modality=args.entity_modality,
            split=i
        )
