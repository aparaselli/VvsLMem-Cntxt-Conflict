import os

# --- 1. SETUP ENVIRONMENT ---
os.environ["HF_HOME"] = "/oscar/scratch/aparasel/hf_cache"
os.environ["HF_HUB_CACHE"] = "/oscar/scratch/aparasel/hf_cache/hub"

import torch
import pandas as pd
import numpy as np
import json
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm

# --- 2. MODEL SETUP ---
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_name(0).startswith(("NVIDIA A100","NVIDIA H100")) else torch.float16

print(f"Loading model: {model_name}...")
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
if model.generation_config.pad_token_id is None:
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

print("Model loaded successfully.")

# --- 3. INPUT PREPARATION (CORRECTED) ---
def get_model_inputs(retrieved_context, user_query, entity_name=None, image=None, ctxt_image=None):
    # A. Build Content (User Instruction ONLY)
    content_list = []
    
    # 1. Context
    if retrieved_context is None:
        text_before = "Given your knowledge, answer the multiple choice question about the following entity.\nEntity: "
        content_list.append({"type": "text", "text": text_before})
    else:
        content_list.append({"type": "text", "text": "Context information is below.\n---------------------\n"})
        if ctxt_image:
             content_list.append({"type": "image", "image": ctxt_image})
        content_list.append({"type": "text", "text": f"{retrieved_context}\n---------------------\n"})
        content_list.append({"type": "text", "text": "Given the context information and your knowledge, answer the multiple choice question about the following entity.\nEntity: "})

    # 2. Entity / Modality
    if entity_name is not None:
        # TEXT PROMPT: "EntityName.\nQuery:" (Note the period)
        content_list.append({"type": "text", "text": f"{entity_name}.\nQuery: {user_query}"})
    elif image is not None:
        # VISION PROMPT: "[Image]\nQuery:" (No period)
        content_list.append({"type": "image", "image": image})
        content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})

    # B. Generate Prompt & Prefill
    messages = [{"role": "user", "content": content_list}]
    
    # Apply template (adds <|user|>...<|assistant|>)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Manually append Assistant Prefill (matches your eval script)
    prompt_text += " Between A, B, C, and D, the answer is "

    # C. Handle Images
    active_images = []
    if ctxt_image: active_images.append(ctxt_image)
    if image: active_images.append(image)

    if not active_images:
        inputs = processor(text=[prompt_text], return_tensors="pt", padding=False)
        inputs = inputs.to(model.device)
        return inputs

    img_inputs_dummy = processor(text=[""] * len(active_images), images=active_images, return_tensors="pt")
    img_inputs_dummy = img_inputs_dummy.to(model.device)
    
    pixel_values = img_inputs_dummy["pixel_values"]
    image_grid_thw = img_inputs_dummy.get("image_grid_thw", None)

    text_inputs = processor(text=[prompt_text], images=active_images, return_tensors="pt", padding=False)
    text_inputs = text_inputs.to(model.device)
    
    text_inputs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        text_inputs["image_grid_thw"] = image_grid_thw
        
    return text_inputs

# --- 4. UTILS ---

def get_right_aligned_map(vis_ids, text_ids):
    """
    Returns mapping from END of sequence backwards.
    Used here to find the 'Anchor' (Start of the Query).
    """
    mapping = []
    vis_len = len(vis_ids)
    text_len = len(text_ids)
    min_len = min(vis_len, text_len)
    
    for i in range(1, min_len + 1):
        v_idx = vis_len - i
        t_idx = text_len - i
        
        if vis_ids[v_idx] == text_ids[t_idx]:
            mapping.append((v_idx, t_idx))
        else:
            break
    # mapping is [Last_Token, ..., First_Common_Token]
    return mapping

class ActivationPatcher:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

    def clear(self):
        for h in self.hooks: h.remove()
        self.hooks = []
        self.activations = {}

    def save_hook(self, layer_idx):
        def fn(module, input, output):
            self.activations[layer_idx] = output[0].detach().cpu()
        return fn

    def patch_hook(self, target_idx, source_vector):
        def fn(module, input, output):
            # Move source to correct device/dtype
            src = source_vector.to(output[0].device).to(output[0].dtype)
            output[0][:, target_idx, :] = src
            return output
        return fn

# --- 5. MAIN EXECUTION ---

def run_patching_experiment(vis_csv, text_csv, json_data_path, out_path):
    print("Loading Data...")
    df_vis = pd.read_csv(vis_csv)
    df_text = pd.read_csv(text_csv)
    
    with open(json_data_path, 'r') as f:
        raw_data = json.load(f)
    data_lookup = {item['ID']: item for item in raw_data}

    ## REMOVE CELEBRITIES THAT ARE UNKNOWN AND QUESTIONS IT DOES NOT KNOW THE ANSWER TO 
    df = pd.read_csv("/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/WHO_VISION_Experiment_Results.csv")
    df['Knows_celeb'] = df.apply(lambda row: row['Instance'] in row['Response'], axis=1)
    unknown_celebs = df[df["Knows_celeb"] == False]['Instance'].unique() 
    unknown_celebs = unknown_celebs[1:] # "Remove The Rock from unknown list bc it knows Dwayne Johnson"
    unknown_celebs

    df_inherent_text = pd.read_csv("/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/No_RAG_Experiment_Results.csv")
    df_inherent_vision = pd.read_csv("/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/No_RAG_VISION_Experiment_Results.csv")
    def get_correct_indices(df):
        prob_cols = ['Prob_A', 'Prob_B', 'Prob_C', 'Prob_D']
        predicted_col = df[prob_cols].idxmax(axis=1)

        predicted_choice = predicted_col.str.replace('Prob_', '')
        

        is_correct = predicted_choice.str.strip() == df['Ground_Truth'].str.strip()
        
        return is_correct


    text_correct_mask = get_correct_indices(df_inherent_text)
    vision_correct_mask = get_correct_indices(df_inherent_vision)


    both_correct_mask = text_correct_mask & vision_correct_mask

    correct_indices = df_inherent_text.index[both_correct_mask].tolist()

    df_vis = df_vis.drop_duplicates(subset=['Instance', 'Category','Mis_Knowledge_Key'])
    df_inherent_text_corr = df_inherent_text.iloc[correct_indices]
    corr_inst_cat = df_inherent_text_corr[['Instance', 'Category']].drop_duplicates()
    df_vis = df_vis.merge(corr_inst_cat, on=['Instance', 'Category'], how='inner')
    df_vis = df_vis[~df_vis['Instance'].isin(unknown_celebs)]

    df_text = df_text.drop_duplicates(subset=['Instance', 'Category','Mis_Knowledge_Key'])
    df_text = df_text.merge(corr_inst_cat, on=['Instance', 'Category'], how='inner')
    df_text = df_text[~df_text['Instance'].isin(unknown_celebs)]
    ######################
    

    # Filter Differences
    prob_cols = ['Prob_A', 'Prob_B', 'Prob_C', 'Prob_D']
    df_vis['Pred'] = df_vis[prob_cols].idxmax(axis=1).str[-1]
    df_text['Pred'] = df_text[prob_cols].idxmax(axis=1).str[-1]
    
    # --- MERGE FIX: Use ID, Category, and Knowledge Key to ensure 1-to-1 matching ---
    print("Merging dataframes...")
    df_merged = pd.merge(
        df_vis, 
        df_text, 
        on=["ID", "Category", "Mis_Knowledge_Key"], 
        suffixes=("_vis", "_text")
    )
    
    # Filter: Vision Correct & Text Wrong
    mask = (df_merged['Pred_vis'] == df_merged['Ground_Truth_vis']) & \
           (df_merged['Pred_text'] != df_merged['Ground_Truth_vis'])
    
    diff_df = df_merged[mask].copy()
    
    # Remove duplicates if any remain to avoid redundant computation
    diff_df = diff_df.drop_duplicates(subset=['ID', 'Category', 'Mis_Knowledge_Key'])
    
    print(f"Found {len(diff_df)} unique instances to patch.")
    print(diff_df)
    # Limit for Testing (Remove this line for full run!)
    #diff_df = diff_df.iloc[[0]]

    # Tokenizer helpers
    choices = [" A", " B", " C", " D"]
    choice_map = {c.strip(): processor.tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices}

    transformer_layers = model.model.language_model.layers

    patch_results = []
    
    # Processing Loop
    for idx, row in tqdm(diff_df.iterrows(), total=len(diff_df)):
        
        item_id = row['ID']
        item = data_lookup.get(item_id)
        if not item: continue

        instance_name = row['Instance_vis']
        gt_token = choice_map[row['Ground_Truth_vis'].strip()]
        
        # Load Image
        pil_image = None
        if item.get('image_path'):
            full_path = os.path.join("/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data", item['image_path'][0])
            if os.path.exists(full_path):
                pil_image = Image.open(full_path)
                if max(pil_image.size) > 1024: pil_image.thumbnail((1024, 1024))
        
        # 1. Prepare Inputs
        inputs_vis = get_model_inputs(row['Context_vis'], row['Query_vis'], image=pil_image)
        inputs_text = get_model_inputs(row['Context_vis'], row['Query_vis'], entity_name=instance_name)

        # 2. Vision Run (Cache)
        patcher = ActivationPatcher(model)
        for i, layer in enumerate(transformer_layers):
            patcher.hooks.append(layer.register_forward_hook(patcher.save_hook(i)))
            
        with torch.inference_mode():
            _ = model(**inputs_vis)
        
        vision_cache = patcher.activations.copy()
        patcher.clear()

        # 3. Text Baseline
        with torch.inference_mode():
            out_base = model(**inputs_text)
            logits_base = out_base.logits[0, -1, :]
            probs_base = torch.softmax(logits_base, dim=0)
            base_gt = (probs_base[gt_token]).item() 

        # 4. Determine Indices
        vis_ids = inputs_vis.input_ids[0].tolist()
        text_ids = inputs_text.input_ids[0].tolist()
        
        alignment_map = get_right_aligned_map(vis_ids, text_ids)
        
        if not alignment_map:
            print(f"Skipping ID {item_id}: Alignment failed.")
            continue
            
        # The first shared token (Anchor)
        v_anchor, t_anchor = alignment_map[-1] 
        
        # Source: Last Token of Vision Entity (Before \nQuery)
        v_source_idx = v_anchor - 1
        
        # Target: Last Token of Text Entity (Before period and \nQuery)
        t_target_idx = t_anchor - 2
        
        # Debug / Validation string extraction
        target_token_str = processor.tokenizer.decode([text_ids[t_target_idx]])
        
        # 5. Patching Loop (Single Position, All Layers)
        WINDOW_SIZE = 3
        
        for start_layer in range(len(transformer_layers) - WINDOW_SIZE + 1):
            
            active_hooks = []
            end_layer = start_layer + WINDOW_SIZE
            
            for i in range(start_layer, end_layer):
                src_vec = vision_cache[i][0, v_source_idx, :]
                h = transformer_layers[i].register_forward_hook(
                    patcher.patch_hook(t_target_idx, src_vec)
                )
                active_hooks.append(h)
            
            with torch.inference_mode():
                out_patch = model(**inputs_text)
                p_logits = out_patch.logits[0, -1, :]
                probs_patch = torch.softmax(p_logits, dim=0)
            
            patched_gt = (probs_patch[gt_token]).item()
            recovery = patched_gt - base_gt
            
            patch_results.append({
                "ID": item_id,
                "Instance": instance_name,
                "Category": row['Category'], # Added for traceability
                "Patched_Token_Str": target_token_str,
                "Window_Start": start_layer,
                "Window_End": end_layer,
                "Window_Size": WINDOW_SIZE,
                "Base_Prob": base_gt,
                "Patched_Prob": patched_gt,
                "Recovery": recovery
            })
            
            for h in active_hooks:
                h.remove()
        
        # Save periodically
        if len(patch_results) >= 500:
            pd.DataFrame(patch_results).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
            patch_results = []

    if patch_results:
        pd.DataFrame(patch_results).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
    print(f"Done. Results saved to {out_path}")

if __name__ == "__main__":
    VIS_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/RAG_VISION_Experiment_Results.csv"
    TEXT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/RAG_Experiment_Results.csv"
    JSON_DATA = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/people_knowledge.json"
    OUT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/Patching_LastEntityToken_Corrected.csv"
    
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
        
    run_patching_experiment(VIS_CSV, TEXT_CSV, JSON_DATA, OUT_CSV)