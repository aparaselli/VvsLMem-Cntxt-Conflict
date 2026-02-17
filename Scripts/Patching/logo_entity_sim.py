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
import torch.nn.functional as F

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

# --- 3. INPUT PREPARATION ---
def get_model_inputs(retrieved_context, user_query, entity_name=None, image=None, ctxt_image=None):
    # A. Build Content
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
        # NOTE: For logos, ensure your text prompt format matches exactly what was used in RAG.
        # Often it is: f"The logo of the company known as {instance_name}" or just "{instance_name}"
        # This script assumes the entity_name passed in is the FULL formatted string.
        content_list.append({"type": "text", "text": f"{entity_name}.\nQuery: {user_query}"})
    elif image is not None:
        content_list.append({"type": "image", "image": image})
        content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})

    # B. Generate Prompt & Prefill
    messages = [{"role": "user", "content": content_list}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_text += " Between A, B, C, and D, the answer is "

    # C. Handle Images
    active_images = []
    if ctxt_image: active_images.append(ctxt_image)
    if image: active_images.append(image)

    if not active_images:
        inputs = processor(text=[prompt_text], return_tensors="pt", padding=False)
        inputs = inputs.to(model.device)
        return inputs, prompt_text

    img_inputs_dummy = processor(text=[""] * len(active_images), images=active_images, return_tensors="pt")
    img_inputs_dummy = img_inputs_dummy.to(model.device)
    
    pixel_values = img_inputs_dummy["pixel_values"]
    image_grid_thw = img_inputs_dummy.get("image_grid_thw", None)

    text_inputs = processor(text=[prompt_text], images=active_images, return_tensors="pt", padding=False)
    text_inputs = text_inputs.to(model.device)
    
    text_inputs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        text_inputs["image_grid_thw"] = image_grid_thw
        
    return text_inputs, prompt_text

# --- 4. ALIGNMENT & METRICS ---

def get_right_aligned_map(vis_ids, text_ids):
    """Aligns sequences from the END to find common suffix."""
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
    return mapping

# --- 5. MAIN EXECUTION ---

def run_similarity_experiment(vis_csv, text_csv, json_data_path, out_path, image_base_dir):
    print("Loading Data...")
    df_vis = pd.read_csv(vis_csv)
    df_text = pd.read_csv(text_csv)
    
    with open(json_data_path, 'r') as f:
        raw_data = json.load(f)
    data_lookup = {item['ID']: item for item in raw_data}

    # Filter Setup
    prob_cols = ['Prob_A', 'Prob_B', 'Prob_C', 'Prob_D']
    df_vis['Pred'] = df_vis[prob_cols].idxmax(axis=1).str[-1]
    df_text['Pred'] = df_text[prob_cols].idxmax(axis=1).str[-1]
    
    print("Merging dataframes...")
    # NOTE: Ensure 'Category' and 'Mis_Knowledge_Key' match in your CSVs.
    # If logos only have 'Category' (time_error, etc.) it should work fine.
    df_merged = pd.merge(
        df_vis, 
        df_text, 
        on=["ID", "Category", "Mis_Knowledge_Key"], 
        suffixes=("_vis", "_text")
    )
    
    # Categorize: Vision Correct AND Text Wrong
    condition_mask = (df_merged['Pred_vis'] == df_merged['Ground_Truth_vis']) & \
                     (df_merged['Pred_text'] != df_merged['Ground_Truth_vis'])
    
    df_merged['Group'] = np.where(condition_mask, 'VisCorrect_TextWrong', 'Others')
    
    # Deduplicate
    analysis_df = df_merged.drop_duplicates(subset=['ID', 'Category', 'Mis_Knowledge_Key']).copy()
    
    n_target = len(analysis_df[analysis_df['Group'] == 'VisCorrect_TextWrong'])
    n_other = len(analysis_df[analysis_df['Group'] == 'Others'])
    
    print(f"Total Unique Instances: {len(analysis_df)}")
    print(f" - Group 'VisCorrect_TextWrong': {n_target}")
    print(f" - Group 'Others': {n_other}")

    transformer_layers = model.model.language_model.layers
    num_layers = len(transformer_layers)
    
    results = []

    # Get period token ID for checking
    period_token_id = processor.tokenizer.encode(".", add_special_tokens=False)[0]

    for idx, row in tqdm(analysis_df.iterrows(), total=len(analysis_df)):
        item_id = row['ID']
        item = data_lookup.get(item_id)
        if not item: continue
        
        instance_name = row['Instance_vis']
        group_label = row['Group']
        cat_label = row['Category']

        # Load Image
        pil_image = None
        if item.get('image_path'):
            # Use the passed image_base_dir
            full_path = os.path.join(image_base_dir, item['image_path'][0])
            if os.path.exists(full_path):
                pil_image = Image.open(full_path)
                if max(pil_image.size) > 1024: pil_image.thumbnail((1024, 1024))
        
        # 1. Prepare Entity Name for Text Input
        # Recreate the exact string used in RAG experiment for logos
        # Assuming RAG used: f"The logo of the company known as {instance_name}"
        entity_text_formatted = f"The logo of the company known as {instance_name}"

        # 2. Get Inputs
        inputs_vis, _ = get_model_inputs(row['Context_vis'], row['Query_vis'], image=pil_image)
        inputs_text, _ = get_model_inputs(row['Context_vis'], row['Query_vis'], entity_name=entity_text_formatted)

        # 3. Determine Indices
        vis_ids = inputs_vis.input_ids[0].tolist()
        text_ids = inputs_text.input_ids[0].tolist()
        
        alignment_map = get_right_aligned_map(vis_ids, text_ids)
        if not alignment_map: continue
        
        # Anchor is the first token of "\nQuery".
        v_anchor, t_anchor = alignment_map[-1] 
        
        v_target_idx = v_anchor - 1 # Last token of Image (or placeholder)

        # Check for Period before anchor to decide offset
        # We look at the token immediately preceding the anchor in the text sequence
        prev_token = text_ids[t_anchor - 1]
        
        if prev_token == period_token_id:
            # Format: "Entity Name.\nQuery" -> Token is at anchor - 2
            t_target_idx = t_anchor - 2
        else:
            # Format: "Entity Name\nQuery" -> Token is at anchor - 1
            t_target_idx = t_anchor - 1

        # 4. Capture Hooks
        vis_vectors = {}
        text_vectors = {}

        def get_capture_hook(storage_dict, layer_i, target_idx):
            def fn(module, input, output):
                # shape: [batch, seq, hidden]
                vec = output[0][0, target_idx, :].detach().float().cpu() 
                storage_dict[layer_i] = vec
            return fn

        # --- Run Vision ---
        hooks = []
        for i, layer in enumerate(transformer_layers):
            hooks.append(layer.register_forward_hook(get_capture_hook(vis_vectors, i, v_target_idx)))
        
        with torch.inference_mode():
            model(**inputs_vis)
        for h in hooks: h.remove()
        
        # --- Run Text ---
        hooks = []
        for i, layer in enumerate(transformer_layers):
            hooks.append(layer.register_forward_hook(get_capture_hook(text_vectors, i, t_target_idx)))
            
        with torch.inference_mode():
            model(**inputs_text)
        for h in hooks: h.remove()

        # 5. Compute Similarity
        for i in range(num_layers):
            v_vec = vis_vectors.get(i)
            t_vec = text_vectors.get(i)
            
            if v_vec is not None and t_vec is not None:
                sim = F.cosine_similarity(v_vec.unsqueeze(0), t_vec.unsqueeze(0)).item()
                
                results.append({
                    "ID": item_id,
                    "Instance": instance_name,
                    "Category": cat_label,
                    "Group": group_label,
                    "Layer": i,
                    "Cosine_Similarity": sim
                })

        # Save periodically
        if len(results) >= 2000: 
            pd.DataFrame(results).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
            results = []

    if results:
        pd.DataFrame(results).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
    print(f"Done. Similarity results saved to {out_path}")

if __name__ == "__main__":
    # --- CONFIGURATION FOR LOGOS ---
    VIS_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/RAG_VISION_Experiment_Results_logo.csv"
    TEXT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/RAG_Experiment_Results_logo.csv"
    JSON_DATA = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/logo_knowledge.json"
    OUT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/Logo_Cosine_Similarity_Grouped.csv"
    
    # Ensure this points to where your logo images actually live
    IMAGE_BASE_DIR = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data" 
    
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
        
    run_similarity_experiment(VIS_CSV, TEXT_CSV, JSON_DATA, OUT_CSV, IMAGE_BASE_DIR)