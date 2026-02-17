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


def get_image_token_span_indices(input_ids, tokenizer):
    candidate_tokens = ["<|image_pad|>", "<image_pad>", "<|vision_pad|>"]

    image_token_id = None
    for tok in candidate_tokens:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                image_token_id = tid
                break
        except Exception:
            pass

    if image_token_id is None:
        raise ValueError(
            "Could not find an image placeholder token id. "
            "Inspect processor.tokenizer.special_tokens_map / added_tokens_encoder."
        )

    # input_ids is a python list (1D)
    img_positions = [i for i, tid in enumerate(input_ids) if tid == image_token_id]
    return img_positions


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

def run_similarity_experiment(vis_csv, text_csv, json_data_path, out_path):
    print("Loading Data...")
    df_vis = pd.read_csv(vis_csv)
    df_text = pd.read_csv(text_csv)
    
    with open(json_data_path, 'r') as f:
        raw_data = json.load(f)

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
    
    data_lookup = {item['ID']: item for item in raw_data}
    prob_cols = ['Prob_A', 'Prob_B', 'Prob_C', 'Prob_D']
    # Identify Predictions
    
    # Merge first to keep everything in one dataframe
    df_merged = pd.merge(df_vis, df_text, on=["ID", "Category", "Mis_Knowledge_Key"], suffixes=("_vis", "_text"))

    #Include the "true" answers
    #print(df_merged)
    def actual_ans(row):
        query_list = row['Query_vis'].split(" ")
        ans_GT = query_list[query_list.index(row['Ground_Truth_vis']+")") + 1]
        ans_Param = query_list[query_list.index(row['Mis_Answer_Label_vis']+")") + 1]
        return ans_GT, ans_Param
    gt_list = []
    param_list = []
    for i in range(df_merged.shape[0]):
        gt, param = actual_ans(df_merged.iloc[i])
        gt_list.append(gt)
        param_list.append(param)
    df_merged['Actual_GT'] = gt_list
    df_merged['Actual_Param'] = param_list
    ##########
    # Extract the predicted letter (A, B, C, or D) from the probability columns
    # We use the _vis and _text suffixes created by the merge
    prob_cols_vis = ['Prob_A_vis', 'Prob_B_vis', 'Prob_C_vis', 'Prob_D_vis']
    prob_cols_text = ['Prob_A_text', 'Prob_B_text', 'Prob_C_text', 'Prob_D_text']
    
    df_merged['Pred_vis'] = df_merged[prob_cols_vis].idxmax(axis=1).str.extract(r'Prob_([A-D])')
    df_merged['Pred_text'] = df_merged[prob_cols_text].idxmax(axis=1).str.extract(r'Prob_([A-D])')

    # 1. Others: Vision and Text modalities agree on the output (Vision is context, text is context)
    same_pred_mask = (df_merged['Pred_vis'] == df_merged['Pred_text']) & (df_merged['Pred_vis'] == df_merged['Mis_Answer_Label_vis'])

    # 2. Target: Vision-Language Conflict (Vision is parametric, Text is context)
    vis_corr_text_wrong_mask = (df_merged['Pred_vis'] == df_merged['Ground_Truth_vis']) & \
                               (df_merged['Pred_text'] != df_merged['Ground_Truth_vis'])

    # 3. Apply grouping
    df_merged['Group'] = 'Exclude'
    df_merged.loc[same_pred_mask, 'Group'] = 'Others'
    df_merged.loc[vis_corr_text_wrong_mask, 'Group'] = 'VisCorrect_TextWrong'

    # 4. Final filter for analysis
    analysis_df = df_merged[df_merged['Group'] != 'Exclude'].copy()
    analysis_df = analysis_df.drop_duplicates(subset=['ID', 'Category', 'Mis_Knowledge_Key'])

    print(f"Processing {len(analysis_df)} items...")
    
    # Count stats
    n_target = len(analysis_df[analysis_df['Group'] == 'VisCorrect_TextWrong'])
    n_other = len(analysis_df[analysis_df['Group'] == 'Others'])
    
    print(f"Total Unique Instances: {len(analysis_df)}")
    print(f" - Group 'VisCorrect_TextWrong': {n_target}")
    print(f" - Group 'Others': {n_other}")
    transformer_layers = model.model.language_model.layers
    num_layers = len(transformer_layers)
    
    results = []

    for idx, row in tqdm(analysis_df.iterrows(), total=len(analysis_df)):
        item_id = row['ID']
        item = data_lookup.get(item_id)
        if not item: continue
        instance_name = row['Instance_vis']
        group_label = row['Group'] # Get the category
        # Load Image
        pil_image = None
        if item.get('image_path'):
            full_path = os.path.join("/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data", item['image_path'][0])
            if os.path.exists(full_path):
                pil_image = Image.open(full_path)
                if max(pil_image.size) > 1024: pil_image.thumbnail((1024, 1024))
        
        inputs_vis = get_model_inputs(row['Context_vis'], row['Query_vis'], image=pil_image)
        inputs_text = get_model_inputs(row['Context_vis'], row['Query_vis'], entity_name=instance_name)
        vis_ids = inputs_vis.input_ids[0].tolist()
        text_ids = inputs_text.input_ids[0].tolist()
        alignment_map = get_right_aligned_map(vis_ids, text_ids)
        if not alignment_map: continue
        
        # Determine "Last Entity Token"
        # Anchor is the first token of "\nQuery".
        v_anchor, t_anchor = alignment_map[-1] 
        v_target_idx = v_anchor - 1 
        t_target_idx = t_anchor - 2 

        vis_vectors = {}
        text_vectors = {}
        def get_capture_hook(storage_dict, layer_i, start_idx, end_idx):
            def fn(module, input, output):
                # output[0] shape: (batch, seq_len, hidden_dim)
                # Slice the span [start:end] and mean across the sequence dimension
                span_vecs = output[0][0, start_idx:end_idx, :].detach().float().cpu()
                storage_dict[layer_i] = span_vecs.mean(dim=0) 
            return fn

        # --- REPLACED LOGIC START ---
        # 1. Vision Span: Typically from start until the alignment anchor (the query)
        # v_anchor is the start of "\nQuery"
        img_pos = get_image_token_span_indices(vis_ids, processor.tokenizer)
        if len(img_pos) == 0:
            continue 

        v_start_idx = img_pos[0]
        v_end_idx   = img_pos[-1] + 1 
        if idx == 0:#Sanity check
            print("Found image tokens:", len(img_pos))
            print("Example tokens:", processor.tokenizer.convert_ids_to_tokens(vis_ids[img_pos[0]:img_pos[0]+5]))


        # 2. Text Span: Find where the entity name starts
        # The prompt is: "...Entity: [Name].\nQuery..."
        # We know the text ends at t_anchor. 
        # We can find the start by subtracting the length of the entity name tokens.
        # A simple way is to find the index of "Entity: " and add its length.
        prompt_ids = processor.tokenizer("Entity: ", add_special_tokens=False).input_ids
        # Find where prompt_ids occurs in text_ids to get the exact start
        t_start_idx = 0
        for i in range(len(text_ids) - len(prompt_ids)):
            if text_ids[i:i+len(prompt_ids)] == prompt_ids:
                t_start_idx = i + len(prompt_ids)
                break
        t_end_idx = t_anchor - 1 # End before the "." or "\nQuery"
        
        # Capture Vision Vectors
        hooks = []
        for i, layer in enumerate(transformer_layers):
            hooks.append(layer.register_forward_hook(get_capture_hook(vis_vectors, i, v_start_idx, v_end_idx)))
        with torch.inference_mode():
            model(**inputs_vis)
        for h in hooks: h.remove()
        b_idx = 16
        baseline_token_id = text_ids[b_idx]
        baseline_word = processor.tokenizer.decode([baseline_token_id]).strip()
        if idx == 0:
            print(f"Sanity Check: Using baseline token '{baseline_word}' at index {b_idx}")
        #random baseline vector 
        baseline_vectors = {}
        b_start_idx, b_end_idx = 16, 17

        # Capture Text Vectors
        hooks = []
        for i, layer in enumerate(transformer_layers):
            hooks.append(layer.register_forward_hook(get_capture_hook(text_vectors, i, t_start_idx, t_end_idx)))
            hooks.append(layer.register_forward_hook(get_capture_hook(baseline_vectors, i, b_start_idx, b_end_idx)))
        with torch.inference_mode():
            model(**inputs_text)
        for h in hooks: h.remove()



        for i in range(num_layers):
            v_vec = vis_vectors[i]
            t_vec = text_vectors[i] 
            b_vec = baseline_vectors[i]   
            sim = F.cosine_similarity(v_vec.unsqueeze(0), t_vec.unsqueeze(0)).item()   
            baseline_sim = F.cosine_similarity(v_vec.unsqueeze(0), b_vec.unsqueeze(0)).item()        
            results.append({
                "ID": item_id,
                "Instance": instance_name,
                "Category": row['Category'],
                "Group": group_label,  
                "Layer": i,
                "Cosine_Similarity": sim,
                "Baseline_Similarity": baseline_sim,
                "Baseline_word": baseline_word
            })
        if len(results) >= 2000: 
            pd.DataFrame(results).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
            results = []

    if results:
        pd.DataFrame(results).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
    print(f"Done. Similarity results grouped by type saved to {out_path}")

if __name__ == "__main__":
    VIS_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/RAG_VISION_Experiment_Results.csv"
    TEXT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/RAG_Experiment_Results.csv"
    JSON_DATA = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/people_knowledge.json"
    OUT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/Entity_Cosine_Similarity_Grouped.csv"
    
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
        
    run_similarity_experiment(VIS_CSV, TEXT_CSV, JSON_DATA, OUT_CSV)