import os
# VISION TAKES MAX TOKEN, TEXT TAKES MEAN TOKEN
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

def run_vision_logit_lens(analysis_df, data_lookup, out_path):
    transformer_layers = model.model.language_model.layers
    final_norm = model.model.language_model.norm
    lm_head = model.lm_head
    
    results = []
    layer_data = {}

    def get_vision_lens_hook(layer_idx, vis_start, vis_end, tid_gt, tid_cont, tid_entity):
        def fn(module, input, output):
            vis_hidden = output[0][0, vis_start:vis_end, :] 
            
            with torch.inference_mode():
                normalized = final_norm(vis_hidden)
                logits = lm_head(normalized) # (num_vis_tokens, vocab_size)
                
                # --- CHANGE HERE: Use max() on logits instead of mean() on probs ---
                max_l_gt = logits[:, tid_gt].max().item() if tid_gt is not None else -float('inf')
                max_l_cont = logits[:, tid_cont].max().item() if tid_cont is not None else -float('inf')
                max_l_entity = logits[:, tid_entity].max().item() if tid_entity is not None else -float('inf')
                
                layer_data[layer_idx] = {
                    "Avg_P_GT": max_l_gt,
                    "Avg_P_Cont": max_l_cont,
                    "Avg_P_Entity": max_l_entity
                }
        return fn

    for idx, row in tqdm(analysis_df.iterrows(), total=len(analysis_df)):
        item = data_lookup.get(row['ID'])
        if not item or not item.get('image_path'): continue
        instance_name = row['Instance_vis']
        group_label = row['Group']
        # Load Image and Get Inputs
        full_path = os.path.join("/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data", item['image_path'][0])
        pil_image = Image.open(full_path).convert("RGB")
        if max(pil_image.size) > 1024: pil_image.thumbnail((1024, 1024))
        
        inputs = get_model_inputs(row['Context_vis'], row['Query_vis'], image=pil_image)
        input_ids = inputs.input_ids[0].tolist()

        # Identify vision token span 
        # In Qwen2-VL, vision tokens are typically represented by specific placeholder IDs
        # Default behavior: vision tokens are at the beginning of the sequence
        # You can find the span by checking where the first text token begins
        # (Usually index 0 to the start of "Context information" or "Entity:")
        
        # Using a simple anchor for the first text token
        prompt_anchor = processor.tokenizer("Context", add_special_tokens=False).input_ids[0]
        try:
            vis_start = 0
            vis_end = input_ids.index(prompt_anchor)
        except ValueError:
            # Fallback if text format differs
            vis_end = 128 # Estimate for Qwen2-VL default vision factor

        # Tokenize target answers
        gt_ids = processor.tokenizer.encode(" " + row['Actual_GT'], add_special_tokens=False)
        cont_ids = processor.tokenizer.encode(" " + row['Actual_Param'], add_special_tokens=False) 
        entity_ids = processor.tokenizer.encode(" " + row['Instance_vis'], add_special_tokens=False) 
        target_gt_id = gt_ids[0] if gt_ids else None
        target_cont_id = cont_ids[0] if cont_ids else None
        target_entity_id = entity_ids[0] if entity_ids else None

        # Hooks and Run
        layer_data = {}
        hooks = []
        for i, layer in enumerate(transformer_layers):
            hooks.append(layer.register_forward_hook(
                get_vision_lens_hook(i, vis_start, vis_end, target_gt_id, target_cont_id, target_entity_id)
            ))

        with torch.inference_mode():
            model(**inputs)
        for h in hooks: h.remove()

        # Save Results
        for i in range(len(transformer_layers)):
            d = layer_data.get(i, {"Avg_P_GT": 0, "Avg_P_Cont": 0, "Avg_P_Entity": 0})
            results.append({
                "ID": row['ID'],
                "Instance": instance_name,
                "Category": row['Category'],
                "Layer": i,
                "Avg_Vision_P_Parametric": d["Avg_P_GT"],
                "Avg_Vision_P_Context": d["Avg_P_Cont"],
                "Avg_Vision_P_Entity": d["Avg_P_Entity"],
                "Parametric_ans": row['Actual_GT'],
                "Context_ans": row['Actual_Param'],
                "Group": row['Group']
            })

    pd.DataFrame(results).to_csv(out_path, index=False)

def run_text_entity_logit_lens(analysis_df, data_lookup, out_path):
    transformer_layers = model.model.language_model.layers
    final_norm = model.model.language_model.norm
    lm_head = model.lm_head
    skips = {"no_item": 0, "no_anchor": 0, "no_entity_span": 0, "ran": 0}

    anchor_variants = ["Entity:", "Entity: ", "\nEntity:", "\nEntity: "]
    anchor_ids_list = [processor.tokenizer.encode(a, add_special_tokens=False) for a in anchor_variants]

    results = []
    
    # 1. Define the anchor token IDs for "Entity:"
    # We include a leading space variant just in case the template formatting shifts
    anchor_text = "Entity:"
    anchor_ids = processor.tokenizer.encode(anchor_text, add_special_tokens=False)

    def get_text_lens_hook(layer_idx, text_start, text_end, tid_gt, tid_cont, tid_entity, storage_dict):
        def fn(module, input, output):
            text_hidden = output[0][0, text_start:text_end, :] 
            
            with torch.inference_mode():
                normalized = final_norm(text_hidden)
                logits = lm_head(normalized) 
                
                # --- CHANGE HERE: Max Logit logic ---
                max_l_gt = logits[:, tid_gt].max().item() if tid_gt is not None else -float('inf')
                max_l_cont = logits[:, tid_cont].max().item() if tid_cont is not None else -float('inf')
                max_l_entity = logits[:, tid_entity].max().item() if tid_entity is not None else -float('inf')
                
                storage_dict[layer_idx] = {
                    "Avg_Text_P_Parametric": max_l_gt,
                    "Avg_Text_P_Context": max_l_cont,
                    "Avg_Text_P_Entity": max_l_entity
                }
        return fn

    for idx, row in tqdm(analysis_df.iterrows(), total=len(analysis_df)):
        item = data_lookup.get(row['ID'])
        if not item: continue
        
        instance_name = row['Instance_vis']
        
        # Generate inputs (Text-only mode for this experiment)
        inputs = get_model_inputs(row['Context_vis'], row['Query_vis'], entity_name=instance_name, image=None) 
        input_ids = inputs.input_ids[0].tolist()

        # --- ANCHORED SEARCH LOGIC ---
        # Find where "Entity:" appears in the prompt
        def find_subseq(haystack, needle, start=0, end=None):
            if end is None:
                end = len(haystack)
            n = len(needle)
            for i in range(start, end - n + 1):
                if haystack[i:i+n] == needle:
                    return i
            return -1

        # --- find anchor (any variant) ---
        anchor_pos = -1
        anchor_len = None
        for anchor_ids in anchor_ids_list:
            pos = find_subseq(input_ids, anchor_ids, start=0)
            if pos != -1:
                anchor_pos = pos
                anchor_len = len(anchor_ids)
                break

        if anchor_pos == -1:
            skips["no_anchor"] += 1
            continue

        search_start = anchor_pos + anchor_len

        # --- find entity span (try variants) ---
        ent_variants = [
            processor.tokenizer.encode(instance_name, add_special_tokens=False),
            processor.tokenizer.encode(" " + instance_name, add_special_tokens=False),
            processor.tokenizer.encode(instance_name + ".", add_special_tokens=False),
            processor.tokenizer.encode(" " + instance_name + ".", add_special_tokens=False),
        ]

        text_start = -1
        best_len = None

        # search within a generous window after anchor (or just all remaining tokens)
        window_end = min(len(input_ids), search_start + 200)

        for ent_ids in ent_variants:
            if not ent_ids:
                continue
            pos = find_subseq(input_ids, ent_ids, start=search_start, end=window_end)
            if pos != -1:
                text_start = pos
                best_len = len(ent_ids)
                break

        # fallback: if not found near anchor, try global search after anchor
        if text_start == -1:
            for ent_ids in ent_variants:
                if not ent_ids:
                    continue
                pos = find_subseq(input_ids, ent_ids, start=search_start)
                if pos != -1:
                    text_start = pos
                    best_len = len(ent_ids)
                    break

        if text_start == -1:
            skips["no_entity_span"] += 1
            continue

        text_end = text_start + best_len


        # Tokenize target answers (with leading space)
        gt_ids = processor.tokenizer.encode(" " + row['Actual_GT'], add_special_tokens=False)
        cont_ids = processor.tokenizer.encode(" " + row['Actual_Param'], add_special_tokens=False) 
        entity_ids = processor.tokenizer.encode(" " + row['Instance_vis'], add_special_tokens=False)
        target_gt_id = gt_ids[0] if gt_ids else None
        target_cont_id = cont_ids[0] if cont_ids else None
        target_entity_id = entity_ids[0] if entity_ids else None

        # Hooks and Inference
        current_layer_data = {}
        hooks = []
        for i, layer in enumerate(transformer_layers):
            hooks.append(layer.register_forward_hook(
                get_text_lens_hook(i, text_start, text_end, target_gt_id, target_cont_id, target_entity_id, current_layer_data)
            ))
        skips["ran"] += 1

        with torch.inference_mode():
            model(**inputs)
        for h in hooks: h.remove()

        # Collect Results
        for i in range(len(transformer_layers)):
            d = current_layer_data.get(i, {"Avg_Text_P_Parametric": 0, "Avg_Text_P_Context": 0, "Avg_Text_P_Entity": 0})
            results.append({
                "ID": row['ID'],
                "Instance": instance_name,
                "Layer": i,
                "Avg_Text_P_Parametric": d["Avg_Text_P_Parametric"],
                "Avg_Text_P_Context": d["Avg_Text_P_Context"],
                "Avg_Text_P_Entity": d["Avg_Text_P_Entity"],
                "Group": row['Group'],
                "Category": row['Category']
            })
    print("SKIPS:", skips)
    print("num results rows:", len(results))

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Text Entity Logit Lens results saved to {out_path}")


def run_similarity_experiment(vis_csv, text_csv, json_data_path, out_path,vision_exp=True):
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

    if vision_exp:
        # 1. Others: Vision and Text modalities agree on the output (Vision is context, text is context)
        same_pred_mask = (df_merged['Pred_vis'] == df_merged['Pred_text']) & (df_merged['Pred_vis'] == df_merged['Mis_Answer_Label_vis'])

        # 2. Target: Vision-Language Conflict (Vision is parametric, Text is context)
        vis_corr_text_wrong_mask = (df_merged['Pred_vis'] == df_merged['Ground_Truth_vis']) & \
                                (df_merged['Pred_text'] != df_merged['Ground_Truth_vis'])

        # 3. Apply grouping
        df_merged['Group'] = 'Exclude'
        df_merged.loc[same_pred_mask, 'Group'] = 'Others'
        df_merged.loc[vis_corr_text_wrong_mask, 'Group'] = 'VisCorrect_TextWrong'
        g1 = 'VisCorrect_TextWrong'
        g2 = 'Others'
    else:
        # Initialize as Exclude
        df_merged['Group'] = 'Exclude'

        # Parametric: Model follows internal knowledge despite context
        parametric_mask = (df_merged['Pred_text'] == df_merged['Ground_Truth_vis'])
        df_merged.loc[parametric_mask, 'Group'] = 'parametric'

        # Context: Model follows the provided (mis-informative) context
        context_mask = (df_merged['Pred_text'] == df_merged['Mis_Answer_Label_vis'])
        df_merged.loc[context_mask, 'Group'] = 'context'
        g1= 'parametric'
        g2 = 'context'


    # 4. Final filter for analysis
    analysis_df = df_merged[df_merged['Group'] != 'Exclude'].copy()
    analysis_df = analysis_df.drop_duplicates(subset=['ID', 'Category', 'Mis_Knowledge_Key'])

    print(f"Processing {len(analysis_df)} items...")
    
    # Count stats
    n_target = len(analysis_df[analysis_df['Group'] == g1])
    n_other = len(analysis_df[analysis_df['Group'] == g2])
    
    print(f"Total Unique Instances: {len(analysis_df)}")
    print(f" - Group '{g1}': {n_target}")
    print(f" - Group '{g2}': {n_other}")
    transformer_layers = model.model.language_model.layers
    num_layers = len(transformer_layers)
    if vision_exp:
        run_vision_logit_lens(analysis_df, data_lookup, out_path)
    else:
        run_text_entity_logit_lens(analysis_df, data_lookup, out_path)

    
if __name__ == "__main__":
    VIS_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/RAG_VISION_Experiment_Results.csv"
    TEXT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/RAG_Experiment_Results.csv"
    JSON_DATA = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/people_knowledge.json"
    VISION_EXPERIMENT = False
    if VISION_EXPERIMENT:
        OUT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/Entity_Logit_Lens_Grouped.csv"
    else:
        OUT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/text-Entity_Logit_Lens_Grouped.csv"
    
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
        
    run_similarity_experiment(VIS_CSV, TEXT_CSV, JSON_DATA, OUT_CSV,VISION_EXPERIMENT)