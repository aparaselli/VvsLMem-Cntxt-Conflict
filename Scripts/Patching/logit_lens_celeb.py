import os
import torch
import pandas as pd
import numpy as np
import json
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm
import torch.nn.functional as F

os.environ["HF_HOME"] = "/oscar/scratch/aparasel/hf_cache"
os.environ["HF_HUB_CACHE"] = "/oscar/scratch/aparasel/hf_cache/hub"

VIS_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/RAG_VISION_Experiment_Results.csv"
TEXT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/RAG_Experiment_Results.csv"
JSON_DATA = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/people_knowledge.json"
OUT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/Last_tok_LogitLens_Grouped.csv"

### TESTS ONLY THE ONES WHERE LANGUAGE ANSWERED WITH CONTEXT ANSWER
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_name(0).startswith(("NVIDIA A100","NVIDIA H100")) else torch.float16

print(f"Loading model: {model_name}...")
model = AutoModelForVision2Seq.from_pretrained(
    model_name, dtype=dtype, device_map="auto", low_cpu_mem_usage=True, attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(model_name)
model.eval()


answer_map = {
    "A": [" A", "A"], "B": [" B", "B"], 
    "C": [" C", "C"], "D": [" D", "D"]
}
target_token_ids = {}
for letter, variants in answer_map.items():
    ids = []
    for var in variants:
        encoded = processor.tokenizer.encode(var, add_special_tokens=False)
        if encoded: ids.append(encoded[-1])
    target_token_ids[letter] = list(set(ids))

print(f"Target Token IDs: {target_token_ids}")

def get_model_inputs(retrieved_context, user_query, image=None):
    content_list = []
    if retrieved_context is None:
        content_list.append({"type": "text", "text": "Given your knowledge, answer the multiple choice question about the following entity.\nEntity: "})
    else:
        content_list.append({"type": "text", "text": f"Context information is below.\n---------------------\n{retrieved_context}\n---------------------\nGiven the context information and your knowledge, answer the multiple choice question about the following entity.\nEntity: "})

    if image is not None:
        content_list.append({"type": "image", "image": image})
        content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})

    messages = [{"role": "user", "content": content_list}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_text += " Between A, B, C, and D, the answer is" 

    active_images = [image] if image else []
    if not active_images:
        return processor(text=[prompt_text], return_tensors="pt", padding=False).to(model.device)
    
    return processor(text=[prompt_text], images=active_images, return_tensors="pt", padding=False).to(model.device)

def get_token_prob(probs_tensor, letter):
    ids = target_token_ids.get(letter, [])
    if not ids: return 0.0
    return probs_tensor[ids].sum().item()


def run_logit_lens_fixed():
    print("Loading Data...")
    df_vis = pd.read_csv(VIS_CSV)
    df_text = pd.read_csv(TEXT_CSV)
    
    with open(JSON_DATA, 'r') as f:
        data_lookup = {item['ID']: item for item in json.load(f)}


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


    transformer_layers = model.model.language_model.layers
    final_norm = model.model.language_model.norm
    lm_head = model.lm_head
    
    results = []

    # Prepare Hook Function (Debugged w/ gemini)
    layer_data = {}
    def get_lens_hook(layer_idx, tid_gt, tid_cont):
        def fn(module, input, output):
            hidden_state = output[0][0, -1, :] 
            with torch.inference_mode():
                normalized_hidden = final_norm(hidden_state) 
                logits = lm_head(normalized_hidden)
                probs = F.softmax(logits, dim=-1)
                
                layer_data[layer_idx] = {
                    "P_GT": probs[tid_gt].item() if tid_gt is not None else 0.0,
                    "P_Cont": probs[tid_cont].item() if tid_cont is not None else 0.0
                }
        return fn

    if os.path.exists(OUT_CSV): os.remove(OUT_CSV)

    for idx, row in tqdm(analysis_df.iterrows(), total=len(analysis_df)):
        item = data_lookup.get(row['ID'])
        if not item: continue
        
        # Load Image
        pil_image = None
        if item.get('image_path'):
            full_path = os.path.join("/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data", item['image_path'][0])
            if os.path.exists(full_path):
                pil_image = Image.open(full_path)
                if max(pil_image.size) > 1024: pil_image.thumbnail((1024, 1024))
        
        inputs = get_model_inputs(row['Context_vis'], row['Query_vis'], image=pil_image)

        #Get first token of answer
        gt_text = row['Actual_GT']
        cont_text = row['Actual_Param']
        gt_ids = processor.tokenizer.encode(" " + gt_text, add_special_tokens=False)
        cont_ids = processor.tokenizer.encode(" " + cont_text, add_special_tokens=False) 
        target_gt_id = gt_ids[0] if gt_ids else None
        target_cont_id = cont_ids[0] if cont_ids else None
        # Register Hooks
        layer_data = {}
        hooks = []
        for i, layer in enumerate(transformer_layers):
            # We pass specific answers to the hook to save only what we need
            hooks.append(layer.register_forward_hook(
                get_lens_hook(i, target_gt_id, target_cont_id)
            ))
        #print(row)
        # Run Model
        with torch.inference_mode():
            model(**inputs)
        for h in hooks: h.remove()
        # Save Results
        for i in range(len(transformer_layers)):
            d = layer_data.get(i, {"P_GT": 0, "P_Cont": 0})
            results.append({
                "ID": row['ID'],
                "Entity": row['Instance_vis'],
                "Category": row['Category'],
                "Group": row['Group'],
                "Layer": i,
                "P_Parametric": d["P_GT"],
                "P_Context": d["P_Cont"],
                "P_Diff": d["P_GT"] - d["P_Cont"]
            })
            
        # Periodic Save
        if len(results) >= 5000:
            pd.DataFrame(results).to_csv(OUT_CSV, mode='a', header=not os.path.exists(OUT_CSV), index=False)
            results = []

    if results:
        pd.DataFrame(results).to_csv(OUT_CSV, mode='a', header=not os.path.exists(OUT_CSV), index=False)
    print("Done.")

if __name__ == "__main__":
    run_logit_lens_fixed()