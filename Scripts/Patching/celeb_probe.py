import os
import torch
import pandas as pd
import numpy as np
import json
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. SETUP ENVIRONMENT ---
os.environ["HF_HOME"] = "/oscar/scratch/aparasel/hf_cache"
os.environ["HF_HUB_CACHE"] = "/oscar/scratch/aparasel/hf_cache/hub"

# --- 2. CONFIG ---
VIS_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/RAG_VISION_Experiment_Results.csv"
TEXT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/RAG_Experiment_Results.csv"
JSON_DATA = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/people_knowledge.json"

# Output for detailed per-example predictions
OUT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/Entity_Probing_Results.csv"

# --- 3. LOAD MODEL ---
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_name(0).startswith(("NVIDIA A100","NVIDIA H100")) else torch.float16

print(f"Loading model: {model_name}...")
model = AutoModelForVision2Seq.from_pretrained(
    model_name, dtype=dtype, device_map="auto", low_cpu_mem_usage=True, attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(model_name)
model.eval()

# --- 4. DATA PREP ---
label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
reverse_label_map = {0: "A", 1: "B", 2: "C", 3: "D"}

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

# --- 5. EXECUTION PHASE 1: EXTRACT VECTORS ---
def extract_hidden_states():
    print("Loading Data for Extraction...")
    df_vis = pd.read_csv(VIS_CSV)
    df_text = pd.read_csv(TEXT_CSV)
    
    with open(JSON_DATA, 'r') as f:
        data_lookup = {item['ID']: item for item in json.load(f)}

    # Predictions & Grouping
    prob_cols = ['Prob_A', 'Prob_B', 'Prob_C', 'Prob_D']
    df_vis['Pred'] = df_vis[prob_cols].idxmax(axis=1).str[-1]
    df_text['Pred'] = df_text[prob_cols].idxmax(axis=1).str[-1]
    
    df_merged = pd.merge(df_vis, df_text, on=["ID", "Category", "Mis_Knowledge_Key"], suffixes=("_vis", "_text"))
    condition_mask = (df_merged['Pred_vis'] == df_merged['Ground_Truth_vis']) & \
                     (df_merged['Pred_text'] != df_merged['Ground_Truth_vis'])
    df_merged['Group'] = np.where(condition_mask, 'VisCorrect_TextWrong', 'Others')
    
    # We keep more metadata now
    analysis_df = df_merged.drop_duplicates(subset=['ID', 'Category', 'Mis_Knowledge_Key']).copy()
    print(f"Extracting vectors for {len(analysis_df)} samples...")

    # Storage
    transformer_layers = model.model.language_model.layers
    num_layers = len(transformer_layers)
    
    X_storage = {i: [] for i in range(num_layers)}
    
    # Lists to store aligned metadata
    meta_ids = []
    meta_categories = []
    meta_groups = []
    meta_labels = []

    # Hook to capture hidden state
    current_vectors = {}
    def get_capture_hook(layer_idx):
        def fn(module, input, output):
            # Capture last token hidden state
            hidden = output[0][0, -1, :].detach().float().cpu().numpy()
            current_vectors[layer_idx] = hidden
        return fn

    # Extraction Loop
    for idx, row in tqdm(analysis_df.iterrows(), total=len(analysis_df)):
        item = data_lookup.get(row['ID'])
        if not item: continue
        
        gt_label = label_map.get(row['Ground_Truth_vis'])
        if gt_label is None: continue

        # Load Image
        pil_image = None
        if item.get('image_path'):
            full_path = os.path.join("/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data", item['image_path'][0])
            if os.path.exists(full_path):
                pil_image = Image.open(full_path)
                if max(pil_image.size) > 1024: pil_image.thumbnail((1024, 1024))

        inputs = get_model_inputs(row['Context_vis'], row['Query_vis'], image=pil_image)

        # Register Hooks
        hooks = []
        for i, layer in enumerate(transformer_layers):
            hooks.append(layer.register_forward_hook(get_capture_hook(i)))
        
        # Forward Pass
        with torch.inference_mode():
            model(**inputs)
        
        # Cleanup Hooks
        for h in hooks: h.remove()

        # Store Data
        for i in range(num_layers):
            X_storage[i].append(current_vectors[i])
        
        # Store Metadata aligned with X
        meta_ids.append(row['ID'])
        meta_categories.append(row['Category'])
        meta_groups.append(row['Group'])
        meta_labels.append(gt_label)

    return X_storage, np.array(meta_labels), np.array(meta_ids), np.array(meta_categories), np.array(meta_groups)

# --- 6. EXECUTION PHASE 2: TRAIN & PREDICT ---
def train_and_predict_detailed(X_storage, y, ids, categories, groups):
    print("\nTraining Probes & Generating Detailed Predictions...")
    
    # Split Indices (Stratified)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    y_train = y[train_idx]
    
    # We only care about saving detailed predictions for the TEST SET
    # (Checking if the probe generalizes to unseen data)
    test_ids = ids[test_idx]
    test_cats = categories[test_idx]
    test_groups = groups[test_idx]
    test_y = y[test_idx]
    
    detailed_results = []
    
    # Iterate Layers
    for layer_idx, vectors_list in tqdm(X_storage.items(), desc="Processing Layers"):
        X = np.array(vectors_list)
        X_train, X_test = X[train_idx], X[test_idx]
        
        # Train
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        # Predict on Test Set
        y_pred = clf.predict(X_test)
        
        # Store PER-EXAMPLE results
        for i in range(len(test_idx)):
            is_correct = (y_pred[i] == test_y[i])
            
            detailed_results.append({
                "ID": test_ids[i],
                "Category": test_cats[i],
                "Group": test_groups[i],
                "Layer": layer_idx,
                "Ground_Truth": reverse_label_map[test_y[i]],
                "Probe_Prediction": reverse_label_map[y_pred[i]],
                "Is_Probe_Correct": int(is_correct)
            })
        
    # Save Results
    res_df = pd.DataFrame(detailed_results)
    res_df.to_csv(OUT_CSV, index=False)
    print(f"Detailed Probing Results saved to {OUT_CSV}")
    print(f"Total Rows: {len(res_df)}")

# --- MAIN ---
if __name__ == "__main__":
    if os.path.exists(OUT_CSV): os.remove(OUT_CSV)
    
    # 1. Extract
    X_data, y_data, id_data, cat_data, group_data = extract_hidden_states()
    
    # 2. Train & Save Detailed Report
    train_and_predict_detailed(X_data, y_data, id_data, cat_data, group_data)