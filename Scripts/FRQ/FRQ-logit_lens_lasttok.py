import os
import torch
import pandas as pd
import numpy as np
import json
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm
import torch.nn.functional as F
import argparse
from FRQ_make_input import make_inputs

os.environ["HF_HOME"] = "/oscar/scratch/aparasel/hf_cache"
os.environ["HF_HUB_CACHE"] = "/oscar/scratch/aparasel/hf_cache/hub"

VIS_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/FRQ_RAG_VISION_Experiment_Results.csv"
TEXT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/FRQ_RAG_Experiment_Results.csv"
VIS_CSV_LOGO = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/FRQ_RAG_VISION_Experiment_Results_logo.csv"
TEXT_CSV_LOGO = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/FRQ_Results/FRQ_RAG_Experiment_Results_logo.csv"
JSON_DATA = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/people_knowledge.json"
OUT_CSV = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/FRQ-Last_tok_LogitLens.csv"
OUT_CSV_logo = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/FRQ-Last_tok_LogitLens_logo.csv"
OUT_CSV_txt = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/FRQ-Last_tok_LogitLens_txt.csv"
OUT_CSV_logo_txt = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/FRQ-Last_tok_LogitLens_logo_txt.csv"

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

# def get_model_inputs(retrieved_context, user_query, image=None, err_cat="Career_error"):
#     # Build messages exactly like rag_model_call (image-path branch)
#     if retrieved_context is None:
#         text_before_image = (
#             "Given your knowledge, answer the question about the following entity in one short phrase.\n"
#             "Entity: "
#         )
#     else:
#         text_before_image = (
#             "Context information is below.\n"
#             "---------------------\n"
#             f"{retrieved_context}\n"
#             "---------------------\n"
#             "Given the context information and your knowledge, answer the question about the following entity in one short phrase.\n"
#             "Entity: "
#         )

#     content_list = [{"type": "text", "text": text_before_image}]
#     if image is not None:
#         content_list.append({"type": "image", "image": image})
#         content_list.append({"type": "text", "text": f"\nQuery: {user_query}"})

#     messages = [{"role": "user", "content": content_list}]
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     # Use the same prefix logic (important: include trailing space)
#     prefix = "If I had to answer in one short phrase, the answer would be "
#     if err_cat == "Temporal_error":
#         prefix = "The person was born in "
#     elif err_cat == "location_error":
#         prefix = "The nationality of the person is "
#     elif err_cat == "Career_error":
#         prefix = "The occupation of the person is "
#     elif err_cat == "time_error":
#         prefix = "The company associated with the entity was established in "
#     elif err_cat == "creator_error":
#         prefix = "The person/people who founded this company is/are "
#     elif err_cat == "content_error":
#         prefix = "The main products of this brand is/are "

#     text += prefix

#     image_inputs = {}
#     if image is not None:
#         image_inputs = {"images": [image], "videos": None}

#     inputs = processor(
#         text=[text],
#         padding=True,                 # match rag_model_call
#         return_tensors="pt",
#         **image_inputs
#     ).to(model.device)

#     return inputs


def get_token_prob(probs_tensor, letter):
    ids = target_token_ids.get(letter, [])
    if not ids: return 0.0
    return probs_tensor[ids].sum().item()


def run_logit_lens_fixed(dataset="celeb", entity_modality="vision"):     
    print("Loading Data...")
    if dataset == "celeb":
        df_vis = pd.read_csv(VIS_CSV)
        df_text = pd.read_csv(TEXT_CSV)
    elif dataset == "logo":
        df_vis = pd.read_csv(VIS_CSV_LOGO)
        df_text = pd.read_csv(TEXT_CSV_LOGO)

    
    with open(JSON_DATA, 'r') as f:
        data_lookup = {item['ID']: item for item in json.load(f)}

    if dataset == "celeb":
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
    elif dataset == 'logo':
        df_inherent_text = pd.read_csv("/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/No_RAG_Experiment_Results_logo.csv")
        df_inherent_vision = pd.read_csv("/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/No_RAG_VISION_Experiment_Results_logo.csv")
        df_inherent_text

        def get_correct_indices(df):
            # 1. Define the probability columns
            prob_cols = ['Prob_A', 'Prob_B', 'Prob_C', 'Prob_D']
            
            # 2. Find the column name with the max value for each row (e.g., "Prob_A")
            # idxmax(axis=1) returns the column label of the maximum value
            predicted_col = df[prob_cols].idxmax(axis=1)
            
            # 3. Clean the prediction to match Ground_Truth format (e.g., "Prob_A" -> "A")
            predicted_choice = predicted_col.str.replace('Prob_', '')
            
            # 4. Compare with Ground Truth
            # We strip whitespace from both just to be safe (e.g. " A" vs "A")
            is_correct = predicted_choice.str.strip() == df['Ground_Truth'].str.strip()
            
            return is_correct

        # Get the boolean mask (True/False) for both dataframes
        text_correct_mask = get_correct_indices(df_inherent_text)
        vision_correct_mask = get_correct_indices(df_inherent_vision)

        # Find where BOTH are True
        # We use the bitwise AND operator '&'
        both_correct_mask = text_correct_mask & vision_correct_mask

        # Extract the actual indices
        correct_indices = df_inherent_text.index[both_correct_mask].tolist()

        print(f"Number of matches: {len(correct_indices)}")
        print("Indices:", correct_indices)

        df_vis = df_vis.drop_duplicates(subset=['Instance', 'Category','Mis_Knowledge_Key'])
        df_inherent_text_corr = df_inherent_text.iloc[correct_indices]
        corr_inst_cat = df_inherent_text_corr[['Instance', 'Category']].drop_duplicates()
        df_vis = df_vis.merge(corr_inst_cat, on=['Instance', 'Category'], how='inner')

        df_text = df_text.drop_duplicates(subset=['Instance', 'Category','Mis_Knowledge_Key'])
        df_inherent_text_corr = df_inherent_text.iloc[correct_indices]
        corr_inst_cat = df_inherent_text_corr[['Instance', 'Category']].drop_duplicates()
        df_text = df_text.merge(corr_inst_cat, on=['Instance', 'Category'], how='inner')
    # Identify Predictions
    
    # Merge first to keep everything in one dataframe
    df_merged = pd.merge(df_vis, df_text, on=["ID", "Category", "Mis_Knowledge_Key"], suffixes=("_vis", "_text"))

    #Include the "true" answers
    #print(df_merged) IN FRQ WE ALREADY SAVED PREDICTIONS AS STRINGS 
    # def actual_ans(row):
    #     query_list = row['Query_vis'].split(" ")
    #     ans_GT = query_list[query_list.index(row['Ground_Truth_vis']+")") + 1]
    #     ans_Param = query_list[query_list.index(row['Mis_Answer_Label_vis']+")") + 1]
    #     return ans_GT, ans_Param
    # gt_list = []
    # param_list = []
    # for i in range(df_merged.shape[0]):
    #     gt, param = actual_ans(df_merged.iloc[i])
    #     gt_list.append(gt)
    #     param_list.append(param)
    df_merged['Actual_GT'] = df_merged['Ground_Truth_vis']
    df_merged['Actual_Param'] = df_merged['Mis_Answer_Label_vis']
    ##########


    # Extract the predicted letter (A, B, C, or D) from the probability columns
    # We use the _vis and _text suffixes created by the merge
    #prob_cols_vis = ['Prob_A_vis', 'Prob_B_vis', 'Prob_C_vis', 'Prob_D_vis']
    #prob_cols_text = ['Prob_A_text', 'Prob_B_text', 'Prob_C_text', 'Prob_D_text']
    
    #df_merged['Pred_vis'] = df_merged[prob_cols_vis].idxmax(axis=1).str.extract(r'Prob_([A-D])')
    #df_merged['Pred_text'] = df_merged[prob_cols_text].idxmax(axis=1).str.extract(r'Prob_([A-D])')

    same_pred_mask = (df_merged['Pred_vis'] == df_merged['Pred_text']) & (df_merged['Pred_vis'] == 'mis_label')
    vis_corr_text_wrong_mask = (df_merged['Pred_vis'] == 'gt') & \
                               (df_merged['Pred_text'] == 'mis_label')
    vistxt_param = (df_merged['Pred_vis'] == df_merged['Pred_text']) & (df_merged['Pred_vis'] == 'gt')
    viscont_text_param = (df_merged['Pred_vis'] == 'mis_label') & \
                               (df_merged['Pred_text'] == 'gt')

    # 3. Apply grouping
    df_merged['Group'] = 'Exclude'
    df_merged.loc[same_pred_mask, 'Group'] = 'VisTxtCont'
    df_merged.loc[vis_corr_text_wrong_mask, 'Group'] = 'VisParam_TxtCont'
    df_merged.loc[vistxt_param, 'Group'] = 'VisTxtParam'
    df_merged.loc[viscont_text_param, 'Group'] = 'VisCont_TxtParam'

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
    def get_lens_hook(layer_idx, tid_gt, tid_cont, tid_output):
        def fn(module, input, output):
            hidden_state = output[0][0, -1, :] 
            with torch.inference_mode():
                normalized_hidden = final_norm(hidden_state) 
                logits = lm_head(normalized_hidden)
                top_id = int(torch.argmax(logits).item())
                top_tok = processor.tokenizer.decode([top_id])
                    
                layer_data[layer_idx] = {
                    "logit_GT": logits[tid_gt].item() if tid_gt is not None else 0.0,
                    "logit_Cont": logits[tid_cont].item() if tid_cont is not None else 0.0,
                    "logit_Output": logits[tid_output].item() if tid_output is not None else 0.0,
                    "top_token_id": top_id,
                    "top_token": top_tok,
                    "top_logit": logits[top_id].item(),
                }
        return fn

    if os.path.exists(OUT_CSV): os.remove(OUT_CSV)

    for idx, row in tqdm(analysis_df.iterrows(), total=len(analysis_df)):
        item = data_lookup.get(row['ID'])
        if not item: continue
        if entity_modality == "vision":
            # Load Image
            pil_image = None
            if item.get('image_path'):
                full_path = os.path.join("/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data", item['image_path'][0])
                if os.path.exists(full_path):
                    pil_image = Image.open(full_path)
                    if max(pil_image.size) > 1024: pil_image.thumbnail((1024, 1024))
            

            inputs = make_inputs(
                processor=processor,
                model_device=model.device,
                retrieved_context=row['Context_vis'],
                user_query=row['Query_vis'],
                entity_modality="vision",
                mock_RAG=True,                        
                bench_type=("people" if dataset == "celeb" else "logo"),
                image=pil_image,
                padding=True,
            )
        elif entity_modality == "text":
            inputs = make_inputs(
                processor=processor,
                model_device=model.device,
                retrieved_context=row['Context_text'],
                user_query=row['Query_text'],
                instance_name = row['Instance_text'],
                entity_modality="text",
                mock_RAG=True,                        
                bench_type=("people" if dataset == "celeb" else "logo"),
                image=None,
                padding=True,
            )
        else: 
            raise ValueError("Modality type not supported")

        #Get first token of answer
        gt_text = row['Actual_GT']
        cont_text = row['Actual_Param']
        output_text = row['Top_token_str_vis']
        gt_ids = processor.tokenizer.encode(" " + str(gt_text), add_special_tokens=False)
        cont_ids = processor.tokenizer.encode(" " + str(cont_text), add_special_tokens=False) 
        output_ids = processor.tokenizer.encode(" " + str(output_text), add_special_tokens=False) 
        target_gt_id = gt_ids[0] if gt_ids else None #Use first token of occupation name
        target_cont_id = cont_ids[0] if cont_ids else None #use first token of occupation
        target_output_id = output_ids[0] if output_ids else None
        # Register Hooks
        layer_data = {}
        hooks = []
        for i, layer in enumerate(transformer_layers):
            hooks.append(layer.register_forward_hook(
                get_lens_hook(i, target_gt_id, target_cont_id, target_output_id)
            ))
        #print(row)
        # Run Model
        with torch.inference_mode():
            model(**inputs)
        for h in hooks: h.remove()
        # Save Results
        for i in range(len(transformer_layers)):
            d = layer_data.get(i, {
                "logit_GT": 0.0, "logit_Cont": 0.0, "logit_Output": 0.0,
                "top_token_id": -1, "top_token": "", "top_logit": 0.0
            })
            results.append({
                "ID": row['ID'],
                "Entity": row['Instance_vis'],
                "Category": row['Category'],
                "Group": row['Group'],
                "Layer": i,
                "Logit_Parametric": d["logit_GT"],
                "Logit_Context": d["logit_Cont"],
                "Logit_Output": d["logit_Output"],
                "Logit_Diff": d["logit_GT"] - d["logit_Cont"],
                "TopTokenID": d["top_token_id"],
                "TopToken": d["top_token"],
                "TopLogit": d["top_logit"],
            })
            
        # Periodic Save
        if len(results) >= 5000 and dataset=='celeb':
            pd.DataFrame(results).to_csv(OUT_CSV, mode='a', header=not os.path.exists(OUT_CSV), index=False)
            results = []
    if entity_modality == "vision":
        if results:
            if dataset == 'celeb':
                pd.DataFrame(results).to_csv(OUT_CSV, mode='a', header=not os.path.exists(OUT_CSV), index=False)
            elif dataset =='logo':
                pd.DataFrame(results).to_csv(OUT_CSV_logo, mode='a', header=not os.path.exists(OUT_CSV_logo), index=False)
        print("Done.")
    if entity_modality == "text":
        if results:
            if dataset == 'celeb':
                pd.DataFrame(results).to_csv(OUT_CSV_txt, mode='a', header=not os.path.exists(OUT_CSV), index=False)
            elif dataset =='logo':
                pd.DataFrame(results).to_csv(OUT_CSV_logo_txt, mode='a', header=not os.path.exists(OUT_CSV_logo), index=False)
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FRQ Logitlens experiment with optional context.")

    parser.add_argument(
        '--dataset', 
        type=str, 
        default="celeb",
        choices=["celeb", "logo"], 
        help='Run on which dataset (celeb or logo)'
    )
    parser.add_argument(
        "--entity_modality",
        type=str,
        default="vision",
        choices=["text", "vision"],
        help="Modality for entity in query (text or vision)."
    )
    args = parser.parse_args()
    run_logit_lens_fixed(dataset=args.dataset, entity_modality=args.entity_modality)