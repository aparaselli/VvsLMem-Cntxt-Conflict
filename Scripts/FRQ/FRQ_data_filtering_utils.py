import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def keep_only_correct(df_vis_corr, df_lang_corr, ctxt_mem_df): 
    df_vis_correct = df_vis_corr[df_vis_corr["Pred"] == "CORRECT"]
    df_lang_correct = df_lang_corr[df_lang_corr["Pred"] == "CORRECT"]
    merge_cols = ["ID", "Instance", "Category", "Query"]

    df_both_correct = pd.merge(
        df_vis_correct,
        df_lang_correct,
        on=merge_cols,
        how="inner",
        suffixes=("_vis", "_lang")
    )
    print(df_both_correct['Category'].value_counts())
    keys = ["Instance", "Category"]

    ctxt_mem_df_filtered = ctxt_mem_df.merge(
        df_both_correct[keys].drop_duplicates(),
        on=keys,
        how="inner"
    )
    print(ctxt_mem_df_filtered['Category'].value_counts())
    return ctxt_mem_df_filtered


def filter_df_for_analysis(df_vis,df_txt,df_vis_corr,df_txt_corr):
    df_vis = keep_only_correct(df_vis_corr, df_txt_corr, df_vis)
    df_text = keep_only_correct(df_vis_corr, df_txt_corr, df_txt)
    df_merged = pd.merge(df_vis, df_text, on=["ID", "Category", "Mis_Knowledge_Key"], suffixes=("_vis", "_text"))

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
    return analysis_df