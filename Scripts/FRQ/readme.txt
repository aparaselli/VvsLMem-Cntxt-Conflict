# Behavioral Experiments
To run the behavioral experiments use FRQ_ctxt_mem_exp.py
You can then score them using LLM as judge using (ctxt_mem_classifier.py for mock_RAG == True) and (fact_recall_classifier.py for mock_RAG == False)
Then you can visualize them using FRQ_Vis.ipynb
Note: You may need to manually comb through the 'neither' responses as they can sometimes be a bit iffy 

# Attribution patching
To run attributiojn patching use FRQ_attribution_patch_gemma.py (it has implemented attribution patching for gemma and qwen for both modalities)
Visualize it using the code in FRQ_attr_patch_vis_refined.ipynb

#Back patching - back patch from layers i to 0
Run backpatchign using FRQ_backpatch.py
Visualize using FRQ_backpatch.ipynb

#Forward patching experiments - forward patch from layers 0 to i
FRQ_FR_Freeze_experiment does the cumalitve layer patching and supports fact_recall and ctxt_mem
Visualizations for this are stored in Forwardpatch.ipynb

#MLP ablation - ablate MLPs from layers 0 to i
ablation experiment using FRQ_MLP_Ablation.py
Visualize using FRQ_MLP_Ablation.ipynb

#MHA masking - patch the with_context run MHA outputs into the no_context run to see where entity is binded to image entity 
run experiment using FRQ_Attn_Head_Patching.py
visualize using Attn_head_patch_vis.ipynb
Rational for patching RAG -> no_RAG: Since we were observing the MLP interaction at the entity tokens we would want to see how much of the answer is determined 
by the corresponding MHA @ the entity tokens. However if we patched teh other way around, it could be the other tokens which are bringing in the context answer, and is therefore
difficult to isolate.
