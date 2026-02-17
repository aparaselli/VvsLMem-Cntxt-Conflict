import os
# Specific to your environment
os.environ["HF_HOME"] = "/oscar/scratch/aparasel/hf_cache"
os.environ["HF_HUB_CACHE"] = "/oscar/scratch/aparasel/hf_cache/hub"

import json
import torch
import pandas as pd
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm

################################
# Configuration & Paths
################################

# Paths based on your previous script
DATA_PATH = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/MLLMKC_data/logo_knowledge.json"
IMAGE_ROOT = "/oscar/scratch/aparasel/VvsLMem-Cntxt-Conflict/MLLMKC_data"
SAVE_PATH = "/users/aparasel/scratch/VvsLMem-Cntxt-Conflict/Scripts/Results/Logo_Identification_Check.csv"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

################################
# Model Loading
################################

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_name(0).startswith(("NVIDIA A100","NVIDIA H100")) else torch.float16

print(f"Loading model: {MODEL_NAME}")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    dtype=dtype,               
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa", 
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.eval()

# Torch optimizations
torch.backends.cuda.matmul.allow_tf32 = True  
torch.set_float32_matmul_precision("high")    
print("Model loaded")

################################
# Main Execution
################################

def run_identification():
    # Load Data
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)

    results = []
    
    # Iterate through every company in the dataset
    print(f"Processing {len(data)} items...")
    
    for item in tqdm(data):
        instance_name = item.get('instance', 'Unknown')
        image_paths = item.get('image_path', [])
        
        # Skip if no image path provided
        if not image_paths:
            print(f"No image path for {instance_name}, skipping.")
            continue

        # Load Image (using the first image in the list)
        rel_path = image_paths[0]
        full_path = os.path.join(IMAGE_ROOT, rel_path)
        
        pil_image = None
        try:
            if os.path.exists(full_path):
                pil_image = Image.open(full_path)
                if max(pil_image.size) > 1024:
                    pil_image.thumbnail((1024, 1024))
            else:
                print(f"Image not found: {full_path}")
                continue
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            continue

        # Prepare Prompt
        prompt_text = "What company is represented by this logo? Provide only the company name."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Prepare Inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Generate Response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=64, # Short generation for just the name
                do_sample=False 
            )
        
        response = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        # Cleanup response (Qwen sometimes repeats the prompt in raw decode depending on version, 
        # but apply_chat_template usually handles this. Just in case, we take the raw output).
        # Note: In standard Qwen VL inference, the output usually includes the full conversation 
        # if not handled carefully, but batch_decode on generated_ids[new_tokens:] is safer. 
        # However, to keep it simple and consistent with your previous script logic:
        
        # Extract the assistant's reply if the model echoes the prompt (common in some pipelines)
        # But Qwen 2.5 VL instruct usually outputs just the answer if using the chat template correctly.
        # We will save the raw response.
        
        # Append to results
        results.append({
            "Entity_Name": instance_name,
            "Image_Path": rel_path,
            "Model_Output": response,
        })

    # Save Results
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(SAVE_PATH, index=False)
    print(f"Identification complete. Saved to {SAVE_PATH}")

if __name__ == "__main__":
    run_identification()