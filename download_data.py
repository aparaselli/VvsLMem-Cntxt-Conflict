import os
from huggingface_hub import snapshot_download

os.environ.setdefault("HF_HOME", "/oscar/scratch/aparasel/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/oscar/scratch/aparasel/hf_cache/hub")

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

snapshot_download(
    repo_id="starjyf/MLLMKC-datasets",
    repo_type="dataset",
    local_dir="./MLLMKC_data",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=2, 
)

print("Download complete!")