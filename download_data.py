from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="starjyf/MLLMKC-datasets", 
    repo_type="dataset", 
    local_dir="./MLLMKC_data",
    local_dir_use_symlinks=False
)
print("Download complete!")