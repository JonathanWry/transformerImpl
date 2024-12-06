import os
from datasets import load_dataset

# Set timeout for Hugging Face Hub requests
os.environ["HF_HUB_HTTP_TIMEOUT"] = "1200"  # Timeout in seconds (e.g., 60 seconds)

# Download the dataset and save it locally
dataset = load_dataset("wmt14_DED", "fr-en")
dataset.save_to_disk("./wmt14_fr")