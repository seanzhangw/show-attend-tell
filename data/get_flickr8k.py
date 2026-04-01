import kagglehub
import shutil
import os

# Always resolve to data/flickr8k next to this script, regardless of cwd
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(_DATA_DIR, "flickr8k")

# Download (goes to cache)
path = kagglehub.dataset_download("adityajn105/flickr8k")

print("Downloaded to:", path)

# Create target directory if it doesn't exist
os.makedirs(TARGET_DIR, exist_ok=True)

# Move contents into your data folder
if not os.listdir(TARGET_DIR):  # avoid overwriting
    shutil.copytree(path, TARGET_DIR, dirs_exist_ok=True)

print("Dataset available at:", TARGET_DIR)
