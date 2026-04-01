import os

# Project root is parent of this package (code/); paths work no matter where you run python from
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "flickr8k")
IMAGE_DIR = os.path.join(DATA_DIR, "Images")
CAPTION_FILE = os.path.join(DATA_DIR, "captions.txt")

MAX_LEN = 20
VOCAB_SIZE = 10_000  # most common word types (excluding special tokens)
BATCH_SIZE = 32