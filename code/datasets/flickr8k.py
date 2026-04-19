import os
import random
import csv
from collections import defaultdict

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import (
    clean_caption,
    encode_caption,
    pad_caption,
    build_vocab,
)

class FlickrDataset(Dataset):
    def __init__(
        self,
        image_dir,
        captions_map,
        word2idx,
        transform=None,
        max_len=20,
    ):
        """
        Args:
            image_dir (str): path to images
            captions_map (dict): image -> list of captions
            word2idx (dict): word -> index
            transform: torchvision transforms
            max_len (int): max caption length
        """
        self.image_dir = image_dir
        self.word2idx = word2idx
        self.transform = transform
        self.max_len = max_len

        self.pad_idx = word2idx["<pad>"]

        # Flatten (image, caption) pairs
        self.samples = []
        for img, captions in captions_map.items():
            for caption in captions:
                self.samples.append((img, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]

        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Process caption
        tokens = clean_caption(caption)
        encoded = encode_caption(tokens, self.word2idx)
        padded = pad_caption(encoded, self.max_len, self.pad_idx)

        caption_tensor = torch.tensor(padded, dtype=torch.long)

        return image, caption_tensor

def load_captions(caption_file):
    """
    Loads captions from the flickr8k captions.txt file.

    Returns:
        dict: {image_name: [caption1, caption2, ...]}
    """
    mapping = defaultdict(list)

    with open(caption_file, "r", encoding="utf-8", newline="") as f:
        f.seek(0)

        reader = csv.DictReader(f)
        for row in reader:
            img = (row.get("image") or "").strip()
            cap = (row.get("caption") or "").strip()
            if not img or not cap:
                continue
            img = img.split("#")[0]
            mapping[img].append(cap.lower())

    return dict(mapping)

def collate_fn(batch):
    """
    Batch = list of (image, caption_tensor)
    """
    images, captions = zip(*batch)

    images = torch.stack(images, dim=0)
    captions = torch.stack(captions, dim=0)

    return images, captions


def split_train_val_images(captions_map, val_ratio=0.1, seed=42):
    """
    Split image ids into train / val (no image appears in both).
    """
    ids = list(captions_map.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def build_flickr8k_dataset_split(
    config, transform=None, val_ratio=0.1, seed=42
):
    """
    Image-level train/val split. Vocabulary is built from **training** captions only.

    Returns:
        train_dataset, val_dataset, val_image_ids (set), full_captions_map, word2idx, idx2word
    """
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    full_map = load_captions(config.CAPTION_FILE)
    train_ids, val_ids = split_train_val_images(full_map, val_ratio, seed)

    train_map = {k: full_map[k] for k in train_ids}
    word2idx, idx2word = build_vocab(train_map, max_vocab_size=config.VOCAB_SIZE)

    train_dataset = FlickrDataset(
        image_dir=config.IMAGE_DIR,
        captions_map=train_map,
        word2idx=word2idx,
        transform=transform,
        max_len=config.MAX_LEN,
    )
    val_map = {k: full_map[k] for k in val_ids}
    val_dataset = FlickrDataset(
        image_dir=config.IMAGE_DIR,
        captions_map=val_map,
        word2idx=word2idx,
        transform=transform,
        max_len=config.MAX_LEN,
    )

    return train_dataset, val_dataset, val_ids, full_map, word2idx, idx2word


def _default_image_transform():
    """Resize + tensor so batches stack; Flickr8k images are not uniform size."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
