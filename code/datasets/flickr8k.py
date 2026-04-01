import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import (
    load_captions,
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
            word2idx (dict)
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


# 🔥 Important: custom collate function
def collate_fn(batch):
    """
    Batch = list of (image, caption_tensor)
    """
    images, captions = zip(*batch)

    images = torch.stack(images, dim=0)
    captions = torch.stack(captions, dim=0)

    return images, captions


def _default_image_transform():
    """Resize + tensor so batches stack; Flickr8k images are not uniform size."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )


# 🚀 Clean builder function
def build_flickr8k_dataset(config, transform=None):
    """
    Builds dataset + vocab in one place.
    If ``transform`` is None, uses resize-to-224 and ``ToTensor()`` so ``collate_fn`` can stack.
    """
    if transform is None:
        transform = _default_image_transform()

    captions_map = load_captions(config.CAPTION_FILE)

    word2idx, idx2word = build_vocab(
        captions_map,
        max_vocab_size=config.VOCAB_SIZE,
    )

    dataset = FlickrDataset(
        image_dir=config.IMAGE_DIR,
        captions_map=captions_map,
        word2idx=word2idx,
        transform=transform,
        max_len=config.MAX_LEN,
    )

    return dataset, word2idx, idx2word