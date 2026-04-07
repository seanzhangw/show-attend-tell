"""
BLEU-4 for image captioning (corpus-level over images).

Each image has up to 5 reference token lists; one greedy hypothesis per image.
"""

import torch

from .caption_metrics import corpus_bleu_n
from .corpus_predictions import gather_greedy_corpus


@torch.no_grad()
def compute_bleu4(
    encoder,
    decoder,
    val_image_ids,
    captions_map,
    image_dir,
    transform,
    word2idx,
    idx2word,
    device,
    max_len=20,
    max_images=None,
):
    """
    Args:
        val_image_ids: iterable of image filenames in the validation split
        captions_map: full map image_name -> list of caption strings (5 refs typical)
        max_images: if set, only evaluate this many val images (sorted order)

    Returns:
        bleu4: float (0-1 scale, NLTK convention)
    """
    list_of_references, hypotheses = gather_greedy_corpus(
        encoder,
        decoder,
        val_image_ids,
        captions_map,
        image_dir,
        transform,
        word2idx,
        idx2word,
        device,
        max_len=max_len,
        max_images=max_images,
    )
    return corpus_bleu_n(list_of_references, hypotheses, 4)
