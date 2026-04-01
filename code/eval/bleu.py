"""
BLEU-4 for image captioning (corpus-level over images).

Each image has up to 5 reference token lists; one greedy hypothesis per image.
"""

import os

import torch
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from .greedy_decode import greedy_decode
from .tokenize import caption_to_bleu_tokens


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
    encoder.eval()
    decoder.eval()

    ids = sorted(val_image_ids)
    if max_images is not None:
        ids = ids[:max_images]

    list_of_references = []
    hypotheses = []

    for img_name in ids:
        refs = captions_map[img_name]
        ref_toks = [caption_to_bleu_tokens(c) for c in refs]
        list_of_references.append(ref_toks)

        path = os.path.join(image_dir, img_name)
        image = Image.open(path).convert("RGB")
        image_tensor = transform(image)

        hyp_words = greedy_decode(
            encoder,
            decoder,
            image_tensor,
            word2idx,
            idx2word,
            device,
            max_len=max_len,
        )
        hypotheses.append(hyp_words)

    smooth = SmoothingFunction().method1
    weights = (0.25, 0.25, 0.25, 0.25)
    score = corpus_bleu(
        list_of_references,
        hypotheses,
        weights=weights,
        smoothing_function=smooth,
    )
    return score
