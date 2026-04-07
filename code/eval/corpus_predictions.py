"""
Greedy-decode the full validation corpus once for metric computation.
"""

import os

import torch
from PIL import Image

from .greedy_decode import greedy_decode
from .tokenize import caption_to_bleu_tokens


@torch.no_grad()
def gather_greedy_corpus(
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
    Build NLTK-style corpus inputs: references per image and one hypothesis per image.

    Args:
        val_image_ids: iterable of image filenames in the validation split
        captions_map: image_name -> list of caption strings (multiple references per image)
        max_images: if set, only evaluate this many val images (sorted order)

    Returns:
        list_of_references: list of ref lists (each ref is a token list)
        hypotheses: list of hypothesis token lists
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

    return list_of_references, hypotheses
