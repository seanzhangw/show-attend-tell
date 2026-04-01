"""Tokenize captions for BLEU (match training normalization, no special tokens)."""

from datasets.utils import clean_caption


def caption_to_bleu_tokens(caption: str):
    """
    Lowercase, strip punctuation, split — same as training — but drop <start>/<end>.
    Returns a list of word strings.
    """
    toks = clean_caption(caption)
    return toks[1:-1]
