import csv
import re
from collections import defaultdict, Counter


SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]

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


def clean_caption(caption):
    """
    Lowercase + remove punctuation
    Returns list of tokens with <start>, <end>
    """
    caption = caption.lower()
    caption = re.sub(r"[^a-z ]", "", caption)

    tokens = caption.split()

    return ["<start>"] + tokens + ["<end>"]


def build_vocab(captions_map, max_vocab_size=10_000):
    """
    Builds vocabulary from captions using the ``max_vocab_size`` most frequent word types.
    Special tokens are always included first and are excluded from the frequency ranking.

    Returns:
        word2idx, idx2word
    """
    counter = Counter()

    for captions in captions_map.values():
        for caption in captions:
            tokens = clean_caption(caption)
            counter.update(tokens)

    special = set(SPECIAL_TOKENS)
    ranked = Counter({w: c for w, c in counter.items() if w not in special})

    top_words = [w for w, _ in ranked.most_common(max_vocab_size)]
    vocab = list(SPECIAL_TOKENS) + top_words

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word


def encode_caption(tokens, word2idx):
    """
    Converts tokens → list of indices
    """
    unk_idx = word2idx["<unk>"]

    return [word2idx.get(token, unk_idx) for token in tokens]


def pad_caption(encoded, max_len, pad_idx):
    """
    Pads or truncates caption to max_len
    """
    if len(encoded) > max_len:
        return encoded[:max_len]

    return encoded + [pad_idx] * (max_len - len(encoded))