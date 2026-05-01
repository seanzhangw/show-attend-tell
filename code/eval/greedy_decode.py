import torch

@torch.no_grad()
def greedy_decode(
    encoder,
    decoder,
    image_tensor,
    word2idx,
    idx2word,
    device,
    max_len=20,
):
    """
    Greedy caption generation (no teacher forcing).

    Args:
        image_tensor: (3, H, W) float tensor (already transformed)

    Returns:
        List of word tokens (no <start>/<end>).
    """
    encoder.eval()
    decoder.eval()

    image_tensor = image_tensor.unsqueeze(0).to(device)
    features = encoder(image_tensor)

    start_id = word2idx["<start>"]
    end_id = word2idx["<end>"]

    sampled_ids, _ = decoder.sample(features, start_token_id=start_id, max_len=max_len)

    out_words = []
    for token_id in sampled_ids[0].tolist():
        if token_id == end_id:
            break
            
        w = idx2word.get(token_id, "<unk>")
        if w not in {"<start>", "<pad>"}:
            out_words.append(w)

    return out_words
