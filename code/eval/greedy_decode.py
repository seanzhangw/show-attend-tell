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

    h, c = decoder.init_hidden_state(features)
    cur = torch.tensor([word2idx["<start>"]], device=device)

    out_ids = []
    for _ in range(max_len):
        emb = decoder.embedding(cur)
        context, _ = decoder.attention(features, h)
        lstm_in = torch.cat([emb, context], dim=1)
        h, c = decoder.lstm_cell(lstm_in, (h, c))
        logits = decoder.fc(decoder.dropout(h))

        next_id = int(logits.argmax(dim=1).item())
        w = idx2word.get(next_id, "<unk>")
        if w == "<end>":
            break
        if w not in {"<start>", "<pad>"}:
            out_ids.append(next_id)
        cur = torch.tensor([next_id], device=device)

    return [idx2word[i] for i in out_ids]
