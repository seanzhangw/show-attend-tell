"""
LSTM decoder with soft attention for image captioning (Show, Attend and Tell style).
"""

import torch
import torch.nn as nn

from .attention import Attention


class Decoder(nn.Module):
    """
    Teacher-forced decoder: at each step, attend with previous hidden state, then
    LSTMCell( [word_emb; context], (h, c) ) -> logits for the next token.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        feature_dim: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.attention = Attention(feature_dim, hidden_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Input = concat(word embedding, context vector) -> (embed_dim + feature_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + feature_dim, hidden_dim)

        # Initialize h0, c0 from mean-pooled image features (each D -> hidden_dim)
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_hidden_state(self, features: torch.Tensor):
        """
        Args:
            features: (B, L, D) spatial CNN features

        Returns:
            h0: (B, hidden_dim), c0: (B, hidden_dim)
        """
        # Mean over spatial locations L -> (B, D)
        mean = features.mean(dim=1)
        h0 = self.init_h(mean)
        c0 = self.init_c(mean)
        return h0, c0

    def forward(self, features: torch.Tensor, captions: torch.Tensor):
        """
        Teacher forcing: inputs are captions[:, :-1], targets for loss are captions[:, 1:].

        Args:
            features: (B, L, D) image features from encoder
            captions: (B, T) full caption token ids (padded), including <start>...<end>

        Returns:
            logits: (B, T-1, vocab_size) prediction for each next token
            alphas: (B, T-1, L) attention weights at each step
        """
        B, T = captions.shape
        if T < 2:
            raise ValueError("captions must have length T >= 2 for teacher forcing")

        # Input tokens: all but last; we predict one step ahead at each position
        # captions_in: (B, T-1)
        captions_in = captions[:, :-1]

        h, c = self.init_hidden_state(features)

        # Embeddings for all input timesteps: (B, T-1, embed_dim)
        emb = self.embedding(captions_in)

        logits_list = []
        alpha_list = []
        num_steps = captions_in.size(1)

        for t in range(num_steps):
            # Current word embedding: (B, embed_dim)
            x_t = emb[:, t, :]

            # Attention uses hidden from *previous* step (h is h_{t-1} before LSTM update;
            # after init, h is h_0 for t=0)
            # context: (B, D), alpha: (B, L)
            context, alpha_t = self.attention(features, h)

            # (B, embed_dim + D)
            lstm_in = torch.cat([x_t, context], dim=1)

            # h, c become h_t, c_t
            h, c = self.lstm_cell(lstm_in, (h, c))

            # Dropout then classify next token
            out = self.dropout(h)
            logits_t = self.fc(out)  # (B, vocab_size)

            logits_list.append(logits_t)
            alpha_list.append(alpha_t)

        # (B, T-1, vocab_size)
        logits = torch.stack(logits_list, dim=1)
        # (B, T-1, L)
        alphas = torch.stack(alpha_list, dim=1)
        return logits, alphas
