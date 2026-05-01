"""
LSTM decoder with soft attention for image captioning (Show, Attend and Tell style).
"""

import torch
import torch.nn as nn

from .attention import Attention, HardAttention


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

        # Deep output layer (Eq. 7)
        self.L_h = nn.Linear(hidden_dim, embed_dim)
        self.L_z = nn.Linear(feature_dim, embed_dim)
        self.L_o = nn.Linear(embed_dim, vocab_size)

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

        # Loop through each generated word in the caption
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
            out = x_t + self.L_h(h) + self.L_z(context) # (B, embed_dim)
            out = self.dropout(out)
            logits_t = self.L_o(out)  # (B, vocab_size)

            logits_list.append(logits_t)
            alpha_list.append(alpha_t)

        # (B, T-1, vocab_size)
        logits = torch.stack(logits_list, dim=1)
        # (B, T-1, L)
        alphas = torch.stack(alpha_list, dim=1)
        return logits, alphas
    
    def sample(self, features: torch.Tensor, start_token_id: int, max_len: int = 20):
        """
        Greedy decoding for inference without teacher forcing.
        
        Args:
            features: (B, L, D) image features from the encoder
            start_token_id: The integer ID for your <start> token in the vocabulary
            max_len: Maximum number of words to generate
            
        Returns:
            sampled_ids: (B, max_len) predicted word IDs
            alphas: (B, max_len, L) attention weights for visualization
        """
        B = features.size(0)
        
        # Initialize hidden state from the image
        h, c = self.init_hidden_state(features)

        # Start sequence by feeding the <start> token to every item in the batch
        # inputs shape: (B,)
        inputs = torch.full((B,), start_token_id, dtype=torch.long, device=features.device)

        sampled_ids = []
        alpha_list = []

        for _ in range(max_len):
            # Embed the current token
            x_t = self.embedding(inputs)  # (B, embed_dim)

            # Get attention context and weights
            context, alpha_t = self.attention(features, h)

            # Update LSTM
            lstm_in = torch.cat([x_t, context], dim=1)
            h, c = self.lstm_cell(lstm_in, (h, c))

            # Deep output layer (Equation 7)
            deep_out = x_t + self.L_h(h) + self.L_z(context)
            
            logits_t = self.L_o(deep_out)  # (B, vocab_size)

            # Pick the word with the highest probability
            predicted_word_id = logits_t.argmax(dim=1)  # (B,)

            # Save the predictions and attention weights
            sampled_ids.append(predicted_word_id)
            alpha_list.append(alpha_t)

            # Set the newly predicted word as the input for the NEXT time step
            inputs = predicted_word_id

        # Stack lists into final tensors
        sampled_ids = torch.stack(sampled_ids, dim=1) # (B, max_len)
        alphas = torch.stack(alpha_list, dim=1)       # (B, max_len, L)

        return sampled_ids, alphas


class HardDecoder(nn.Module):
    """
    Teacher-forced decoder with hard (stochastic) attention.

    Identical architecture to Decoder but uses HardAttention instead of
    soft Attention.  The extra log_prob output from HardAttention is
    collected at every step and returned so the training loop can compute
    the REINFORCE gradient.

    forward() -> (logits, alphas, log_probs)
      logits:    (B, T-1, vocab_size)
      alphas:    (B, T-1, L)   full attention distribution (for entropy / vis)
      log_probs: (B, T-1)      log p(selected location) at each step

    sample() -> (sampled_ids, alphas)   same signature as Decoder.sample()
      Drop-in compatible with greedy_decode and corpus_predictions.
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

        self.attention = HardAttention(feature_dim, hidden_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + feature_dim, hidden_dim)
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.L_h = nn.Linear(hidden_dim, embed_dim)
        self.L_z = nn.Linear(feature_dim, embed_dim)
        self.L_o = nn.Linear(embed_dim, vocab_size)

    def init_hidden_state(self, features: torch.Tensor):
        mean = features.mean(dim=1)
        return self.init_h(mean), self.init_c(mean)

    def forward(self, features: torch.Tensor, captions: torch.Tensor):
        """
        Args:
            features: (B, L, D)
            captions: (B, T)

        Returns:
            logits:    (B, T-1, vocab_size)
            alphas:    (B, T-1, L)
            log_probs: (B, T-1)
        """
        B, T = captions.shape
        if T < 2:
            raise ValueError("captions must have length T >= 2 for teacher forcing")

        captions_in = captions[:, :-1]
        h, c = self.init_hidden_state(features)
        emb = self.embedding(captions_in)

        logits_list, alpha_list, log_prob_list = [], [], []

        for t in range(captions_in.size(1)):
            x_t = emb[:, t, :]
            context, alpha_t, log_prob_t = self.attention(features, h)

            lstm_in = torch.cat([x_t, context], dim=1)
            h, c = self.lstm_cell(lstm_in, (h, c))

            out = x_t + self.L_h(h) + self.L_z(context)
            out = self.dropout(out)
            logits_t = self.L_o(out)

            logits_list.append(logits_t)
            alpha_list.append(alpha_t)
            log_prob_list.append(log_prob_t)

        logits = torch.stack(logits_list, dim=1)       # (B, T-1, V)
        alphas = torch.stack(alpha_list, dim=1)         # (B, T-1, L)
        log_probs = torch.stack(log_prob_list, dim=1)   # (B, T-1)
        return logits, alphas, log_probs

    def sample(self, features: torch.Tensor, start_token_id: int, max_len: int = 20):
        """
        Greedy decoding for inference — same return signature as Decoder.sample().
        HardAttention uses argmax in eval mode, so this is deterministic.
        """
        B = features.size(0)
        h, c = self.init_hidden_state(features)
        inputs = torch.full((B,), start_token_id, dtype=torch.long, device=features.device)

        sampled_ids, alpha_list = [], []

        for _ in range(max_len):
            x_t = self.embedding(inputs)
            context, alpha_t, _ = self.attention(features, h)
            lstm_in = torch.cat([x_t, context], dim=1)
            h, c = self.lstm_cell(lstm_in, (h, c))
            deep_out = x_t + self.L_h(h) + self.L_z(context)
            logits_t = self.L_o(deep_out)
            predicted = logits_t.argmax(dim=1)
            sampled_ids.append(predicted)
            alpha_list.append(alpha_t)
            inputs = predicted

        sampled_ids = torch.stack(sampled_ids, dim=1)   # (B, max_len)
        alphas = torch.stack(alpha_list, dim=1)          # (B, max_len, L)
        return sampled_ids, alphas