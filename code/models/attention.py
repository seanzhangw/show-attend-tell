import torch
import torch.nn as nn


class HardAttention(nn.Module):
    """
    Hard (stochastic) attention from Show, Attend and Tell (Section 4.2.1).

    During training: samples one spatial location per example from the
    attention distribution using a Categorical draw. This is non-differentiable,
    so the decoder must be trained with REINFORCE (see loop.train_one_epoch_hard).

    During eval: deterministically picks the argmax location.

    Returns (context, alpha, log_prob) instead of (context, alpha).
      alpha    – the full soft distribution over L locations (used for entropy
                 regularisation and visualisation, NOT for the weighted sum)
      log_prob – log p(sampled/argmax location), shape (B,), used by REINFORCE
    """

    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super().__init__()
        self.feature_att = nn.Linear(feature_dim, attention_dim)
        self.hidden_att = nn.Linear(hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.f_beta = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, hidden):
        """
        features: (B, L, D)
        hidden:   (B, hidden_dim)

        returns:
            context:  (B, D)   — feature at the selected location, gated by beta
            alpha:    (B, L)   — full attention distribution (for entropy / visualisation)
            log_prob: (B,)     — log p of the selected location
        """
        att1 = self.feature_att(features)              # (B, L, attention_dim)
        att2 = self.hidden_att(hidden).unsqueeze(1)    # (B, 1, attention_dim)
        att = self.tanh(att1 + att2)                   # (B, L, attention_dim)
        e = self.full_att(att).squeeze(2)              # (B, L)
        alpha = self.softmax(e)                        # (B, L)

        if self.training:
            dist = torch.distributions.Categorical(probs=alpha)
            idx = dist.sample()                        # (B,)
            log_prob = dist.log_prob(idx)              # (B,)
        else:
            idx = alpha.argmax(dim=1)                  # (B,)
            log_prob = alpha.log().gather(1, idx.unsqueeze(1)).squeeze(1)  # (B,)

        # Hard selection: context = feature at the single chosen location
        one_hot = torch.zeros_like(alpha)
        one_hot.scatter_(1, idx.unsqueeze(1), 1.0)    # (B, L)
        context = (features * one_hot.unsqueeze(2)).sum(dim=1)  # (B, D)

        beta = self.sigmoid(self.f_beta(hidden))
        context = context * beta

        return context, alpha, log_prob


class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_dim):
        """
        feature_dim: 2048 (from ResNet)
        hidden_dim: LSTM hidden size (e.g. 512)
        attention_dim: internal projection size (e.g. 512)
        """
        super().__init__()

        # Linear projections
        self.feature_att = nn.Linear(feature_dim, attention_dim)
        self.hidden_att = nn.Linear(hidden_dim, attention_dim)

        # Final scoring layer (v^T in paper)
        self.full_att = nn.Linear(attention_dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.f_beta = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, hidden):
        """
        features: (B, L=49, D=2048)
        hidden:   (B, hidden_dim)

        returns:
            context: (B, D)
            alpha:   (B, L)
        """

        # Project image features
        att1 = self.feature_att(features)        # (B, L, attention_dim)

        # Project hidden state
        att2 = self.hidden_att(hidden)           # (B, attention_dim)
        att2 = att2.unsqueeze(1)                 # (B, 1, attention_dim)

        # Combine + nonlinearity (tanh per paper)
        # Intuitively, this is combining the image features and the hidden state of the decoder
        # ie. What is in the image and what the decoder is currently thinking about
        att = self.tanh(att1 + att2)             # (B, L, attention_dim)

        # Compute attention scores
        e = self.full_att(att).squeeze(2)        # (B, L)

        # Normalize to get attention weights
        alpha = self.softmax(e)                 # (B, L)

        # Compute context vector (weighted sum) 
        # Equation (13) in Show, Attend and Tell
        context = (features * alpha.unsqueeze(2)).sum(dim=1)  # (B, D)

        # TODO: Add gating scalar beta
        beta = self.sigmoid(self.f_beta(hidden))
        context = context * beta
        return context, alpha