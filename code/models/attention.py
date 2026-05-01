import torch
import torch.nn as nn


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