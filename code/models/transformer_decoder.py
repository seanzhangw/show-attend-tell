import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class TransformerCaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, feature_dim=768, num_layers=3, nheads=8, dropout=0.5, max_len=50):
        super().__init__()
        self.embed_dim = embed_dim

        # Project ViT features (768) down to match Transformer width (512)
        self.feature_proj = nn.Linear(feature_dim, embed_dim)

        # Text Embedding & Positional Encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len)

        # The PyTorch Transformer Decoder Engine
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nheads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True # CRITICAL: Keeps batch size as the first dimension!
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final word predictor
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    
    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        
        float_mask = torch.zeros(sz, sz, device=device)
        float_mask.masked_fill_(mask, float('-inf'))
        
        return float_mask
    
    def forward(self, features, captions):
        memory = self.feature_proj(features) # (Batch, 196, 512)

        tgt_input = captions[:, :-1]

        tgt = self.embedding(tgt_input) * math.sqrt(self.embed_dim)
        tgt = self.pos_encoder(tgt) # (Batch, Seq_Len - 1, 512)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)

        out = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)

        predictions = self.fc_out(out) 
        
        batch_size = features.size(0)
        seq_len = tgt_input.size(1)
        dummy_alphas = torch.zeros(batch_size, seq_len, 196, device=features.device)
        
        return predictions, dummy_alphas

    @torch.no_grad()
    def sample(self, features, start_token_id, max_len=20):
        # Autoregressive generation for validation/inference
        device = features.device
        batch_size = features.size(0)

        memory = self.feature_proj(features)
        sampled_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt = self.embedding(sampled_ids) * math.sqrt(self.embed_dim)
            tgt = self.pos_encoder(tgt)
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device)

            out = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            preds = self.fc_out(out[:, -1, :]) # Look only at the newest predicted word
            
            predicted_id = preds.argmax(dim=-1, keepdim=True)
            sampled_ids = torch.cat([sampled_ids, predicted_id], dim=1)

        dummy_alphas = torch.ones(batch_size, max_len, 196, device=device) / 196.0
        return sampled_ids, dummy_alphas