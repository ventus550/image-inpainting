import torch
import torch.nn as nn
import math

# Embedding layer: combines token and position embeddings
class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len):
        super(Embeddings, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.position_embed = nn.Embedding(max_len, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        token_embeddings = self.token_embed(input_ids)
        position_embeddings = self.position_embed(position_ids)

        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)

# Self-Attention Mechanism
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert self.head_dim * num_heads == embed_size, "embed_size must be divisible by num_heads"

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, value, key, query, mask=None):
        N, seq_len, embed_size = query.shape

        Q = self.query(query).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)

        out = out.transpose(1, 2).contiguous().view(N, seq_len, embed_size)
        return self.fc_out(out)