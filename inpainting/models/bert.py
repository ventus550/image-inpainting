import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput
from .components import Embeddings, MultiHeadSelfAttention


# Transformer Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden)
        self.fc2 = nn.Linear(ff_hidden, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# Transformer Encoder Layer
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, ff_hidden)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask=mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.ff(x)
        return self.dropout(self.norm2(forward + x))


# BERT Model
class BERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_layers,
        num_heads,
        ff_hidden,
        max_len,
        dropout,
        ce_weights=None,
        embeddings="default",
    ):
        super().__init__()
        self.ce_weights = torch.nn.Parameter(
            torch.Tensor(ce_weights), requires_grad=False
        )

        if embeddings == "default":
            self.embed = Embeddings(vocab_size, embed_size, max_len)
        elif embeddings == "linear":
            self.embed = nn.Linear(vocab_size, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, ff_hidden, dropout)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(embed_size, vocab_size)

    def forward(self, input_ids, labels=None, mask=None) -> MaskedLMOutput:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, x, x, mask=mask)
        x = self.classifier(x)

        loss = None
        if labels is not None:
            b, n, p = x.shape
            logits = x.view(b * n, p)
            targets = labels.flatten()
            loss = torch.nn.functional.cross_entropy(
                logits, targets, weight=self.ce_weights
            )

        return MaskedLMOutput(loss=loss, logits=x)
