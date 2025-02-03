import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_heads,
        num_layers,
        max_len,
        patches,
        ce_weights=None,
        dropout=0.0,
    ):
        super().__init__()
        self.patches = torch.nn.Parameter(torch.Tensor(patches), requires_grad=False)
        self.ce_weights = torch.nn.Parameter(
            torch.Tensor(ce_weights), requires_grad=False
        )
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_size))
        self.embedding = nn.Embedding(vocab_size, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            embed_size, num_heads, 4 * embed_size, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size - 1, bias=False)

    def forward(self, input_ids, labels=None, attention_mask=None) -> MaskedLMOutput:
        x = (
            self.embedding(input_ids)
            + self.positional_encoding[:, : input_ids.size(1), :]
        )

        x = self.transformer_encoder(x, mask=attention_mask, src_key_padding_mask=None)
        x = self.lm_head(self.ln_f(x))

        probs = torch.softmax(x, dim=-1)
        patches = probs @ self.patches

        loss = None
        if labels is not None:
            logits = patches.flatten(start_dim=1, end_dim=-1)
            targets = self.patches[labels].flatten(start_dim=1, end_dim=-1)
            loss = nn.MSELoss()(logits, targets)

        return MaskedLMOutput(loss=loss, logits=x)
