import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput

class GPT(nn.Module):
	def __init__(self, vocab_size, embed_size=768, num_heads=12, num_layers=12, max_len=512, ce_weights=None):
		super().__init__()
		self.ce_weights = torch.nn.Parameter(
            torch.Tensor(ce_weights), requires_grad=False
        )
		
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.position_embedding = nn.Embedding(max_len, embed_size)
		self.transformer_blocks = nn.ModuleList([
			nn.TransformerEncoderLayer(embed_size, num_heads, dim_feedforward=4*embed_size, activation='gelu')
			for _ in range(num_layers)
		])
		self.ln_f = nn.LayerNorm(embed_size)
		self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
		
	def forward(self, input_ids, labels=None, attention_mask=None) -> MaskedLMOutput:
		device = input_ids.device
		seq_length = input_ids.size(1)
		position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)
		
		x = self.embedding(input_ids) + self.position_embedding(position_ids)
		x = x.permute(1, 0, 2)  # (seq_len, batch, dim) for Transformer
		
		for layer in self.transformer_blocks:
			x = layer(x, src_key_padding_mask=attention_mask)
		
		x = self.ln_f(x)
		x = self.lm_head(x)  # Predict masked tokens
		
		loss = None
		if labels is not None:
			b, n, p = x.shape
			logits = x.view(b * n, p)
			targets = labels.flatten()
			loss = torch.nn.functional.cross_entropy(
				logits, targets, weight=self.ce_weights
			)
        
		return MaskedLMOutput(loss=loss, logits=x)
