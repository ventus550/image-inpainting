"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader


class Trainer:
	def __init__(self, model, dataset):
		self.dataset = dataset
		# self.callbacks = defaultdict(list)
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = model.to(self.device)
		print("running on device", self.device)

	def train(
		self,
		epochs=1,
		lr=3e-4,
		weight_decay=0.1,
		batch_size=64,
		workers=4,
		shuffle=True,
	):
		model = self.model

		# setup the optimizer
		optimizer = torch.optim.Adam(
			model.parameters(), lr=lr, weight_decay=weight_decay
		)

		# setup the dataloader
		loader = DataLoader(
			self.dataset,
			# sampler=torch.utils.data.RandomSampler(
			# 	self.dataset, replacement=True, num_samples=int(1e10)
			# ),
			shuffle=shuffle,
			pin_memory=True,
			batch_size=batch_size,
			num_workers=workers,
		)

		model.train()
		for e in range(epochs):
			for x, y in iter(loader):
				x, y = x.to(self.device), y.to(self.device)
				# mask = x != y
				# logits = model(x)[mask]
				# targets = y[mask]

				logits = model(x)
				b, n, p = logits.shape

				# print(logits.shape)
				# print(y.shape)

				logits = logits.view(b*n, p)
				targets = y.flatten()
				
				loss = torch.nn.functional.cross_entropy(logits, targets)
				loss.backward()
				optimizer.step()
				model.zero_grad()

			print(f"Epoch: {e:<10}\t\tLoss: {loss.item()}")
