"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import torch
from torch.utils.data.dataloader import DataLoader
from transformers import TrainerCallback
from transformers.optimization import get_linear_schedule_with_warmup
from dataclasses import dataclass, field
from torch.utils.data import Dataset

from .callbacks import TrainingMonitor, OracleEstimator


@dataclass
class State:
    log_history: list = field(default_factory=list)
    logging_steps: int = 0
    epoch: int = 0


@dataclass
class Trainer:
    model: torch.nn.Module
    dataset: Dataset
    logging_steps: int = 10
    callbacks: list[TrainerCallback] = field(default_factory=lambda: [TrainingMonitor, OracleEstimator])

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state = State(logging_steps=self.logging_steps)
        # Instantiate callbacks if they are not already instances
        self.callbacks = [
            callback() if isinstance(callback, type) else callback
            for callback in self.callbacks
        ]
        print("running on device", self.device)

    def train(
        self,
        epochs=1,
        lr=3e-4,
        weight_decay=0.1,
        batch_size=64,
        workers=4,
        shuffle=True,
        warmup_steps=0,
    ):
        model = self.model.to(self.device)
        model.device = self.device

        # setup the optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=epochs * len(self.dataset) // batch_size
        )

        # setup the dataloader
        loader = DataLoader(
            self.dataset,
            shuffle=shuffle,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=workers,
        )

        model.train()
        for self.state.epoch in range(epochs):
            for batch in iter(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                del batch["number"]

                outputs = model(**batch)

                outputs.loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                for callback in self.callbacks:
                    callback.on_step_end(
                        args=None,
                        state=self.state,
                        control=None,
                        model=model,
                        train_dataloader=loader,
                        lr_scheduler=scheduler,
                    )
        model.eval()
