import torch
from transformers import TrainerCallback
from dataclasses import dataclass
from tensorflow.keras.saving import load_model
import numpy as np

def highlight(string):
    return f"\033[33m\033[1m{string}\033[0m"

@dataclass
class OracleEstimator(TrainerCallback):
    step: int = 0
    sample_size: int = 10
    oracle_path: str = "./mnist_oracle.keras"

    def __post_init__(self):
        self.oracle = load_model(self.oracle_path)
    
    def run_oracle(self, model, loader):
        same_predictions, correct = 0, 0
        
        sampled_indices = np.random.randint(0, len(loader.dataset.data), size=self.sample_size)
        sampled_images = loader.dataset.data[sampled_indices]

        true_labels = loader.dataset.targets[sampled_indices]
        src_predictions = np.argmax(self.oracle.predict(sampled_images, verbose=0), axis=1)

        input_ids = torch.stack([loader.dataset[i]["input_ids"] for i in sampled_indices])
        target_ids = torch.stack([loader.dataset[i]["labels"] for i in sampled_indices])

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(model.device),
                labels=target_ids.to(model.device)
            )

            restored_imgs = np.array([loader.dataset.restore_image_from_patch_indices(i) for i in outputs.logits.argmax(dim=-1).cpu().numpy()])
        restored_predictions = np.argmax(self.oracle.predict(restored_imgs, verbose=0), axis=1)

        same_predictions = self.sample_size - np.count_nonzero(src_predictions - restored_predictions)
        correct = self.sample_size - np.count_nonzero(true_labels - restored_predictions)

        print(
                f"{highlight("Same classification:")} {same_predictions/self.sample_size:.2f}",
                f"{highlight("Correct classification:")} {correct/self.sample_size:.2e}",
                sep="\t"
            )

    def on_step_end(self, args, state, control, model=None, train_dataloader=None, lr_scheduler = None, **kwargs):
        self.step += 1
        if state.logging_steps and self.step % state.logging_steps:
            return
                
        model.eval()
        
        self.run_oracle(model, train_dataloader)

        model.train()


@dataclass
class TrainingMonitor(TrainerCallback):
    step: int = 0

    def on_step_end(self, args, state, control, model=None, train_dataloader=None, lr_scheduler = None, **kwargs):
        self.step += 1
        if state.logging_steps and self.step % state.logging_steps:
            return
        
        input_ids, target_ids, _ = next(iter(train_dataloader)).values()  # Get the first batch
        model.eval()
        
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids.to(model.device),
                    labels=target_ids.to(model.device)
                )
                
                output_patches = train_dataloader.dataset.itop(outputs.logits.argmax(dim=-1).cpu().numpy())
                target_patches = train_dataloader.dataset.itop(target_ids.cpu().numpy())

                errors = (output_patches - target_patches)**2
                batch_size = train_dataloader.batch_size or state.train_batch_size
                rmse = errors.reshape(batch_size, -1).mean(-1).__pow__(0.5).mean()

            print(
                f"{highlight("Epoch:")} {state.epoch:.2f}",
                f"{highlight("RMSE:")} {rmse:.2f}",
                f"{highlight("Loss:")} {outputs.loss:.2e}",
                f"{highlight("Learning Rate:")} {lr_scheduler.get_lr()[0]:.2e}",
                sep="\t"
            )
        except Exception as e:
            print(e)

        model.train()