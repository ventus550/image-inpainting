import torch
from transformers import TrainerCallback
from dataclasses import dataclass

def highlight(string):
    return f"\033[33m\033[1m{string}\033[0m"

@dataclass
class TrainingMonitor(TrainerCallback):
    step: int = 0

    def on_step_end(self, args, state, control, model=None, train_dataloader=None, lr_scheduler = None, **kwargs):
        self.step += 1
        if state.logging_steps and self.step % state.logging_steps:
            return
        
        input_ids, target_ids = next(iter(train_dataloader)).values()  # Get the first batch
        
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