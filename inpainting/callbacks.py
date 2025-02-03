import torch
from transformers import TrainerCallback
from dataclasses import dataclass
import numpy as np

def highlight(string):
    return f"\033[33m\033[1m{string}\033[0m"

@dataclass
class TrainingMonitor(TrainerCallback):
    step: int = 0

    def run_oracle(self, inputs, targets, labels, oracle, loader):
        same_predictions, correct = 0, 0

        for inp, target, label in zip(inputs, targets, labels):
            inp_img = loader.dataset.restore_image_from_patches(inp)
            target_img = loader.dataset.restore_image_from_patches(target)

            inp_predict = oracle.predict(inp_img[None,:], verbose=0)
            target_predict = oracle.predict(target_img[None,:], verbose=0)
            if np.argmax(inp_predict) == np.argmax(target_predict):
                same_predictions += 1
            
                if np.argmax(inp_predict) == label:
                    correct += 1

        print(
                f"{highlight("Same classification:")} {same_predictions/len(inputs):.2f}",
                f"{highlight("Correct classification:")} {correct/len(inputs):.2e}",
                sep="\t"
            )


    def on_step_end(self, args, state, control, model=None, train_dataloader=None, lr_scheduler = None, oracle=None, **kwargs):
        self.step += 1
        if state.logging_steps and self.step % state.logging_steps:
            return
        
        input_ids, target_ids, numbers = next(iter(train_dataloader)).values()  # Get the first batch
        
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
            self.run_oracle(output_patches, target_patches, numbers, oracle, train_dataloader)
        except Exception as e:
            print(e)

        model.train()