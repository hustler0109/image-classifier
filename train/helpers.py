import numpy as np
import torch

def save_model(path, epoch, model, optimizer, loss, accuracy):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
        },
        path,
    )
    return

class EarlyStopping:

    def __init__(self, patience=7, delta=0, trace_func=print):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_val_loss = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
