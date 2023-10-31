import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .helpers import save_model

def train_model_custom(
    custom_model,
    data_loader,
    custom_criterion,
    custom_optimizer,
    custom_device,
    custom_save_path,
    num_epochs=25,
    custom_scheduler=None,
    custom_early_stopping=None,
):

    log_data = pd.DataFrame(
        {"epoch": [], "train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    )
    log_filename = f'{custom_save_path.split(".")[0]}_logs.csv'

    train_loss_history = []
    train_accuracy_history = []
    validation_loss_history = []
    validation_accuracy_history = []

    best_accuracy = 0.0
    best_loss = np.inf

    if custom_early_stopping:
        custom_early_stopping = custom_early_stopping

    tbar_epoch = tqdm(range(num_epochs), desc="Epochs")
    for epoch in tbar_epoch:

        if custom_early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        # Each epoch has a training and validation phase
        for phase in ["train", "validation"]:
            if phase == "train":
                custom_model.train()  # Set model to training mode
            else:
                custom_model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            tbar_train = tqdm(
                data_loader[phase], position=1, leave=False, desc=f"{phase}"
            )
            accuracies = []
            for inputs, labels in tbar_train:
                inputs = inputs.to(custom_device)
                labels = labels.to(custom_device)
                custom_optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = custom_model(inputs)
                    loss = custom_criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                    if phase == "train":
                        loss.backward()
                        custom_optimizer.step()

                temp_loss = loss.item() * inputs.size(0)
                running_loss += temp_loss
                temp_accuracy = torch.sum(preds == labels.data)
                accuracies.append(temp_accuracy.item() / inputs.size(0))
                tbar_train.set_postfix({"Accuracy": f"{np.mean(accuracies):.3%}"})
                running_corrects += temp_accuracy

            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_accuracy = running_corrects.double() / len(data_loader[phase].dataset)

            if phase == "validation" and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
            if phase == "validation" and epoch_loss < best_loss:
                best_loss = epoch_loss
                tbar_epoch.write(f"Saving model - best loss")
                tbar_epoch.set_postfix(
                    {f"{phase}_loss": epoch_loss, f"{phase}_accuracy": epoch_accuracy}
                )
                save_model(custom_save_path, epoch, custom_model, custom_optimizer, epoch_loss, epoch_accuracy)
            if phase == "train":
                train_accuracy_history.append(epoch_accuracy)
                train_loss_history.append(epoch_loss)
            if phase == "validation":
                validation_accuracy_history.append(epoch_accuracy)
                validation_loss_history.append(epoch_loss)

                if custom_scheduler:
                    custom_scheduler.step(epoch_loss)

                if custom_early_stopping:
                    custom_early_stopping(epoch_loss, custom_model)
                    if custom_early_stopping.early_stop:
                        print("Early stopping triggered!")
                        break
                log_data = log_data.append(
                    pd.DataFrame(
                        {
                            "epoch": [epoch],
                            "train_loss": [train_loss_history[-1]],
                            "validation_loss": [validation_loss_history[-1]],
                            "train_accuracy": [train_accuracy_history[-1].cpu()],
                            "validation_accuracy": [validation_accuracy_history[-1].cpu()],
                        }
                    )
                )
                log_data.to_csv(log_filename)

    checkpoint = torch.load(custom_save_path)
    print("Best validation Accuracy: {:4f}".format(checkpoint["accuracy"]))
    print(f"Logs saved to {log_filename}")

    custom_model.load_state_dict(checkpoint["model_state_dict"])

    return custom_model
