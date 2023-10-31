import argparse
import warnings

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data import cifar10_dataset
from data.transformations import get_test_transformations, get_train_transformations
from download.unpickle_cifar10 import unpickle_cifar10, unpickle_cifar10_test
from model import CIFAR10Model
from trainer import EarlyStopping, train_model

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = False
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True  # Helps optimize training w/ GPU
    PIN_MEMORY = True  # use this only w/ GPU


def main(args):
    images, labels = unpickle_cifar10()

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.1, random_state=713, stratify=labels
    )

    train_dataset = cifar10_dataset(
        X_train, y_train, transforms=get_train_transformations()
    )
    val_dataset = cifar10_dataset(X_val, y_val, transforms=get_test_transformations())

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            pin_memory=PIN_MEMORY,
            num_workers=4,
        ),
        "val": DataLoader(
            val_dataset, batch_size=64, pin_memory=PIN_MEMORY, num_workers=4
        ),
    }

    model = CIFAR10Model(num_classes=10)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(
        model.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.9
    )
    l1_scheduler = lr_scheduler.ReduceLROnPlateau(optim, patience=5)
    trained_model = train_model(
        model=model,
        dataloader=dataloaders,
        criterion=criterion,
        optimizer=optim,
        device=DEVICE,
        save_path="model.pth",
        num_epochs=50,
        scheduler=l1_scheduler,
        early_stopping=EarlyStopping(patience=10),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args)
