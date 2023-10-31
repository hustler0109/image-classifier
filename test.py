import argparse
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cifar import cifar10_dataset
from data.transformations import get_test_transformations
from download.unpickle_cifar10 import unpickle_cifar10_test
from model import CIFAR10Model
from utils import load_weights

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = False
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True  # Helps optimize training w/ GPU
    PIN_MEMORY = True  # use this only w/ GPU


def main(args):
    # Unpickle the CIFAR-10 dataset
    images, labels = unpickle_cifar10_test()

    test_dataset = cifar10_dataset(
        images, labels, transforms=get_test_transformations()
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = CIFAR10Model(num_classes=10)
    model.to(DEVICE)
    load_weights(model, "model.pth", device=DEVICE)

    # Evaluate the model on the test set
    evaluate(model, test_loader)


def evaluate(model, dataloader):
    """Evaluate the model on the test set"""
    correct = 0
    total = 0
    with torch.no_grad():
        tbar = tqdm(dataloader, desc="Evaluating", unit="batch")
        for images, labels in tbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            tbar.set_postfix({"acc": correct / total})
    print(f"Accuracy of the network on the {total} test images: {correct / total:.2%}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
