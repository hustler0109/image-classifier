import argparse
import random
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.cifar import cifar10_dataset
from data.transformations import get_test_transformations
from download.unpickle_cifar10 import unpickle_cifar10_test
from model.model import CIFAR10Model
from utils import load_weights

warnings.filterwarnings("ignore")

CLASS_CONVERT_MAP = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = False
if DEVICE == torch.device("cuda"):
    torch.backends.cudnn.benchmark = True  
    PIN_MEMORY = True 


def read_image(fp: str) -> np.ndarray:
    """
    Reads an image from a file path and returns it as a numpy array.
    """
    img = Image.open(fp).convert("RGB")
    return np.array(img)


def main(args):
    if not args.fp:
        images, _ = unpickle_cifar10_test()
        i = random.randint(0, len(images) - 1)
        Image.fromarray(images[i]).save("docs/prediction.png")
        images = [images[i]]
    else:
        images = [read_image(args.fp)]

    test_dataset = cifar10_dataset(
        images, [None], transforms=get_test_transformations()
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = CIFAR10Model(num_classes=10)
    model.to(DEVICE)
    load_weights(model, "model.pth", device=DEVICE)

    # Evaluate the model on the test set
    for image in test_loader:
        with torch.no_grad():
            image = image.to(DEVICE)
            output = model(image)
            _, predicted = torch.max(output, 1)
            print("Most possible class:", CLASS_CONVERT_MAP[predicted.item()])
            probs = torch.softmax(output, dim=1)[0]
            print(
                "Probabilities:",
                {CLASS_CONVERT_MAP[i]: f"{probs[i].item():.2f}" for i in range(10)},
            )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="model.pth")
    argparser.add_argument("--fp", type=str)
    args = argparser.parse_args()
    main(args)
