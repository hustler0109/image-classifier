import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        pil_img = Image.fromarray(img).convert("RGB")
        label = self.labels[idx]

        if self.transforms:
            pil_img = self.transforms(pil_img)

        img_tensor = torch.tensor(np.array(pil_img), dtype=torch.float).permute(2, 0, 1) / 255.0  # Convert to tensor and normalize

        if label is None:
            return img_tensor
        return img_tensor, torch.tensor(label, dtype=torch.long)
