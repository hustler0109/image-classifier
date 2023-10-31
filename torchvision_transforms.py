from torchvision import transforms

def get_train_transformations():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop((224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.550, 0.470, 0.321],  
                std=[0.211, 0.355, 0.482],  
            ),
        ]
    )

def get_test_transformations():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.550, 0.470, 0.321], std=[0.211, 0.355, 0.482]),
        ]
    )
