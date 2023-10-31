import pickle
import numpy as np

def unpickle(file_path: str) -> dict:
    with open(file_path, "rb") as file:
        data_dict = pickle.load(file, encoding="bytes")
    return data_dict

def load_cifar10_data(verbose: bool = False):
    num_batches = 5
    batch_size = 10000
    total_samples = num_batches * batch_size
    
    images = np.empty((total_samples, 32, 32, 3), dtype=np.uint8)
    labels = np.empty((total_samples,))
    counter = 0
    
    for i in range(1, num_batches + 1):
        batch_data = unpickle(f"./cifar-10-batches-py/data_batch_{i}")
        batch_images = batch_data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        batch_labels = np.array(batch_data[b"labels"])
        
        images[counter : counter + batch_size] = batch_images
        labels[counter : counter + batch_size] = batch_labels
        counter += batch_size
    
    if verbose:
        print(f"{num_batches} batches unpickled")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        
    return images, labels

def load_cifar10_test_data(verbose: bool = False):
    test_data = unpickle(f"./cifar-10-batches-py/test_batch")
    test_images = test_data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = np.array(test_data[b"labels"])
    
    if verbose:
        print(f"Test Images shape: {test_images.shape}")
        print(f"Test Labels shape: {test_labels.shape}")
        
    return test_images, test_labels

if __name__ == "__main__":
    load_cifar10_data(verbose=True)
    load_cifar10_test_data(verbose=True)
