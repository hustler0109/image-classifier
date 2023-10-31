import torch


def load_weights(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device)["model_state_dict"])
    model.eval()
    return model
