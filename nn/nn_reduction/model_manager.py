import torch


def save_model(model, path=''):
    if not path:
        path = 'model.pth'
    torch.save(model.state_dict(), path)
    return path


def load_model(path):
    if not path:
        return None
    return torch.load(path)

