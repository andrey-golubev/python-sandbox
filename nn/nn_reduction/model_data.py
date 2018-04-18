import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


def init_model_data_globals(parser_args):
    global args
    args = parser_args


class MnistNetwork(nn.Module):
    def __init__(self, conv1_out_size, conv2_out_size, fc1_out, fc2_out):
        self._out_sizes = {
            'conv1': conv1_out_size,
            'conv2': conv2_out_size,
            'fc1': fc1_out,
            'fc2': fc2_out
        }
        super(MnistNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_out_size, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_out_size, conv2_out_size, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4*4*conv2_out_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 4*4*self._out_sizes['conv2'])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def train_loader():
        return torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True)

    @staticmethod
    def test_loader():
        return torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True)


def get_data(key):
    """Model data by dataset"""
    get_data.data = {
        'mnist': {
            'init_callable': MnistNetwork,
            'init_params': (10, 20, 50, 10),
            'train_loader': MnistNetwork.train_loader(),
            'test_loader': MnistNetwork.test_loader(),
            'optimization_data': [
                ('conv1', 'conv2', 0),
                ('conv2', 'fc1', 1)
            ],
        }
    }
    return get_data.data[key]
