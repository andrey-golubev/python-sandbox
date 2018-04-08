from __future__ import print_function
import argparse
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils.flops_benchmark import add_flops_counting_methods

import model_manager as mm


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--create-model', action='store_true')
parser.add_argument('--load-model', action='store_true')
parser.add_argument('--opt-model', action='store_true')

parser.add_argument('file', nargs='?', default='mnist_0.pth', help='File containing saved model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Network(nn.Module):
    def __init__(self, conv1_out_size, conv2_out_size, fc1_out, fc2_out):
        self._out_sizes = {
            'conv1': conv1_out_size,
            'conv2': conv2_out_size,
            'fc1': fc1_out,
            'fc2': fc2_out
        }
        super(Network, self).__init__()
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


def train(model, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data[0]))


def test(model, printing=True):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    if printing:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 1. * correct / len(test_loader.dataset)


model_file_path = args.file  # 'mnist_0.pth'


def make_model():
    model = Network(10, 20, 50, 10)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model = add_flops_counting_methods(model)
    if args.cuda:
        model.cuda()

    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, epoch)
        pass

    # consider flops counting on test forward pass
    model.start_flops_count()
    print('-'*100)
    test(model)
    print('FLOPS:', model.compute_average_flops_cost())
    print('Model:', model)
    mm.save_model(model, model_file_path)
    print('Saved to: {0}'.format(model_file_path))


########################################################################################################################
def main_train():
    make_model()


def main_load():
    print('Loaded from: {0}'.format(model_file_path))
    data = mm.load_model(model_file_path)
    model = Network(10, 20, 50, 10)
    model.load_state_dict(data)
    model = add_flops_counting_methods(model)
    model.start_flops_count()
    test(model)
    print('-'*100)
    print(model)
    print('FLOPS:', model.compute_average_flops_cost())


class Optimizer:
    def __init__(self, model_initializer, baseline_acc, params):
        self._init_callable = model_initializer
        self._baseline_acc = baseline_acc
        self._baseline_params = params
        self._best_score = (0.0, math.inf)
        self._best_params = None
        self._best_state_dict = None
        self._epsilon = 0.07

    def _decision_function(self, score1, score2):
        b_acc, _ = self._baseline_acc
        (acc1, flops1), (acc2, flops2) = score1, score2
        bad1, bad2 = False, False
        if math.abs(acc1 - b_acc) > self._epsilon:
            bad1 = True
        if math.abs(acc2 - b_acc) > self._epsilon:
            bad2 = True
        if bad1 and bad2:
            return self._best_score, self._best_params
        if bad1 or flops2 < flops1:
            return score2
        if bad2 or flops1 < flops2:
            return score1

    def _run_test(self, state, model_params):
        model = self._init_callable(*model_params)
        model.load_state_dict(state)
        model = add_flops_counting_methods(model)
        model.start_flops_count()
        return test(model, printing=False), model.compute_average_flops_cost()

    @staticmethod
    def _reduce_state_opt(orig_state, indices, size_pos):
        new_size = list(orig_state.size())
        new_size[size_pos] = len(indices)
        state = torch.FloatTensor(*new_size)
        for i, target_i in enumerate(indices):
            state[i] = orig_state[target_i]
        return state

    @staticmethod
    def _reduce_state_chg(orig_state, indices, size_pos):
        new_size = list(orig_state.size())
        new_size[size_pos] = len(indices)
        state = torch.FloatTensor(*new_size)
        for row, orig_row in zip(state, orig_state):
            for i, target_i in enumerate(indices):
                row[i] = orig_row[target_i]
        return state

    @staticmethod
    def _update_state(orig_state, indices, layer_names):
        opt_layer, chg_layer = layer_names
        opt_weight_s = '{0}.weight'.format(opt_layer)
        opt_bias_s = '{0}.bias'.format(opt_layer)
        chg_weight_s = '{0}.weight'.format(chg_layer)
        state = copy.deepcopy(orig_state)
        state[opt_weight_s] = Optimizer._reduce_state_opt(
            state[opt_weight_s],
            indices,
            0
        )
        state[opt_bias_s] = Optimizer._reduce_state_opt(
            state[opt_bias_s],
            indices,
            0
        )
        state[chg_weight_s] = Optimizer._reduce_state_chg(
            state[chg_weight_s],
            indices,
            1
        )
        return state


    # def optimize(self, model_state_dict, layers_to_optimize, layers_to_adjust):
    #     for layer_to_opt, layer_to_adjust in zip(layers_to_optimize, layers_to_adjust):
    #         local_state_dict = copy.deepcopy(model_optim_state_dict)
    #         local_params = list(range(0, len(local_state_dict['{0}.weights'.format(layer_to_opt)])))
    #         test_score = self._run_test(local_state_dict, local_params)
    #         decided_score = self._decision_function(
    #             self._best_score,
    #             test_score
    #         )

    #         # score actually changed:
    #         if self._best_score != decided_score:
    #             self._best_score = decided_score
    #             self._best_params = local_params
    #             self._best_state_dict = local_state_dict

    #     if self._best_params is None or self._best_state_dict is None:
    #         print('Could not optimized with gived data')
    #         return self._baseline_params, model_state_dict

    #     return self._best_params, self._best_state_dict

    def optimize(self, model_state_dict, optimization_input):
        for (opt_layer, chg_layer, p_index) in optimization_input:
            opt_weight_s = '{0}.weight'.format(opt_layer)
            opt_bias_s = '{0}.bias'.format(opt_layer)
            chg_weight_s = '{0}.weight'.format(chg_layer)
            local_state = model_state_dict
            indices = list(range(0, len(local_state[opt_weight_s])))

            # 1: calculate metrics per index
            metric_per_param = []
            for index in indices:
                # changing state:
                # state = copy.deepcopy(local_state)
                # opt_weight = np.array(state[opt_weight_s][index:index+1])
                # opt_bias = np.array(state[opt_bias_s][index:index+1])
                # chg_weight = np.array(state[chg_weight_s])
                # chg_weight = np.array([row[index:index+1] for row in chg_weight])

                # state[opt_weight_s] = torch.from_numpy(opt_weight)
                # state[opt_bias_s] = torch.from_numpy(opt_bias)
                # state[chg_weight_s] = torch.from_numpy(chg_weight)
                state = Optimizer._update_state(
                    local_state,
                    [index],
                    (opt_layer, chg_layer)
                )

                params = list(self._baseline_params)
                params[p_index] = 1
                acc, flops = self._run_test(state, params)
                metric_per_param.append((index, acc, flops))

            # 2: sort metrics
            metric_per_param = list(reversed(sorted(metric_per_param, key=lambda e: e[1])))

            # 3: reduction
            last_viable_params = self._baseline_params
            last_viable_state = local_state
            for limit in reversed(indices[1:len(indices)-1]):
                sln_indices = [v[0] for v in metric_per_param[:limit]]
                state = Optimizer._update_state(
                    local_state,
                    sln_indices,
                    (opt_layer, chg_layer)
                )
                params = list(self._baseline_params)
                params[p_index] = len(sln_indices)
                acc, flops = self._run_test(state, params)
                if (self._baseline_acc - acc) > self._epsilon:
                    break
                last_viable_params = params
                last_viable_state = state

            # 4: decision
            return last_viable_params, last_viable_state


def main_optimize():
    print('-' * 100)
    print('Loaded from: {0}'.format(model_file_path))
    print('-' * 100)
    data = mm.load_model(model_file_path)
    params = (10, 20, 50, 10)
    model = Network(*params)
    model.load_state_dict(data)
    model = add_flops_counting_methods(model)
    model.start_flops_count()
    print('BASELINE')
    baseline = test(model)
    print(model)
    print('FLOPS:', model.compute_average_flops_cost())

    model_optimizer = Optimizer(Network, baseline, params)
    opt_params, opt_data = model_optimizer.optimize(data, [('conv1', 'conv2', 0)])
    print('-'*100)
    print('OPTIMIZATION')
    print('Got params:', opt_params)

    optimized_model = Network(*opt_params)
    optimized_model.load_state_dict(opt_data)
    optimized_model = add_flops_counting_methods(optimized_model)
    optimized_model.start_flops_count()
    # print('Found reduced model')
    # print('Params:', opt_params)
    test(optimized_model)
    print(optimized_model)
    print('FLOPS:', optimized_model.compute_average_flops_cost())


if __name__ == "__main__":
    if args.create_model:
        main_train()
    if args.load_model:
        main_load()
    if args.opt_model:
        main_optimize()
