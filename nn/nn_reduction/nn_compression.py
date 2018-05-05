#!/usr/bin/env python3

from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils.flops_benchmark import add_flops_counting_methods

import model_data
import model_manager as mm
from model_optimizer import Optimizer, init_optimizer_globals
from model_decisioner import init_decisioner_globals


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

parser.add_argument('--opt-func',
                    nargs='?',
                    default=0, const=0,
                    type=int,
                    choices=[0, 1, 2, 3],
                    help='Different optimization functions')
parser.add_argument('--epsilon', type=float, default=0.07,
                    help='Epsilon between baseline accuracy and reduction algo accuracy')

parser.add_argument('--dataset', type=str, default='mnist',
                    help='Dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)


train_loader = None
test_loader = None
network_init_callable = None
network_init_params = None
optimization_data = None


def train(model, optimizer, epoch=0):
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


########################################################################################################################
def main_train():
    model = network_init_callable(*network_init_params)
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


def main_load():
    print('Loaded from: {0}'.format(model_file_path))
    data = mm.load_model(model_file_path)
    model_params = [v.size()[0] for k, v in data.items() if 'bias' in k]
    model = network_init_callable(*model_params)
    model.load_state_dict(data)
    model = add_flops_counting_methods(model)
    model.start_flops_count()
    test(model)
    print('-'*100)
    print(model)
    print('FLOPS:', model.compute_average_flops_cost())


def main_optimize():
    print('-' * 100)
    print('Loaded from: {0}'.format(model_file_path))
    print('-' * 100)
    data = mm.load_model(model_file_path)
    params = network_init_params
    model = network_init_callable(*params)
    model.load_state_dict(data)
    model = add_flops_counting_methods(model)
    model.start_flops_count()
    print('BASELINE')
    baseline = test(model)
    print(model)
    baseline_flops = model.compute_average_flops_cost()
    print('FLOPS:', baseline_flops)

    model_optimizer = Optimizer(
        network_init_callable,
        baseline,
        baseline_flops,
        params,
        args.opt_func,
        args.epsilon
    )
    opt_params, opt_data = model_optimizer.optimize(data, optimization_data)
    print('-'*100)
    print('OPTIMIZATION')
    print('Got params:', opt_params)

    optimized_model = network_init_callable(*opt_params)
    optimized_model.load_state_dict(opt_data)
    optimized_model = add_flops_counting_methods(optimized_model)
    optimized_model.start_flops_count()
    test(optimized_model)
    print(optimized_model)
    print('FLOPS:', optimized_model.compute_average_flops_cost())
    optimized_model_file_path = 'optim_{0}'.format(model_file_path)
    print('-'*100)
    print('Saved to:', optimized_model_file_path)
    mm.save_model(optimized_model, optimized_model_file_path)


if __name__ == "__main__":
    model_data.init_model_data_globals(args)
    data = model_data.get_data(args.dataset)
    network_init_callable = data['init_callable']
    network_init_params = data['init_params']
    train_loader = data['train_loader']
    test_loader = data['test_loader']
    optimization_data = data['optimization_data']
    if args.create_model:
        main_train()
    if args.load_model:
        main_load()
    if args.opt_model:
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        init_optimizer_globals(
            (
                train,
                optim.SGD,
                {'lr': args.lr, 'momentum': args.momentum},
                int(args.epochs / 10)
            ),
            test)
        init_decisioner_globals(train, test)
        main_optimize()
