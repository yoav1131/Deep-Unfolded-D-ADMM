import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import LoadData
import numpy as np
import DADMM_utils


def data(args):
    if args.data == 'mnist':
        train_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False),
            batch_size=args.batch_size * args.P, shuffle=False)
    else:
        train_data = LoadData.SimulatedData(args.train_size, args.snr)
        test_loader = LoadData.SimulatedData(args.test_size, args.snr)
        test_loader = torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return train_data, test_loader

def data_split(dataset, args):
    # split train, validation
    if args.case == 'dlr':
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset,
                                                                     [int(0.8 * dataset.train_data.shape[0]),
                                                                      int(0.2 * dataset.train_data.shape[0])],
                                                                     generator=torch.Generator().manual_seed(args.seed))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size * args.P, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size * args.P, shuffle=False, drop_last=True)
    else:
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset,
                                                                     [int(0.8 * args.train_size),
                                                                      int(0.2 * args.train_size)],
                                                                     generator=torch.Generator().manual_seed(args.seed))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader

def train_one_epoch(model, train_loader, neighbors, color_partition, args):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, amsgrad=False)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.train()
    train_accuracy = 0
    train_loss_iter, valid_loss_iter = [], []
    valid_acc_iter, train_acc_iter = [], []
    for iter, (inputs, labels) in enumerate(train_loader):
        if args.case == 'dlr':
            inputs = inputs.reshape(args.P, args.batch_size, 28 * 28, 1)
            inputs = (inputs / 255).type(torch.double).to(args.device)
            labels = labels.type(torch.double).to(args.device)
            labels = labels.reshape(args.P, int(labels.shape[0] / args.P), 1, 1)
            vars_prob = DADMM_utils.Vars(inputs, 0.0)
            a, omega, _, __ = model(vars_prob, neighbors, color_partition, args, labels, test=False)
            y_hat_train = (torch.transpose(a, 2, 3) @ inputs + omega)[:, :, 0]
            loss = error(labels, y_hat_train, args)
            acc = accuracy(labels, y_hat_train)
            train_acc_iter.append(acc)
        else:
            vars_prob = DADMM_utils.Vars(inputs, 0.0)
            X, __, _ = model(vars_prob, neighbors, color_partition, args, labels, test=False)
            loss = error(labels, X, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_iter.append(loss.data.item())
        if args.lr_scheduler:
            scheduler.step(loss)

    train_loss = np.mean(np.array(train_loss_iter))
    if args.case == 'dlr':
        train_accuracy = np.mean(np.array(train_acc_iter))
    return train_loss, train_accuracy


def test(test_loader, model, neighbors, color_partition, args):
    model.eval()
    for iter, (inputs, labels) in enumerate(test_loader):
        if args.case == 'dlr':
            inputs = inputs.reshape(args.P, args.batch_size, 28 * 28, 1)
            inputs = (inputs / 255).type(torch.double).to(args.device)
            labels = labels.type(torch.double).to(args.device)
            labels = labels.reshape(args.P, args.batch_size, 1, 1)
            vars_prob = DADMM_utils.Vars(inputs, 0.0)
        else:
            vars_prob = DADMM_utils.Vars(inputs, 0.0)
        if args.valid:
            if args.case == 'dlr':
                a_test, omega_test, __, _ = model(vars_prob, neighbors, color_partition, args, labels, False)
                y_hat_val = (torch.transpose(a_test, 2, 3) @ vars_prob.inputs + omega_test)[:, :, 0]
                test_loss_arr = error(labels, y_hat_val, args)
                test_acc_arr = accuracy(labels, y_hat_val)
            else:
                X_val, __, _ = model(vars_prob, neighbors, color_partition, args, labels, False)
                test_loss_arr = error(labels, X_val, args)
                test_acc_arr = 0
        else:
            if args.case == 'dlr':
                _, __, test_loss_arr, test_acc_arr = model(vars_prob, neighbors, color_partition, args, labels, True)
            else:
                _, __, test_loss_arr = model(vars_prob, neighbors, color_partition, args, labels, True)
                test_acc_arr = 0
        return test_loss_arr, test_acc_arr

def error(labels, y_hat, args):
    loss_ = 0
    mse_loss = nn.MSELoss()
    for ii in range(y_hat.shape[0]):
        for jj in range(y_hat.shape[1]):
            if args.case == 'dlr':
                loss_ += mse_loss(torch.abs(y_hat[ii][jj].flatten()), labels[ii][jj].flatten())
            else:
                loss_ += mse_loss(torch.abs(y_hat[ii][jj].flatten()), labels[jj].flatten())
    loss = loss_ / (y_hat.shape[0] * y_hat.shape[1])
    return loss

def accuracy(labels, y_hat):
    correct = 0
    for ii in range(y_hat.shape[0]):
        for idx in range(y_hat.shape[1]):
            if torch.abs(y_hat[ii][idx]) % 1 > 0.8 or torch.abs(y_hat[ii][idx]) % 1 < 0.2:
                correct += (torch.round(torch.abs(y_hat[ii][idx].flatten())) == labels[ii][idx].flatten()).item()
    accur = 100 * correct / (y_hat.shape[0] * y_hat.shape[1])
    return accur

def init_best_path(args):
    return f'checkpoints/{args.exp_name}/model.seq_num_{args.seq_num}.best.t7'
def initializations(args):
    #  reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Graph documentation
    if not os.path.isdir(f'graphs data'):
        os.mkdir('graphs data')
    if not os.path.isdir(f'graphs data/{args.graph_type}'):
        os.mkdir(f'graphs data/{args.graph_type}')


    # documentation
    if not os.path.isdir(f'checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.isdir(f'checkpoints/{args.case}'):
        os.mkdir(f'checkpoints/{args.case}')
    if not os.path.isdir(f'checkpoints/{args.case}/{args.exp_name}'):
        os.mkdir(f'checkpoints/{args.case}/{args.exp_name}')
    if not os.path.isdir(f'checkpoints/{args.case}/{args.exp_name}/{args.model}'):
        os.mkdir(f'checkpoints/{args.case}/{args.exp_name}/{args.model}')
    if not os.path.isdir(f'checkpoints/{args.case}/{args.exp_name}/{args.model}/results'):
        os.mkdir(f'checkpoints/{args.case}/{args.exp_name}/{args.model}/results')

    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')

    best_val_loss = np.PINF


    return boardio, textio, best_val_loss

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()