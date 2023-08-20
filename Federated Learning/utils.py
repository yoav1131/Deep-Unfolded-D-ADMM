import os
import torch
from statistics import mean
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import numpy as np

def data(args):
    if args.data == 'mnist':
        train_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True),
            batch_size=args.test_batch_size, shuffle=False)
    return train_data, test_loader


def data_split(data, amount, args):
    # split train, validation
    train_data, val_data = torch.utils.data.random_split(data, [len(data) - amount, amount])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False)

    # input, output sizes
    in_channels, dim1, dim2 = data[0][0].shape  # images are dim1 x dim2 pixels
    input = dim1 * dim2 if args.model == 'mlp' or args.model == 'linear' else in_channels
    output = data[0][0].shape[0]

    return input, output, train_data, val_loader


def train_one_epoch(train_loader, model, optimizer, creterion, device, iterations):
    model.train()
    losses = []
    correct = 0
    acc = []
    if iterations is not None:
        local_iteration = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # send to device
        data, label = data.to(device), label.to(device).to(torch.float32)
        output = model(data)
        loss = creterion(torch.abs(output.flatten()), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for idx in range(output.shape[0]):
            if torch.abs(output[idx]) % 1 > 0.8 or torch.abs(output[idx]) % 1 < 0.2:
                correct += (torch.round(torch.abs(output[idx].flatten())) == label[idx].flatten()).item()

        accuracy = correct / output.shape[0]
        losses.append(loss.item())
        acc.append(accuracy)

        if iterations is not None:
            local_iteration += 1
            if local_iteration == iterations:
                break
    return mean(losses), mean(acc) * 100


def test(test_loader, model, creterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)  # send to device

        output = model(data)
        test_loss += creterion(torch.abs(output.flatten()), label).item()  # sum up batch loss
        for idx in range(output.shape[0]):
            if torch.abs(output[idx]) % 1 > 0.8 or torch.abs(output[idx]) % 1 < 0.2:
                correct += (torch.round(torch.abs(output[idx].flatten())) == label[idx].flatten()).item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy, test_loss


def initializations(args):
    #  reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    #  documentation
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists(f'checkpoints/{args.num_users}'):
        os.makedirs(f'checkpoints/{args.num_users}')
    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')

    best_val_acc = np.NINF
    path_best_model = 'checkpoints/' + args.exp_name + '/model.best.t7'

    return boardio, textio, best_val_acc, path_best_model


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()
