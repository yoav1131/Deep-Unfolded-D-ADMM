import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Linear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.linear = torch.nn.Linear(input_size, output_size, bias=True)
        # np.random.seed(1234)
        self.linear.bias.data.fill_(np.random.randint(10) * np.random.rand())
        torch.nn.init.xavier_normal_(self.linear.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = self.linear(x)
        return out


class FC2Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FC2Layer, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

