# ---- Deep Unfolded D-ADMM for limited communications ----
# --- Yoav Noah (yoavno@post.bgu.ac.il) and Nir Shlezinger (nirshl@bgu.ac.il) ---

import scipy.io as sio
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import networkx as nx
import sys


np.random.seed(1234)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.autograd.set_detect_anomaly(True)
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
direct_data = f'{__location__}/GenerateData/ProblemData'
mat_data = sio.loadmat(f'{direct_data}/CompressedSensing/GaussianData.mat')


""" 
Graph data initialization:
All the nodes in the graph need to be connected, 
therefore playing with the probability of the Erods-Rényi is important.
If the number of nodes is high then the probability can be low and 
if the number of nodes is low then the probability should be high.
"""

def graph2array(net1):
    edge_list = list(net1.edges)
    edge_dict = {}
    test_arr = []
    for i in range(1, len(edge_list)):
        if edge_list[i-1][0] < edge_list[i-1][1]:
            test_arr.append(edge_list[i-1][1])
        if edge_list[i][0] > edge_list[i-1][0]:
            n = edge_list[i-1][0]
            edge_dict[n] = test_arr
            n += 1
            test_arr = []
        if i == len(edge_list) - 1:
            test_arr.append(edge_list[i][1])
            n = edge_list[i][0]
            edge_dict[n] = test_arr
    for i in range(len(edge_list)):
            if edge_list[i][1] in edge_dict:
                edge_dict[edge_list[i][1]].append(edge_list[i][0])
            else:
                edge_dict[edge_list[i][1]] = [edge_list[i][0]]
    keys_list = list(edge_dict.keys())
    neighbs = []
    for idx in range(len(keys_list)):
        try:
            t = np.array(sorted(edge_dict[idx]), dtype='uint8')
        except KeyError:
            print("One or more nodes in the graph are not connected\n"
                  "Please increase the probability of the graph and run again")
            sys.exit()
        neighbs.append(t)
    neighbs = np.array(neighbs, dtype=object)
    return neighbs


def proper_coloring_algorithm(network):
    colors = [
        "lightcoral", "gray", "lightgray", "firebrick", "red", "chocolate", "darkorange", "moccasin", "gold", "yellow",
        "darkolivegreen", "chartreuse", "forestgreen", "lime", "mediumaquamarine", "turquoise", "teal", "cadetblue",
        "dogerblue", "blue", "slateblue", "blueviolet", "magenta", "lightsteelblue"]
    nodes = list(network.nodes())
    random.shuffle(nodes)  # random ordering
    color_dict = {}
    color_partition = []
    for node in nodes:
        dict_neighbors = dict(network[node])
        # gives names of nodes that are neighbors
        nodes_neighbors = list(dict_neighbors.keys())
        forbidden_colors = []
        for neighbor in nodes_neighbors:
            if len(network.nodes.data()[neighbor].keys()) != 0:  # if the neighbor has a color, this color is forbidden
                forbidden_color = network.nodes.data()[neighbor]
                forbidden_color = forbidden_color['color']
                forbidden_colors.append(forbidden_color)
        for color in colors:  # assign the first color that is not forbidden
            # start everytime at the top of the colors, so that the smallest number of colors is used
            if color not in forbidden_colors:
                color_dict[color] = node
                network.nodes[node]['color'] = color  # color one node at the time
                break
    color_dict = {}
    for v, data in network.nodes(data=True):
        if data['color'] in color_dict:
            color_dict[data['color']].append(v)
        else:
            color_dict[data['color']] = [v]
    colors_list = color_dict.keys()
    for idx in colors_list:
        t = np.array(color_dict[idx], dtype='uint8')
        color_partition.append(t)
    color_partition = np.array(color_partition, dtype=object)
    return color_partition


class NetworkVars:
    def __init__(self, P, neighbors, partition_colors):
        self.P = P
        self.neighbors = neighbors
        self.partition_colors = partition_colors

# P = 50
# graph_probability = 0.12  #Erdos-Rényi graph with probability 0.12

P = 5
graph_probability = 0.8

def net_vars(P, graph_prob):
    net1 = nx.erdos_renyi_graph(P, graph_prob) #Erdos-Rényi graph with probability 0.8
    color_partition = proper_coloring_algorithm(net1)
    neighbors = graph2array(net1)
    network_vars = NetworkVars(P, neighbors, color_partition)  # create a struct with the graph variables
    return network_vars


vars_network = net_vars(P, graph_probability)


"""---- Data initialization ----"""
class Vars:
    def __init__(self, A_BPDN, b_BPDN, m_p, P):
        self.A_BPDN = A_BPDN
        self.b_BPDN = b_BPDN
        self.m_p = m_p
        self.P = P


class DADMM(nn.Module):
    def __init__(self, hyp):
        super(DADMM, self).__init__()
        self.hyp = nn.Parameter(hyp)

    def forward(self, vars_prob, vars_network, MAX_ITER, labels, test):
        """
        Inputs:
         - vars_prob is a class that contains:
         # The sensing matrix A (500x2000)
         # The data inputs (observations)
         # m_p - splitting factor according to nodes number
         # P - No. of nodes

         - vars_network is a class containing information about the network:
         # P - No. of nodes
         # neighbors - array of size of the number of nodes P, and each entry i contains the neighbors of node i
         # color partition - array of the size of the number of (proper) colors of the network graph. Each
           entry contains a vector with the nodes that have that color.
           For example, for a network with 6 nodes,
           {[1,2], [3,4,5], 6} would mean that we have three colors:
           nodes 1 and 2 get the first color, 3, 4, and 5 the second,
           and node 6 gets the third color.
         - MAX_ITER - number of iterations of the D-ADMM algorithm

        """
        #--- initializing variables
        m, n = vars_prob.A_BPDN.shape
        P = vars_network.P
        m_p = m / P  # Number of rows of A that each node stores
        vars_prob.m_p = m_p
        X = torch.empty((P, batch_size, 2000, 1), dtype=torch.float64)
        U = torch.empty((P, batch_size, 2000, 1), dtype=torch.float64)
        # --- Network variables ---
        neighbors = vars_network.neighbors
        partiotion_colors = vars_network.partition_colors

        """Algorithm"""
        size = X.shape[1]
        n = X.shape[2]
        num_colors = len(partiotion_colors)
        loss_arr = []

        """DADMM algorithm"""
        for k in range(MAX_ITER):
            for color in range(num_colors):
                X_aux = X.clone()
                for p in partiotion_colors[color]:
                    neighbs = neighbors[p]
                    Dp = len(neighbs)
                    # --- Determine the sum of the X's of the neighbors ---
                    sum_neighbs = torch.zeros(size, n, 1)
                    for j in range(Dp):
                        sum_neighbs = sum_neighbs + X[neighbs[j]].clone().detach()
                    X_aux[p] = self.bpdn_rp_solver(p, U[p], sum_neighbs, Dp, X[p], vars_prob, k)
                X = X_aux.clone()

            # Output
            neighbs = neighbors[0]
            Dp = len(neighbs)
            # --- Determine the sum of the X's of the neighbors ---
            sum_neighbs = torch.zeros(size, n, 1)
            for j in range(Dp):
                sum_neighbs = sum_neighbs + X[neighbs[j]]
            U_ = [U[0].clone().detach() + (torch.abs(self.hyp[k][0][-1])) * (Dp * X[0] - sum_neighbs)]
            for pp in range(1, P):
                neighbs = neighbors[pp]
                Dp = len(neighbs)
                # --- Determine the sum of the X's of the neighbors ---
                sum_neighbs = torch.zeros(size, n, 1)
                for j in range(Dp):
                    sum_neighbs = sum_neighbs + X[neighbs[j]]
                U_.append(U[pp].clone().detach() + (torch.abs(self.hyp[k][pp][-1])) * (Dp * X[pp] - sum_neighbs))
            U = torch.stack(U_)
            if test:
                loss_ = 0
                mse_loss = nn.MSELoss()
                for ii in range(X.shape[0]):
                    for jj in range(labels.shape[0]):
                        loss_ += mse_loss(X[ii][jj].flatten(), labels[jj].flatten())
                loss = loss_ / (X.shape[0] * labels.shape[0])
                loss_arr.append(loss.item())
        return X, U, loss_arr

    def bpdn_rp_solver(self, p, U, sum_neighbs, Dp, X, vars_prob, k):
        A_full = vars_prob.A_BPDN
        b_full = vars_prob.b_BPDN
        m_p = vars_prob.m_p
        Ap = A_full[int(p * m_p): int((p + 1) * m_p)]
        bp = b_full[:, int(p * m_p): int((p + 1) * m_p)]
        return X.clone().detach() - (torch.abs(self.hyp[k][p][1])) * (Ap.T @ Ap @ X - Ap.T @ bp +
                                                                Dp * (torch.abs(self.hyp[k][p][0])) * X +
                                                                (torch.abs(self.hyp[k][p][2])) * torch.sign(X) + Dp * U -
                                                                (torch.abs(self.hyp[k][p][0])) * sum_neighbs)


# ---- Data generalization ----
class SimulatedData(Dataset):
    def __init__(self, idx):
        lst1 = []
        lst2 = []
        dataset = np.load('dataset.npy', allow_pickle=True)  # Loading the dataset with 0 SNR
        # dataset = np.load('dataset_2_snr.npy', allow_pickle=True)  # Loading the dataset with 2 SNR
        for x, y in dataset:
            lst1.append(x)
            lst2.append(y)
        data, label = np.array(lst1), np.array(lst2)
        data = data[:1200]
        label = label[:1200]
        if idx >= 0.7 * data.shape[0]:
            self.x = torch.from_numpy(data[:idx, :, :])
            self.y = torch.from_numpy(label[:idx, :, :])
        else:
            self.x = torch.from_numpy(data[-idx:, :, :])
            self.y = torch.from_numpy(label[-idx:, :, :])
        self.samples_num = idx

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.samples_num


def error(x_opt, X):
    loss_ = 0
    mse_loss = nn.MSELoss()
    for ii in range(X.shape[0]):
        for jj in range(x_opt.shape[0]):
            loss_ += mse_loss(X[ii][jj].flatten(), x_opt[jj].flatten())
    loss = loss_ / (X.shape[0] * x_opt.shape[0])
    return loss

def train_model(model, train_dataset, valid_dataset, num_epochs, batch_size, mat_data):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, betas=(0.9, 0.999), weight_decay=0.06, amsgrad=False)
    train_losses, valid_losses = np.zeros((num_epochs,)), np.zeros((num_epochs,))
    for epoch in range(num_epochs):
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        model.train()
        train_loss = 0
        train_loss_iter, valid_loss_iter = [], []
        for iter, (inputs, labels) in enumerate(train_loader):
            vars_prob = Vars(torch.from_numpy(mat_data['A_BP']), inputs, 0.0, P)
            X, __, _ = model(vars_prob, vars_network, num_iter, labels, test=False)
            loss = error(labels, X)
            optimizer.zero_grad()
            print(f'Epoch No. {epoch}; iteration No.: {iter}, Loss: {loss}')
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            train_loss_iter.append(loss.data.item())
        train_losses[epoch] = (train_loss / len(train_loader.dataset))
        with torch.no_grad():
            val_loss = 0
            for iter, (inputs, labels) in enumerate(valid_loader):
                vars_prob = Vars(torch.from_numpy(mat_data['A_BP']), inputs, 0.0, P)
                X_val, __, _ = model(vars_prob, vars_network, num_iter, labels, test=False)
                valid_loss = error(labels, X_val)
                val_loss += valid_loss.data.item()
                valid_loss_iter.append(valid_loss.data.item())
            valid_losses[epoch] = (val_loss / len(valid_loader.dataset))
        if epoch % 2 == 0:
            print(
                "Epoch %d, Train loss %.8f, Validation loss %.8f"
                % (epoch, train_losses[epoch], valid_losses[epoch])
            )

    return train_losses, valid_losses, train_loss_iter, valid_loss_iter


""" --- Main --- """

#   Initializations
batch_size = 100
train_size = 1000
test_size = 200
num_epochs = 100

# --- Data generalization ---
train_ds = SimulatedData(train_size)
train_dataset, valid_dataset = torch.utils.data.random_split(train_ds, [int(0.8 * train_size), int(0.2 * train_size)], generator=torch.Generator().manual_seed(42))
test_dataset = SimulatedData(test_size)

# --- D-ADMM parameters
num_iter = 25
learn_params = torch.tensor([[0.2603, 0.2713, 0.1142, 0.0867]] * P, requires_grad=True).unsqueeze(0).repeat([num_iter, 1, 1])
# --- Classical D-ADMM
classic_dadmm = DADMM(learn_params)
classic_dadmm.eval()
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
vars_prob = Vars(torch.from_numpy(mat_data['A_BP']), test_loader.dataset.x[:batch_size], 0.0, P)
__, __, dadmm_loss_arr = classic_dadmm.forward(vars_prob, vars_network, num_iter, test_loader.dataset.y[:batch_size], test=True)
y_dadmm = np.array(dadmm_loss_arr)
x_dadmm = np.array(list(range(len(dadmm_loss_arr))))


plt.figure()
plt.plot(x_dadmm, y_dadmm, '-b', label='Classic D-ADMM loss')
plt.grid()
plt.title(f'Loss Curve, Num of DADMM iterations = {num_iter}')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()

#  --- Unfolded D-ADMM

model = DADMM(learn_params)
train_losses, valid_losses, train_loss_iter, valid_loss_iter = train_model(model, train_dataset, valid_dataset, num_epochs, batch_size, mat_data)


"""Plot the learning curves"""
y_t = np.array(train_losses) * batch_size
x_t = np.array(list(range(len(train_losses))))
y_v = np.array(valid_losses) * batch_size
x_v = np.array(list(range(len(valid_losses))))
plt.figure()
plt.plot(x_t, y_t, 'b--', label='Train')
plt.plot(x_v, y_v, '-*', label='Valid')
plt.grid()
plt.title(f'Loss Curve, Num Epochs = {num_epochs}, Batch Size = {batch_size} \n Num of Iterations of DADMM = {num_iter}')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.show()

#--- Unfolded D-ADMM on the test set ---
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
model.eval()
test_los = 0
test_loss_iter, test_losses = [], []
for iter, (inputs, labels) in enumerate(test_loader):
    vars_prob = Vars(torch.from_numpy(mat_data['A_BP']), inputs, 0.0, P)
    X_test, __, test_loss_arr = model(vars_prob, vars_network, num_iter, labels, test=True)
    test_loss = error(labels, X_test)
    test_los += test_loss.data.item()
    valid_loss_iter.append(test_loss.data.item())
    test_losses.append((test_los / len(test_loader.dataset)))

y_test = test_loss_arr
x_test = np.array(list(range(len(test_loss_arr))))


plt.figure()
plt.plot(x_dadmm, y_dadmm, '-b', label='Classic D-ADMM loss')
plt.grid()
plt.title(f'Loss Curve, Num of DADMM iterations = {num_iter}')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()
