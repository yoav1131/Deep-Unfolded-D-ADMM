# ---- Deep Unfolded D-ADMM for limited communications ----
# --- Yoav Noah (yoavno@post.bgu.ac.il) and Nir Shlezinger (nirshl@bgu.ac.il) ---
import gc
import psutil
import scipy.io as sio
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import random
import networkx as nx
import sys


torch.manual_seed(10)
np.random.seed(10)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = f'{os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))}/model_weights'
checkpoint_path = f'{path}/four_nodes/new weights''/ckpt-{}.pk'

""" 
Graph data initialization:
All the nodes in the graph need to be connected, 
therefore playing with the probability of the Erods-Rényi is important.
If the number of nodes is high then the probability can be low and 
if the number of nodes is low then the probability should be high.
"""

def graph2array(net1):
    global P
    edge_list = list(net1.edges)
    edge_dict = {}
    test_arr = []
    #---- For P > 2!!! ----
    if P > 2:
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
    # ---- For P=2 ----
    else:
        edge_dict[edge_list[0][0]] = [edge_list[0][1]]
        edge_dict[edge_list[0][1]] = [edge_list[0][0]]
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


def net_vars(P, graph_prob):
    net1 = nx.erdos_renyi_graph(P, graph_prob) #Erdos-Rényi graph with probability 0.8
    # nx.draw_networkx(net1)
    # plt.show()
    color_partition = proper_coloring_algorithm(net1)
    # color_partition = [np.array(range(P))]
    neighbors = graph2array(net1)
    network_vars = NetworkVars(P, neighbors, color_partition)  # create a struct with the graph variables
    return network_vars





"""---- Data initialization ----"""
class Vars:
    def __init__(self, inputs, m_p, P):
        # self.a = a
        # self.omega = omega
        self.inputs = inputs
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
         # The sensing matrix A
         # The bias omega
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
        m = vars_prob.inputs.shape[0]
        P = vars_network.P
        m_p = m / P  # Number of rows of A that each node stores
        # vars_prob.m_p = m_p
        mu = torch.empty((P, int(batch_size/P), 28*28, 1), dtype=torch.float64).to(device)  # Dual variable of a
        lamda = torch.empty((P, int(batch_size/P), 1, 1), dtype=torch.float64).to(device)  # Dual variable of omega
        a = torch.empty((P, int(batch_size/P), 28 * 28, 1), dtype=torch.float64)
        omega = torch.empty((P, int(batch_size/P), 1, 1), dtype=torch.float64)
        torch.manual_seed(1)
        a = torch.nn.init.normal_(a).to(device)
        omega = torch.nn.init.uniform_(omega).to(device)
        # --- Network variables ---
        neighbors = vars_network.neighbors
        partiotion_colors = vars_network.partition_colors


        """Algorithm"""
        # size = vars_prob.a.shape[1]
        size = int(m_p)
        n_a = a.shape[2]
        n_omega = omega.shape[2]
        num_colors = len(partiotion_colors)
        loss_arr, acc_arr = [], []

        """DADMM algorithm"""
        for k in range(MAX_ITER):
            # print(k)
            # if k == 0 and test == False:
            #     print(self.hyp[-1])
            if k + 1 == MAX_ITER:
                size = int(m_p) + m % P
            # for color in range(num_colors):
            for color in range(num_colors):
                a_aux = a.clone()
                omega_aux = omega.clone()
                for p in partiotion_colors[color]:
                    neighbs = neighbors[p]
                    Dp = len(neighbs)
                    # --- Determine the sum of the X's of the neighbors ---
                    sum_neighbs_a = torch.zeros(size, n_a, 1).to(device)
                    sum_neighbs_omega = torch.zeros(size, n_omega, 1).to(device)
                    for j in range(Dp):
                        # sum_neighbs_a = sum_neighbs_a + a[neighbs[j]].clone().detach()
                        # sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j]].clone().detach()
                        sum_neighbs_a = sum_neighbs_a + a[neighbs[j]]
                        sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j]]
                        # sum_neighbs_a = sum_neighbs_a + a[neighbs[j], int(size * p): int(size * (p+1))].clone().detach()
                        # sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j], int(size * p): int(size * (p+1))].clone().detach()
                    a_aux[p] = self.grad_alpha(p, k, a[p], omega[p], mu[p], labels[p], sum_neighbs_a, Dp, size, vars_prob.inputs[p])
                    omega_aux[p] = self.grad_omega(p, k, a[p], omega[p], lamda[p], labels[p], sum_neighbs_omega, Dp, size, vars_prob.inputs[p])
                    if float(torch.max(torch.abs(omega_aux[p][:, 0, 0]))) > 100:
                        print(torch.max(torch.abs(omega_aux[p][:, 0, 0])))
                    # del sum_neighbs_a
                    # del sum_neighbs_omega
                    # gc.collect()
                a = a_aux.clone()
                omega = omega_aux.clone()
                # del a_aux
                # del omega_aux
                # gc.collect()

            # Output
            neighbs = neighbors[0]
            Dp = len(neighbs)
            # --- Determine the sum of the X's of the neighbors ---
            sum_neighbs_a = torch.zeros(size, n_a, 1).to(device)
            sum_neighbs_omega = torch.zeros(size, n_omega, 1).to(device)
            for j in range(Dp):
                # sum_neighbs_a = sum_neighbs_a + a[neighbs[j]].clone().detach()
                # sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j]].clone().detach()
                sum_neighbs_a = sum_neighbs_a + a[neighbs[j]]
                sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j]]
                # sum_neighbs_a = sum_neighbs_a + a[neighbs[j], int(size * p): int(size * (p + 1))].clone().detach()
                # sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j],
                #                                         int(size * p): int(size * (p + 1))].clone().detach()
            mu_ = [mu[0].clone().detach() + (torch.abs(self.hyp[k][0][3])) * (Dp * a[0] - sum_neighbs_a)]
            lamda_ = [lamda[0].clone().detach() + (torch.abs(self.hyp[k][0][4])) * (Dp * omega[0] - sum_neighbs_omega)]
            for pp in range(1, P):
                neighbs = neighbors[pp]
                Dp = len(neighbs)
                # --- Determine the sum of the X's of the neighbors ---
                sum_neighbs_a = torch.zeros(size, n_a, 1).to(device)
                sum_neighbs_omega = torch.zeros(size, n_omega, 1).to(device)
                for j in range(Dp):
                    # sum_neighbs_a = sum_neighbs_a + a[neighbs[j]].clone().detach()
                    # sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j]].clone().detach()
                    sum_neighbs_a = sum_neighbs_a + a[neighbs[j]]
                    sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j]]
                    # sum_neighbs_a = sum_neighbs_a + a[neighbs[j], int(size * p): int(size * (p + 1))].clone().detach()
                    # sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j],
                    #                                         int(size * p): int(size * (p + 1))].clone().detach()
                mu_.append(mu[pp].clone().detach() + (torch.abs(self.hyp[k][pp][3])) * (Dp * a[pp] - sum_neighbs_a))
                lamda_.append(lamda[pp].clone().detach() + (torch.abs(self.hyp[k][pp][4])) * (Dp * omega[pp] - sum_neighbs_omega))
            mu = torch.stack(mu_)
            lamda = torch.stack(lamda_)
            # del mu_
            # del lamda_
            # del sum_neighbs_a
            # del sum_neighbs_omega
            # gc.collect()
            if test:
                loss_, acc_ = 0, 0
                mse_loss = nn.MSELoss()
                for jj in range(a.shape[0]):
                    y_hat = (torch.transpose(a[jj], 1, 2) @ inputs[jj] + omega[jj])[:, :, 0]
                # print(f'')
                    for ii in range(y_hat.shape[0]):
                        # print(f'Iter: {ii}; Loss: {mse_loss(y_hat[ii].flatten(), labels[jj][ii].flatten())}')
                        loss_ += mse_loss(y_hat[ii].flatten(), labels[jj][ii].flatten())
                        acc_ += (torch.round(y_hat[ii].flatten()) == labels[jj][ii].flatten()).sum()
                        if mse_loss(y_hat[ii].flatten(), labels[jj][ii].flatten()) == 'nan':
                            print(f'Iter: {ii}; Loss: {mse_loss(y_hat[ii].flatten(), labels[jj][ii].flatten())}')
                loss = loss_ / (a.shape[0] * y_hat.shape[0])
                acc = acc_ / (a.shape[0] * y_hat.shape[0])
                print(f'Iter: {k}; Loss: {loss}, accuracy: {acc}')
                loss_arr.append(loss.item())
                acc_arr.append(acc.item())
            # print(f'Iter: {k}, RAM memory % used: {psutil.virtual_memory().percent}, CPU memory % used: {psutil.cpu_percent()}')
        return a, omega, loss_arr, acc_arr

    def grad_alpha(self, p, k, a, omega, mu, labels, sum_neighbs, Dp, size, inputs):
        # inputs = vars_prob.inputs
        # labels_p = labels[int(p * size): int((p + 1) * size), :]
        # observ_p = full_observ[int(p * size): int((p + 1) * size), :]
        # a_p = a[int(p * size): int((p + 1) * size), :]
        # omega_p = omega[int(p * size): int((p + 1) * size), :]
        return a - (torch.abs(self.hyp[k][p][1])) * (inputs @ torch.transpose(inputs, 1, 2) @ a
                                                                                 + inputs @ omega - inputs @ labels +
                                                                                torch.abs(self.hyp[k][p][0]) * a * Dp +
                                                                      Dp * mu - torch.abs(self.hyp[k][p][0]) * sum_neighbs)

    def grad_omega(self, p, k, a, omega, lamda, labels, sum_neighbs, Dp, size, inputs):
        # inputs = vars_prob.inputs
        # observ_p = full_observ[int(p * size): int((p + 1) * size), :]
        # labels_p = labels[int(p * size): int((p + 1) * size), :]
        # a_p = a[int(p * size): int((p + 1) * size), :]
        # omega_p = omega[int(p * size): int((p + 1) * size), :]
        return omega - (torch.abs(self.hyp[k][p][-1])) * ((torch.transpose(a, 1, 2) @ inputs)
                                                                                 + omega - labels +
                                                                (torch.abs(self.hyp[k][p][2])) * omega * Dp + lamda * Dp
                                                                          - (torch.abs(self.hyp[k][p][2])) * sum_neighbs)


# def error(labels, a, omega, inputs):
# def error(labels, a, omega, inputs):
#     loss_ = 0
#     mse_loss = nn.MSELoss()
#     for jj in range(a.shape[0]):
#         y_hat = (torch.transpose(a[jj], 1, 2) @ inputs[jj] + omega[jj])[:, :, 0]
#         for ii in range(y_hat.shape[0]):
#             # print(f'Iter: {ii}; Loss: {mse_loss(y_hat[ii], labels[jj][ii])}')
#             loss_ += mse_loss(y_hat[ii].flatten(), labels[jj][ii].flatten())
#     loss = loss_ / (a.shape[0] * y_hat.shape[0])
#     return loss
#
# def accuracy(labels, a, omega, inputs):
#     acc_ = 0
#     for jj in range(a.shape[0]):
#         y_hat = (torch.transpose(a[jj], 1, 2) @ inputs[jj] + omega[jj])[:, :, 0]
#         for ii in range(y_hat.shape[0]):
#             if y_hat[ii] % 1 > 0.8 or y_hat[ii] % 1 < 0.2:
#                 acc_ += (torch.round(y_hat[ii].flatten()) == labels[jj][ii].flatten()).sum()
#     accur = 100 * (acc_ / (a.shape[0] * y_hat.shape[0]))
#     return accur.item()

def error(labels, y_hat):
    loss_ = 0
    mse_loss = nn.MSELoss()
    for ii in range(y_hat.shape[0]):
        for jj in range(y_hat.shape[1]):
            loss_ += mse_loss(y_hat[ii][jj].flatten(), labels[ii][jj].flatten())
    loss = loss_ / (y_hat.shape[0] * y_hat.shape[1])
    return loss

def accuracy(labels, y_hat):
    correct = 0
    for ii in range(y_hat.shape[0]):
        for idx in range(y_hat.shape[1]):
            if y_hat[ii][idx] % 1 > 0.8 or y_hat[ii][idx] % 1 < 0.2:
                correct += (torch.round(y_hat[ii][idx].flatten()) == labels[ii][idx].flatten()).item()
            # if torch.abs(y_hat[ii][idx]) % 1 > 0.5:
            #     correct += (torch.round(y_hat[ii][idx].flatten()) == labels[ii][idx].flatten()).item()
    accur = 100 * correct / (y_hat.shape[0] * y_hat.shape[1])
    return accur

def train_model(model, train_dataset, valid_dataset, num_epochs, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, betas=(0.9, 0.999), weight_decay=0.02, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # train_losses, valid_losses = np.zeros((num_epochs,)), np.zeros((num_epochs,))
    train_losses, valid_losses, train_acc, valid_accuracy = np.zeros((num_epochs,)), np.zeros((num_epochs,)), np.zeros((num_epochs,)), np.zeros((num_epochs,))
    for epoch in range(num_epochs):
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        model.train()
        train_loss = 0
        train_accuracy = 0
        train_loss_iter, valid_loss_iter = [], []
        valid_acc_iter, train_acc_iter = [], []
        for iter, (inputs, labels) in enumerate(train_loader):
            if iter == 1:
                break
            # inputs = inputs.reshape(batch_size, 28 * 28, 1)
            inputs = inputs.reshape(P, int(batch_size / P), 28 * 28, 1)
            # inputs = (inputs / 255 - 0.5).type(torch.double)
            inputs = (inputs / 255).type(torch.double).to(device)
            labels = labels.type(torch.double).to(device)
            # labels = labels.reshape(labels.shape[0], 1, 1)
            labels = labels.reshape(P, int(labels.shape[0] / P), 1, 1)
            vars_prob = Vars(inputs, 0.0, P)
            a, omega, _, __ = model(vars_prob, vars_network, num_iter, labels, test=False)
            y_hat_train = (torch.transpose(a, 2, 3) @ inputs + omega)[:, :, 0]
            loss = error(labels, y_hat_train)
            acc = accuracy(labels, y_hat_train)
            optimizer.zero_grad()
            # if iter % 10 == 0:
            # print(f'Epoch No. {epoch}; iteration No.: {iter}, Loss: {loss}, accuracy: {acc}')
            loss.backward()
            optimizer.step()
            torch.save(model.state_dict(), checkpoint_path.format(epoch))
            train_loss += loss.data.item()
            train_accuracy += acc
            train_loss_iter.append(loss.data.item())
            train_acc_iter.append(acc)
            # if epoch % 10 == 0:
            #     scheduler.step()
            # if iter == 20:
            #     break
        # train_losses[epoch] = (train_loss / len(train_loader.dataset)) * batch_size
        # train_acc[epoch] = (train_accuracy / len(train_loader.dataset)) * batch_size
        train_losses[epoch] = np.mean(np.array(train_loss_iter))
        train_acc[epoch] = np.mean(np.array(train_acc_iter))
        # scheduler.step()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            # a_val = a.clone()
            # omega_val = omega.clone()
            for iter, (inputs, labels) in enumerate(valid_loader):
                if iter == 1:
                    break
                # inputs = inputs.reshape(batch_size, 28 * 28, 1)
                inputs = inputs.reshape(P, int(batch_size / P), 28 * 28, 1)
                # inputs = (inputs / 255 - 0.5).type(torch.double)
                inputs = (inputs / 255).type(torch.double).to(device)
                labels = labels.type(torch.double).to(device)
                # labels = labels.reshape(labels.shape[0], 1, 1)
                labels = labels.reshape(P, int(labels.shape[0] / P), 1, 1)
                vars_prob = Vars(inputs, 0.0, P)
                a_val, omega_val, _, __ = model(vars_prob, vars_network, num_iter, labels, test=False)
                y_hat_val = (torch.transpose(a_val, 2, 3) @ inputs + omega_val)[:, :, 0]
                valid_loss = error(labels, y_hat_val)
                valid_acc = accuracy(labels,  y_hat_val)
                val_acc += valid_acc
                val_loss += valid_loss.data.item()
                valid_loss_iter.append(valid_loss.data.item())
                valid_acc_iter.append(valid_acc)
                # if iter % 10 == 0:
                # print(f'Epoch No. {epoch}; iteration No.: {iter}, Val Loss: {valid_loss}, Val accuracy: {valid_acc}')
                # if iter == 20:
                #     break
            # valid_losses[epoch] = (val_loss / len(valid_loader.dataset)) * (batch_size)
            # valid_accuracy[epoch] = (val_acc / len(valid_loader.dataset)) * (batch_size)
            valid_losses[epoch] = np.mean(np.array(valid_loss_iter))
            valid_accuracy[epoch] = np.mean(np.array(valid_acc_iter))
        if epoch % 1 == 0:
            print(
                "Epoch %d; [Train loss %.8f, Validation loss %.8f], [Train accuracy %.8f, Validation accuracy %.8f] "
                % (epoch, train_losses[epoch], valid_losses[epoch], train_acc[epoch], valid_accuracy[epoch])
            )
            gc.collect()

    # return train_losses, valid_losses, train_loss_iter, valid_loss_iter, a, omega, learned_params
    return train_losses, valid_losses, train_acc, valid_accuracy, train_loss_iter, valid_loss_iter, train_acc_iter, valid_acc_iter


""" --- Main --- """

#   Initializations
# P = 50
# graph_probability = 0.12  #Erdos-Rényi graph with probability 0.12

P = 4
graph_probability = 0.7
vars_network = net_vars(P, graph_probability)
# ----Try lower batch size with more DADMM iterarions (e.g. batch size of 1200 and 13 DADMM iterations)!!!! ----
batch_size = 800
mod = batch_size % P
if mod != 0:
    print("batch size mod P must be 0!!, Please select an appropriate P and batch size!")
    sys.exit()

num_epochs = 700

# --- Data generalization ---
dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
# test_dataset = MNIST(root='./data', train=False, transform=transforms.ToTensor())
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(0.8 * dataset.train_data.shape[0]), int(0.2 * dataset.train_data.shape[0])], generator=torch.Generator().manual_seed(42))
del dataset
gc.collect()

# --- D-ADMM parameters
num_iter = 21
# learn_params = torch.tensor([[9.1017e-03, 9.0766e-02, 9.9245e-04, 9.4113e-12, 9.4366e-12, 7.8107e-02]] * P, requires_grad=True).unsqueeze(0).repeat([num_iter, 1, 1]).to(device)  # rho, alpha, beta, eta1, eta2, P=12 fully connected
learn_params = torch.tensor([[2.4231e-04, 4.3877e-01, 1.2665e-03, 1.1221e-06, 1.1797e-06, 1.2260e-01]] * P, requires_grad=True).unsqueeze(0).repeat([num_iter, 1, 1]).to(device) # P=4
# learn_params = torch.tensor([[1.9231e-04, 4.0877e-02, 1.5665e-03, 9.7221e-05, 1.0797e-06, 1.0260e-01]] * P, requires_grad=True).unsqueeze(0).repeat([num_iter, 1, 1]).to(device)
# learn_params = torch.tensor([[0.01808, 0.02239, 0.00500, 0.00319, 0.00243, 0.02239]] * P, requires_grad=True).unsqueeze(0).repeat([num_iter, 1, 1])  # rho, alpha, beta, eta1, eta2
# learn_params = torch.tensor([[7.1545e-05, 1.1120e-01, 2.1993e-02, 2.5715e-11, 5.2461e-12, 6.0542e-02]] * P, requires_grad=True).unsqueeze(0).repeat([num_iter, 1, 1])
# a = torch.rand((P, int(batch_size/P), 28 * 28, 1), dtype=torch.float64).to(device)
# w = torch.empty(P, int(batch_size/P), 28 * 28, 1)
# a = torch.nn.init.xavier_normal_(w).to(device)
# b = torch.empty(P, int(batch_size/P), 28 * 28, 1)
# omega = nn.init.uniform_(b).to(device)
# omega = torch.rand((batch_size, P, 1, 1), dtype=torch.float64)
# a = torch.normal(0, 0.75, size=(P, int(batch_size/P), 28 * 28, 1), dtype=torch.double).to(device)
# omega = torch.rand((P, int(batch_size/P), 1, 1), dtype=torch.double).to(device)
# --- Classical D-ADMM
# classic_dadmm = DADMM(learn_params)
# classic_dadmm.eval()
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
# for iter, (inputs, labels) in enumerate(test_loader):
#     inputs = inputs.reshape(P, int(batch_size/P), 28 * 28, 1)
#     # inputs = (inputs / 255 - 0.5).type(torch.double)
#     inputs = (inputs / 255).type(torch.double)
#     labels = labels.type(torch.double)
#     labels = labels.reshape(P, int(labels.shape[0]/P), 1, 1)
#     vars_prob = Vars(a, omega, inputs, 0.0, P)
#     a_test, omega_test, dadmm_loss_arr, __ = classic_dadmm.forward(vars_prob, vars_network, num_iter, labels, test=True)
#     # y_hat_test = (torch.transpose(a_test, 1, 2) @ inputs + omega_test)[:, :, 0]
#     # loss = error(labels, a_test, omega_test)
#     break
# y_dadmm = np.array(dadmm_loss_arr)
# x_dadmm = np.array(list(range(len(dadmm_loss_arr))))

# plt.figure()
# plt.plot(x_dadmm, y_dadmm, '-b', label='Classic D-ADMM loss')
# plt.grid()
# plt.title(f'Loss Curve, Num of DADMM iterations = {num_iter}')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend(loc='best')
# plt.show()

#  --- Unfolded D-ADMM

model = DADMM(learn_params)
model.to(device)
train_losses, valid_losses, train_acc, valid_accuracy, train_loss_iter, valid_loss_iter, train_acc_iter, valid_acc_iter = train_model(model, train_dataset, valid_dataset, num_epochs, batch_size)


"""Plot the learning curves"""
y_t = np.array(train_losses)
x_t = np.array(list(range(len(train_losses))))
y_v = np.array(valid_losses)
x_v = np.array(list(range(len(valid_losses))))
plt.figure()
plt.plot(x_t, y_t, 'b--', label='Train')
plt.plot(x_v, y_v, '-*', label='Valid')
plt.grid()
plt.title(f'Loss Curve, Num Epochs = {num_epochs}, Batch Size = {batch_size} \n Num of Iterations of DADMM = {num_iter}')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.show()

y_acc_t = np.array(train_acc)
x_acc_t = np.array(list(range(len(train_acc))))
y_acc_v = np.array(valid_accuracy)
x_acc_v = np.array(list(range(len(valid_accuracy))))
plt.figure()
plt.plot(x_acc_t, y_acc_t, '--s', label='Train accuracy')
plt.plot(x_acc_v, y_acc_v, '-s', label='Validation accuracy')
plt.grid()
plt.title(f'Accuracy Curve, Num Epochs = {num_epochs}, Batch Size = {batch_size} \n Num of Iterations of DADMM = {num_iter}')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.show()


#--- Unfolded D-ADMM on the test set ---.
test_dataset = MNIST(root='./data', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
model.eval()
test_los = 0
test_loss_iter, test_losses = [], []
for iter, (inputs, labels) in enumerate(test_loader):
    # inputs = inputs.reshape(batch_size, 28 * 28, 1)
    inputs = inputs.reshape(P, int(batch_size / P), 28 * 28, 1)
    # inputs = (inputs / 255 - 0.5).type(torch.double)
    inputs = (inputs / 255).type(torch.double)
    labels = labels.type(torch.double)
    # labels = labels.reshape(labels.shape[0], 1, 1)
    labels = labels.reshape(P, int(labels.shape[0] / P), 1, 1)
    vars_prob = Vars(inputs, 0.0, P)
    a_test, omega_test, dadmm_loss_arr, dadmm_acc_arr = model.forward(vars_prob, vars_network, num_iter, labels, test=True)
    # y_hat_test = (torch.transpose(a_test, 1, 2) @ inputs + omega_test)[:, :, 0]
    test_loss = error(labels, a_test, omega_test, inputs)
    test_los += test_loss.data.item()
    valid_loss_iter.append(test_loss.data.item())
    test_losses.append((test_los / len(test_loader.dataset)))
    # if iter == 20:
    #     break

y_test = dadmm_loss_arr
x_test = np.array(list(range(len(dadmm_loss_arr))))


plt.figure()
plt.plot(x_test, y_test, '-b', label='Classic D-ADMM loss')
plt.grid()
plt.title(f'Loss Curve, Num of DADMM iterations = {num_iter}')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()
