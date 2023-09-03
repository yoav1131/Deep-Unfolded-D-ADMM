import os
import scipy.io as sio
import numpy as np
import sys
import random
import networkx as nx
import torch
import matplotlib.pyplot as plt


"""---- Data initialization ----"""
class Vars:
    def __init__(self, inputs, m_p):
        self.inputs = inputs
        self.m_p = m_p
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        direct_data = f'{__location__}/GenerateData/ProblemData'
        self.A_BPDN = torch.from_numpy(sio.loadmat(f'{direct_data}/CompressedSensing/GaussianData.mat')['A_BP'])


class CreateGraph:
    def __init__(self, args):
        self.args = args
        self.net1 = nx.erdos_renyi_graph(args.P, args.graph_prob) # Creates erdos renyi graph

    @staticmethod
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

    @staticmethod
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

        # Plot the graph
        # colors_nodes = [data['color'] for v, data in network.nodes(data=True)]
        # nx.draw(network, node_color=colors_nodes, with_labels=True)
        # plt.show()

        for idx in colors_list:
            t = np.array(color_dict[idx], dtype='uint8')
            color_partition.append(t)
        color_partition = np.array(color_partition, dtype=object)
        return color_partition