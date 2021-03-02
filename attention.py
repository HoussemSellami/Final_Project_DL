import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric import utils
import matplotlib.pyplot as plt
from models import TransfEnc


def plot_train(dataset, graph):
    fig = plt.figure(figsize=(36, 32))
    node_colors = np.zeros(len(graph))
    for i in range(len(node_colors)):
        if dataset.data.train_mask[i]:
            node_colors[i] = -1
        if dataset.data.val_mask[i]:
            node_colors[i] = 1
    graph = nx.bfs_tree(graph, 100)
    types = nx.get_node_attributes(graph, 'y')
    pos = nx.layout.fruchterman_reingold_layout(graph)
    p = nx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('coolwarm'),
                               node_color=node_colors[graph.nodes], vmin=-0.1, vmax=0.2)
    plt.colorbar(p)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos, types)
    plt.show()
    # plt.savefig('train_val_test.png')


def plot_attention(model, dataset, graph):
    """ plot the attention weights on a graph

    model:  TransfEnc model
    dataset: torch_geometric.data.Data object
    graph: networkx.graph instance

    """
    # load full batch dataset and calculate the scores (attention weights) for each node
    loader = DataLoader(dataset, batch_size=len(dataset))
    for data in loader:
        scores = model.get_att(data)
        scores = scores[:, 4, :]
    id = 100
    fig = plt.figure(figsize=(36, 32))
    node_colors = np.zeros(len(graph))
    neighbors = [n for n in graph.neighbors(id)]
    node_colors[neighbors] = scores[id, 0].detach().numpy()
    neighbors_all = [id]
    neighbors_all.extend(neighbors)
    for i in range(scores.size()[1]-1):
        new_neighbors = []
        for j, neigh in enumerate(neighbors):
            new_neighbors.extend([n for n in graph.neighbors(neigh) if n not in neighbors_all and n not in new_neighbors])
        neighbors_all.extend(new_neighbors)
        node_colors[new_neighbors] = scores[id, i + 1].detach().numpy()
        neighbors = new_neighbors.copy()
    subgraph = graph.subgraph(neighbors_all)
    types = nx.get_node_attributes(subgraph, 'y')
    node_colors[id] = 0
    pos = nx.layout.fruchterman_reingold_layout(subgraph)
    neighbors_all.sort()
    p = nx.draw_networkx_nodes(subgraph, pos, cmap=plt.get_cmap('coolwarm'),
                               node_color=node_colors[subgraph.nodes], vmin=-0.1, vmax=0.2)
    plt.colorbar(p)
    nx.draw_networkx_edges(subgraph, pos)
    nx.draw_networkx_labels(subgraph, pos, types)
    nx.draw_networkx_labels(graph.subgraph(id), pos, {id: types[id]}, font_color='r')
    # plt.savefig('graph.png', dpi=fig.dpi)
    plt.show()

    fig2, ax = plt.subplots()
    x = np.arange(10)
    ax.bar(x, list(scores[id, :].detach().numpy()), width=0.3)
    ax.set_ylabel('Attention weights')
    ax.set_title('Attention weights ')
    ax.set_xticks(x)
    plt.grid()
    # plt.savefig('weigths.png', dpi=fig2.dpi)
    plt.show()


if __name__ == '__main__':
    # define path to dataset
    dataset = 'CiteSeer'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    # load dataset
    dataset = Planetoid(path, dataset, "full")
    processed_dir = os.path.join(path, 'CiteSeer/processed')

    # load the trained model
    model = TransfEnc(input_dim=3703, num_layers=10, dropout=0.3, hidden_dim=128, num_outputs=6, heads=1)
    with open(os.path.join(processed_dir, 'best_model_TransfEnc.pt'), 'rb') as fp:
        state_dict = torch.load(fp)
        model.load_state_dict(state_dict, strict=False)

    graph = utils.to_networkx(dataset[0], node_attrs=['x', 'y', 'train_mask', 'test_mask', 'val_mask'])
    plot_train(dataset, graph)
    # plot attention weights
    plot_attention(model, dataset, graph)
