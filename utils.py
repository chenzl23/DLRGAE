from texttable import Texttable
import torch
from torch_sparse import SparseTensor
from sklearn.metrics import f1_score 
import numpy as np

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def to_normalized_sparsetensor(edge_index, N, mode='DAD'):
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5) 
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    if mode == 'DA':
        return deg_inv_sqrt.view(-1,1) * deg_inv_sqrt.view(-1,1) * adj
    if mode == 'DAD':
        return deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

def torch_dense_to_sparse(adj):
    edge_index = np.argwhere(adj > 0)
    values = adj[adj > 0]
    row, col = edge_index
    N = adj.shape[0]
    adj = SparseTensor(row=row, col=col, value=values, sparse_sizes=(N, N))
    return adj

def generate_label_matrix(graph, classes_num):
    Y = torch.zeros(graph.y.size(0), classes_num)
    for i in range(graph.y.size(0)):
        Y[i, graph.y[i]] = 1

    return Y

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def F1_value(output, labels):
    preds = output.max(1)[1].type_as(labels)
    F1 = f1_score(preds, labels, average='macro')
    return F1

# def normalize_adj_torch(adj):
