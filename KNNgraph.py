import os
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos
'''
KNN Graph generated by AM-GCN
'''


def construct_graph(dataset, features, topk):
    if not os.path.exists('../data/KNN/' + dataset):
        os.makedirs('../data/KNN/' + dataset)
    fname = '../data/KNN/' + dataset + '/tmp.txt'
    print(fname)
    f = open(fname, 'w')

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()


def generate_knn(dataset, data, topk=10):
    construct_graph(dataset, data, topk)
    f1 = open('../data/KNN/' + dataset + '/tmp.txt','r')
    f2 = open('../data/KNN/' + dataset + '/c' + str(topk) + '.txt', 'w')
    lines = f1.readlines()
    for line in lines:
        start, end = line.strip('\n').split(' ')
        if int(start) < int(end):
            f2.write('{} {}\n'.format(start, end))
    f2.close()



def load_graph(dataset, knns, n):
    featuregraph_path = '../data/KNN/' + dataset + '/c' + str(knns) + '.txt'

    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(n, n), dtype=np.float32).todense()
    fadj = torch.from_numpy(fadj)

    return fadj


