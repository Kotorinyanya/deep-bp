from functools import reduce
import hashlib

import torch
import logging
from torch import nn
import numpy as np
from itertools import permutations, product
import networkx as nx
from torch_geometric.data import Data
import os.path as osp
from scipy.sparse import coo_matrix, csr_matrix


def my_uuid(string):
    return hashlib.sha256(str(string).encode('utf-8')).hexdigest()

def use_logging(level='info'):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if level == 'warn':
                logging.warning("%s is running" % func.__name__)
            elif level == "info":
                logging.info("%s is running" % func.__name__)
            return func(*args)

        return wrapper

    return decorator


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = x * torch.log(x)
        b = -1.0 * b.sum(dim=1).mean()
        return b


def get_model_log_dir(comment, model_name):
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = osp.join(
        current_time + '_' + socket.gethostname() + '_' + comment + '_' + model_name)
    return log_dir


def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if a_set & b_set:
        return True
    else:
        return False


def csr_matrix_equal(a1, a2):
    return (np.array_equal(a1.indptr, a2.indptr) and
            np.array_equal(a1.indices, a2.indices) and
            np.array_equal(a1.data, a2.data))


def overlap(s1, s2):
    max_overlap = 0
    perms = list(permutations(range(0, np.max(s1) + 1)))
    combs = list(product(perms, perms))
    for perm1, perm2 in combs:
        overlap = 0
        for i in range(np.max(s1) + 1):
            overlap += len(np.intersect1d(np.where(s1 == perm1[i]), np.where(s2 == perm2[i])))
        max_overlap = overlap if overlap > max_overlap else max_overlap
    max_overlap /= len(s1)
    return max_overlap


def normalized_overlap(s1, s2, max_na):
    over_lap = overlap(s1, s2)
    norm_overlap = (over_lap - max_na) / (1 - max_na)
    return norm_overlap


def compute_modularity(G, marginal_psi):
    """
    This modularity can't be used with auto_grad as assignment (line 2) is disperse.
    :return:
    """
    m = G.number_of_edges()
    adjacency_matrix = nx.to_scipy_sparse_matrix(G).astype(np.float)
    mean_w = adjacency_matrix.mean()
    _, assignment = torch.max(marginal_psi, 1)

    modularity = torch.tensor([0], dtype=torch.float)
    for i, j in G.edges():
        delta = 1 if assignment[i] == assignment[j] else 0
        modularity = modularity + adjacency_matrix[i, j] * delta - mean_w * delta
    modularity = modularity / m
    return modularity


def compute_reg(marginal_psi, adjacency_matrix, edge_index):
    """
    continues version of negative modularity (with positive value)
    :return:
    """
    reg = torch.tensor([0], dtype=torch.float)
    for i, j in edge_index.t():
        reg += adjacency_matrix[i, j] * torch.pow((marginal_psi[i] - marginal_psi[j]), 2).sum()
    return reg


def compute_free_energy():
    """
    Q: is free_energy required as an output ?
    :return:
    """
    pass


def add_self_loops_with_edge_attr(edge_index, edge_attr, num_nodes=None):
    dtype, device = edge_index.dtype, edge_index.device
    loop = torch.arange(0, num_nodes, dtype=dtype, device=device)
    loop = loop.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop], dim=1)
    ones = torch.ones([edge_index.shape[1] - edge_attr.shape[0], edge_attr.shape[1]], dtype=edge_attr.dtype,
                      device=edge_attr.device)
    edge_attr = torch.cat([edge_attr, ones], dim=0)
    assert edge_index.shape[1] == edge_attr.shape[0]
    return edge_index, edge_attr


def networkx_to_data(G, node_feature_dim=0):
    """
    convert a networkx graph (without node features) to torch geometric data
    """
    n = G.number_of_nodes()
    x_shape = (n, node_feature_dim)

    edge_index = list()
    for edge in G.edges:
        edge_index.append(list(edge))

    edge_attr = list()
    for edge in edge_index:
        edge_feature = list()
        for key in G[edge[0]][edge[1]].keys():
            edge_feature.append(G[edge[0]][edge[1]][key])
        edge_attr.append(edge_feature)

    x = torch.ones(x_shape, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def edge_index_to_csr(edge_index, num_nodes, edge_attr=None):
    """
    coo_matrix(coordinate matrix) is fast to get row
    :param edge_index: Tensor
    :param edge_attr: Tensor
    :return:
    """
    # num_nodes = int(edge_index.max()) + 1
    if edge_index.min() < 0:
        edge_index -= edge_index.min()
    shape = (num_nodes, num_nodes)
    if edge_attr is not None:
        assert edge_attr.shape[1] == 1
    edge_attr = edge_attr.reshape(-1) if edge_attr is not None else None
    row, col = edge_index
    data = torch.ones(edge_index.shape[1], dtype=torch.float) if edge_attr is None else edge_attr.float()
    adj = csr_matrix((data.cpu(), (row.cpu(), col.cpu())), shape=shape)
    return adj


def csr_to_symmetric(csr_matrix):
    """
    This is suuuuupper slow
    :param csr_matrix:
    :return:
    """
    rows, cols = csr_matrix.nonzero()
    csr_matrix[cols, rows] = csr_matrix[rows, cols]
    return csr_matrix


def adj_to_edge_index(adj):
    A = coo_matrix(adj)
    edge_index = torch.tensor(np.stack([A.row, A.col]), dtype=torch.long)
    edge_attr = torch.tensor(A.data).unsqueeze(-1)

    return edge_index, edge_attr


def pad_with_zero(max_x_size, data):
    """
    pad to same size for a batch run
    :param data:
    :param max_x_size:
    :return:
    """
    x = data.x
    data.num_nodes = max_x_size
    padded_x = torch.zeros(max_x_size, x.shape[1])
    padded_x[:x.shape[0], :] = x
    data.x = padded_x
    return data


def count_memory(tensors):
    total = 0
    for obj in tensors:
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if len(obj.size()) > 0:
                    if obj.type() == 'torch.FloatTensor':
                        total += reduce(lambda x, y: x * y, obj.size()) * 32
                    elif obj.type() == 'torch.LongTensor':
                        total += reduce(lambda x, y: x * y, obj.size()) * 64
                    elif obj.type() == 'torch.IntTensor':
                        total += reduce(lambda x, y: x * y, obj.size()) * 32
                    # else:
                    # Few non-cuda tensors in my case from dataloader
        except Exception as e:
            pass
    print("{} GB".format(total / ((1024 ** 3) * 8)))
