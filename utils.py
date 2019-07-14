import torch
from torch import nn
import numpy as np
from itertools import permutations, product
import networkx as nx


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = x * torch.log(x)
        b = -1.0 * b.sum(dim=1).mean()
        return b


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


def compute_reg(G, marginal_psi):
    """
    continues version of negative modularity (with positive value)
    :return:
    """
    m = G.number_of_edges()
    adjacency_matrix = nx.to_scipy_sparse_matrix(G).astype(np.float)
    reg = torch.tensor([0], dtype=torch.float)
    for i, j in G.edges():
        reg += adjacency_matrix[i, j] * torch.pow((marginal_psi[i] - marginal_psi[j]), 2).sum()
    reg = reg / m
    return reg


def compute_free_energy():
    """
    Q: is free_energy required as an output ?
    :return:
    """
    pass
