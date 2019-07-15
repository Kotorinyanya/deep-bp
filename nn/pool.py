import torch
from torch import nn
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn import global_sort_pool
from torch_geometric.nn.inits import glorot

from utils import adj_to_edge_index

from boxx import timeit


class SortPool(torch.nn.Module):

    def __init__(self, k):
        super(SortPool, self).__init__()
        self.k = k

    def forward(self, x):
        return global_sort_pool(x=x,
                                batch=torch.tensor([0 for i in range(x.size()[0])], dtype=torch.long, device=x.device),
                                k=self.k)

    def __repr__(self):
        return '{}(k_nodes_to_keep={})'.format(self.__class__.__name__,
                                               self.k,
                                               )


class DIFFPool(torch.nn.Module):
    """
    Differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper.

    """

    def __init__(self):

        super(DIFFPool, self).__init__()

    def forward(self, x, adj, s, short_cut=False):
        """
        Returns pooled node feature matrix, coarsened adjacency matrix and the
        auxiliary link prediction objective
        Args:
            adj: Adjacency matrix with shape [num_nodes, num_nodes]
        """
        out_x, out_adj, reg = dense_diff_pool(x, adj, s)
        out_adj = out_adj.squeeze(0) if out_adj.dim() == 3 else out_adj
        out_x = out_x.squeeze(0) if out_x.dim() == 3 else out_x
        if not short_cut:
            out_edge_index, out_edge_attr = adj_to_edge_index(out_adj.detach())
        else:
            out_edge_index, out_edge_attr = None, None

        return out_x, out_edge_index, out_edge_attr, out_adj, reg

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
