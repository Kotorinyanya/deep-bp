import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import softmax, remove_self_loops
from torch_scatter import scatter_add

from utils import add_self_loops_with_edge_attr
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add

import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.nn.inits import glorot, zeros


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 type=1,
                 improved=False,
                 cached=False,
                 bias=True,
                 **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.type = type
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, type, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        if type == 1:
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif type == 2:
            deg_inv = deg.pow(-1)
            norm = deg_inv[row] * edge_weight
        elif type == 3:
            norm = edge_weight

        return edge_index, norm

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.type, self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class EGATConv(torch.nn.Module):
    """Adaptive Edge Features Graph Attentional Layer from the `"Adaptive Edge FeaturesGraph Attention Networks (GAT)"
    <https://arxiv.org/abs/1809.02709`_ paper.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions. (default:
            :obj:`1`)
        concat (bool, optional): Whether to concat or average multi-head
            attentions (default: :obj:`True`)
        negative_slope (float, optional): LeakyRELU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout propbability of the normalized
            attention coefficients, i.e. exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(EGATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.drop = nn.Dropout(dropout)

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att_weight = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(out_channels * heads))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.heads * self.in_channels
        uniform(size, self.weight)
        uniform(size, self.att_weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr=None, save=False):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x = torch.mm(x, self.weight)
        x = x.view(-1, self.heads, self.out_channels)

        # Add self-loops to adjacency matrix.
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops_with_edge_attr(edge_index, edge_attr, num_nodes=x.size(0))
        row, col = edge_index

        # Compute attention coefficients.
        alpha = torch.cat([x[row], x[col]], dim=-1)
        alpha = alpha * self.att_weight

        if save:
            try:
                s_alpha_list = torch.load("s_alpha.pkl")
            except Exception as e:
                s_alpha_list = []
            s_alpha_list.append(alpha.detach().cpu())
            torch.save(s_alpha_list, "s_alpha.pkl")

        alpha = alpha.sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if save:
            try:
                plain_alpha_list = torch.load("plain_alpha.pkl")
            except Exception as e:
                plain_alpha_list = []
            plain_alpha_list.append(alpha.detach().cpu())
            torch.save(plain_alpha_list, "plain_alpha.pkl")

        # This will broadcast edge_attr across all attentions
        alpha = torch.mul(alpha, edge_attr.float())

        if save:
            try:
                alpha_list = torch.load("alpha.pkl")
                edge_index_list = torch.load("edge_index.pkl")
            except Exception as e:
                alpha_list, edge_index_list = [], []
            alpha_list.append(alpha.detach().cpu())
            edge_index_list.append(edge_index.detach().cpu())
            torch.save(alpha_list, "alpha.pkl")
            torch.save(edge_index_list, "edge_index.pkl")

        # Sample attention coefficients stochastically.
        alpha = self.drop(alpha)

        # Sum up neighborhoods.
        out = alpha.view(-1, self.heads, 1) * x[col]
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.sum(dim=1) / self.heads

        if self.bias is not None:
            out = out + self.bias

        return out, edge_index, alpha

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class MEGATConv(torch.nn.Module):
    # Multi-dimension versiion of EGAT
    """
    Adaptive Edge Features Graph Attentional Layer from the `"Adaptive Edge FeaturesGraph Attention Networks (GAT)"
    <https://arxiv.org/abs/1809.02709`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions. (default:
            :obj:`1`)
        concat (bool, optional): Whether to concat or average multi-head
            attentions (default: :obj:`True`)
        negative_slope (float, optional): LeakyRELU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients, i.e. exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        edge_attr_dim (int, required): The dimension of edge features. (default: :obj:`1`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 edge_attr_dim=1):
        super(MEGATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_attr_dim = edge_attr_dim

        self.weight = nn.Parameter(
            torch.Tensor(in_channels, self.edge_attr_dim * out_channels))
        self.att_weight = nn.Parameter(torch.Tensor(1, edge_attr_dim, 2 * out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels * edge_attr_dim))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.edge_attr_dim * self.in_channels
        uniform(size, self.weight)
        uniform(size, self.att_weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x = torch.mm(x, self.weight)
        x = x.view(-1, self.edge_attr_dim, self.out_channels)

        row, col = edge_index

        # Compute attention coefficients
        alpha = torch.cat([x[row], x[col]], dim=-1)
        alpha = (alpha * self.att_weight).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, row, num_nodes=x.size(0))
        # This will broadcast edge_attr across all attentions
        alpha = torch.mul(alpha, edge_attr.float())
        alpha = F.normalize(alpha, p=1, dim=1)

        # Sample attention coefficients stochastically.
        dropout = self.dropout if self.training else 0
        alpha = F.dropout(alpha, p=dropout, training=True)

        # Sum up neighborhoods.
        out = alpha.view(-1, self.edge_attr_dim, 1) * x[col]
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))

        if self.concat is True:
            out = out.view(-1, self.out_channels * self.edge_attr_dim)
        else:
            out = out.sum(dim=1) / self.edge_attr_dim

        if self.bias is not None:
            out = out + self.bias

        return out, alpha

    def __repr__(self):
        return '{}({}, {}, edge_attr_dim={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.edge_attr_dim
                                                     )
