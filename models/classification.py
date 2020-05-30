import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import dense_diff_pool

from belief_propagation import BeliefPropagation
from torch_geometric.nn import GCNConv, GINConv
from dataset import SBM4
# from nn.conv import GCNConv
from utils import *


class Net(nn.Module):

    def __init__(self, writer, num_clusters, in_dim, out_dim, dropout=0.0):
        super(Net, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = 30
        self.out_dim = out_dim
        self.num_clusters = num_clusters

        self.conv11 = GINConv(
            nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                # nn.BatchNorm1d(self.hidden_dim)
            ))

        self.conv12 = GINConv(
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                # nn.BatchNorm1d(self.hidden_dim)
            ))

        self.conv13 = GINConv(
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                # nn.BatchNorm1d(self.hidden_dim)
            ))

        self.pool_dim = self.num_clusters

        bp_params = dict(mean_degree=None,
                         # summary_writer=writer,
                         max_num_iter=5,
                         verbose_init=False,
                         is_logging=True,
                         verbose_iter=False,
                         save_full_init=True,
                         bp_max_diff=2e-1
                         )

        self.bp1 = BeliefPropagation(2, **bp_params)
        self.bp2 = BeliefPropagation(3, **bp_params)
        self.bp3 = BeliefPropagation(4, **bp_params)

        self.pool_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 + 3 + 4, 100),
            nn.ReLU(),
            nn.Linear(100, self.pool_dim),
            nn.Softmax(1)
        )

        self.conv21 = GINConv(
            nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                # nn.BatchNorm1d(self.hidden_dim)
            ))

        self.conv22 = GINConv(
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                # nn.BatchNorm1d(self.hidden_dim)
            ))

        self.conv23 = GINConv(
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                # nn.BatchNorm1d(self.hidden_dim)
            ))

        self.final_fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * 6),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 6, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, self.out_dim),
        )

    def forward(self, data_list):
        assert len(data_list) == 1
        batch = data_list[0]

        x, edge_index, edge_attr = batch.x.to(self.device), batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr
        adj = torch.stack([
            torch.tensor(edge_index_to_csr(data.edge_index, data.x.shape[0], data.edge_attr).todense(),
                         dtype=torch.float,
                         device=self.device)
            for data in batch.to_data_list()], dim=0)

        x11 = self.conv11(x, edge_index, )
        x12 = self.conv12(x11, edge_index, )
        x13 = self.conv13(x12, edge_index, )
        x1 = torch.cat([x11, x12, x13], dim=-1)
        x1 = x1.reshape(batch.num_graphs, int(batch.num_nodes / batch.num_graphs), -1)

        # global pooling
        x1_out, _ = torch.max(x1, dim=1)
        # x1_out += torch.mean(x1, dim=1)

        num_nodes = batch.num_nodes
        _, s1, _ = self.bp1(edge_index, num_nodes, edge_attr)
        _, s2, _ = self.bp2(edge_index, num_nodes, edge_attr)
        _, s3, _ = self.bp3(edge_index, num_nodes, edge_attr)

        ss = [s1, s2, s3]

        s = self.pool_fc(torch.cat(ss, dim=1))

        reg = sum([real_modularity(s, edge_index, edge_attr) for s in ss])

        # pooling
        s = s.reshape(batch.num_graphs, int(batch.num_nodes / batch.num_graphs), -1)

        p1_x = s.transpose(1, 2) @ x1
        p1_adj = s.transpose(1, 2) @ adj @ s

        data_list = []
        for i in range(p1_adj.shape[0]):
            edge_index, edge_attr = from_2d_tensor_adj(p1_adj[i])
            data_list.append(Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=self.num_clusters))
        p1_batch = Batch.from_data_list(data_list)
        p1_edge_index, p1_edge_attr = p1_batch.edge_index, p1_batch.edge_attr
        p1_x = p1_x.reshape(-1, p1_x.shape[-1])

        x21 = self.conv21(p1_x, p1_edge_index, )
        x22 = self.conv22(x21, p1_edge_index, )
        x23 = self.conv23(x22, p1_edge_index, )
        x2 = torch.cat([x21, x22, x23], dim=-1)
        x2 = x2.reshape(p1_batch.num_graphs, int(p1_batch.num_nodes / p1_batch.num_graphs), -1)
        # global pooling
        x2_out, _ = torch.max(x2, dim=1)
        # x2_out += torch.mean(x2, dim=1)

        conv_out = torch.cat([x1_out, x2_out], dim=-1)

        out = self.final_fc(conv_out)

        assert (out != out).sum() == 0  # nan
        # print(reg)
        return out, reg

    @property
    def device(self):
        return self.final_fc[3].weight.device
