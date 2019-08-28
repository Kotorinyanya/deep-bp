import torch.nn.functional as F
from boxx import timeit
# from torch_geometric.nn import GCNConv
from nn.conv import SAGEConv, GCNConv
import torch_geometric
from torch_geometric.nn import dense_diff_pool
from torch_geometric.data import Batch

from belief_propagation import BeliefPropagation
from utils import *


class Net(nn.Module):

    def __init__(self, writer, dropout=0.0):
        super(Net, self).__init__()
        self.conv11 = GCNConv(3, 30)
        self.norm11 = nn.BatchNorm1d(30)
        self.conv12 = GCNConv(30, 30)
        self.norm12 = nn.BatchNorm1d(30)
        self.conv13 = GCNConv(30, 30)
        self.norm13 = nn.BatchNorm1d(30)

        bp_params = dict(mean_degree=3.8,
                         summary_writer=None,
                         max_num_iter=10,
                         verbose_init=False,
                         is_logging=True,
                         verbose_iter=False,
                         save_full_init=True)

        self.bp1 = BeliefPropagation(2, bp_max_diff=2e-1, **bp_params)
        self.bp2 = BeliefPropagation(4, bp_max_diff=4e-1, **bp_params)
        self.bp3 = BeliefPropagation(8, bp_max_diff=2e-1, **bp_params)
        self.bp4 = BeliefPropagation(16, bp_max_diff=1e-1, **bp_params)
        self.bp5 = BeliefPropagation(32, bp_max_diff=1e-1, **bp_params)
        self.bp6 = BeliefPropagation(64, bp_max_diff=5e-2, **bp_params)
        # self.bp7 = BeliefPropagation(128, bp_max_diff=1e-2, **bp_params)

        self.pool_dim = 100
        self.pool_fc = nn.Linear(126, self.pool_dim)

        self.conv21 = GCNConv(30, 30)
        self.norm21 = nn.BatchNorm1d(30)
        self.conv22 = GCNConv(30, 30)
        self.norm22 = nn.BatchNorm1d(30)
        self.conv23 = GCNConv(30, 30)
        self.norm23 = nn.BatchNorm1d(30)

        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(30 * 6, 50)
        # self.bn1 = nn.BatchNorm1d(30)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(50, 6)

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        x, edge_index, edge_attr = batch.x.to(self.device), batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr
        adj = torch.stack([
            torch.tensor(edge_index_to_csr(data.edge_index, data.x.shape[0], data.edge_attr).todense(),
                         dtype=torch.float,
                         device=self.device)
            for data in batch.to_data_list()], dim=0)

        x11 = self.conv11(x, edge_index, edge_attr)
        x11 = self.norm11(x11)
        x12 = self.conv12(x11, edge_index, edge_attr)
        x12 = self.norm12(x12)
        x13 = self.conv13(x12, edge_index, edge_attr)
        x13 = self.norm13(x13)
        x1 = torch.cat([x11, x12, x13], dim=-1)
        # max pooling
        x1_out, _ = torch.max(x1.reshape(batch.num_graphs, int(batch.num_nodes / batch.num_graphs), -1), dim=1)

        _, s11, _ = self.bp1(edge_index, x.shape[0])
        _, s12, _ = self.bp2(edge_index, x.shape[0])
        _, s13, _ = self.bp3(edge_index, x.shape[0])
        _, s14, _ = self.bp4(edge_index, x.shape[0])
        _, s15, _ = self.bp5(edge_index, x.shape[0])
        _, s16, _ = self.bp6(edge_index, x.shape[0])
        s1 = torch.cat([s11, s12, s13, s14, s15, s16], dim=-1)
        # s1 = self.batch_bp(batch.to_data_list())
        s1 = self.pool_fc(s1)

        x13 = x13.reshape(batch.num_graphs, int(batch.num_nodes / batch.num_graphs), -1)
        s1 = s1.reshape(batch.num_graphs, int(batch.num_nodes / batch.num_graphs), -1)

        p1_x, p1_adj, p1_ll, p1_el = dense_diff_pool(x13, adj, s1)

        data_list = []
        for i in range(p1_adj.shape[0]):
            edge_index, edge_attr = adj_to_edge_index(p1_adj[i].cpu().detach().numpy())
            edge_index, edge_attr = edge_index.to(self.device), edge_attr.to(self.device)
            data_list.append(Data(edge_index=edge_index, edge_attr=edge_attr))
        p1_batch = Batch.from_data_list(data_list)
        p1_edge_index, p1_edge_attr = p1_batch.edge_index, p1_batch.edge_attr
        p1_edge_attr = p1_edge_attr.view(-1)
        p1_x = p1_x.reshape(-1, p1_x.shape[-1])

        x21 = self.conv21(p1_x, p1_edge_index, p1_edge_attr)
        x21 = self.norm21(x21)
        x22 = self.conv22(x21, p1_edge_index, p1_edge_attr)
        x22 = self.norm22(x22)
        x23 = self.conv23(x22, p1_edge_index, p1_edge_attr)
        x23 = self.norm23(x23)
        x2 = torch.cat([x21, x22, x23], dim=-1)
        # max pooling
        x2_out, _ = torch.max(x2.reshape(batch.num_graphs, self.pool_dim, -1), dim=1)

        conv_out = torch.cat([x1_out, x2_out], dim=-1)
        out = self.fc1(self.drop1(conv_out))
        out = self.fc2(F.relu(self.drop2(out)))

        # reg = p1_ll
        reg = torch.tensor([0.], device=self.device)

        return out, reg

    def batch_bp(self, data_list):
        all_s = []
        for data in data_list:
            edge_index, edge_attr = data.edge_index, data.edge_attr
            _, s_out, _ = self.bp(edge_index, data.x.shape[0])
            s = torch.zeros(data.x.shape[0], s_out.shape[1], device=self.device)
            s[:s_out.shape[0], :] = s_out
            all_s.append(s)
        batch_s = torch.stack(all_s, dim=0)
        return batch_s

    @property
    def device(self):
        return self.bp1.beta.device


class SAGE_DIFFPOOL(nn.Module):

    def __init__(self, writer, dropout=0.0):
        super(SAGE_DIFFPOOL, self).__init__()

        self.conv11 = GCNConv(3, 30)
        self.norm11 = nn.BatchNorm1d(30)
        self.conv12 = GCNConv(30, 30)
        self.norm12 = nn.BatchNorm1d(30)
        self.conv13 = GCNConv(30, 30)
        self.norm13 = nn.BatchNorm1d(30)

        self.pool_conv11 = GCNConv(3, 30)
        self.norm_p11 = nn.BatchNorm1d(30)
        self.pool_conv12 = GCNConv(30, 30)
        self.norm_p12 = nn.BatchNorm1d(30)
        self.pool_conv13 = GCNConv(30, 100)
        self.norm_p13 = nn.BatchNorm1d(100)
        self.pool_fc = nn.Linear(160, 100)

        self.conv21 = GCNConv(30, 30)
        self.norm21 = nn.BatchNorm1d(30)
        self.conv22 = GCNConv(30, 30)
        self.norm22 = nn.BatchNorm1d(30)
        self.conv23 = GCNConv(30, 30)
        self.norm23 = nn.BatchNorm1d(30)

        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(30 * 6, 50)
        # self.bn1 = nn.BatchNorm1d(30)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(50, 6)

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        x, edge_index, edge_attr = batch.x.to(self.device), batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr
        adj = torch.stack([
            torch.tensor(edge_index_to_csr(data.edge_index, data.x.shape[0], data.edge_attr).todense(),
                         dtype=torch.float,
                         device=self.device)
            for data in batch.to_data_list()], dim=0)

        x11 = self.conv11(x, edge_index, edge_attr)
        x11 = self.norm11(x11)
        x12 = self.conv12(x11, edge_index, edge_attr)
        x12 = self.norm12(x12)
        x13 = self.conv13(x12, edge_index, edge_attr)
        x13 = self.norm13(x13)
        x1 = torch.cat([x11, x12, x13], dim=-1)
        # max pooling
        x1_out, _ = torch.max(x1.reshape(batch.num_graphs, int(batch.num_nodes / batch.num_graphs), -1), dim=1)

        s11 = self.pool_conv11(x, edge_index)
        s11 = self.norm_p11(s11)
        s12 = self.pool_conv12(s11, edge_index)
        s12 = self.norm_p12(s12)
        s13 = self.pool_conv13(s12, edge_index)
        s13 = self.norm_p13(s13)
        s1 = self.pool_fc(torch.cat([s11, s12, s13], dim=-1))

        x13 = x13.reshape(batch.num_graphs, int(batch.num_nodes / batch.num_graphs), -1)
        s1 = s1.reshape(batch.num_graphs, int(batch.num_nodes / batch.num_graphs), -1)

        p1_x, p1_adj, p1_ll, p1_el = dense_diff_pool(x13, adj, s1)

        data_list = []
        for i in range(p1_adj.shape[0]):
            edge_index, edge_attr = adj_to_edge_index(p1_adj[i].cpu().detach().numpy())
            edge_index, edge_attr = edge_index.to(self.device), edge_attr.to(self.device)
            data_list.append(Data(edge_index=edge_index, edge_attr=edge_attr))
        p1_batch = Batch.from_data_list(data_list)
        p1_edge_index, p1_edge_attr = p1_batch.edge_index, p1_batch.edge_attr
        p1_edge_attr = p1_edge_attr.view(-1)
        p1_x = p1_x.reshape(-1, p1_x.shape[-1])

        x21 = self.conv21(p1_x, p1_edge_index, p1_edge_attr)
        x21 = self.norm21(x21)
        x22 = self.conv22(x21, p1_edge_index, p1_edge_attr)
        x22 = self.norm22(x22)
        x23 = self.conv23(x22, p1_edge_index, p1_edge_attr)
        x23 = self.norm23(x23)
        x2 = torch.cat([x21, x22, x23], dim=-1)
        # max pooling
        x2_out, _ = torch.max(x2.reshape(batch.num_graphs, 100, -1), dim=1)

        conv_out = torch.cat([x1_out, x2_out], dim=-1)
        out = self.fc1(self.drop1(conv_out))
        out = self.fc2(F.relu(self.drop2(out)))

        # reg = p1_ll
        reg = torch.tensor([0.], device=self.device)

        return out, reg

    @property
    def device(self):
        return self.conv11.weight.device
