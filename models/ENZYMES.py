import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from nn.conv import GCNConv
import torch_geometric
from torch_geometric.nn import dense_diff_pool
from torch_geometric.data import Batch

from belief_propagation import BeliefPropagation
from utils import *


class Net(nn.Module):

    def __init__(self, writer, dropout=0.0):
        super(Net, self).__init__()
        num_groups = 4
        self.conv1 = GCNConv(3, 8)
        self.conv2 = GCNConv(8, 8)
        self.bp = BeliefPropagation(num_groups,
                                    mean_degree=3.8,
                                    summary_writer=writer,
                                    verbose_init=False)
        self.entropy = EntropyLoss()
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(8 * num_groups, 32)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 16)
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(16, 6)

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)
        x, edge_index = batch.x.to(self.device), batch.edge_index.to(self.device)

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        _, s, _ = self.bp(edge_index, x.shape[0])
        ent_loss = self.entropy(s)

        x = x.reshape(batch.num_graphs, int(batch.num_nodes / batch.num_graphs), -1)
        s = s.reshape(batch.num_graphs, int(batch.num_nodes / batch.num_graphs), -1)

        x = torch.matmul(s.transpose(1, 2), x)
        x = x.view(batch.num_graphs, -1)
        x = F.relu(self.fc1(self.drop1(x)))
        x = F.relu(self.fc2(self.drop2(x)))
        x = self.fc3(self.drop3(x))
        return x, ent_loss

    @property
    def device(self):
        return self.bp.beta.device


class SAGE_DIFFPOOL(nn.Module):

    def __init__(self, writer, dropout=0.0):
        super(SAGE_DIFFPOOL, self).__init__()
        self.conv11 = SAGEConv(3, 256)
        self.norm11 = nn.BatchNorm1d(256)
        self.conv12 = SAGEConv(256, 256)
        self.norm12 = nn.BatchNorm1d(256)
        self.conv13 = SAGEConv(256, 256)
        self.norm13 = nn.BatchNorm1d(256)
        self.pool_conv11 = SAGEConv(3, 256)
        self.norm_p11 = nn.BatchNorm1d(256)
        self.pool_conv12 = SAGEConv(256, 4)
        self.norm_p12 = nn.BatchNorm1d(4)
        self.conv21 = SAGEConv(256, 256)
        self.norm21 = nn.BatchNorm1d(256)
        self.conv22 = SAGEConv(256, 256)
        self.norm22 = nn.BatchNorm1d(256)
        # self.pool_conv21 = SAGEConv(256, 256)
        # self.pool_conv22 = SAGEConv(256, 2)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(4 * 256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, 6)

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        x, edge_index = batch.x.to(self.device), batch.edge_index.to(self.device)
        adj = torch.stack([
            torch.tensor(edge_index_to_csr(data.edge_index, data.x.shape[0], data.edge_attr).todense(),
                         dtype=torch.float,
                         device=self.device)
            for data in batch.to_data_list()], dim=0)

        x11 = self.conv11(x, edge_index)
        x11 = self.norm11(x11)
        x12 = self.conv12(x11, edge_index)
        x12 = self.norm12(x12)
        x13 = self.conv13(x12, edge_index)
        x13 = self.norm13(x13)
        s11 = self.pool_conv11(x, edge_index)
        s11 = self.norm_p11(s11)
        s12 = self.pool_conv12(s11, edge_index)
        s12 = self.norm_p12(s12)

        x13 = x13.reshape(batch.num_graphs, int(batch.num_nodes / batch.num_graphs), -1)
        s12 = s12.reshape(batch.num_graphs, int(batch.num_nodes / batch.num_graphs), -1)
        p1_x, p1_adj, p1_ll, p1_el = dense_diff_pool(x13, adj, s12)
        p1_edge_index = Batch.from_data_list(
            [Data(edge_index=adj_to_edge_index(p1_adj[i].cpu().detach().numpy())[0].to(self.device))
             for i in range(p1_adj.shape[0])]
        ).edge_index
        p1_x = p1_x.reshape(-1, p1_x.shape[-1])

        x21 = self.conv21(p1_x, p1_edge_index)
        x21 = self.norm21(x21)
        x22 = self.conv22(x21, p1_edge_index)
        x22 = self.norm22(x22)
        # s21 = self.pool_conv21(p1_x, p1_edge_index)
        # s22 = self.pool_conv22(s21, p1_edge_index)
        # p2_x, p2_adj, p2_ll, p2_el = dense_diff_pool(x22, p1_adj, s22)
        conv_out = x22.reshape(batch.num_graphs, -1)
        out = self.bn1(self.fc1(F.relu(self.drop1(conv_out))))
        out = self.bn2(self.fc2(F.relu(self.drop2(out))))
        out = self.fc3(F.relu(self.drop3(out)))

        reg = p1_ll * 1000 + p1_el

        return out, reg * 100

    @property
    def device(self):
        return self.conv11.weight.device
