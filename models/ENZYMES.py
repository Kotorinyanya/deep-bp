import torch.nn.functional as F
from boxx import timeit
# from torch_geometric.nn import SAGEConv
from nn.conv import GCNConv, SAGEConv
import torch_geometric
from torch_geometric.nn import dense_diff_pool
from torch_geometric.data import Batch

from belief_propagation import BeliefPropagation
from utils import *


class Net(nn.Module):

    def __init__(self, writer, dropout=0.0):
        super(Net, self).__init__()
        num_groups = 4
        self.conv1 = GCNConv(3, 30)
        self.conv2 = GCNConv(30, 30)
        self.conv3 = GCNConv(30, 30)
        self.bp = BeliefPropagation(num_groups,
                                    mean_degree=3.8,
                                    summary_writer=writer,
                                    max_num_iter=10,
                                    bp_max_diff=4e-1,
                                    verbose_init=False,
                                    is_logging=True)
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

        self.conv11 = SAGEConv(3, 30)
        self.norm11 = nn.BatchNorm1d(30)
        self.conv12 = SAGEConv(30, 30)
        self.norm12 = nn.BatchNorm1d(30)
        self.conv13 = SAGEConv(30, 30)
        self.norm13 = nn.BatchNorm1d(30)

        self.pool_conv11 = SAGEConv(3, 30)
        self.norm_p11 = nn.BatchNorm1d(30)
        self.pool_conv12 = SAGEConv(30, 30)
        self.norm_p12 = nn.BatchNorm1d(30)
        self.pool_conv13 = SAGEConv(30, 100)
        self.norm_p13 = nn.BatchNorm1d(100)
        self.pool_fc = nn.Linear(160, 100)

        self.conv21 = SAGEConv(30, 30)
        self.norm21 = nn.BatchNorm1d(30)
        self.conv22 = SAGEConv(30, 30)
        self.norm22 = nn.BatchNorm1d(30)
        self.conv23 = SAGEConv(30, 30)
        self.norm23 = nn.BatchNorm1d(30)

        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(30 * 6, 50)
        # self.bn1 = nn.BatchNorm1d(30)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(50, 6)

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

        p1_edge_index = Batch.from_data_list(
            [Data(edge_index=adj_to_edge_index(p1_adj[i].cpu().detach().numpy())[0].to(self.device))
             for i in range(p1_adj.shape[0])]
        ).edge_index
        p1_x = p1_x.reshape(-1, p1_x.shape[-1])

        x21 = self.conv21(p1_x, p1_edge_index)
        x21 = self.norm21(x21)
        x22 = self.conv22(x21, p1_edge_index)
        x22 = self.norm22(x22)
        x23 = self.conv23(x22, p1_edge_index)
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
