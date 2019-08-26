import torch.nn.functional as F
from torch_geometric.data import Batch
# from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv
from nn.conv import GCNConv
from utils import *


class FullGCN(nn.Module):

    def __init__(self, writer, dropout=0.0):
        super(FullGCN, self).__init__()
        self.conv11 = GCNConv(1, 4, type=1)
        self.conv21 = GCNConv(4, 4, type=1)
        # self.norm1 = nn.BatchNorm1d(4)
        self.conv12 = GCNConv(1, 4, type=2)
        self.conv22 = GCNConv(4, 4, type=2)
        # self.norm2 = nn.BatchNorm1d(4)
        self.conv13 = GCNConv(1, 4, type=3)
        self.conv23 = GCNConv(4, 4, type=3)
        # self.norm3 = nn.BatchNorm1d(4)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(4 * 3 * 50, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        x, edge_index = batch.x.to(self.device), batch.edge_index.to(self.device)
        x1 = self.conv21(self.conv11(x, edge_index), edge_index)
        x2 = self.conv22(self.conv12(x, edge_index), edge_index)
        x3 = self.conv23(self.conv13(x, edge_index), edge_index)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = x.view(batch.num_graphs, -1)
        x = self.bn1(F.relu(self.fc1(self.drop1(x))))
        x = self.bn2(F.relu(self.fc2(self.drop2(x))))
        x = self.fc3(self.drop3(x))

        reg = torch.tensor([], device=self.device)
        return x, reg

    @property
    def device(self):
        return self.conv11.weight.device


class Net(nn.Module):

    def __init__(self, writer, dropout=0.0):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, 12)
        self.norm1 = nn.BatchNorm1d(12)
        self.conv2 = GCNConv(12, 12)
        self.norm2 = nn.BatchNorm1d(12)
        self.conv3 = GCNConv(12, 12)
        self.norm3 = nn.BatchNorm1d(12)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(12 * 3 * 50, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        x, edge_index = batch.x.to(self.device), batch.edge_index.to(self.device)
        x1 = self.norm1(self.conv1(x, edge_index))
        x2 = self.norm2(self.conv2(x1, edge_index))
        x3 = self.norm3(self.conv3(x2, edge_index))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = x.view(batch.num_graphs, -1)
        x = self.bn1(F.relu(self.fc1(self.drop1(x))))
        x = self.bn2(F.relu(self.fc2(self.drop2(x))))
        x = self.fc3(self.drop3(x))

        reg = torch.tensor([], device=self.device)
        return x, reg

    @property
    def device(self):
        return self.conv1.weight.device
