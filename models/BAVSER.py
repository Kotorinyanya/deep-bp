import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv

from utils import *


class Net(nn.Module):

    def __init__(self, writer, dropout=0.0):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, 1)
        # self.conv2 = SGConv(32, 8)
        # self.conv3 = SGConv(8, 1)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(50 * 1, 2)
        # self.drop2 = nn.Dropout(dropout)
        # self.fc2 = nn.Linear(32, 16)
        # self.drop3 = nn.Dropout(dropout)
        # self.fc3 = nn.Linear(16, 2)

    def forward(self, input):
        if type(input) == list:
            assert len(input) == 1
            data = input[0]
            data.num_graphs = 1
        else:
            data = input
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)

        self.saved_x = self.conv1(x, edge_index)
        # x2 = self.conv2(x1, edge_index)
        # x3 = self.conv3(x2, edge_index)
        # x = torch.cat([x1, x2, x3], dim=-1)
        x = self.saved_x.view(data.num_graphs, -1)
        # print(x[:, 0])
        # print("x: ", x[:10])
        # x = F.relu(self.fc1(self.drop1(x)))
        # x = F.relu(self.fc2(self.drop2(x)))
        # x = self.fc3(self.drop3(x))
        x = self.fc1(x)

        reg = torch.tensor([], device=self.device)
        return x, reg

    @property
    def device(self):
        return self.conv1.weight.device
