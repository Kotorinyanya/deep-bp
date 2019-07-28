import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from belief_propagation import BeliefPropagation
from utils import *


class Net(nn.Module):

    def __init__(self, writer, dropout=0.0):
        super(Net, self).__init__()
        num_groups = 4
        self.conv1 = SAGEConv(3, 8)
        self.bp = BeliefPropagation(num_groups,
                                    mean_degree=3.8,
                                    summary_writer=writer,
                                    verbose_init=False)
        self.entropy = EntropyLoss()
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(8 * num_groups, 8)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(8, 6)

    def forward(self, data_list):
        assert len(data_list) == 1
        data = data_list[0]
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        x = self.conv1(x, edge_index)
        num_nodes = x.shape[0]
        _, s, _ = self.bp(edge_index, num_nodes)
        ent_loss = self.entropy(s)
        x = torch.matmul(s.t(), x)
        x = x.view(-1)
        #         x, out_adj, link_loss, entropy_loss = dense_diff_pool(x, adj, s)
        x = F.relu(self.fc1(self.drop1(x)))
        x = self.fc2(self.drop2(x))
        return x, ent_loss

    @property
    def device(self):
        return self.bp.beta.device