import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import dense_diff_pool

from belief_propagation import BeliefPropagation
from torch_geometric.nn import GCNConv, GINConv
from dataset import SBM4
# from nn.conv import GCNConv
from utils import *


class CD_BP_Net(nn.Module):

    def __init__(self, writer=None, dropout=0.0):
        super(CD_BP_Net, self).__init__()

        self.writer = writer

        bp_params = dict(mean_degree=None,
                         summary_writer=writer,
                         max_num_iter=10,
                         verbose_init=True,
                         is_logging=True,
                         verbose_iter=True,
                         save_full_init=True,
                         bp_max_diff=4e-1
                         )

        # self.bp1 = BeliefPropagation(2, **bp_params)
        # self.bp2 = BeliefPropagation(3, **bp_params)
        # self.bp3 = BeliefPropagation(4, **bp_params)
        # self.final_fc = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(9, 4),
        #     nn.Softmax(dim=-1)
        # )

        self.bp1 = BeliefPropagation(4, **bp_params)

    def forward(self, data_list):
        assert len(data_list) == 1
        batch = data_list[0]

        x, edge_index, edge_attr = batch.x.to(self.device), batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr

        # _, s1, _ = self.bp1(edge_index, batch.num_nodes, edge_attr)
        # _, s2, _ = self.bp2(edge_index, batch.num_nodes, edge_attr)
        # _, s3, _ = self.bp3(edge_index, batch.num_nodes, edge_attr)
        # s = self.final_fc(torch.cat([s1, s2, s3], dim=-1))

        _, s, _ = self.bp1(edge_index, batch.num_nodes, edge_attr)

        q = real_modularity(s, edge_index, edge_attr)

        return s, q

    @property
    def device(self):
        return self.bp1.beta.device


class CD_GCN_Net(nn.Module):

    def __init__(self, writer=None, dropout=0.0):
        super(CD_GCN_Net, self).__init__()

        self.writer = writer

        self.in_dim = 1
        self.hidden_dim = 10
        self.out_dim = 4

        # self.conv11 = GCNConv(self.in_dim, self.hidden_dim)
        # self.conv12 = GCNConv(self.hidden_dim, self.hidden_dim)
        # self.conv13 = GCNConv(self.hidden_dim, self.hidden_dim)

        self.conv11 = GINConv(
            nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim)
            ))

        self.conv12 = GINConv(
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim)
            ))

        self.conv13 = GINConv(
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim)
            ))

        self.final_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * self.hidden_dim, self.out_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, data_list):
        assert len(data_list) == 1
        batch = data_list[0]

        x, edge_index, edge_attr = batch.x.to(self.device), batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr

        x11 = self.conv11(x, edge_index,)
        x12 = self.conv12(x11, edge_index, )
        x13 = self.conv13(x12, edge_index, )

        s = self.final_fc(torch.cat([x11, x12, x13], dim=-1))

        q = real_modularity(s, edge_index, edge_attr)

        return s, q

    @property
    def device(self):
        return self.final_fc[1].weight.device

# if __name__ == '__main__':
#     dataset = SBM4(root='datasets/SBM4')
#     model = CD_GCN_Net
#     train_cross_validation(model, dataset, comment='debug', batch_size=2,
#                            num_epochs=100, patience=100, dropout=0, lr=1e-2, weight_decay=0,
#                            use_gpu=True, dp=True, ddp=False, device_ids=[4], is_reg=True)
