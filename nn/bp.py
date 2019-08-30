import os
import threading
import time

from IPython.core.magics import logging
from boxx import timeit
import random
from scipy.sparse import csr_matrix

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_scipy_sparse_matrix
from tqdm import tqdm
from torch_geometric.datasets import KarateClub, Reddit
from multiprocessing import Pool, Manager
import torch.multiprocessing as mp

from bp_helper import create_job_list_parallel_fast, _node_to_slice_and_index_message, \
    _node_to_slice_and_index_psi, _update_manager_node_dd, node_list_to_slice_and_index_message, \
    node_list_to_slice_and_index_psi, save_node_list_dict, _save_node_dict
from bp_lagecy import *
from bp_helper import *
from bp_lagecy import _create_job_list_parallel, _update_message_create_slice, _update_psi_create_index

from utils import *


class BP(nn.Module):

    def __init__(self,
                 num_groups,
                 mean_degree=None,
                 max_num_iter=10,
                 bp_max_diff=5e-1,
                 bp_dumping_rate=1.0):

        super(BP, self).__init__()

        self.bp_dumping_rate = bp_dumping_rate
        self.bp_max_diff = bp_max_diff
        self.max_num_iter = max_num_iter
        self.num_groups = num_groups

        # initialize beta, the learning parameter
        if mean_degree is None:
            self.logger.warning(
                "Initializing beta without mean_degree\n"
                "BP requires a rough mean degree of the network to initialize, "
                "which makes learning process faster and more stable. "
                "Without mean degree being specified, "
                "beta will be initialized to `1.0`.\n"
                "Please pass the mean degree of the network by argument "
                "when calling, for example BeliefPropagation(num_groups=q, mean_degree=c)\n"
                "Otherwise beta will be initialized when calling forward()"
            )
            beta = torch.tensor(1.0, dtype=torch.float)
            self.is_beta_init = False
        elif mean_degree > 0:
            # beta = torch.log(
            #     self.num_groups / (torch.sqrt(torch.tensor(mean_degree, dtype=torch.float)) - 1) + 1
            # ).unsqueeze(-1)
            beta = np.log(self.num_groups / (np.sqrt(mean_degree) - 1) + 1)
            beta = torch.tensor(beta, dtype=torch.float)
            self.is_beta_init = True
        else:
            raise Exception("What???")
        self.beta = nn.Parameter(data=beta, requires_grad=(True if not self.disable_gradient else False))

        # self.logger.info('\t mean_degree \t {0:.2f} \t beta \t {1} \t '.format(mean_degree, self.beta.data.item()))
        self.logger.warning('beta initialized to {0}'.format(self.beta.data))

    def forward(self, batch):
        """

        :return:
        """
        num_messages = batch.num_edges + 1
        num_nodes = batch.num_nodes
        self.init_bp(edge_index, num_nodes, edge_attr)

        num_iter, max_diff = self.bp_infer_parallel()

        if self.is_logging:
            self.bp_logging(num_iter, max_diff)

        if (self.beta < 0.05).sum() > 0:
            raise Exception("beta={0}, indicating a paramagnetic (P) phase in BP"
                            .format(self.beta.data))

        return self.message_map, self.marginal_psi

    def bp_logging(self, num_iter, max_diff):
        is_converged = True if max_diff < self.bp_max_diff else False
        self.logger.info("BP STATUS: \t beta \t {0}".format(self.beta.data))
        self.logger.info("BP STATUS: is_converge \t {0} \t iterations \t {1} \t max_diff \t {2:.2e}"
                         .format(is_converged, num_iter, max_diff))
        if not is_converged:
            self.logger.info("SG:BP failed to converge with max_num_iter={0}, beta={1:.5f}, max_diff={2:.2e}. "
                             .format(num_iter, self.beta.data, max_diff))
            self.logger.info("parallel_max_node_percent={0:.2f}, in case of parallelization, "
                             "reducing this percentage is good for BP to converge"
                             .format(self.parallel_max_node_percent))
        if (self.beta < 0.05).sum() > 0:
            self.logger.critical("P:beta={0}, indicating paramagnetic phase in BP, "
                                 "please consider adding the weight for entropy_loss".format(self.beta.data))
