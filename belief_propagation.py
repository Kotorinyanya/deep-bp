import concurrent.futures
import logging
import threading
import time
from random import shuffle

import networkx as nx
import numpy as np
import torch
from torch import nn

from utils import *


class BeliefPropagation(nn.Module):

    def __init__(self,
                 num_groups,
                 mean_degree=None,
                 max_num_iter=10,
                 bp_max_diff=1e-2,
                 bp_dumping_rate=1.0,
                 num_workers=1,
                 bp_type='approximate',
                 bp_implementation_type='parallel',
                 seed=0):
        """
        Belief Propagation for pooling on graphs

        :param num_groups: pooling size
        :param mean_degree: not required, but highly recommend
        :param max_num_iter: max BP iteration to converge
        :param bp_max_diff: max BP diff to converge
        :param bp_dumping_rate: BP dumping rate, 1.0 as default
        :param num_workers: multi-threading workers
        :param bp_type: 'approximate' with external field h, or 'exact'
        :param bp_implementation_type: 'parallel` with torch_scatter, or 'legacy'
        :param seed:
        """

        super(BeliefPropagation, self).__init__()
        assert bp_type in ['exact', 'approximate']
        assert bp_implementation_type in ['parallel', 'legacy']
        if bp_implementation_type == 'parallel':
            global scatter_mul
            from torch_scatter import scatter_mul

        self.bp_type = bp_type
        self.bp_implementation_type = bp_implementation_type
        self.seed = seed
        self.logger = logging.getLogger(str(self.__class__.__name__))
        self.num_workers = num_workers
        self.bp_dumping_rate = bp_dumping_rate
        self.bp_max_diff = bp_max_diff  # max_diff = max_diff / num_groups ?
        self.max_num_iter = max_num_iter
        self.num_groups = num_groups

        # initialize beta, the learning parameter
        if mean_degree is None:
            self.logger.error(
                "Initializing `beta` to `1.0` without mean_degree\n"
                "BP requires a rough mean degree of the network to initialize "
                "the learning parameter `beta` to beta_ast, "
                "which makes learning faster and more stable. "
                "Without mean degree being specified, "
                "`beta` will be initialized to `1.0`.\n"
                "Please pass the mean degree of the network by argument "
                "when calling, for example BeliefPropagation(number_of_groups=q, mean_degree=c)"
            )
            beta = torch.tensor(1.0, dtype=torch.float)
        elif mean_degree > 0:
            # beta = torch.log(
            #     self.num_groups / (torch.sqrt(torch.tensor(mean_degree, dtype=torch.float)) - 1) + 1
            # ).unsqueeze(-1)
            beta = np.log(self.num_groups / np.sqrt(mean_degree - 1) + 1)
            beta = torch.tensor(beta, dtype=torch.float)
        self.beta = nn.Parameter(data=beta)

        # self.logger.info('\t mean_degree \t {0:.2f} \t beta \t {1} \t '.format(mean_degree, self.beta.data.item()))
        self.logger.info('beta initialized to {0}'.format(self.beta.data))

        # initialized later on calling forward
        self.G, self.W, self.mean_w, self.N, self.mean_degree, self.message_index_list = None, None, None, None, None, None
        self.marginal_psi, self.message_map, self.h, self.node_id_to_index, self.w_indexed = None, None, None, None, None
        self._lock_psi, self._lock_mmap, self._lock_h = None, None, None
        self.job_list, self._update_message_slice_and_index, self._update_psi_index = None, None, None
        self.is_init = False

        # constrain add to modularity regularization
        self.entropy_loss = EntropyLoss()

    def forward(self, adjacency_matrix):
        self.init_bp(adjacency_matrix)
        if self.bp_implementation_type == 'parallel':
            num_iter, max_diff = self.bp_infer_parallel()
        elif self.bp_implementation_type == 'legacy':
            num_iter, max_diff = self.bp_infer_legacy()
        # num_iter, max_diff = self.bp_infer()
        is_converge = True if max_diff < self.bp_max_diff else False
        assignment_matrix = self.marginal_psi  # weighted
        # _, assignment = torch.max(self.marginal_psi, 1)  # unweighted
        modularity = self.compute_modularity()
        reg = self.compute_reg() * np.sqrt(self.num_groups)
        entropy_loss = self.entropy_loss(self.marginal_psi) / np.log(self.num_groups)
        # or * self.mean_degree * self.mean_degree / self.num_groups

        self.logger.info("BP STATUS: \t beta \t {0}".format(self.beta.data))
        self.logger.info("BP STATUS: is_converge \t {0} \t iterations \t {1}".format(is_converge, num_iter))
        self.logger.info("BP STATUS: max_diff \t {0:.5f} \t modularity \t {1}".format(max_diff, modularity))
        if not is_converge:
            self.logger.warning(
                "SG:BP failed to converge with max_num_iter={0}, beta={1}, max_diff={2}. "
                "Indicating a spin-glass (SG) phase in BP (which is not good)".format(
                    num_iter, self.beta.data, max_diff))
        if (self.beta < 0.05).sum() > 0:
            self.logger.critical("P:beta={0}, indicating paramagnetic phase in BP (which is not good), "
                                 "please consider adding the weight for entropy_loss".format(self.beta.data))
            raise Exception("beta={0}, indicating a paramagnetic (P) phase in BP (which is not good), "
                            "please consider adding the weight for entropy_loss".format(self.beta.data))

        return assignment_matrix, reg, entropy_loss, modularity

    def init_bp(self, adjacency_matrix):
        """
        Initialize BP contents
        :param adjacency_matrix:
        :return:
        """
        self.G = nx.to_networkx_graph(adjacency_matrix)
        if self.is_init is True:
            if not csr_matrix_equal(nx.to_scipy_sparse_matrix(self.G).astype(np.float), self.W):  # check not equal
                # if the input adjacency_matrix changes, job_list need to be re initialized
                self.is_init = False
        self.W = nx.to_scipy_sparse_matrix(self.G).astype(np.float)  # connectivity matrix
        self.mean_w = self.W.mean()
        self.N = self.G.number_of_nodes()
        self.mean_degree = torch.tensor(self.W.mean() * self.N)
        self.message_index_list = list(self.G.to_directed().edges())

        # seed for stable gradient descent
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # initialize by random psi
        marginal_psi = torch.rand(self.N, self.num_groups)  # the marginal probability of node i

        self.marginal_psi = marginal_psi / marginal_psi.sum(1).reshape(-1, 1)
        if self.bp_implementation_type == 'legacy':
            self.message_map = self.init_messages()  # messages (bi-edges)
        elif self.bp_implementation_type == 'parallel':
            self.message_map, self.node_id_to_index, self.w_indexed = self.init_messages()
        # initialize external field
        self.h = (-self.beta * self.mean_w * self.marginal_psi.clone()).sum(0)

        if not self.is_init:
            if self.bp_implementation_type == 'legacy':
                # multi-thread synchronization using lock
                self._lock_psi = [threading.Lock() for _ in range(len(self.marginal_psi))]
                self._lock_mmap = [[threading.Lock() for _ in range(len(mmap))] for mmap in self.message_map]
                self._lock_h = threading.Lock()
            elif self.bp_implementation_type == 'parallel':
                # job_list to avoid race
                # job -> (edges, nodes)
                self.job_list = self._create_job_list_parallel()
                self._update_message_slice_and_index = [self._update_message_create_slice(job[0])
                                                        for job in self.job_list]
                self._update_psi_index = [self._update_psi_create_index(job[1])
                                          for job in self.job_list]
            self.is_init = True

    def bp_infer_parallel(self):
        """
        Faster BP Inference with torch_scatter
        :return:
        """
        for num_iter in range(self.max_num_iter):
            max_diff = -np.inf

            start = time.time()
            for i, (messagess, nodes) in enumerate(self.job_list):
                # i: step (create job_list and slice_index_list at init for faster inference
                diff = self.bp_infer_step_fast(i)
                max_diff = diff if diff > max_diff else max_diff
            end = time.time()

            self.logger.info(
                "num_iter \t {:3d} \t time \t {:.2f} \t max_diff {}".format(num_iter, end - start, max_diff))

            if max_diff < self.bp_max_diff:
                return num_iter, max_diff

        return num_iter, max_diff

    def bp_infer_step_fast(self, i):
        if self.bp_type == 'exact':
            raise NotImplementedError("BP exact if not implemented for BP parallel")

        diff = self.update_message_fast(i)
        self.update_psi_h_fast(i)

        return diff

    def update_psi_h_fast(self, i):
        slice_tensor, index_tensor = self._update_psi_index[i]
        # sum all messages
        src = 1 + self.message_map.clone() * (torch.exp(self.beta * self.w_indexed) - 1)
        out = self.marginal_psi.new_ones((self.marginal_psi.shape[0] + 1, self.marginal_psi.shape[1]))
        out = scatter_mul(src, index_tensor, out=out, dim=0)
        out = out[:-1][slice_tensor]
        out = out * torch.exp(self.h)
        out = out / out.sum(-1).reshape(-1, 1)
        # new_marginal_psi = out[slice_tensor[:torch.nonzero(slice_tensor).max() + 1]]
        # subtract the old psi
        self.h -= -self.beta * self.mean_w * self.marginal_psi[slice_tensor].clone().sum(0)
        # update marginal_psi
        self.marginal_psi[slice_tensor] = out
        # add the new psi
        self.h += -self.beta * self.mean_w * self.marginal_psi[slice_tensor].clone().sum(0)

    def _update_psi_create_index(self, nodes):
        # for updating marginal_psi
        slice_tensor = torch.zeros(self.N, dtype=torch.uint8)
        # for torch_scatter
        index_tensor = torch.ones(self.message_map.shape[0], dtype=torch.long) * \
                       self.marginal_psi.shape[0]  # a trick solve "Invalid index in gather at", but redundant
        for dst_node in nodes:
            slice_tensor[dst_node] = 1  # mark message to write
            neighbors = list(self.G.neighbors(dst_node))
            messages_to_dst_node = [(src_node, dst_node) for src_node in neighbors]
            message_indexes = []
            for i, j in messages_to_dst_node:
                i_neighbors = list(self.G.neighbors(i))
                i_to_j_message_index = self.node_id_to_index[i] + i_neighbors.index(j)
                message_indexes.append(i_to_j_message_index)
            index_tensor[message_indexes] = int(dst_node)  # index message to read

            # assert index_tensor.max() == -1 or index_tensor.max() == torch.nonzero(slice_tensor).max()

        return slice_tensor, index_tensor

    def update_message_fast(self, i):
        slice_tensor, index_tensor = self._update_message_slice_and_index[i]
        # sum all messages
        src = 1 + self.message_map.clone() * (torch.exp(self.beta * self.w_indexed) - 1)
        out = self.message_map.new_ones((self.message_map.shape[0] + 1, self.message_map.shape[1]))
        out = scatter_mul(src, index_tensor, out=out, dim=0)
        out = out[:-1]
        out = out * torch.exp(self.h)
        out = out / out.sum(-1).reshape(-1, 1)
        # new_message = out[slice_tensor[:torch.nonzero(slice_tensor).max() + 1]]
        # updated_message = out[slice_tensor].clone()

        max_diff = (out[slice_tensor].detach() - self.message_map[slice_tensor].detach()).abs().sum(1).max()
        # update messages
        self.message_map[slice_tensor] = out[slice_tensor]
        return max_diff

    def _update_message_create_slice(self, target_list):

        # for updating message_map
        slice_tensor = torch.zeros(self.message_map.shape[0], dtype=torch.uint8)
        # for torch_scatter
        index_tensor = torch.ones(self.message_map.shape[0], dtype=torch.long) * \
                       self.message_map.shape[0]  # a trick solve "Invalid index in gather at", but redundant

        for src_node, dst_node in target_list:
            src_neighbors = list(self.G.neighbors(src_node))
            src_to_dst_message_index = self.node_id_to_index[src_node] + src_neighbors.index(dst_node)
            if self.bp_type == 'approximate':
                src_messages = [(k, src_node) for k in self.G.neighbors(src_node) if k != dst_node]
                src_message_indexes = []
                for i, j in src_messages:
                    i_neighbors = list(self.G.neighbors(i))
                    i_to_j_message_index = self.node_id_to_index[i] + i_neighbors.index(j)
                    src_message_indexes.append(i_to_j_message_index)
                if len(src_message_indexes) > 0:
                    slice_tensor[src_to_dst_message_index] = 1  # mark message to write
                    index_tensor[src_message_indexes] = src_to_dst_message_index  # index message to read

            elif self.bp_type == 'exact':
                src_messages1 = [(k, src_node) for k in self.G.neighbors(src_node) if k != dst_node]
                src_messages2 = [(k, src_node) for k in self.G.nodes() if k not in [src_node, dst_node]]
                raise NotImplementedError()

            # assert index_tensor.max() == -1 or index_tensor.max() == torch.nonzero(slice_tensor).max()
        # assert index_tensor.max() == torch.nonzero(slice_tensor).max()
        return slice_tensor, index_tensor

    def _create_job_list_parallel(self):
        job_list = []
        todo_list = self.message_index_list
        while len(todo_list) > 0:
            writing, reading = [], []
            for e in todo_list:
                if e in reading:
                    continue
                i, j = e
                if self.bp_type == 'approximate':
                    to_read = [(k, i) for k in self.G.neighbors(i) if k != j]
                elif self.bp_type == 'exact':
                    to_read = [(k, i) for k in self.G.nodes() if k not in [i, j]]
                if common_member(to_read, writing):
                    continue
                writing.append(e)
                reading += to_read
            todo_list = [e for e in todo_list if e not in writing]
            for e in self.message_index_list:
                if e in reading + writing:
                    continue
                i, j = e
                if self.bp_type == 'approximate':
                    to_read = [(k, i) for k in self.G.neighbors(i) if k != j]
                elif self.bp_type == 'exact':
                    to_read = [(k, i) for k in self.G.nodes() if k not in [i, j]]
                if common_member(to_read, writing):
                    continue
                writing.append(e)
                reading += to_read
            job_list.append(writing)
        for index, lst in enumerate(job_list):
            edges = lst
            nodes = list(set([edge[-1] for edge in edges]))
            job_list[index] = (edges, nodes)
        return job_list

    def bp_infer_legacy(self):
        """
        Belief Propagation for probabilistic graphical inference
        :return:
        """
        if self.bp_type == 'exact':
            raise NotImplementedError("BP exact is not Implemented for legacy")
        job_list = list(self.G.edges())[:]
        np.random.seed(self.seed)
        np.random.shuffle(job_list)
        for num_iter in range(self.max_num_iter):
            max_diff = -np.inf

            start = time.time()
            with concurrent.futures.ThreadPoolExecutor(self.num_workers) as executor:
                for result in executor.map(lambda args: self.bp_iter_step(*args),
                                           job_list):
                    diff = result
                    max_diff = diff if diff > max_diff else max_diff
            end = time.time()
            self.logger.info(
                "num_iter \t {:3d} \t time \t {:.2f} \t max_diff {}".format(num_iter, end - start, max_diff))

            if max_diff < self.bp_max_diff:
                return num_iter, max_diff

        return num_iter, max_diff

    def bp_iter_step(self, i, j):
        """
        Legacy
        :param i:
        :param j:
        :return:
        """
        message_i_to_j = self.update_message_i_to_j(i, j)
        marginal_psi_i = self.update_marginal_psi(i)

        i_to_j = list(self.G.neighbors(i)).index(j)
        diff = (message_i_to_j - self.message_map[i][i_to_j]).abs().sum()

        with self._lock_mmap[i][i_to_j]:
            self.message_map[i][i_to_j] = self.bp_dumping_rate * message_i_to_j.clone() \
                                          + (1 - self.bp_dumping_rate) * self.message_map[i][i_to_j].clone()
        with self._lock_h:
            self.h -= -self.beta * self.mean_w * self.marginal_psi[j].clone()
            self.h += -self.beta * self.mean_w * marginal_psi_i.clone()
        with self._lock_psi[j]:
            self.marginal_psi[j] = marginal_psi_i.clone()

        self.logger.debug("bp_iter_step node {} to {} \t diff {:.6f}".format(i, j, diff))

        return diff

    def update_message_i_to_j(self, i, j):
        """
        Legacy
        :param i:
        :param j:
        :return:
        """
        message_i_to_j = torch.zeros(self.num_groups)
        # all neighbors except j
        neighbors = list(self.G.neighbors(i))
        neighbors.remove(j)

        # sum all message to i
        for q in range(self.num_groups):
            this_value = 1.0
            for k in neighbors:
                i_to_k = list(self.G.neighbors(i)).index(k)
                k_to_i = list(self.G.neighbors(k)).index(i)
                #             print(i, i_to_k, k, k_to_i)
                this_value *= (1 + self.message_map[k][k_to_i][q].clone() *
                               (torch.exp(self.beta * self.W[i, k]) - 1))
            this_value *= torch.exp(self.h[q])
            message_i_to_j[q] = this_value
        message_i_to_j = message_i_to_j.clone() / message_i_to_j.clone().sum()

        self.logger.debug("update_message_i_to_j: ", i, j, message_i_to_j)
        return message_i_to_j

    def update_marginal_psi(self, i):
        marginal_psi_i = torch.zeros(self.num_groups)
        neighbors = list(self.G.neighbors(i))
        for q in range(self.num_groups):
            this_value = 1.0
            for j in neighbors:
                j_to_i = list(self.G.neighbors(j)).index(i)
                this_value *= (1 + self.message_map[j][j_to_i][q].clone() *
                               (torch.exp(self.beta * self.W[i, j]) - 1))
            this_value *= torch.exp(self.h[q])
            marginal_psi_i[q] = this_value
        marginal_psi_i = marginal_psi_i.clone() / marginal_psi_i.clone().clone().sum()
        logging.debug("update_marginal_psi: ", i, marginal_psi_i)
        return marginal_psi_i

    def init_messages(self):

        if self.bp_implementation_type == 'legacy':
            message_map = []
            for i in self.G.nodes():
                message_map_at_i = torch.rand(len(list(self.G.neighbors(i))), self.num_groups)
                message_map_at_i = message_map_at_i / message_map_at_i.sum(1).reshape(-1, 1)
                message_map.append(message_map_at_i)
            return message_map

        elif self.bp_implementation_type == 'parallel':
            node_id_to_index = dict()
            sum_index = 0
            for i in self.G.nodes():
                node_id_to_index[i] = sum_index
                sum_index += len(list(self.G.neighbors(i)))
            num_messages = len(list(self.G.to_directed().edges()))

            message_map = torch.rand(num_messages, self.num_groups)
            message_map = message_map / message_map.sum(1).reshape(-1, 1)

            w_indexed = torch.zeros(num_messages)
            for i in self.G.nodes():
                for j in self.G.neighbors(i):
                    j_th_neibor = list(self.G.neighbors(i)).index(j)
                    w_indexed[node_id_to_index[i] + j_th_neibor] = self.W[i, j]

            return message_map, node_id_to_index, w_indexed.reshape(-1, 1)

    def compute_modularity(self):
        """
        This modularity can't be used with auto_grad as assignment (line 2) is disperse.
        :return:
        """
        m = self.G.number_of_edges()
        _, assignment = torch.max(self.marginal_psi, 1)

        modularity = torch.tensor([0], dtype=torch.float)
        for i, j in self.G.edges():
            delta = 1 if assignment[i] == assignment[j] else 0
            modularity = modularity + self.W[i, j] * delta - self.mean_w * delta
        modularity = modularity / m
        return modularity

    def compute_reg(self):
        """
        continues version of negative modularity (with positive value)
        :return:
        """
        m = self.G.number_of_edges()
        reg = torch.tensor([0], dtype=torch.float)
        for i, j in self.G.edges():
            reg += self.W[i, j] * torch.pow((self.marginal_psi[i] - self.marginal_psi[j]), 2).sum()
        reg = reg / m
        return reg

    def compute_free_energy(self):
        """
        Q: is free_energy required as an output ?
        :return:
        """
        pass

    def __repr__(self):
        return '{0}(num_groups={1}, max_num_iter={2}, bp_max_diff={3}, bp_dumping_rate={4})'.format(
            self.__class__.__name__, self.num_groups, self.max_num_iter, self.bp_max_diff, self.bp_dumping_rate)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    sizes = [50, 50, 50, 50]
    P = np.ones((4, 4)) * 0.006
    for i in range(len(P)):
        P[i][i] = P[i][i] * 10
    g = nx.stochastic_block_model(sizes, P)
    adj = nx.to_scipy_sparse_matrix(g)

    bp = BeliefPropagation(4)
    assignment_matrix, reg, entropy_loss, modularity = bp(adj)
    entropy_loss.backward(retain_graph=True)

    print("reg \t", reg)
    print("bp.beta.grad \t", bp.beta.grad)
    print()
