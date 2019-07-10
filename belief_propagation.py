import concurrent.futures
import logging
import threading
import time
from random import shuffle

import networkx as nx
import numpy as np
import torch
from torch import nn

from utils import EntropyLoss


class BeliefPropagation(nn.Module):

    def __init__(self,
                 num_groups,
                 mean_degree=None,
                 max_num_iter=10,
                 bp_max_diff=1e-2,
                 bp_dumping_rate=1.0,
                 num_workers=1,
                 seed=0):
        """
        Belief Propagation for pooling on graphs

        :param num_groups: pooling size
        :param mean_degree: not required, but highly recommend
        :param max_num_iter: max BP iteration to converge
        :param bp_max_diff: max BP diff to converge
        :param bp_dumping_rate: BP dumping rate, 1.0 as default
        :param num_workers: multi-threading workers
        :param seed:
        """

        super(BeliefPropagation, self).__init__()

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
            beta = torch.tensor(1.0, dtype=torch.float).unsqueeze(-1)
        elif mean_degree > 0:
            mean_degree = torch.tensor(mean_degree)
            beta = torch.log(
                self.num_groups / (torch.sqrt(torch.tensor(mean_degree, dtype=torch.float)) - 1) + 1
            ).unsqueeze(-1)
        self.beta = nn.Parameter(data=beta)

        # self.logger.info('\t mean_degree \t {0:.2f} \t beta \t {1} \t '.format(mean_degree, self.beta.data.item()))
        self.logger.info('beta initialized to {0}'.format(self.beta.data))

        # initialized later on calling forward
        self.G, self.W, self.mean_w, self.N, self.mean_degree = None, None, None, None, None
        self.marginal_psi, self.message_map, self.h = None, None, None
        self._lock_psi, self._lock_mmap, self._lock_h = None, None, None

        # constrain add to modularity regularization
        self.entropy_loss = EntropyLoss()

    def forward(self, adjacency_matrix):
        self.init_bp(adjacency_matrix)
        num_iter, max_diff = self.bp_infer()
        is_converge = True if max_diff < self.bp_max_diff else False
        assignment_matrix = self.marginal_psi  # weighted
        # _, assignment = torch.max(self.marginal_psi, 1)  # unweighted
        modularity = self.compute_modularity()
        reg = self.compute_reg()
        entropy_loss = self.entropy_loss(self.marginal_psi) * \
                       4 * np.log(1 / 2) / np.log(1 / self.num_groups)
        # or * self.mean_degree * self.mean_degree / self.num_groups

        self.logger.info("BP STATUS: \t beta \t {0}".format(self.beta.data))
        self.logger.info("BP STATUS: is_converge \t {0} \t iterations \t {1}".format(is_converge, num_iter))
        self.logger.info("BP STATUS: max_diff \t {0:.5f} \t modularity \t {1}".format(max_diff, modularity))
        if not is_converge:
            self.logger.warning(
                "SG:BP failed to converge with max_num_iter={0}, beta={1}, max_diff={2}. "
                "Indicating a spin-glass (SG) phase in BP (which is not good)".format(
                    num_iter, self.beta, max_diff))
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
        self.W = nx.to_scipy_sparse_matrix(self.G)  # connectivity matrix
        self.mean_w = self.W.mean()
        self.N = self.G.number_of_nodes()
        self.mean_degree = torch.tensor(self.W.mean() * self.N)

        # seed for stable gradient descent
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # initialize by random messages
        self.marginal_psi, self.message_map = self.init_messages()
        # initialize external field
        self.h = self.init_h()

        # multi-thread synchronization using lock
        self._lock_psi = [threading.Lock() for _ in range(len(self.marginal_psi))]
        self._lock_mmap = [[threading.Lock() for _ in range(len(mmap))] for mmap in self.message_map]
        self._lock_h = threading.Lock()

    def bp_infer(self):
        """
        Belief Propagation for probabilistic graphical inference
        :return:
        """
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
        marginal_psi = torch.rand(self.N,
                                  self.num_groups)  # the marginal probability of node i (in N) at block q (in Q), normalized
        marginal_psi = marginal_psi / marginal_psi.sum(1).reshape(-1, 1)
        message_map = []
        for i in range(self.N):
            message_map_at_i = torch.rand(len(list(self.G.neighbors(i))), self.num_groups)
            message_map_at_i = message_map_at_i / message_map_at_i.sum(1).reshape(-1, 1)
            message_map.append(message_map_at_i)
        return marginal_psi, message_map

    def init_h(self):
        h = torch.empty(self.num_groups)
        for q in range(self.num_groups):
            h_q = -self.beta * self.W.mean() * self.marginal_psi[:, q].sum()
            h[q] = h_q
        return h

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
    G = nx.stochastic_block_model(sizes, P)

    bp = BeliefPropagation(nx.to_scipy_sparse_matrix(G), num_of_groups=4,
                           bp_max_diff=2e-2, max_num_iter=10, num_workers=1)
    S, reg = bp()
    reg.backward(retain_graph=True)

    print("reg \t", reg)
    print("bp.beta.grad \t", bp.beta.grad)
    print()
