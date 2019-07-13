import concurrent.futures
import logging
import threading
import time
from boxx import timeit
import random
from scipy.sparse import csr_matrix

import networkx as nx
import numpy as np
import torch
from torch import nn

from utils import *
from bp_helper import _create_job_list_parallel


class BeliefPropagation(nn.Module):

    def __init__(self,
                 num_groups,
                 mean_degree=None,
                 max_num_iter=100,
                 bp_max_diff=1e-5,
                 bp_dumping_rate=1.0,
                 num_workers=1,
                 is_logging=True,
                 verbose=False,
                 bp_type='approximate',
                 bp_implementation_type='parallel',
                 parallel_min_iter=100,
                 summary_writer=None,
                 histogram_dim=1,
                 seed=0):
        """
        Belief Propagation for pooling on graphs

        :param num_groups: pooling size
        :param mean_degree: not required, but highly recommend
        :param max_num_iter: max BP iteration to converge,
                            NOTE: large number of iteration blows up RAM
        :param bp_max_diff: max BP diff to converge
        :param bp_dumping_rate: BP dumping rate, 1.0 as default
        :param verbose: logging level
        :param num_workers: multi-threading workers (for legacy)
        :param is_logging: if logging
        :param bp_type: 'approximate' with external field h, or 'exact'
        :param bp_implementation_type: 'parallel` with torch_scatter, or 'legacy'
        :param parallel_min_iter: minimum number of iteration for BP approximation,
                                few value means faster inference, large value means worse approximation
        :param summary_writer: TensorBoardX
        :param histogram_dim: 1 is fast, 0 is extremely slow but detailed
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
        self.parallel_min_iter = parallel_min_iter
        self.seed = seed
        self.is_logging = is_logging
        self.logger = logging.getLogger(str(self.__class__.__name__))
        if verbose:
            logging.basicConfig(level=logging.INFO)
        self.writer = summary_writer
        self.histogram_dim = histogram_dim
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
        self.G, self.adjacency_matrix, self.mean_w, self.num_nodes, self.mean_degree = None, None, None, None, None
        self.marginal_psi, self.message_map, self.h, self.w_indexed, self.message_index_list = None, None, None, None, None
        self.node_id_to_index, self.num_messages = None, None
        self._lock_psi, self._lock_mmap, self._lock_h = None, None, None
        self.job_list, self._update_message_slice_and_index, self._update_psi_index = None, None, None
        self.is_init, self.global_step = False, 0

    def forward(self, adjacency_matrix):
        """

        :param adjacency_matrix: csr_matrix, or numpy array
        :return:
        """
        self.init_bp(adjacency_matrix)

        with timeit(name="parallel_min_iter={0}".format(self.parallel_min_iter)):
            if self.bp_implementation_type == 'parallel':
                num_iter, max_diff = self.bp_infer_parallel()
            elif self.bp_implementation_type == 'legacy':
                num_iter, max_diff = self.bp_infer_legacy()

        if self.is_logging:
            self.bp_logging(num_iter, max_diff)

        if (self.beta < 0.05).sum() > 0:
            raise Exception("beta={0}, indicating a paramagnetic (P) phase in BP (which is not good), "
                            "please consider adding the weight for entropy_loss".format(self.beta.data))

        return self.message_map, self.marginal_psi, self.message_index_list

    def bp_logging(self, num_iter, max_diff):
        is_converged = True if max_diff < self.bp_max_diff else False
        self.logger.info("BP STATUS: \t beta \t {0}".format(self.beta.data))
        self.logger.info("BP STATUS: is_converge \t {0} \t iterations \t {1} \t max_diff \t {2:.5f}"
                         .format(is_converged, num_iter, max_diff))
        if self.writer is not None:
            self.writer.add_scalar("beta", self.beta.item(), self.global_step)
            if self.global_step > 0:  # at step 0, gradient is not computed
                self.writer.add_scalar("beta_grad", self.beta.grad.item(), self.global_step)
            self.writer.add_scalar("num_iter", num_iter, self.global_step)
            self.writer.add_scalar("max_diff", max_diff, self.global_step)
            if self.bp_implementation_type == 'parallel':
                if self.histogram_dim == 1:
                    for i in range(self.num_groups):
                        self.writer.add_histogram("message_dim{}".format(i), self.message_map[:, i].flatten(),
                                                  self.global_step)
                        self.writer.add_histogram("psi_dim{}".format(i), self.marginal_psi[:, i].flatten(),
                                                  self.global_step)
                elif self.histogram_dim == 0:
                    for e in range(self.num_messages):
                        i, j = self.message_index_list[e]
                        self.writer.add_histogram("message_{}_to_{}".format(i, j), self.message_map[e, :].flatten(),
                                                  self.global_step)
                    for n in range(self.num_nodes):
                        self.writer.add_histogram("psi_{}".format(n), self.marginal_psi[n, :].flatten(),
                                                  self.global_step)
            self.global_step += 1
        if not is_converged:
            self.logger.warning(
                "SG:BP failed to converge with max_num_iter={0}, beta={1}, max_diff={2}. "
                "Indicating a spin-glass (SG) phase in BP".format(
                    num_iter, self.beta.data, max_diff))
        if (self.beta < 0.05).sum() > 0:
            self.logger.critical("P:beta={0}, indicating paramagnetic phase in BP (which is not good), "
                                 "please consider adding the weight for entropy_loss".format(self.beta.data))

    def init_bp(self, adjacency_matrix):
        """
        Initialize BP contents
        :param adjacency_matrix:
        :return:
        """
        # seed for a stable gradient descent
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if self.is_init:
            if not csr_matrix_equal(csr_matrix(adjacency_matrix).astype(np.float), self.adjacency_matrix):
                # if the input adjacency_matrix changes, job_list need to be re initialized
                self.is_init = False
        if not self.is_init:
            self.logger.info("Initializing BP")
            self.G = nx.to_networkx_graph(adjacency_matrix)
            self.adjacency_matrix = nx.to_scipy_sparse_matrix(self.G).astype(np.float)  # connectivity matrix
            self.mean_w = self.adjacency_matrix.mean()
            self.num_nodes = self.G.number_of_nodes()
            self.mean_degree = torch.tensor(self.mean_w * self.num_nodes, device=self.beta.device)
            self.message_index_list = list(self.G.to_directed().edges())  # set is way more faster than list
            self.num_messages = len(self.message_index_list)

            if self.bp_implementation_type == 'parallel':
                self.node_id_to_index, self.w_indexed = self.init_node_w()

            if self.bp_implementation_type == 'legacy':
                # multi-thread synchronization using lock
                self._lock_psi = [threading.Lock() for _ in range(len(self.marginal_psi))]
                self._lock_mmap = [[threading.Lock() for _ in range(len(mmap))] for mmap in self.message_map]
                self._lock_h = threading.Lock()
            elif self.bp_implementation_type == 'parallel':
                self.logger.info("Initializing Job List")
                with timeit(name="Initialize Job List"):
                    # job_list to avoid race, job -> (edges, nodes)
                    self.job_list = _create_job_list_parallel(self.G, self.message_index_list, self.seed,
                                                              self.bp_type, self.is_logging, self.logger)
                self.logger.info("Creating slice and index")
                with timeit(name="Create slice and index"):
                    self._update_message_slice_and_index = [self._update_message_create_slice(job[0])
                                                            for job in self.job_list]
                    self._update_psi_index = [self._update_psi_create_index(job[1])
                                              for job in self.job_list]
            self.is_init = True

        self.logger.info("Initializing Messages")
        # with timeit(name="Initialize Messages"):
        # initialize by random psi
        marginal_psi = torch.rand(self.num_nodes, self.num_groups,
                                  device=self.beta.device)  # the marginal probability of node i
        self.marginal_psi = marginal_psi / marginal_psi.sum(1).reshape(-1, 1)
        self.message_map = self.init_messages()  # messages (bi-edges)
        # initialize external field
        self.h = (-self.beta * self.mean_w * self.marginal_psi.clone()).sum(0)

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
                "num_iter \t {:3d} \t time \t {:.2f}ms \t max_diff {}".format(num_iter, (end - start) * 1000, max_diff))

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
        write_nodes_slice_tensor, read_message_slice_tensor, index_tensor = self._update_psi_index[i]
        # sum all messages
        src = 1 + self.message_map[read_message_slice_tensor].clone() * \
              (torch.exp(self.beta * self.w_indexed[read_message_slice_tensor]) - 1)
        out = self.marginal_psi.new_ones((self.marginal_psi.shape[0], self.marginal_psi.shape[1]))
        out = scatter_mul(src, index_tensor, out=out, dim=0)
        out = out[write_nodes_slice_tensor]
        out = out * torch.exp(self.h)
        out = out / out.sum(-1).reshape(-1, 1)
        # new_marginal_psi = out[slice_tensor[:torch.nonzero(slice_tensor).max() + 1]]
        # subtract the old psi
        self.h -= -self.beta * self.mean_w * self.marginal_psi[write_nodes_slice_tensor].clone().sum(0)
        # update marginal_psi
        self.marginal_psi[write_nodes_slice_tensor] = out
        # add the new psi
        self.h += -self.beta * self.mean_w * self.marginal_psi[write_nodes_slice_tensor].clone().sum(0)

    def _update_psi_create_index(self, nodes):
        # for updating marginal_psi
        write_nodes_slice_tensor = torch.zeros(self.num_nodes, dtype=torch.uint8, device=self.beta.device)
        # for reading messages
        read_message_slice_tensor = torch.zeros(self.num_messages, dtype=torch.uint8, device=self.beta.device)
        # for torch_scatter
        index_tensor = torch.ones(self.num_messages, dtype=torch.long, device=self.beta.device) * \
                       (-1)  # a trick

        # O(nk) time
        for dst_node in nodes:
            write_nodes_slice_tensor[dst_node] = 1  # mark message to write
            neighbors = list(self.G.neighbors(dst_node))
            messages_to_dst_node = [(src_node, dst_node) for src_node in neighbors]
            message_indexes = []
            for i, j in messages_to_dst_node:
                i_neighbors = list(self.G.neighbors(i))
                i_to_j_message_index = self.node_id_to_index[i] + i_neighbors.index(j)
                message_indexes.append(i_to_j_message_index)
            index_tensor[message_indexes] = int(dst_node)  # index message to read for torch_scatter
            read_message_slice_tensor[message_indexes] = 1  # for slicing input message

            assert index_tensor.max() == -1 or index_tensor.max() == torch.nonzero(write_nodes_slice_tensor).max()
        index_tensor = index_tensor[index_tensor != -1]
        return write_nodes_slice_tensor, read_message_slice_tensor, index_tensor

    def update_message_fast(self, i):
        write_messages_slice_tensor, read_messages_slice_tensor, index_tensor = self._update_message_slice_and_index[i]
        # sum all messages
        src = 1 + self.message_map[read_messages_slice_tensor].clone() * \
              (torch.exp(self.beta * self.w_indexed[read_messages_slice_tensor]) - 1)
        out = self.message_map.new_ones((self.num_messages, self.message_map.shape[1]))
        out = scatter_mul(src, index_tensor, out=out, dim=0)
        out = out[write_messages_slice_tensor]
        out = out * torch.exp(self.h)
        out = out / out.sum(-1).reshape(-1, 1)
        # new_message = out[slice_tensor[:torch.nonzero(slice_tensor).max() + 1]]
        # updated_message = out[slice_tensor].clone()

        max_diff = (out.detach() - self.message_map[write_messages_slice_tensor].detach()).abs().max()
        # update messages
        self.message_map[write_messages_slice_tensor] = out
        return max_diff

    def _update_message_create_slice(self, target_list):
        # for updating message_map
        write_messages_slice_tensor = torch.zeros(self.num_messages, dtype=torch.uint8, device=self.beta.device)
        # for reading message_map
        read_messages_slice_tensor = torch.zeros(self.num_messages, dtype=torch.uint8, device=self.beta.device)
        # for torch_scatter
        index_tensor = torch.ones(self.num_messages, dtype=torch.long, device=self.beta.device) * \
                       (-1)  # a trick

        # O(mk) time
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
                    write_messages_slice_tensor[src_to_dst_message_index] = 1
                    read_messages_slice_tensor[src_message_indexes] = 1
                    index_tensor[src_message_indexes] = src_to_dst_message_index

            elif self.bp_type == 'exact':
                src_messages1 = [(k, src_node) for k in self.G.neighbors(src_node) if k != dst_node]
                src_messages2 = [(k, src_node) for k in self.G.nodes() if k not in [src_node, dst_node]]
                raise NotImplementedError()
        # assert index_tensor.max() == -1 or index_tensor.max() == torch.nonzero(write_messages_slice_tensor).max()
        # if index_tensor.max() != torch.nonzero(write_messages_slice_tensor).max():
        #     pass
        index_tensor = index_tensor[index_tensor != -1]
        return write_messages_slice_tensor, read_messages_slice_tensor, index_tensor

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
                "num_iter \t {:3d} \t time \t {:.2f}ms \t max_diff {}".format(num_iter, (end - start) * 1000, max_diff))

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
                               (torch.exp(self.beta * self.adjacency_matrix[i, k]) - 1))
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
                               (torch.exp(self.beta * self.adjacency_matrix[i, j]) - 1))
            this_value *= torch.exp(self.h[q])
            marginal_psi_i[q] = this_value
        marginal_psi_i = marginal_psi_i.clone() / marginal_psi_i.clone().clone().sum()
        logging.debug("update_marginal_psi: ", i, marginal_psi_i)
        return marginal_psi_i

    def init_messages(self):
        if self.bp_implementation_type == 'legacy':
            message_map = []
            for i in self.G.nodes():
                message_map_at_i = torch.rand(len(list(self.G.neighbors(i))), self.num_groups, device=self.beta.device)
                message_map_at_i = message_map_at_i / message_map_at_i.sum(1).reshape(-1, 1)
                message_map.append(message_map_at_i)
        elif self.bp_implementation_type == 'parallel':
            message_map = torch.rand(self.num_messages, self.num_groups, device=self.beta.device)
            message_map = message_map / message_map.sum(1).reshape(-1, 1)

        return message_map

    def init_node_w(self):
        node_id_to_index = dict()
        sum_index = 0
        for i in self.G.nodes():
            node_id_to_index[i] = sum_index
            sum_index += len(list(self.G.neighbors(i)))

        w_indexed = torch.zeros(self.num_messages, device=self.beta.device)
        for i in self.G.nodes():
            i_neighbors = list(self.G.neighbors(i))
            for index, j in enumerate(self.G.neighbors(i)):
                i_to_j_index = node_id_to_index[i] + index
                w_indexed[i_to_j_index] = self.adjacency_matrix[i, j]
        w_indexed = torch.stack([w_indexed for _ in range(self.num_groups)], dim=-1)

        return node_id_to_index, w_indexed

    def __repr__(self):
        return '{0}(num_groups={1}, max_num_iter={2}, bp_max_diff={3}, bp_dumping_rate={4})'.format(
            self.__class__.__name__, self.num_groups, self.max_num_iter, self.bp_max_diff, self.bp_dumping_rate)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    num_groups = 4
    sizes = [100] * num_groups
    P = np.ones((num_groups, num_groups)) * 0.05
    for i in range(len(P)):
        P[i][i] = P[i][i] * 5
    g = nx.stochastic_block_model(sizes, P, seed=0)
    adj = nx.to_scipy_sparse_matrix(g)
    mean_degree = adj.mean() * g.number_of_nodes()
    for parallel_min_iter in range(50, 300, 50):
        bp = BeliefPropagation(num_groups, parallel_min_iter=parallel_min_iter)
        entropy = EntropyLoss()
        # bp = bp.cuda()
        message_map, marginal_psi, message_index_list = bp(adj)
        entropy_loss = entropy(marginal_psi)
        entropy_loss.backward(retain_graph=True)
