import threading
import time
from boxx import timeit
import random
from scipy.sparse import csr_matrix

import networkx as nx
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from multiprocessing import Pool

from bp_helper import _create_job_list_parallel
from utils import *
from bp_lagecy import *


class BeliefPropagation(nn.Module):

    def __init__(self,
                 num_groups,
                 mean_degree=None,
                 max_num_iter=100,
                 bp_max_diff=1e-2,
                 bp_dumping_rate=1.0,
                 num_workers=1,
                 is_logging=True,
                 verbose_iter=False,
                 verbose_init=True,
                 bp_type='approximate',
                 bp_implementation_type='parallel',
                 parallel_max_node_percent=None,
                 summary_writer=None,
                 histogram_dim=1,
                 disable_gradient=False,
                 seed=0):
        """
        Belief Propagation for pooling on graphs

        :param num_groups: pooling size
        :param mean_degree: not required, but highly recommend
        :param max_num_iter: max BP iteration to converge,
                            NOTE: large number of iteration blows up RAM for gradient descent.
        :param bp_max_diff: max BP diff to converge
        :param bp_dumping_rate: BP dumping rate, 1.0 as default
        :param verbose_iter: verbose during iteration
        :param verbose_init: verbose during initialization
        :param num_workers: multi-threading workers (for legacy)
        :param is_logging: if logging
        :param bp_type: 'approximate' with external field h, or 'exact'
        :param bp_implementation_type: 'parallel` with torch_scatter, or 'legacy'
        :param parallel_max_node_percent: maximum percent of nodes to parallelize,
                                few value means faster inference, large value means worse approximation (h)
        :param summary_writer: TensorBoardX
        :param histogram_dim: 1 is fast, 0 is extremely slow but detailed
        :param disable_gradient: to save memory for large number of iterations
        :param seed:
        """

        super(BeliefPropagation, self).__init__()

        if disable_gradient and bp_implementation_type == 'legacy':
            raise NotImplementedError()
        assert bp_type in ['exact', 'approximate']
        assert bp_implementation_type in ['parallel', 'legacy']
        if bp_implementation_type == 'parallel':
            global scatter_mul
            from torch_scatter import scatter_mul

        self.bp_type = bp_type
        self.bp_implementation_type = bp_implementation_type
        self.disable_gradient = disable_gradient
        self.parallel_max_node_percent = parallel_max_node_percent
        self.seed = seed
        self.is_logging = is_logging
        self.logger = logging.getLogger(str(self.__class__.__name__))
        if verbose_iter:
            logging.basicConfig(level=logging.INFO)
        self.verbose_init = verbose_init
        self.writer = summary_writer
        self.histogram_dim = histogram_dim
        self.num_workers = num_workers
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
            raise Exception("What are u doing???")
        self.beta = nn.Parameter(data=beta, requires_grad=(True if not self.disable_gradient else False))

        # self.logger.info('\t mean_degree \t {0:.2f} \t beta \t {1} \t '.format(mean_degree, self.beta.data.item()))
        self.logger.warning('beta initialized to {0}'.format(self.beta.data))

        self.is_init, self.global_step = False, 0

    def forward(self, edge_index, edge_attr=None):
        """

        :param edge_index:
        :param edge_attr:
        :return:
        """
        self.init_bp(edge_index, edge_attr)

        if self.bp_implementation_type == 'parallel':
            num_iter, max_diff = self.bp_infer_parallel()
        elif self.bp_implementation_type == 'legacy':
            num_iter, max_diff = bp_infer_legacy(self)

        if self.is_logging:
            self.bp_logging(num_iter, max_diff)

        if (self.beta < 0.05).sum() > 0:
            raise Exception("beta={0}, indicating a paramagnetic (P) phase in BP"
                            .format(self.beta.data))

        return self.message_map, self.marginal_psi, self.message_index_set

    def bp_logging(self, num_iter, max_diff):
        is_converged = True if max_diff < self.bp_max_diff else False
        self.logger.info("BP STATUS: \t beta \t {0}".format(self.beta.data))
        self.logger.info("BP STATUS: is_converge \t {0} \t iterations \t {1} \t max_diff \t {2:.2e}"
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
                    for index in range(self.num_messages):
                        i, j = self.message_to_index_map_inverse[index]
                        self.writer.add_histogram("message_{}_to_{}".format(i, j), self.message_map[index, :].flatten(),
                                                  self.global_step)
                    for n in range(self.num_nodes):
                        self.writer.add_histogram("psi_{}".format(n), self.marginal_psi[n, :].flatten(),
                                                  self.global_step)
            self.global_step += 1
        if not is_converged:
            self.logger.warning("SG:BP failed to converge with max_num_iter={0}, beta={1:.5f}, max_diff={2:.2e}. "
                                .format(num_iter, self.beta.data, max_diff))
            self.logger.warning("parallel_max_node_percent={0:.2f}, in case of parallelization, "
                                "reducing this percentage is good for BP to converge"
                                .format(self.parallel_max_node_percent))
        if (self.beta < 0.05).sum() > 0:
            self.logger.critical("P:beta={0}, indicating paramagnetic phase in BP, "
                                 "please consider adding the weight for entropy_loss".format(self.beta.data))

    def init_bp(self, edge_index, edge_attr=None):
        """

        :param edge_index:
        :param edge_attr:
        :return:
        """
        # seed for a stable gradient descent
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if self.is_init:
            # if not csr_matrix_equal(csr_matrix(adjacency_matrix).astype(np.float), self.adjacency_matrix):
            if not (torch.all(torch.eq(edge_index, self.edge_index)) and
                    torch.all(torch.eq(edge_attr, self.edge_attr))):
                # if the input adjacency_matrix changes, job_list need to be re initialized
                self.is_init = False
        if not self.is_init:
            self.logger.info("Initializing BP")
            adjacency_matrix = edge_index_to_adj(edge_index, edge_attr)
            self.G = nx.to_networkx_graph(adjacency_matrix)
            self.adjacency_matrix = adjacency_matrix
            self.num_nodes = self.G.number_of_nodes()
            self.mean_w = self.adjacency_matrix.mean()
            self.mean_degree = torch.tensor(self.mean_w * self.num_nodes, device=self.beta.device)
            self.message_to_index_map = {e: i for i, e in enumerate(self.G.to_directed().edges())}
            self.message_to_index_map_inverse = {v: k for k, v in self.message_to_index_map.items()}
            self.message_index_set = set(self.message_to_index_map.keys())  # set is way more faster than list
            self.num_messages = len(self.message_index_set)
            self.parallel_max_node_percent = (1 / torch.sqrt(self.mean_degree + 1)
                                              if self.parallel_max_node_percent is None else
                                              self.parallel_max_node_percent)

        self.logger.info("Initializing Messages")
        # with timeit(name="Initialize Messages"):
        # initialize by random psi
        marginal_psi = torch.rand(self.num_nodes, self.num_groups,
                                  device=self.beta.device)  # the marginal probability of node i
        self.marginal_psi = marginal_psi / marginal_psi.sum(1).reshape(-1, 1)
        self.message_map = self.init_messages() if self.bp_implementation_type == 'parallel' \
            else init_messages_legacy(self)  # messages (bi-edges)
        # initialize external field
        self.h = (-self.beta * self.mean_w * (
            self.marginal_psi.clone()
            if not self.disable_gradient else
            self.marginal_psi
        )).sum(0)

        if not self.is_init:
            if not self.is_beta_init:
                self.logger.warning("Initializing beta again with "
                                    "'torch.log(self.num_groups / (torch.sqrt(self.mean_degree) - 1) + 1)'")
                beta = torch.log(self.num_groups / (torch.sqrt(self.mean_degree) - 1) + 1)
                beta = beta if not torch.isnan(beta).sum() > 0 else torch.tensor(1.0, dtype=torch.float)  # nan
                self.beta.data = beta
                self.logger.warning('beta initialized to {0}'.format(self.beta.data))

            if self.bp_implementation_type == 'parallel':
                self.logger.info("Initializing indexes")
                self.w_indexed = self.init_node_w()

            if self.bp_implementation_type == 'legacy':
                # multi-thread synchronization using lock
                self._lock_psi = [threading.Lock() for _ in range(len(self.marginal_psi))]
                self._lock_mmap = [[threading.Lock() for _ in range(len(mmap))] for mmap in self.message_map]
                self._lock_h = threading.Lock()
            elif self.bp_implementation_type == 'parallel':
                self.logger.info("Initializing Job List")
                # job_list to avoid race, job -> (edges, nodes)
                self.job_list = _create_job_list_parallel(
                    self.G, self.message_index_set,
                    self.parallel_max_node_percent * self.num_nodes,
                    self.seed, self.bp_type, self.verbose_init
                )
                self.logger.info("Creating slice and index")
                # self._update_message_slice_and_index, self._update_psi_slice_and_index = \
                #     self.create_slice_and_index_mp()
                self._update_message_slice_and_index = [self._update_message_create_slice(job[0])
                                                        for job in (tqdm(self.job_list, desc="create message slice")
                                                                    if self.verbose_init else self.job_list)]
                self._update_psi_slice_and_index = [self._update_psi_create_index(job[1])
                                                    for job in (tqdm(self.job_list, desc="create psi slice")
                                                                if self.verbose_init else self.job_list)]
            self.is_init = True

        self.logger.info("BP Initialization Completed")

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
                "num_iter \t {:3d} \t time \t {:.2f}ms \t max_diff {:.2e}".format(num_iter, (end - start) * 1000,
                                                                                  max_diff))

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
        write_nodes_slice_tensor, read_messages_slice_tensor, index_tensor = self._update_psi_slice_and_index[i]
        if read_messages_slice_tensor.sum() == 0:  # isolated node
            return
        # sum all messages
        src = 1 + (
            self.message_map[read_messages_slice_tensor].clone()
            if not self.disable_gradient else
            self.message_map[read_messages_slice_tensor]
        ) * (torch.exp(self.beta * self.w_indexed[read_messages_slice_tensor]) - 1)
        out = self.marginal_psi.new_ones((self.num_nodes, self.num_groups))
        out = scatter_mul(src, index_tensor, out=out, dim=0)
        out = out[write_nodes_slice_tensor]
        out = out * torch.exp(self.h)
        out = out / out.sum(-1).reshape(-1, 1)

        # subtract the old psi
        self.h -= -self.beta * self.mean_w * (
            self.marginal_psi[write_nodes_slice_tensor].clone()
            if not self.disable_gradient else
            self.marginal_psi[write_nodes_slice_tensor]
        ).sum(0)
        # update marginal_psi
        self.marginal_psi[write_nodes_slice_tensor] = out
        # add the new psi
        self.h += -self.beta * self.mean_w * (
            self.marginal_psi[write_nodes_slice_tensor].clone()
            if not self.disable_gradient else
            self.marginal_psi[write_nodes_slice_tensor]
        ).sum(0)

    def _update_psi_create_index(self, nodes):
        # for updating marginal_psi
        write_nodes_slice_tensor = torch.zeros(self.num_nodes, dtype=torch.uint8, device=self.beta.device)
        # for reading messages
        read_message_slice_tensor = torch.zeros(self.num_messages, dtype=torch.uint8, device=self.beta.device)
        # for torch_scatter
        index_tensor = torch.ones(self.num_messages, dtype=torch.long, device=self.beta.device) * \
                       (-1)  # a trick

        # O(nk)
        for dst_node in nodes:
            write_nodes_slice_tensor[dst_node] = 1  # mark message to write
            src_message_indexes = [self.message_to_index_map[(src_node, dst_node)]
                                   for dst_node, src_node in self.G.edges(dst_node)]
            if len(src_message_indexes) > 0:
                index_tensor[src_message_indexes] = int(dst_node)  # index message to read for torch_scatter
                read_message_slice_tensor[src_message_indexes] = 1  # for slicing input message

        # assert index_tensor.max() == -1 or index_tensor.max() == torch.nonzero(write_nodes_slice_tensor).max()
        index_tensor = index_tensor[index_tensor != -1]  # remove redundant indexes
        return write_nodes_slice_tensor, read_message_slice_tensor, index_tensor

    def update_message_fast(self, i):
        write_messages_slice_tensor, read_messages_slice_tensor, index_tensor = self._update_message_slice_and_index[i]
        if read_messages_slice_tensor.sum() == 0:  # isolated node
            return 0
        # sum all messages
        src = 1 + (
            self.message_map[read_messages_slice_tensor].clone()
            if not self.disable_gradient else
            self.message_map[read_messages_slice_tensor]
        ) * (torch.exp(self.beta * self.w_indexed[read_messages_slice_tensor]) - 1)
        out = self.message_map.new_ones((self.num_messages, self.num_groups))
        out = scatter_mul(src, index_tensor, out=out, dim=0)
        out = out[write_messages_slice_tensor]
        out = out * torch.exp(self.h)
        out = out / out.sum(-1).reshape(-1, 1)

        max_diff = (out.detach() - self.message_map[write_messages_slice_tensor].detach()).abs().max()
        # update messages
        self.message_map[write_messages_slice_tensor] = out
        return max_diff

    def _update_message_create_slice(self, edge_list):
        # for updating message_map
        write_messages_slice_tensor = torch.zeros(self.num_messages, dtype=torch.uint8, device=self.beta.device)
        # for reading message_map
        read_messages_slice_tensor = torch.zeros(self.num_messages, dtype=torch.uint8, device=self.beta.device)
        # for torch_scatter
        index_tensor = torch.ones(self.num_messages, dtype=torch.long, device=self.beta.device) * \
                       (-1)  # a trick

        # O(mk)
        for src_node, dst_node in edge_list:
            src_to_dst_message_index = self.message_to_index_map[(src_node, dst_node)]
            if self.bp_type == 'approximate':
                src_message_indexes = [self.message_to_index_map[(k, src_node)]
                                       for k in self.G.neighbors(src_node) if k != dst_node]
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
        index_tensor = index_tensor[index_tensor != -1]  # remove redundant indexes
        return write_messages_slice_tensor, read_messages_slice_tensor, index_tensor

    def create_slice_and_index_mp(self):
        _update_message_slice_and_index, _update_psi_slice_and_index = [], []
        with Pool() as p:
            for result in (
                    tqdm(p.imap(self._create_slice_and_index_mp, self.job_list),
                         desc="create slice and index mp",
                         total=len(self.job_list)
                         ) if self.verbose_init else
                    p.imap(self._create_slice_and_index_mp, self.job_list)
            ):
                _update_message_slice_and_index.append(result[0])
                _update_psi_slice_and_index.append(result[1])
        return _update_message_slice_and_index, _update_psi_slice_and_index

    def _create_slice_and_index_mp(self, job):
        edge_list, node_list = job
        return self._update_message_create_slice(edge_list), self._update_psi_create_index(node_list)

    def init_messages(self):
        message_map = torch.rand(self.num_messages, self.num_groups, device=self.beta.device)
        message_map = message_map / message_map.sum(1).reshape(-1, 1)
        return message_map

    def init_node_w(self):

        w_indexed = torch.tensor([
            self.adjacency_matrix[i, j] for k, (i, j) in (
                tqdm(self.message_to_index_map_inverse.items(), desc="w_indexed")
                if self.verbose_init else
                self.message_to_index_map_inverse.items()
            )], device=self.beta.device)
        w_indexed = torch.stack([w_indexed for _ in range(self.num_groups)], dim=-1)

        return w_indexed

    def __repr__(self):
        return '{0}(num_groups={1}, max_num_iter={2}, bp_max_diff={3}, bp_dumping_rate={4})'.format(
            self.__class__.__name__, self.num_groups, self.max_num_iter, self.bp_max_diff, self.bp_dumping_rate)


if __name__ == '__main__':
    num_groups = 2
    sizes = np.asarray([50] * num_groups)
    epslion = 0.3
    P = np.ones((num_groups, num_groups)) * 0.01
    for i in range(len(P)):
        P[i][i] = P[i][i] / epslion
    G = nx.stochastic_block_model(sizes, P, seed=0)
    adj = nx.to_scipy_sparse_matrix(G)
    edge_index, edge_attr = adj_to_edge_index(adj)
    c = nx.to_scipy_sparse_matrix(G).mean() * G.number_of_nodes()
    ground_truth = []
    for i in G.nodes():
        ground_truth.append(G.node[i]['block'])
    print(c)
    epslion_ast = (np.sqrt(c) - 1) / (np.sqrt(c) - 1 + num_groups)
    print(epslion, epslion_ast, epslion < epslion_ast)
    percent = 1 / np.sqrt(c + 1)
    print(percent)

    bp = BeliefPropagation(num_groups, mean_degree=c, verbose_iter=True,
                           max_num_iter=100,
                           parallel_max_node_percent=None,
                           bp_max_diff=1e-2, disable_gradient=True, )
    # bp.beta.data = torch.tensor(1.)
    entropy = EntropyLoss()
    # bp = bp.cuda()
    message_map, marginal_psi, message_index_list = bp(edge_index, edge_attr)
    entropy_loss = entropy(marginal_psi)
    print(entropy_loss)
