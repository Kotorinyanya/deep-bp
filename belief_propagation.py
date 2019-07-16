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
from tqdm import tqdm
from multiprocessing import Pool

from bp_helper import _create_job_list_parallel_fast, _node_list_to_slice_and_index_message, \
    _node_list_to_slice_and_index_psi
from bp_lagecy import *
from bp_helper import *
from bp_lagecy import _create_job_list_parallel, _update_message_create_slice, _update_psi_create_index

from utils import *


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
                 disable_gradient=False,
                 bp_init_type='new',
                 save_job=True,
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
        :param disable_gradient: to save memory for large number of iterations
        :param seed:
        """

        super(BeliefPropagation, self).__init__()

        self.save_job = save_job
        self.bp_init_type = bp_init_type
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

        return self.message_map, self.marginal_psi, \
               self.message_index_set if self.bp_init_type == 'old' else self.csc_map

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
                for i in range(self.num_groups):
                    self.writer.add_histogram("message_dim{}".format(i), self.message_map[:, i].flatten(),
                                              self.global_step)
                    self.writer.add_histogram("psi_dim{}".format(i), self.marginal_psi[:, i].flatten(),
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
            self.adjacency_matrix = edge_index_to_csr(edge_index, edge_attr)
            if edge_attr is not None:
                rand_int = np.random.randint(edge_index.shape[1])
                assert edge_attr.shape[1] == 1
                edge_attr = edge_attr.reshape(-1)
                # check edge_index and edge_attr is in ascending order:
                assert self.adjacency_matrix[edge_index[0, rand_int], edge_index[1, rand_int]] == edge_attr[rand_int]
            # check if symmetric
            assert self.adjacency_matrix[0, 1] == self.adjacency_matrix[1, 0]
            if self.bp_implementation_type == 'legacy':
                self.G = nx.to_networkx_graph(self.adjacency_matrix)
            self.num_nodes = self.adjacency_matrix.shape[0]
            self.mean_w = self.adjacency_matrix.mean()
            self.mean_degree = torch.tensor(self.mean_w * self.num_nodes, device=self.beta.device)

            self.logger.info("Initializing Message Indexes")
            if self.bp_init_type == 'new':
                self.csc_map = adj_to_csc_map(self.adjacency_matrix)
                self.num_messages = self.adjacency_matrix.count_nonzero() + 1  # plus one for csr_map starts with 1
            if self.bp_init_type == 'old':
                self.message_to_index_map = {e: i for i, e in enumerate(
                    tqdm(zip(*self.adjacency_matrix.nonzero()), desc="indexes",
                         total=self.adjacency_matrix.count_nonzero())
                    if self.verbose_init else
                    zip(*self.adjacency_matrix.nonzero())
                )}
                self.message_index_set = set(self.message_to_index_map.keys())  # set is way more faster than list
                self.num_messages = len(self.message_index_set)
            self.parallel_max_node_percent = (1 / torch.sqrt(self.mean_degree + 1)
                                              if self.parallel_max_node_percent is None else
                                              self.parallel_max_node_percent)
            self.w_indexed = torch.stack([(edge_attr
                                           if edge_attr is not None else
                                           torch.ones(edge_index.shape[1], dtype=torch.float, device=self.beta.device))
                                          for _ in range(self.num_groups)]).reshape(-1, self.num_groups)
            if self.bp_init_type == 'new':
                self.w_indexed = torch.cat(
                    [torch.ones(1, self.num_groups, dtype=torch.float, device=self.beta.device), self.w_indexed],
                    dim=0)

        self.logger.info("Initializing Messages")
        # initialize psi
        self.marginal_psi = torch.rand(self.num_nodes, self.num_groups,
                                       device=self.beta.device)  # the marginal probability of node i
        self.marginal_psi = self.marginal_psi / self.marginal_psi.sum(1).reshape(-1, 1)
        # initialize message
        if self.bp_implementation_type == 'parallel':
            self.message_map = torch.rand(self.num_messages, self.num_groups, device=self.beta.device)
            self.message_map = self.message_map / self.message_map.sum(1).reshape(-1, 1)
        elif self.bp_implementation_type == 'legacy':
            self.message_map = init_messages_legacy(self)
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

            if self.bp_implementation_type == 'legacy':
                # multi-thread synchronization using lock
                self._lock_psi = [threading.Lock() for _ in range(len(self.marginal_psi))]
                self._lock_mmap = [[threading.Lock() for _ in range(len(mmap))] for mmap in self.message_map]
                self._lock_h = threading.Lock()
            elif self.bp_implementation_type == 'parallel':
                self.logger.info("Initializing Job List")
                if self.bp_init_type == 'old':
                    # job_list to avoid race, job -> (edges, nodes)
                    self.job_list = _create_job_list_parallel(
                        self.adjacency_matrix, self.message_index_set,
                        self.parallel_max_node_percent * self.num_nodes,
                        self.seed, self.verbose_init
                    )
                    self.logger.info("Creating slice and index")
                    # self._update_message_slice_and_index, self._update_psi_slice_and_index = \
                    #     self.create_slice_and_index_mp()
                    self._update_message_slice_and_index = [_update_message_create_slice(self, job[0])
                                                            for job in (tqdm(self.job_list, desc="create message slice")
                                                                        if self.verbose_init else self.job_list)]
                    self._update_psi_slice_and_index = [_update_psi_create_index(self, job[1])
                                                        for job in (tqdm(self.job_list, desc="create psi slice")
                                                                    if self.verbose_init else self.job_list)]
                if self.bp_init_type == 'new':
                    self.job_list = _create_job_list_parallel_fast(self.csc_map,
                                                                   self.parallel_max_node_percent * self.num_nodes,
                                                                   self.seed,
                                                                   self.verbose_init)
                    if self.save_job:
                        iter_items = enumerate(_node_list_to_slice_and_index_message(
                            range(self.num_nodes), self.csc_map, self.num_messages,
                            self.beta.device
                        ))
                        self._saved_message_update_dict = {
                            n: (w, r, i)
                            for n, (w, r, i) in (
                                tqdm(iter_items, desc="message update dict", total=self.num_nodes)
                                if self.verbose_init else
                                iter_items
                            )
                        }
                        iter_items = enumerate(_node_list_to_slice_and_index_psi(
                            range(self.num_nodes), self.csc_map, self.num_messages,
                            self.beta.device
                        ))
                        self._saved_psi_update_dict = {
                            n: (w, r, i)
                            for n, (w, r, i) in (
                                tqdm(iter_items, desc="psi update dict", total=self.num_nodes)
                                if self.verbose_init else
                                iter_items
                            )
                        }

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
            for i in range(len(self.job_list)):
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

    def bp_infer_step_fast(self, idx):
        if self.bp_type == 'exact':
            raise NotImplementedError("BP exact if not implemented for BP parallel")

        write_slice, read_slice, index = self._update_message_slice_and_index[idx] \
            if self.bp_init_type == 'old' else [
            (
                torch.cat(w), torch.cat(r), torch.cat(i)
            ) for (w, r, i) in [(
                zip(*(self._saved_message_update_dict[n]
                      for n in self.job_list[idx])))]
        ][0]
        diff = self.update_message_fast(write_slice, read_slice, index)

        write_slice, read_slice, index = self._update_psi_slice_and_index[idx] \
            if self.bp_init_type == 'old' else [
            (
                torch.cat(w), torch.cat(r), torch.cat(i)
            ) for (w, r, i) in [(
                zip(*(self._saved_psi_update_dict[n]
                      for n in self.job_list[idx])))]
        ][0]
        self.update_psi_h_fast(write_slice, read_slice, index)

        return diff

    def update_psi_h_fast(self, write_nodes_slice_tensor, read_messages_slice_tensor, index_tensor):
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

    def update_message_fast(self, write_messages_slice_tensor, read_messages_slice_tensor, index_tensor):
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

    bp = BeliefPropagation(num_groups, mean_degree=None, verbose_iter=True,
                           max_num_iter=100,
                           parallel_max_node_percent=0.2,
                           bp_max_diff=1e-2, disable_gradient=True,
                           bp_init_type='new', save_job=True)
    # bp.beta.data = torch.tensor(1.75)
    entropy = EntropyLoss()
    # bp = bp.cuda()
    message_map, marginal_psi, message_index_list = bp(edge_index)
    entropy_loss = entropy(marginal_psi)
    print(entropy_loss)
