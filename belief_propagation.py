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


class BeliefPropagation(nn.Module):

    def __init__(self,
                 num_groups,
                 mean_degree=None,
                 max_num_iter=10,
                 bp_max_diff=5e-1,
                 bp_dumping_rate=1.0,
                 save_init_path='.cache/',
                 dataset_unique_name='',
                 is_logging=True,
                 is_writing_hist=False,
                 verbose_iter=False,
                 verbose_init=True,
                 bp_type='approximate',
                 bp_implementation_type='parallel',
                 parallel_max_node_percent=0.1,
                 summary_writer=None,
                 disable_gradient=False,
                 bp_init_type='new',
                 multi_processing=False,
                 save_node_dict=False,
                 node_job_slice=None,
                 save_full_init=False,
                 batch_run=False,
                 num_workers=1,
                 mp_chunksize=2,
                 mp_processes_count=4,
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
        :param node_job_slice: for parallel init
        :param seed:
        """

        super(BeliefPropagation, self).__init__()

        self.is_writing_hist = is_writing_hist  # tb
        self.batch_run = batch_run
        self.node_job_slice = node_job_slice
        if disable_gradient and bp_implementation_type == 'legacy':
            raise NotImplementedError()
        assert bp_type in ['exact', 'approximate']
        assert bp_implementation_type in ['parallel', 'legacy']
        assert bp_init_type in ['old', 'new']
        if bp_implementation_type == 'parallel':
            global scatter_mul
            from torch_scatter import scatter_mul

        self.dataset_unique_name = dataset_unique_name if dataset_unique_name != '' else None
        # TODO: change for dataset_unique_name
        self.dataset_uuid = None
        self.save_full_init = save_full_init
        self.save_node_dict = save_node_dict
        self.multi_processing = multi_processing
        self.save_init_path = save_init_path
        self.mp_processes_count = mp_processes_count if mp_processes_count is not None else os.cpu_count()
        self.mp_chunksize = mp_chunksize
        self.bp_init_type = bp_init_type
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
            raise Exception("What???")
        self.beta = nn.Parameter(data=beta, requires_grad=(True if not self.disable_gradient else False))

        # self.logger.info('\t mean_degree \t {0:.2f} \t beta \t {1} \t '.format(mean_degree, self.beta.data.item()))
        self.logger.warning('beta initialized to {0}'.format(self.beta.data))

        self.job_attrs = [
            'G',
            'adjacency_matrix',
            'mean_w',
            'num_nodes',
            'mean_degree',
            'is_weighted',
            'marginal_psi',
            'message_map',
            'h',
            'w_indexed',
            'message_index_list',
            'node_id_to_index',
            'num_messages',
            'message_index_set',
            'dok_map',
            '_lock_psi',
            '_lock_mmap',
            '_lock_h',
            'csc_map',
            'job_list',
            '_update_message_slice_and_index',
            '_update_psi_slice_and_index',
            '_saved_message_update_dict',
            '_saved_psi_update_dict'
        ]
        for name in self.job_attrs:
            setattr(self, name, None)

        self.is_init, self.global_step = False, 0
        self.edge_index, self.edge_attr = None, None

    def forward(self, edge_index, num_nodes, edge_attr=None, slices=None):
        """

        :param edge_index:
        :param edge_attr:
        :param slices: for batch run `TO BE IMPLEMENTED`
        :return:
        """
        self.init_bp(edge_index, num_nodes, edge_attr, slices)

        if self.bp_implementation_type == 'parallel':
            num_iter, max_diff = self.bp_infer_parallel()
        elif self.bp_implementation_type == 'legacy':
            num_iter, max_diff = bp_infer_legacy(self)

        if self.is_logging:
            self.bp_logging(num_iter, max_diff)

        if (self.beta < 0.05).sum() > 0:
            raise Exception("beta={0}, indicating a paramagnetic (P) phase in BP"
                            .format(self.beta.data))

        return self.message_map, self.marginal_psi, self.csc_map

    def init_bp(self, edge_index, num_nodes, edge_attr=None, slices=None):
        # uuid for saving init
        d = locals()
        self.dataset_uuid = my_uuid(str(dict((k, d[k]) for k in ('edge_index', 'num_nodes', 'edge_attr'))))

        # seed for a stable gradient descent
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if self.is_init:
            try:
                # check if the graph changes
                if (not torch.all(torch.eq(edge_index, self.edge_index))) or \
                        (edge_attr is not None and torch.all(torch.eq(edge_attr, self.edge_attr))):
                    self.is_init = False
            except:
                self.is_init = False

        if not self.is_init:
            if self.save_full_init:
                self._init_bp_with_save(edge_index, num_nodes, edge_attr, slices)
            else:
                self._init_bp(edge_index, num_nodes, edge_attr, slices)
            self.is_init = True

        # messages need to be initialized every single run with the same random value (for gradients)
        self._init_messages()

        self.logger.info("BP Initialization Completed")

    def bp_logging(self, num_iter, max_diff):
        self.num_iter = num_iter
        self.is_converged = True if max_diff < self.bp_max_diff else False
        self.logger.info("BP STATUS: \t beta \t {0}".format(self.beta.data))
        self.logger.info("BP STATUS: is_converge \t {0} \t iterations \t {1} \t max_diff \t {2:.2e}"
                         .format(self.is_converged, num_iter, max_diff))
        if self.writer is not None:
            self.writer.add_scalar("beta", self.beta.item(), self.global_step)
            if self.beta.grad is not None:  # at step 0, gradient is not computed
                self.writer.add_scalar("beta_grad", self.beta.grad.item(), self.global_step)
            self.writer.add_scalar("num_iter", num_iter, self.global_step)
            self.writer.add_scalar("max_diff", max_diff, self.global_step)
            if self.bp_implementation_type == 'parallel' and self.is_writing_hist:
                for i in range(self.num_groups):
                    self.writer.add_histogram("bp_hist/message_dim{}".format(i), self.message_map[:, i].flatten(),
                                              self.global_step)
                    self.writer.add_histogram("bp_hist/psi_dim{}".format(i), self.marginal_psi[:, i].flatten(),
                                              self.global_step)
            self.global_step += 1
        if not self.is_converged:
            self.logger.info("SG:BP failed to converge with max_num_iter={0}, beta={1}, max_diff={2}. "
                             .format(num_iter, self.beta.data, max_diff))
            self.logger.info("parallel_max_node_percent={0:.2f}, in case of parallelization, "
                             "reducing this percentage is good for BP to converge"
                             .format(self.parallel_max_node_percent))
        if (self.beta < 0.05).sum() > 0:
            self.logger.critical("P:beta={0}, indicating paramagnetic phase in BP, "
                                 "please consider adding the weight for entropy_loss".format(self.beta.data))

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
        ) * (torch.exp(
            self.beta * (
                self.w_indexed[read_messages_slice_tensor]
                if self.is_weighted else
                1
            )
        ) - 1)
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
        ) * (torch.exp(
            self.beta * (
                self.w_indexed[read_messages_slice_tensor]
                if self.is_weighted else
                1
            )
        ) - 1)
        out = self.message_map.new_ones((self.num_messages, self.num_groups))
        out = scatter_mul(src, index_tensor, out=out, dim=0)
        out = out[write_messages_slice_tensor]
        out = out * torch.exp(self.h)
        out = out / out.sum(-1).reshape(-1, 1)

        max_diff = (out.detach() - self.message_map[write_messages_slice_tensor].detach()).abs().max()
        # update messages
        self.message_map[write_messages_slice_tensor] = out
        return max_diff

    """ ----------------------------------------------------------- """
    """ BP needs a very complicated initialization to run very fast """
    """ ----------------------------------------------------------- """

    @use_logging(level='info')
    def _init_bp_with_save(self, edge_index, num_nodes, edge_attr=None, slices=None):
        try:
            self._load_full_bp_init()
            self.logger.info("Succeed to load BP Job from file, skipping init...")
            if not self.is_beta_init:
                self._init_beta()
                self.is_beta_init = True
        except Exception as e:
            self.logger.info(e)
            self.logger.info("Failed to load BP Job from file, running init...")
            self._init_bp(edge_index, num_nodes, edge_attr, slices)
            try:
                self._save_full_bp_init()
                self.logger.info("Saved BP Job to {}".format(self.save_init_path))
            except Exception as e:
                self.logger.info(e)
                self.logger.info("Failed to save BP Job to file")

    @use_logging(level='info')
    def _init_bp(self, edge_index, num_nodes, edge_attr=None, slices=None):
        if self.batch_run and self.bp_implementation_type == 'parallel' and self.bp_init_type == 'new':
            self._init_bp_batch_run(edge_index, slices, num_nodes, edge_attr)
        else:
            self._init_graph(edge_index, num_nodes, edge_attr)
            if not self.is_beta_init:
                self._init_beta()
                self.is_beta_init = True
            if self.bp_implementation_type == 'legacy':
                self._init_lock_legacy()
            elif self.bp_implementation_type == 'parallel':
                if self.bp_init_type == 'old':
                    self._init_message_indexes_old()
                    self._init_job_list_old()
                if self.bp_init_type == 'new':
                    self._init_message_indexes_new()
                    self._init_job_list_new()

    @use_logging(level='info')
    def _init_bp_batch_run(self, edge_index_list, slices, num_nodes, edge_attr_list=None):
        # raise NotImplementedError("TODO")
        if edge_attr_list is None:
            edge_attr_list = [None] * len(edge_index_list)
        batch_job_list, batch_saved_message_update_dict, batch_saved_psi_update_dict = [], [], []

        for edge_index, edge_attr in zip(edge_index_list, edge_attr_list):
            self._init_graph(edge_index, num_nodes, edge_attr)
            self._init_message_indexes_new()
            self._init_job_list_new()
            batch_job_list.append(self.job_list)
            batch_saved_message_update_dict.append(self._saved_message_update_dict)
            batch_saved_psi_update_dict.append(self._saved_psi_update_dict)

        self._init_graph(edge_index, edge_attr)
        self._init_message_indexes_new()

    @use_logging(level='info')
    def _load_full_bp_init(self):
        path = osp.join(self.save_init_path, self.dataset_uuid)
        with open(path, 'rb') as inf:
            self.logger.info("loading init {}".format(path))
            load_dict = torch.load(inf)
            for k in self.job_attrs:
                self.__dict__[k] = load_dict[k]

    @use_logging(level='info')
    def _save_full_bp_init(self):
        # try:
        #     os.mkdir(self.save_init_path)
        # except Exception as e:
        #     print(e)
        with open(osp.join(self.save_init_path, self.dataset_uuid), 'wb') as ouf:
            save_dict = {k: v for k, v in self.__dict__.items() if k in self.job_attrs}
            torch.save(save_dict, ouf)

    @use_logging(level='info')
    def _init_graph(self, edge_index, num_nodes=None, edge_attr=None):
        self.adjacency_matrix = edge_index_to_csr(edge_index, num_nodes, edge_attr)
        if edge_attr is not None:
            rand_int = np.random.randint(edge_index.shape[1])
            assert edge_attr.shape[1] == 1
            edge_attr = edge_attr.reshape(-1)
            # check edge_index and edge_attr is in ascending order:
            assert self.adjacency_matrix[edge_index[0, rand_int], edge_index[1, rand_int]] == edge_attr[rand_int]
            # TODO: size num_groups in w is still dummy
            self.w_indexed = torch.stack([edge_attr for _ in range(self.num_groups)]).reshape(-1, self.num_groups)
            if self.bp_init_type == 'new':  # dummy at 0
                self.w_indexed = torch.cat(
                    [torch.ones(1, self.num_groups, dtype=torch.float, device=self.beta.device), self.w_indexed],
                    dim=0)
            self.is_weighted = True
        else:
            self.is_weighted = False
        # check if symmetric
        assert self.adjacency_matrix[0, 1] == self.adjacency_matrix[1, 0]
        if self.bp_implementation_type == 'legacy':  # deprecated
            self.logger.warning("legacy is deprecated")
            self.G = nx.to_networkx_graph(self.adjacency_matrix)
        self.num_nodes = num_nodes if num_nodes is not None else self.adjacency_matrix.shape[-1]
        self.mean_w = self.adjacency_matrix.sum() / np.sum(self.adjacency_matrix.sum(axis=1) > 0) ** 2
        # self.mean_w = self.adjacency_matrix.mean()
        self.mean_degree = torch.tensor(self.mean_w * self.num_nodes, device=self.beta.device)
        self.parallel_max_node_percent = (1 / torch.sqrt(self.mean_degree + 1)
                                          if self.parallel_max_node_percent is None else
                                          self.parallel_max_node_percent)
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    @use_logging(level='info')
    def _init_message_indexes_new(self):
        self.csc_map, row, col, data = adj_to_csc_map(self.adjacency_matrix)
        iter_items = tqdm(zip(*(row, col, data)), total=data.size, desc="dok_map") \
            if self.verbose_init else \
            zip(*(row, col, data))
        self.dok_map = {(i, j): n for i, j, n in iter_items}
        self.num_messages = self.adjacency_matrix.count_nonzero() + 1  # plus one for csr_map starts with 1

    @use_logging(level='info')
    def _init_message_indexes_old(self):
        self.dok_map = {e: i for i, e in enumerate(
            tqdm(zip(*self.adjacency_matrix.nonzero()), desc="indexes",
                 total=self.adjacency_matrix.count_nonzero())
            if self.verbose_init else
            zip(*self.adjacency_matrix.nonzero())
        )}
        self.message_index_set = set(self.dok_map.keys())  # set is way more faster than list
        self.num_messages = len(self.message_index_set)

    @use_logging(level='info')
    def _init_messages(self):
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

    @use_logging(level='info')
    def _init_beta(self):
        self.logger.warning("Initializing beta again with "
                            "'torch.log(self.num_groups / (torch.sqrt(self.mean_degree) - 1) + 1)'")
        beta = torch.log(self.num_groups / (torch.sqrt(self.mean_degree) - 1) + 1)
        beta = beta if not torch.isnan(beta).sum() > 0 else torch.tensor(1.0, dtype=torch.float)  # nan
        self.beta.data = beta
        self.logger.warning('beta initialized to {0}'.format(self.beta.data))

    @use_logging(level='info')
    def _init_lock_legacy(self):
        # multi-thread synchronization using lock
        self._lock_psi = [threading.Lock() for _ in range(len(self.marginal_psi))]
        self._lock_mmap = [[threading.Lock() for _ in range(len(mmap))] for mmap in self.message_map]
        self._lock_h = threading.Lock()

    @use_logging(level='info')
    def _init_job_list_old(self):
        # job_list to avoid race, job -> (edges, nodes)
        self.job_list = _create_job_list_parallel(
            self.adjacency_matrix, self.message_index_set,
            self.parallel_max_node_percent * self.num_nodes,
            self.seed, self.verbose_init
        )
        self._init_job_indexes_slices()

    @use_logging(level='info')
    def _init_job_indexes_slices(self):
        self._update_message_slice_and_index = [_update_message_create_slice(self, job[0])
                                                for job in (tqdm(self.job_list, desc="create message slice")
                                                            if self.verbose_init else self.job_list)]
        self._update_psi_slice_and_index = [_update_psi_create_index(self, job[1])
                                            for job in (tqdm(self.job_list, desc="create psi slice")
                                                        if self.verbose_init else self.job_list)]

    @use_logging(level='info')
    def _init_job_list_new(self):
        if self.multi_processing:
            if self.save_node_dict:
                self.__init_node_update_slice_dict_mp_save()
            else:
                self.__init_node_update_slice_dict_new_mp()
        else:
            if self.save_node_dict:
                self.__init_node_update_slice_dict_new_save()
            else:
                self.__init_node_update_slice_dict_new()
        self.job_list = create_job_list_parallel_fast(
            self.csc_map, self.parallel_max_node_percent * self.num_nodes, self.seed, self.verbose_init)

    @use_logging(level='info')
    def __init_node_update_slice_dict_new_save(self):
        all_nodes = list(range(self.num_nodes))
        all_nodes = all_nodes[self.node_job_slice] if self.node_job_slice is not None else all_nodes
        todo_list = self.__load_node_dict_(all_nodes)
        if len(todo_list) > 0:
            parent_path = osp.join(self.save_init_path, self.dataset_uuid)
            for dst_node in (tqdm(todo_list) if self.verbose_init else todo_list):
                save_path = parent_path + '-node-{0}'.format(dst_node)
                _save_node_dict(self.csc_map, self.dok_map, self.num_messages, self.beta.device, dst_node, save_path)

            if len(self.__load_node_dict_(todo_list)) > 0:
                raise Exception("Failed to load saved node dicts")

    @use_logging(level='info')
    def __init_node_update_slice_dict_new(self):
        iter_items = enumerate(node_list_to_slice_and_index_message(
            range(self.num_nodes), self.csc_map, self.dok_map, self.num_messages,
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
        iter_items = enumerate(node_list_to_slice_and_index_psi(
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

    @use_logging(level='info')
    def __init_node_update_slice_dict_new_mp(self):
        with Manager() as manager:
            dd = manager.dict()
            dd['message'], dd['psi'] = manager.dict(), manager.dict()
            with Pool(processes=self.mp_processes_count) as p:
                func = partial(_update_manager_node_dd,
                               dd, self.csc_map, self.dok_map, self.num_messages, self.beta.device)
                # chunksize = np.ceil(self.num_nodes / os.cpu_count() / 10)
                iter_items = p.imap_unordered(func, range(self.num_nodes),
                                              chunksize=self.mp_chunksize)
                if self.verbose_init:
                    with tqdm(total=self.num_nodes, desc="nodes") as pbar:
                        for _ in iter_items:
                            pbar.update()
                else:
                    _ = list(iter_items)

            self._saved_message_update_dict = dd['message'].copy()
            self._saved_psi_update_dict = dd['psi'].copy()

    @use_logging(level='info')
    def __init_node_update_slice_dict_mp_save(self):
        # TODO: dok_map (30GB in memory) blows up the RAM
        all_nodes = list(range(self.num_nodes))
        todo_list = self.__load_node_dict_(all_nodes)
        if len(todo_list) > 0:
            # mp save
            ctx = mp.get_context('spawn')
            chucks = np.array_split(todo_list, self.mp_processes_count)
            workers = [ctx.Process(
                target=save_node_list_dict,
                args=(self.csc_map, self.dok_map, self.num_messages, self.beta.device, node_list,
                      osp.join(self.save_init_path, self.dataset_uuid))
            ) for node_list in chucks]
            for w in workers:
                w.start()
            for w in workers:
                w.join()

            if len(self.__load_node_dict_(todo_list)) > 0:
                raise Exception("Failed to load saved node dicts")

    @use_logging(level='info')
    def __load_node_dict_(self, todo_list):
        self._saved_message_update_dict, self._saved_psi_update_dict = dict(), dict()
        failed_nodes = []
        with tqdm(desc="load node dict", total=len(todo_list), disable=(not self.verbose_init)) as pbar:
            for node in todo_list:
                try:
                    saved_path = osp.join(self.save_init_path, self.dataset_uuid) + '-node-{0}'.format(node)
                    # TODO: check but not load
                    d = torch.load(saved_path)
                    if self.beta.device == 'cpu':
                        self._saved_message_update_dict[node] = d[node]['message']
                        self._saved_psi_update_dict[node] = d[node]['psi']
                    else:
                        self._saved_message_update_dict[node] = tuple(
                            t.to(self.beta.device) for t in d[node]['message'])
                        self._saved_psi_update_dict[node] = tuple(t.to(self.beta.device) for t in d[node]['psi'])
                    pbar.update()
                except Exception as e:
                    failed_nodes.append(node)
                    continue
        if len(failed_nodes) > 0:
            self.logger.info("nodes failed to load from file: {}".format(len(failed_nodes)))
        return failed_nodes

    def __repr__(self):
        return '{}(num_groups={})'.format(self.__class__.__name__, self.num_groups)


if __name__ == '__main__':
    num_groups = 2
    sizes = np.asarray([50] * num_groups)
    epslion = 0.1
    P = np.ones((num_groups, num_groups)) * 0.1
    for i in range(len(P)):
        P[i][i] = P[i][i] / epslion
    G = nx.stochastic_block_model(sizes, P, seed=0)
    adj = nx.to_scipy_sparse_matrix(G)
    edge_index, edge_attr = adj_to_edge_index(adj)
    c = nx.to_scipy_sparse_matrix(G).mean() * G.number_of_nodes()
    ground_truth = []
    for i in G.nodes():
        ground_truth.append(G.nodes[i]['block'])
    print(c)
    epslion_ast = (np.sqrt(c) - 1) / (np.sqrt(c) - 1 + num_groups)
    print(epslion, epslion_ast, epslion < epslion_ast)
    percent = 1 / np.sqrt(c + 1)
    print(percent)

    # reddit = Reddit('datasets/Reddit')
    # num_groups = int(reddit.data.y.max() - reddit.data.y.min())
    # edge_index = reddit.data.edge_index
    # ground_truth = reddit.data.y

    bp = BeliefPropagation(num_groups, mean_degree=None, verbose_iter=True,
                           max_num_iter=100, verbose_init=True,
                           parallel_max_node_percent=0.1,
                           bp_max_diff=4e-1, disable_gradient=False,
                           bp_init_type='new', save_node_dict=False,
                           save_full_init=True,
                           mp_chunksize=1, mp_processes_count=4,
                           node_job_slice=None,
                           batch_run=False)
    # bp.beta.data = torch.tensor(1.75)
    entropy = EntropyLoss()
    # bp = bp.cuda()
    message_map, marginal_psi, message_index_list = bp(edge_index, G.number_of_nodes())
    entropy_loss = entropy(marginal_psi)
    print(entropy_loss)
