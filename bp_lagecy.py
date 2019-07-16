import concurrent.futures
import logging
import time
from multiprocessing.pool import Pool

from tqdm import tqdm

from utils import *

from functools import partial


def bp_infer_legacy(self):
    """
    Belief Propagation for probabilistic graphical inference
    :return:
    """
    if self.bp_type == 'exact':
        raise NotImplementedError("BP exact is not Implemented for legacy")
    job_list = list(self.G.edges())[:]
    bp_job = partial(bp_iter_step_legacy, self)
    np.random.seed(self.seed)
    np.random.shuffle(job_list)
    for num_iter in range(self.max_num_iter):
        max_diff = -np.inf

        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(self.num_workers) as executor:
            for result in executor.map(lambda args: bp_job(*args),
                                       job_list):
                diff = result
                max_diff = diff if diff > max_diff else max_diff
        end = time.time()
        self.logger.info(
            "num_iter \t {:3d} \t time \t {:.2f}ms \t max_diff {}".format(num_iter, (end - start) * 1000, max_diff))

        if max_diff < self.bp_max_diff:
            return num_iter, max_diff

    return num_iter, max_diff


def bp_iter_step_legacy(self, i, j):
    """
    Legacy
    :param i:
    :param j:
    :return:
    """
    message_i_to_j = update_message_i_to_j(self, i, j)
    marginal_psi_i = update_marginal_psi(self, i)

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


def init_messages_legacy(self):
    message_map = []
    for i in self.G.nodes():
        message_map_at_i = torch.rand(len(list(self.G.neighbors(i))), self.num_groups, device=self.beta.device)
        message_map_at_i = message_map_at_i / message_map_at_i.sum(1).reshape(-1, 1)
        message_map.append(message_map_at_i)
    return message_map


def _create_slice_and_index_mp(self):
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


def __create_slice_and_index_mp(self, job):
    edge_list, node_list = job
    return _update_message_create_slice(self, edge_list), _update_psi_create_index(self, node_list)


class BPJob(object):

    def __init__(self, max_job_count):
        self.max_job_count = max_job_count
        self.writing_edges_set = set()
        self.reading_edges_set = set()
        self.writing_nodes_set = set()

    def __len__(self):
        return len(self.writing_edges_set)

    def add_job(self, write_edge, write_node, read_edges):
        self.writing_edges_set.add(write_edge)
        self.writing_nodes_set.add(write_node)
        self.reading_edges_set |= read_edges

    def check_is_full(self):
        if len(self.writing_nodes_set) >= self.max_job_count:
            return True
        return False

    def check_writable(self, edge_to_write):
        i, j = edge_to_write
        if edge_to_write in self.reading_edges_set \
                or j in self.writing_nodes_set:
            return False
        return True

    def check_readable(self, edges_to_read):
        if common_member(edges_to_read, self.writing_edges_set):
            return False
        return True

    def get_job(self):
        return self.writing_edges_set, self.writing_nodes_set


def _create_job_list_parallel(adj, todo_list, max_node_count, seed, verbose):
    """

    :param adj: csr_matrix
    :param todo_list: edges
    :param max_node_count:
    :param seed:
    :param bp_type:
    :param verbose:
    :return:
    """
    np.random.seed(seed)
    # set is faster than list to do subtraction
    todo_set, doing_set = set(todo_list), set()
    bp_job_empty_list = [BPJob(max_node_count)]
    bp_job_full_list = []

    for edge_to_write in (tqdm(todo_set, desc="todo_set") if verbose else todo_set):
        succeed = False
        i, j = edge_to_write  # i -> j

        for num, bp_job in enumerate(list(bp_job_empty_list)):
            edges_to_read = set(zip(*adj.getrow(i).nonzero()))
            # edges_to_read = set((k, i) for k in G.neighbors(i) if k != j)
            if not bp_job.check_writable(edge_to_write):
                continue
            elif not bp_job.check_readable(edges_to_read):
                continue
            else:
                succeed = True
                break

        if not succeed:
            new_bp_job = BPJob(max_node_count)
            new_bp_job.add_job(edge_to_write, j, edges_to_read)
            bp_job_empty_list.append(new_bp_job)
        else:
            bp_job.add_job(edge_to_write, j, edges_to_read)
            if bp_job.check_is_full():
                bp_job_empty_list.remove(bp_job)
                bp_job_full_list.append(bp_job)
    job_list = [job.get_job() for job in bp_job_full_list + bp_job_empty_list]

    return job_list


# before optimization
def __create_job_list_parallel(G, todo_list, parallel_max_edges, seed, bp_type, verbose):
    np.random.seed(seed)
    # set is faster than list to do subtraction
    todo_set, doing_set = set(todo_list), set()
    job_list = []
    if verbose:
        pbar = tqdm(total=len(todo_set), desc="todo_set")
    while len(todo_set) > 0:
        writing_edges_set, reading_edges_set, writing_nodes_set = set(), set(), set()
        # avoid racing
        for i_to_j in todo_set:
            i, j = i_to_j
            if i_to_j in reading_edges_set or j in writing_nodes_set:
                continue
            if bp_type == 'approximate':
                to_read_edge_set = set((k, i) for k in G.neighbors(i) if k != j)
            elif bp_type == 'exact':
                to_read_edge_set = set((k, i) for k in G.nodes() if k not in [i, j])
            if common_member(to_read_edge_set, writing_edges_set):
                continue
            writing_edges_set.add(i_to_j)
            writing_nodes_set.add(j)
            reading_edges_set |= to_read_edge_set
            if len(writing_edges_set) > parallel_max_edges:
                break
        todo_set -= writing_edges_set
        doing_set |= writing_edges_set
        job_list.append((writing_edges_set, writing_nodes_set))

        if verbose:
            pbar.update(len(writing_edges_set))
    if verbose:
        pbar.close()
    return job_list


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
                                   for k in self.adjacency_matrix.getrow(src_node).nonzero()[1]
                                   if k != dst_node]
            if len(src_message_indexes) > 0:
                write_messages_slice_tensor[src_to_dst_message_index] = 1
                read_messages_slice_tensor[src_message_indexes] = 1
                index_tensor[src_message_indexes] = src_to_dst_message_index

        elif self.bp_type == 'exact':
            raise NotImplementedError()
    # assert index_tensor.max() == -1 or index_tensor.max() == torch.nonzero(write_messages_slice_tensor).max()
    index_tensor = index_tensor[index_tensor != -1]  # remove redundant indexes
    return write_messages_slice_tensor, read_messages_slice_tensor, index_tensor


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
                               for src_node in self.adjacency_matrix.getrow(dst_node).nonzero()[1]]
        if len(src_message_indexes) > 0:
            index_tensor[src_message_indexes] = int(dst_node)  # index message to read for torch_scatter
            read_message_slice_tensor[src_message_indexes] = 1  # for slicing input message

    # assert index_tensor.max() == -1 or index_tensor.max() == torch.nonzero(write_nodes_slice_tensor).max()
    index_tensor = index_tensor[index_tensor != -1]  # remove redundant indexes
    return write_nodes_slice_tensor, read_message_slice_tensor, index_tensor
