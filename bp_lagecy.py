import concurrent.futures
import logging
import time

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
