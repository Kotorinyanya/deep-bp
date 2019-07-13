from utils import *
from boxx import timeit
from tqdm import tqdm


class BPJob():

    def __init__(self):
        self.writing_edges_set = set()
        self.reading_edges_set = set()
        self.writing_nodes_set = set()

    def __len__(self):
        return len(self.writing_edges_set)

    def add_job(self, write_edge, write_node, read_edges):
        self.writing_edges_set.add(write_edge)
        self.writing_nodes_set.add(write_node)
        self.reading_edges_set |= read_edges

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


def _create_job_list_parallel(G, todo_list, parallel_max_edges, seed, bp_type, verbose):
    np.random.seed(seed)
    # set is faster than list to do subtraction
    todo_set, doing_set = set(todo_list), set()
    bp_job_empty_list = [BPJob()]
    bp_job_full_list = []

    for edge_to_write in (tqdm(todo_set, desc="todo_set") if verbose else todo_set):
        succeed = False
        i, j = edge_to_write  # i -> j

        for num, bp_job in enumerate(list(bp_job_empty_list)):
            edges_to_read = set((k, i) for k in G.neighbors(i) if k != j)
            if not bp_job.check_writable(edge_to_write):
                continue
            elif not bp_job.check_readable(edges_to_read):
                continue
            else:
                succeed = True
                break

        if not succeed:
            new_bp_job = BPJob()
            new_bp_job.add_job(edge_to_write, j, edges_to_read)
            bp_job_empty_list.append(new_bp_job)
        else:
            bp_job.add_job(edge_to_write, j, edges_to_read)
            if len(bp_job) >= parallel_max_edges:
                bp_job_empty_list.remove(bp_job)
                bp_job_full_list.append(bp_job)
    job_list = [job.get_job() for job in bp_job_full_list + bp_job_empty_list]

    return job_list


def __create_job_list_parallel(G, todo_list, parallel_max_edges, seed, bp_type, verbose):
    np.random.seed(seed)
    # set is faster than list to do subtraction
    todo_set, doing_set = set(todo_list), set()
    job_list = []
    if verbose:
        pbar = tqdm(total=len(todo_set), desc="create job list")
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


# appending random nodes have a negative affect for BP to converge
"""
append_list = list(doing_set - writing_edges_set)
np.random.shuffle(append_list)
# extend jobs in one iter
for i_to_j in append_list:
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
"""
