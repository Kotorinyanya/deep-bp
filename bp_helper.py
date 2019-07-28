import random

from scipy.sparse import csc_matrix, dok_matrix
from tqdm import tqdm

from utils import *


def adj_to_csc_map(adjacency_matrix):
    """

    :param adjacency_matrix:
    :return: row: out messages, col: in messages
    """
    row, col = adjacency_matrix.nonzero()
    data = np.arange(adjacency_matrix.count_nonzero()) + 1
    csc_map = csc_matrix((data, (row, col)), shape=adjacency_matrix.shape)
    return csc_map, row, col, data


def create_job_list_parallel_fast(csc_map, max_node_count, seed, verbose=True):
    all_nodes = set(np.unique(csc_map.nonzero()[0]))
    global_todo_nodes = set(all_nodes)
    job_list = []

    if verbose:
        pbar = tqdm(total=len(global_todo_nodes), desc="todo_nodes")

    seeed = 0
    while len(global_todo_nodes) > 0:
        nodes_to_choose = set(global_todo_nodes)
        blocked_nodes, local_doing_nodes = set(), set()
        while len(nodes_to_choose) > 0:
            random.seed(seed + seeed)
            seeed += 1 + len(global_todo_nodes) + len(blocked_nodes)  # just a random number
            dst_node = random.sample(nodes_to_choose, 1)[0]
            neighbors = set(csc_map.getcol(dst_node).nonzero()[0])
            local_doing_nodes.add(dst_node)
            blocked_nodes.add(dst_node)
            blocked_nodes |= neighbors
            nodes_to_choose -= blocked_nodes
            if verbose:
                pbar.update()
            if len(local_doing_nodes) >= max_node_count:
                break
        job_list.append(local_doing_nodes)
        global_todo_nodes -= local_doing_nodes
        seeed += 1

    if verbose:
        pbar.close()

    return job_list


def node_list_to_slice_and_index_message(node_list, csc_map, dok_map, num_messages, device):
    for dst_node in node_list:
        w, r, i = _node_to_slice_and_index_message(dst_node, csc_map, dok_map, num_messages, device)
        yield w, r, i


def node_list_to_slice_and_index_psi(node_list, csc_map, num_messages, device):
    for dst_node in node_list:
        w, r, i = _node_to_slice_and_index_psi(dst_node, csc_map, num_messages, device)
        yield w, r, i


def _node_to_slice_and_index_message(dst_node, csc_map, dok_map, num_messages, device):
    is_isolated = True
    # for torch_scatter
    index_tensor = torch.ones(num_messages, dtype=torch.long, device=device) * (-1)

    src_nodes = csc_map.getcol(dst_node).nonzero()[0]

    write_messages_index_list, read_messages_index_list = [], []
    for src_node in src_nodes:
        col = csc_map.getcol(src_node)
        col[dst_node, 0] = 0
        col.eliminate_zeros()
        src_index = col.data
        if src_index.size == 0:  # if no messages
            continue
        else:
            is_isolated = False
            src_index = torch.tensor(src_index, dtype=torch.long, device=device)
            dst_index = int(dok_map[src_node, dst_node])  # dok is O(1), csc is O(n)
            index_tensor[src_index] = dst_index
            read_messages_index_list.append(src_index)
            write_messages_index_list.append(dst_index)

    if is_isolated:
        empty_tensor = torch.tensor([], dtype=torch.long, device=device)
        return empty_tensor, empty_tensor, empty_tensor
    else:
        # for reading message_map
        write_messages_slice = torch.unique(torch.tensor(write_messages_index_list, device=device), sorted=True)
        # for updating message_map
        read_messages_slice = torch.unique(torch.cat(read_messages_index_list), sorted=True)
        # clean up
        index_tensor = index_tensor[index_tensor != -1]
        return write_messages_slice, read_messages_slice, index_tensor


def _node_to_slice_and_index_psi(dst_node, csc_map, num_messages, device):
    # for torch_scatter
    index_tensor = torch.ones(num_messages, dtype=torch.long, device=device) * (-1)

    col = csc_map.getcol(dst_node)
    src_index = col.data
    if src_index.size == 0:  # if no messages
        empty_tensor = torch.tensor([], dtype=torch.long, device=device)
        return empty_tensor, empty_tensor, empty_tensor
    else:
        src_index = torch.tensor(src_index, dtype=torch.long, device=device)
        dst_index = int(dst_node)
        index_tensor[src_index] = dst_index
        # for reading message_map
        write_messages_slice = torch.tensor([dst_index], device=device)
        # for updating message_map
        read_messages_slice = src_index
        # clean up
        index_tensor = index_tensor[index_tensor != -1]
        return write_messages_slice, read_messages_slice, index_tensor


def _update_manager_node_dd(dd, csc_map, dok_map, num_messages, device, dst_node):
    w, r, i = _node_to_slice_and_index_message(dst_node, csc_map, dok_map, num_messages, device)
    w, r, i = w.share_memory_(), r.share_memory_(), i.share_memory_()
    dd['message'][dst_node] = (w, r, i)
    w, r, i = _node_to_slice_and_index_psi(dst_node, csc_map, num_messages, device)
    w, r, i = w.share_memory_(), r.share_memory_(), i.share_memory_()
    dd['psi'][dst_node] = (w, r, i)


def save_node_list_dict(csc_map, dok_map, num_messages, device, node_list, parent_path):
    for dst_node in node_list:
        save_path = parent_path + '-node-{0}'.format(dst_node)
        _save_node_dict(csc_map, dok_map, num_messages, device, dst_node, save_path)


def _save_node_dict(csc_map, dok_map, num_messages, device, dst_node, save_path):
    # device=cpu, load to CUDA on use
    d = dict()
    d[dst_node] = dict()
    w, r, i = _node_to_slice_and_index_message(dst_node, csc_map, dok_map, num_messages, device='cpu')
    d[dst_node]['message'] = (w, r, i)
    w, r, i = _node_to_slice_and_index_psi(dst_node, csc_map, num_messages, device='cpu')
    d[dst_node]['psi'] = (w, r, i)
    torch.save(d, save_path)
