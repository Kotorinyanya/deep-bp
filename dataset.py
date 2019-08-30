import random

from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from bp_helper import pre_transfrom
from utils import *

import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.read import read_tu_data


class SBM2v4(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 seed=0):
        self.seed = seed
        self.num_graphs = 100
        self.N = 20
        # self.ba_ms = [1, int(self.N / 8), int(2 * self.N / 8), int(3 * self.N / 8), int(4 * self.N / 8)]
        # self.er_ps = [1 / self.N, 1 / 8, 2 / 8, 3 / 8, 4 / 8]

        super(SBM2v4, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ''

    @property
    def processed_file_names(self):
        return 'SBM24-{}.pt'.format(self.N)

    def download(self):
        pass

    def process(self):
        data_list = self._generate_smb(2) + self._generate_smb(4)
        random.seed(self.seed)
        random.shuffle(data_list)

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def _generate_smb(self, num_groups):

        sizes = np.asarray([int(self.N / num_groups)] * num_groups)
        epslion = 0.5
        P = np.ones((num_groups, num_groups)) * 0.1
        for i in range(len(P)):
            P[i][i] = P[i][i] / epslion

        seeed = self.seed
        data_list = []
        label = 0 if num_groups == 2 else 1
        todo_count = self.num_graphs
        with tqdm(desc="_generate_sbm_{}".format(num_groups), total=todo_count) as pbar:
            while todo_count > 0:
                G = nx.stochastic_block_model(sizes, P, seed=seeed)
                data_list.append(self._g_to_data(G, label))
                seeed += 1
                todo_count -= 1
                pbar.update()
        return data_list

    @staticmethod
    def _g_to_data(G, label):
        adj = nx.to_numpy_array(G)
        edge_index, _ = adj_to_edge_index(adj)
        x = torch.ones(G.number_of_nodes(), 3)
        y = torch.tensor([label], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)


class BAvsER(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 seed=0):
        self.seed = seed
        self.num_ba = 2500
        self.num_er = 2500
        self.N = 50
        # self.ba_ms = [1, int(self.N / 8), int(2 * self.N / 8), int(3 * self.N / 8), int(4 * self.N / 8)]
        # self.er_ps = [1 / self.N, 1 / 8, 2 / 8, 3 / 8, 4 / 8]
        self.ba_ms = [int(self.N / 8)]
        self.er_ps = [1 / 8]

        super(BAvsER, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ''

    @property
    def processed_file_names(self):
        return 'BAvsER-{}-{}+{}.pt'.format(self.N, self.num_ba, self.num_er)

    def download(self):
        pass

    def process(self):
        data_list = self._generate_ba() + self._generate_er()
        random.seed(self.seed)
        random.shuffle(data_list)

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def _generate_ba(self):
        data_list = []
        todo_count, seeed = self.num_ba, self.seed
        with tqdm(desc="_generate_ba", total=todo_count) as pbar:
            while todo_count > 0:
                for m in self.ba_ms:
                    G = nx.barabasi_albert_graph(self.N, m, seeed)
                    data_list.append(self._g_to_data(G, 0))
                    seeed += 1
                    todo_count -= 1
                    pbar.update()
        return data_list

    def _generate_er(self):
        data_list = []
        todo_count, seeed = self.num_ba, self.seed
        with tqdm(desc="_generate_er", total=todo_count) as pbar:
            while todo_count > 0:
                for p in self.er_ps:
                    G = nx.erdos_renyi_graph(self.N, p, seeed)
                    data_list.append(self._g_to_data(G, 1))
                    seeed += 1
                    todo_count -= 1
                    pbar.update()
        return data_list

    @staticmethod
    def _g_to_data(G, label):
        adj = nx.to_numpy_array(G)
        edge_index, _ = adj_to_edge_index(adj)
        x = torch.ones(G.number_of_nodes(), 1)
        y = torch.tensor([label], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)


class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <http://graphkernels.cs.tu-dortmund.de>`_.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name <http://graphkernels.cs.tu-dortmund.de>`_ of
            the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
    """

    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/' \
          'graphkerneldatasets'

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False):
        self.use_node_attr = use_node_attr
        self.name = name
        super(TUDataset, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0

        for i in range(self.data.x.size(1)):
            if self.data.x[:, i:].sum().item() == self.data.x.size(0):
                return self.data.x.size(1) - i

        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0

        return self.data.x.size(1) - self.num_node_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url('{}/{}.zip'.format(self.url, self.name), self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.data.x is not None and not self.use_node_attr:
            self.data.x = self.data.x[:, self.num_node_attributes:]

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


if __name__ == '__main__':
    dataset = TUDataset(root='datasets/ENZYMES', name='ENZYMES',
                        pre_transform=pre_transfrom)
    pass
