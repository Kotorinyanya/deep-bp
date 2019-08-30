from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import random

from utils import *


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


if __name__ == '__main__':
    dataset = SBM2v4('datasets/SBM2v4')
    pass
