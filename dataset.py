from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from utils import *


class BAvsER(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 seed=0):
        self.seed = seed
        self.num_ba = 2500
        self.num_er = 2500
        self.N = 50
        self.ba_ms = [1, int(self.N / 8), int(2 * self.N / 8), int(3 * self.N / 8), int(4 * self.N / 8)]
        self.er_ps = [1 / self.N, 1 / 8, 2 / 8, 3 / 8, 4 / 8]

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
        y = torch.tensor([label], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y)


if __name__ == '__main__':
    dataset = BAvsER('datasets/BAvsER')
    pass
