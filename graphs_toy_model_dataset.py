import random
from copy import deepcopy
import networkx as nx
from torch.utils.data import Dataset
from torch import Tensor


class ToyModel(Dataset):
    def __init__(self, graphs0, graphs1):
        self.graphs0 = graphs0
        self.graphs1 = graphs1
        self.dataset_dict = {}
        self.set_dataset_dict()

    def __getitem__(self, index):
        index_value = self.dataset_dict[index]
        values = deepcopy(index_value['values'])
        adjacency_matrix = deepcopy(index_value['adjacency_matrix'])
        label = self.dataset_dict[index]['label']
        return Tensor(values), Tensor(adjacency_matrix), label

    def __len__(self):
        return len(self.dataset_dict)

    def __repr__(self):
        return "Toy_Model_" + "len" + str(len(self))

    def get_vector_size(self):
        return self.dataset_dict[0]['adjacency_matrix'].shape[1]

    def nodes_number(self):
        return self.dataset_dict[0]['adjacency_matrix'].shape[0]

    def set_dataset_dict(self):
        for i, graph in enumerate(self.graphs0):
            attr_mat = [[value for value in list(values_dict.values())] for node, values_dict in graph.nodes(data=True)]
            self.dataset_dict[i] = {'subject': i,
                                    'label': 0,
                                    'adjacency_matrix': nx.adjacency_matrix(graph).todense(),
                                    'values': attr_mat,
                                    'graph': graph}
        for i, graph in enumerate(self.graphs1):
            attr_mat = [[value for value in list(values_dict.values())] for node, values_dict in graph.nodes(data=True)]
            self.dataset_dict[i + len(self.graphs0)] = {'subject': i + len(self.graphs0),
                                                        'label': 1,
                                                        'adjacency_matrix': nx.adjacency_matrix(graph).todense(),
                                                        'values': attr_mat,
                                                        'graph': graph}

        l = list(self.dataset_dict.items())
        # Shuffle the dataset dict so that the class0&class1 will not be class0 after class1
        random.shuffle(l)
        self.dataset_dict = dict(l)

