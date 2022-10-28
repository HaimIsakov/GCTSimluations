import torch
import torch.nn as nn


class VM(nn.Module):
    def __init__(self, nodes_number, data_size, RECEIVED_PARAMS, num_classes=2):
        super(VM, self).__init__()
        self.data_size = data_size
        self.nodes_number = nodes_number
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.fc1 = nn.Linear(self.data_size * self.nodes_number, int(self.RECEIVED_PARAMS["layer_1"]))  # input layer
        self.fc2 = nn.Linear(int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"]))
        self.fc3 = nn.Linear(int(self.RECEIVED_PARAMS["layer_2"]), num_classes)

        self.activation_func_dict = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])
        self.activation_func = self.RECEIVED_PARAMS['activation']
        self.classifier = nn.Sequential(
            self.fc1,
            self.activation_func_dict[self.activation_func],
            self.dropout,
            self.fc2,
            self.activation_func_dict[self.activation_func],
            )

    def forward(self, x, adjacency_matrix):
        x = torch.flatten(x, start_dim=1)  # flatten the tensor
        x = self.classifier(x)
        x = self.fc3(x)
        return x
