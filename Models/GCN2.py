import numpy as np
from Models.utils import *


class GCN2(nn.Module):
    def __init__(self, nodes_number, feature_size, RECEIVED_PARAMS, device, normalize_adj=True, num_classes=2,
                 learnt_alpha=True):
        super(GCN2, self).__init__()
        self.feature_size = feature_size
        self.nodes_number = nodes_number
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.normalize_adj = normalize_adj
        self.minimum = 1e-10
        self.learnt_alpha = learnt_alpha

        self.gcn = nn.Linear(self.feature_size, int(self.RECEIVED_PARAMS["gcn"]))
        if "gcn2" not in self.RECEIVED_PARAMS:
            gcn2_dim = int(self.RECEIVED_PARAMS["gcn"]) + int(
                abs(int(self.RECEIVED_PARAMS["gcn"]) - int(self.RECEIVED_PARAMS["layer_1"])) / 2)
            print("GCN 2 DIM:", gcn2_dim)
        else:
            gcn2_dim = int(self.RECEIVED_PARAMS["gcn2"])

        self.gcn2 = nn.Linear(int(self.RECEIVED_PARAMS["gcn"]), gcn2_dim)
        self.fc1 = nn.Linear(gcn2_dim * self.nodes_number,
                             int(self.RECEIVED_PARAMS["layer_1"]))
        self.fc2 = nn.Linear(int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"]))
        self.fc3 = nn.Linear(int(self.RECEIVED_PARAMS["layer_2"]), num_classes)
        self.activation_func = self.RECEIVED_PARAMS['activation']
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])

        if self.learnt_alpha:
            noise = np.random.normal(0, 0.1)
            self.alpha = nn.Parameter(torch.tensor([1+noise], requires_grad=True, device=self.device).float())
        else:
            self.alpha = torch.tensor([1], device=self.device)

        self.activation_func_dict = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}

        self.gcn_layer = nn.Sequential(
            self.gcn,
            self.activation_func_dict[self.activation_func]
        )
        self.gcn_layer2 = nn.Sequential(
            self.gcn2,
            self.activation_func_dict[self.activation_func]
        )
        self.classifier = nn.Sequential(
            self.fc1,
            self.activation_func_dict[self.activation_func],
            self.dropout,
            self.fc2,
            self.activation_func_dict[self.activation_func],
        )

    def forward(self, x, adjacency_matrix):
        a, b, c = adjacency_matrix.shape
        d, e, f = x.shape
        if self.alpha.item() < self.minimum:
            self.alpha.data = torch.clamp(self.alpha, min=self.minimum)

        I = torch.eye(b).to(self.device)
        # multiply the matrix adjacency_matrix by (learnt scalar) self.alpha
        alpha_I = I * self.alpha.expand_as(I)  # 𝛼I
        if self.normalize_adj:
            normalized_adjacency_matrix = normalize_adjacency_matrix(adjacency_matrix)  # Ã
        else:
            normalized_adjacency_matrix = adjacency_matrix

        alpha_I_plus_A = alpha_I + normalized_adjacency_matrix  # 𝛼I + Ã

        x_input = torch.matmul(alpha_I_plus_A, x)  # (𝛼I + Ã)·x
        # First GCN Layer
        x_input = self.gcn_layer(x_input)
        # Second GCN Layer
        x_input = torch.matmul(alpha_I_plus_A, x_input)  # (𝛼I + Ã)·x
        x_input = self.gcn_layer2(x_input)

        x_input = torch.flatten(x_input, start_dim=1)  # flatten the tensor
        x_input = self.classifier(x_input)
        x_input = self.fc3(x_input)
        return x_input
