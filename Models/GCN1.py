import numpy as np

from Models.utils import *


class GCN1(nn.Module):
    def __init__(self, nodes_number, feature_size, RECEIVED_PARAMS, device, num_classes=1, normalize_adj=True,
                 learnt_alpha=True):
        super(GCN1, self).__init__()
        self.feature_size = feature_size
        self.nodes_number = nodes_number
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.normalize_adj = normalize_adj
        self.minimum = 1e-10
        self.learnt_alpha = learnt_alpha

        self.gcn = nn.Linear(self.feature_size, int(self.RECEIVED_PARAMS["gcn"]))
        self.fc1 = nn.Linear(int(self.RECEIVED_PARAMS["gcn"]) * self.nodes_number, int(self.RECEIVED_PARAMS["layer_1"]))  # input layer
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
        I = torch.eye(b).to(self.device)
        if self.alpha.item() < self.minimum:
            self.alpha.data = torch.clamp(self.alpha, min=self.minimum)

        alpha_I = I * self.alpha.expand_as(I)  # 𝛼I
        if self.normalize_adj:
            normalized_adjacency_matrix = normalize_adjacency_matrix(adjacency_matrix, self.device)  # Ã
        else:
            normalized_adjacency_matrix = adjacency_matrix
        alpha_I_plus_normalized_A = alpha_I + normalized_adjacency_matrix  # 𝛼I + Ã̃

        x = torch.matmul(alpha_I_plus_normalized_A, x)  # (𝛼I + Ã̃)·x

        x = self.gcn_layer(x)
        x = torch.flatten(x, start_dim=1)  # flatten the tensor
        x = self.classifier(x)
        x = self.fc3(x)
        return x
