import numpy as np
from Models.utils import *


class GM(nn.Module):
    def __init__(self, nodes_number, feature_size, RECEIVED_PARAMS, device, normalize_adj=True, num_classes=2,
                 learnt_alpha=True):
        super(GM, self).__init__()
        self.feature_size = feature_size  # the dimension of the features for each node
        self.nodes_number = nodes_number  # the number of nodes for each graph
        self.RECEIVED_PARAMS = RECEIVED_PARAMS  # dictionary of hyper-parameters
        self.gcn = nn.Linear(self.feature_size, int(self.RECEIVED_PARAMS["gcn"]))
        self.fc1 = nn.Linear(int(self.RECEIVED_PARAMS["gcn"]) * self.nodes_number,
                             int(self.RECEIVED_PARAMS["layer_1"]))  # input layer
        self.fc2 = nn.Linear(int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"]))
        self.fc3 = nn.Linear(int(self.RECEIVED_PARAMS["layer_2"]), num_classes)
        self.device = device
        self.normalize_adj = False
        self.minimum = 1e-10
        self.learnt_alpha = learnt_alpha

        self.activation_func = self.RECEIVED_PARAMS['activation']
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])
        self.activation_func_dict = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}

        if self.learnt_alpha:
            noise = np.random.normal(0, 0.1)
            self.alpha = nn.Parameter(torch.tensor([1+noise], requires_grad=True, device=self.device).float())
        else:
            self.alpha = torch.tensor([1], device=self.device)

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
        # multiply the matrix adjacency_matrix by (learnt scalar) self.alpha
        a, b, c = adjacency_matrix.shape
        d, e, f = x.shape
        I = torch.eye(b).to(self.device)

        if self.alpha.item() < self.minimum:
            self.alpha.data = torch.clamp(self.alpha, min=self.minimum)

        normalized_adjacency_matrix = adjacency_matrix

        alpha_I = I * self.alpha.expand_as(I)  # 𝛼I
        alpha_I_plus_A = alpha_I + normalized_adjacency_matrix  # 𝛼I + Ã
        # x = torch.sign(x)
        x = torch.ones(x.shape, device=self.device)
        x = torch.matmul(alpha_I_plus_A, x)  # (𝛼I + Ã)·x

        x = self.gcn_layer(x)
        x = torch.flatten(x, start_dim=1)  # flatten the tensor
        x = self.classifier(x)
        x = self.fc3(x)
        return x
