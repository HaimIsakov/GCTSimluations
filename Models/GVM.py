import numpy as np
from Models.utils import *

class GVM(nn.Module):
    def __init__(self, nodes_number, feature_size, RECEIVED_PARAMS, device, num_classes=2, normalize_adj=True,
                 learnt_alpha=True):
        super(GVM, self).__init__()
        self.feature_size = feature_size
        self.nodes_number = nodes_number
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.normalize_adj = normalize_adj
        self.learnt_alpha = learnt_alpha
        self.minimum = 1e-10

        self.gcn = nn.Linear(self.feature_size, int(self.RECEIVED_PARAMS["gcn"]))
        self.fc1 = nn.Linear(self.feature_size, int(self.RECEIVED_PARAMS["layer_1"]))  # input layer
        self.fc2 = nn.Linear(int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"]))
        self.fc3 = nn.Linear(int(self.RECEIVED_PARAMS["gcn"]) * self.nodes_number +
                             int(self.RECEIVED_PARAMS["layer_2"]) * self.nodes_number, num_classes)
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
        # multiply the matrix adjacency_matrix by (learnt scalar) self.alpha
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

        alpha_I_plus_normalized_A = alpha_I + normalized_adjacency_matrix  # 𝛼I + Ã

        ones_vector = torch.ones(x.shape).to(self.device)
        gcn_output = torch.matmul(alpha_I_plus_normalized_A, ones_vector)  # (𝛼I + Ã)·1

        gcn_output = self.gcn_layer(gcn_output)
        gcn_output = torch.flatten(gcn_output, start_dim=1)  # flatten the tensor

        fc_output = self.classifier(x)
        fc_output = torch.flatten(fc_output, start_dim=1)  # flatten the tensor
        concat_graph_and_values = torch.cat((gcn_output, fc_output), 1)
        final_output = self.fc3(concat_graph_and_values)
        return final_output


