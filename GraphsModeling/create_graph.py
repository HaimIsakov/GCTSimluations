import networkx as nx
import numpy as np
from networkx import erdos_renyi_graph
from scipy.stats import bernoulli
from tqdm import tqdm


def create_graph(n, p):
    graph = erdos_renyi_graph(n, p)
    return graph

def associate_values_to_graphs(graphs_list, n, mu=0, sigma=1, features_dim=1, sigma_values=0, corr_values=False):
    m = len(graphs_list)
    features_matrix = []
    for k in tqdm(range(n), desc="Create values for graphs"):
        for j in range(features_dim):
            mu = np.random.normal(0, sigma_values)  # For graphs from class0 it will be zero,
            # and for graphs from class 1 it will be distributed from N(...,1)
            feature_list = generate_features(number=m, mu=mu, sigma=sigma)
            features_matrix.append(feature_list)
    tqdm._instances.clear()

    features_matrix = np.stack(features_matrix).T
    if corr_values:
        new_feature_matrix = []
        random_mat = np.random.rand(n, n)
        # random_vec = np.random.rand(n, 1)
        # corr_mat = 0.5 * np.repeat(random_vec, n, axis=1).T + 0.5 * random_mat
        corr_mat = random_mat
        for value in features_matrix:
            x = corr_mat @ value
            new_feature_matrix.append(x)
        features_matrix = np.array(new_feature_matrix)
    for i, graph in enumerate(graphs_list):
        features_dict = dict(enumerate(dict(enumerate(row)) for row in np.expand_dims(features_matrix[i], axis=1)))
        # Set nodes with generated features
        nx.set_node_attributes(graph, features_dict)
    return graphs_list


def delete_edges_according_to_values(graphs_list, beta=1):
    # Not used
    new_graph_list = []
    for graph in tqdm(graphs_list, desc="Delete edges according to nodes\' values"):
        graph_edges_to_delete = []
        nodes_values = graph.nodes(data=True)
        for node1, node2 in graph.edges():
            node1_values = np.array(list(nodes_values[node1].values()))
            node2_values = np.array(list(nodes_values[node2].values()))
            distance_node1_and_node2 = np.linalg.norm(node1_values - node2_values)
            delete_edge_prob = np.exp(-distance_node1_and_node2/beta)
            remain_or_not = bernoulli.rvs(delete_edge_prob)
            if not remain_or_not:
                graph_edges_to_delete.append((node1, node2))
                # graph_edges_to_delete.append((node2, node1))
        # print("Number of edges before delete", graph.number_of_edges())
        for edge in graph_edges_to_delete:
            graph.remove_edge(*edge)
        # print("Number of edges after delete", graph.number_of_edges())
        new_graph_list.append(graph)
    tqdm._instances.clear()
    return new_graph_list


def create_collection_of_graphs(n=400, m=100, p=0.1, mu=0, sigma=1, features_dim=5, sigma_values=0, corr_values=False):
    graphs_list = []
    for i in tqdm(range(m), desc="Creating graphs without values"):
        # cur_graph = create_graph_with_values(n, p, mu=mu, sigma=sigma, alpha=alpha, features_dim=features_dim, beta=beta)
        cur_graph = create_graph(n, p)
        graphs_list.append(cur_graph)
    tqdm._instances.clear()

    graphs_list = associate_values_to_graphs(graphs_list, n, sigma_values=sigma_values, sigma=sigma, features_dim=features_dim
                                       , corr_values=corr_values)
    # graphs_list = delete_edges_according_to_values(graphs_list, beta=beta)
    return graphs_list


def generate_features(number=400, mu=0, sigma=1):
    feature_list = np.random.normal(mu, sigma, number)
    return feature_list

# if __name__ == "__main__":
    # m = 1
    # sigma_values = 0.1
    # mu_0 = 0
    # mu_1 = np.random.normal(0, sigma_values)
    # sigma_0 = 1
    # sigma_1 = 1
    # features_dim = 5
    # epsilon = 0.01
    # # graphs_list_0 = create_collection_of_graphs(m=m, p=p, mu=mu_0, sigma=sigma_0, features_dim=features_dim)
    # graphs_list_1 = create_collection_of_graphs(m=m, p=p + epsilon, mu=mu_1, sigma=sigma_1, features_dim=features_dim)
    # x=1