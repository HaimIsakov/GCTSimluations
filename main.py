import argparse
import os
from datetime import datetime
import itertools
import json
from sklearn.model_selection import train_test_split

from Models.GCN1 import *
from Models.GCN2 import *
from Models.GM import *
from Models.VM import *
from Models.GVM import *
from graphs_toy_model_dataset import ToyModel

from train_test import train_model, plot_acc_loss
import torch.nn as nn
import torch.utils.data as data
from GraphsModeling.create_graph import *

def model_menu(model_name, n, features_dim, RECEIVED_PARAMS, device, normalize_adj, learnt_alpha):
    if model_name == "GCN1":
        model = GCN1(n, features_dim, RECEIVED_PARAMS, device, num_classes=2, normalize_adj=normalize_adj,
                     learnt_alpha=learnt_alpha)
    elif model_name == "GCN2":
        model = GCN2(n, features_dim, RECEIVED_PARAMS, device, num_classes=2, normalize_adj=normalize_adj,
                     learnt_alpha=learnt_alpha)
    elif model_name == "GVM":
        model = GVM(n, features_dim, RECEIVED_PARAMS, device, num_classes=2, normalize_adj=normalize_adj,
                    learnt_alpha=learnt_alpha)
    elif model_name == "GM":
        model = GM(n, features_dim, RECEIVED_PARAMS, device, num_classes=2, normalize_adj=normalize_adj,
                   learnt_alpha=learnt_alpha)
    elif model_name == "VM":
        model = VM(n, features_dim, RECEIVED_PARAMS, num_classes=2)

    model = model.to(device)
    return model

def graph_dealing(toy_models_params, corr_values, RECEIVED_PARAMS):
    m = toy_models_params["m"]
    n = toy_models_params["n"]
    p = toy_models_params["p"]
    mu_0 = toy_models_params["mu_0"]
    # mu_1 = toy_models_params["mu_1"]
    sigma_0 = toy_models_params["sigma_0"]
    sigma_1 = toy_models_params["sigma_1"]
    epsilon = toy_models_params["epsilon"]
    features_dim = toy_models_params["features_dim"]
    sigma_values = toy_models_params["sigma_values"]
    # mu_1 = np.random.normal(0, sigma_values)
    train_test_graphs_list_0 = create_collection_of_graphs(n=n, m=m, p=p, sigma_values=mu_0, sigma=sigma_0,
                                                           features_dim=features_dim, corr_values=corr_values)

    train_test_graphs_list_1 = create_collection_of_graphs(n=n, m=m, p=p + epsilon, sigma_values=sigma_values,
                                                           sigma=sigma_1, features_dim=features_dim,
                                                           corr_values=corr_values)

    train_graphs_list_0, test_graphs_list_0 = train_test_split(train_test_graphs_list_0, test_size=0.5)
    train_graphs_list_1, test_graphs_list_1 = train_test_split(train_test_graphs_list_1, test_size=0.5)

    train_dataset = ToyModel(train_graphs_list_0, train_graphs_list_1)
    train_data_loader = data.DataLoader(train_dataset, batch_size=int(RECEIVED_PARAMS["batch_size"]), shuffle=True)

    test_dataset = ToyModel(test_graphs_list_0, test_graphs_list_1)
    test_data_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_data_loader, test_data_loader


def run_trial(RECEIVED_PARAMS, toy_models_params, device="cpu", epochs=20, model_name="GCN1",
              plot_figures=False, normalize_adj=True, corr_values=False, learnt_alpha=True):
    n = toy_models_params["n"]
    epsilon = toy_models_params["epsilon"]
    sigma = toy_models_params["sigma_values"]
    features_dim = toy_models_params["features_dim"]
    model = model_menu(model_name, n, features_dim, RECEIVED_PARAMS, device, normalize_adj, learnt_alpha)
    loss_module = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=RECEIVED_PARAMS["learning_rate"],
                                 weight_decay=RECEIVED_PARAMS["regularization"])
    train_data_loader, test_data_loader = graph_dealing(toy_models_params, corr_values, RECEIVED_PARAMS)
    train_loss, val_loss, train_acc, val_acc, alpha_list = train_model(model, optimizer, train_data_loader, test_data_loader,
                                                           loss_module, device, model_name, num_epochs=epochs)
    if plot_figures:
        # Not updated
        paramters_str = f"ε_{epsilon: .4f}_σ_{sigma: .4f}_node_{n}_graphs_{m}_feat_{features_dim}"
        plot_acc_loss(train_loss, val_loss, train_acc, val_acc, alpha_list, paramters_str)
    return train_loss, val_loss, train_acc, val_acc, alpha_list


def run_backbone_with_std(model_name, fc_hyper_parameters, hyper_parameters_dict, device, normalize_adj=False,
                          corr_values=False, learnt_alpha=True, trials=5):
    now = datetime.now()
    dt_string2 = now.strftime("%d_%m_%Y_%H_%M_%S")
    result_files = []
    for trial in range(trials):
        print("Trial", trial)
        file_name = f"trial_{trial}_{model_name}_normalize_{normalize_adj}_corr_{corr_values}_toy_model_grid_search_{dt_string2}.csv"
        file_name = run_backbone(model_name, fc_hyper_parameters, hyper_parameters_dict, device, normalize_adj=normalize_adj,
                     corr_values=corr_values, learnt_alpha=learnt_alpha, file_name=file_name)
        result_files.append(file_name)
    return result_files


def run_backbone(model_name, fc_hyper_parameters, hyper_parameters_dict, device, normalize_adj=False, corr_values=False,
                 file_name=None, learnt_alpha=True):
    if file_name is None:
        now = datetime.now()
        dt_string2 = now.strftime("%d_%m_%Y_%H_%M_%S")
        file_name = f"{model_name}_normalize_{normalize_adj}_corr_{corr_values}_toy_model_grid_search_{dt_string2}.csv"
    results_file = open(file_name, "w")
    hyper_parameters_names = list(hyper_parameters_dict.keys())
    a = list(hyper_parameters_dict.values())
    all_combination_hyper_parameters = list(itertools.product(*a))
    for idx, hyper_parameters_set in enumerate(tqdm(all_combination_hyper_parameters, desc=
    'Run different hyper-parameters')):
        toy_models_params = {}
        # create toy_models_params dict
        for i, hyper_parameter in enumerate(hyper_parameters_set):
            toy_models_params[hyper_parameters_names[i]] = hyper_parameter
        print(toy_models_params)
        train_loss, val_loss, train_acc, val_acc, alpha_list = run_trial(fc_hyper_parameters, toy_models_params,
                                                                         device=device, epochs=40, model_name=model_name,
                                                                         plot_figures=False,
                                                                         normalize_adj=normalize_adj,
                                                                         corr_values=corr_values,
                                                                         learnt_alpha=learnt_alpha)
        train_metric = train_acc[-1]
        test_metric = val_acc[-1]
        if model_name != "VM":
            alpha_value = alpha_list[-1]
        # result_str represents the line will be written to result file
        if idx == 0:
            headers = list(toy_models_params.keys())
            if model_name == "VM":
                headers += ["train_acc", "test_acc"]
            else:
                headers += ["train_acc", "test_acc", "alpha_value"]

            headers_str = ",".join(headers)
            results_file.write(f"{headers_str}\n")

        result_str = ""
        for i, (k, v) in enumerate(toy_models_params.items()):
            result_str += str(v) + ","
        result_str += str(train_metric) + ","
        result_str += str(test_metric) + ","
        if model_name != "VM":
            result_str += str(alpha_value)
        result_str += "\n"
        results_file.write(result_str)
        results_file.flush()
        os.fsync(results_file.fileno())
    results_file.close()
    return file_name


def run_grid(hyper_parameters_dict, model_name="GVM", normalize_adj=True, corr_values=False, cuda_number=0, trials=1
             , learnt_alpha=True):
    device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
    print(device)    #
    params_file_path = os.path.join("GridSearchResults", f"best_hp_{model_name}.json")
    fc_hyper_parameters = json.load(open(params_file_path, 'r'))
    if trials == 1:
        file_name = run_backbone(model_name, fc_hyper_parameters, hyper_parameters_dict, device,
                                 normalize_adj=normalize_adj, corr_values=corr_values, learnt_alpha=learnt_alpha)
    else:
        file_name = run_backbone_with_std(model_name, fc_hyper_parameters, hyper_parameters_dict, device,
                                          normalize_adj=normalize_adj, corr_values=corr_values, trials=trials
                                          , learnt_alpha=learnt_alpha)
    return file_name


# def get_hyper_parameters(model):
#     hyper_parameters_dict = {}
#     if model == "VM":
#         hyper_parameters_dict = {"learning_rate": [1e-6, 16-5, 1e-3],
#                                  "batch_size": [8, 16],
#                                  "dropout": [0.1, 0.2, 0.5],
#                                  "activation": ["relu", "tanh"],
#                                  "regularization": [1e-6, 1e-4, 1e-3],
#                                  "layer_1": [16, 64],
#                                  "layer_2": [16, 64]}
#     elif model in ["GCN1", "GVM", "GM"]:
#         hyper_parameters_dict = {"learning_rate": [1e-6, 16-5, 1e-3],
#                                  "batch_size": [8, 16],
#                                  "dropout": [0.1, 0.2, 0.5],
#                                  "activation": ["relu", "tanh"],
#                                  "regularization": [1e-6, 1e-4, 1e-3],
#                                  "layer_1": [16, 64],
#                                  "layer_2": [16, 64],
#                                  "gcn": [5, 10, 20]}
#     elif model == "GCN2":
#         hyper_parameters_dict = {"learning_rate": [1e-6, 1e-4, 1e-3],
#                                  "batch_size": [8, 16],
#                                  "dropout": [0.1, 0.2, 0.5],
#                                  "activation": ["relu", "tanh"],
#                                  "regularization": [1e-6, 1e-4, 1e-3],
#                                  "layer_1": [16, 64],
#                                  "layer_2": [16, 64],
#                                  "gcn": [5, 10, 20],
#                                  "gcn2": [5, 10, 20]}
#     return hyper_parameters_dict


# def create_df(file_name):
#     usecols = ["sigma_values", "epsilon", "test_acc", "train_acc"]
#     result_df = pd.read_csv(file_name, usecols=usecols)
#     sigma_values = np.unique(list(result_df["sigma_values"]))
#     epsilon = np.unique(list(result_df["epsilon"]))
#     test_acc = list(result_df["test_acc"])
#     train_acc = list(result_df["train_acc"])
#
#     test_acc = np.reshape(test_acc, (len(sigma_values), len(epsilon)))
#     train_acc = np.reshape(train_acc, (len(sigma_values), len(epsilon)))
#
#     sigma_values = [str(round(x, 2)) for x in sigma_values]
#     epsilon = [str(round(x, 2)) for x in epsilon]
#     new_result_df_test = pd.DataFrame(data=test_acc, index=sigma_values, columns=epsilon)
#     new_result_df_train = pd.DataFrame(data=train_acc, index=sigma_values, columns=epsilon)
#     return new_result_df_test, new_result_df_train


# def calc_mean_and_std(file_name1, file_name2):
#     new_result_df_test1, new_result_df_train1 = create_df(file_name1)
#     new_result_df_test2, new_result_df_train2 = create_df(file_name2)
#
#     test_mean_auc = np.concatenate((np.ravel(new_result_df_test1.values), np.ravel(new_result_df_test2.values))).mean()
#     test_std_auc = np.concatenate((np.ravel(new_result_df_test1.values), np.ravel(new_result_df_test2.values))).std()
#
#     train_mean_auc = np.concatenate((np.ravel(new_result_df_train1.values), np.ravel(new_result_df_train2.values))).mean()
#     train_std_auc = np.concatenate((np.ravel(new_result_df_train1.values), np.ravel(new_result_df_train2.values))).std()
#     return train_mean_auc, train_std_auc, test_mean_auc, test_std_auc


# def fc_hyper_parameters_grid_search(model_params_dict, hyper_parameters_dict, model, device, normalize_adj=False,
#                                     corr_values=False):
#     hyper_parameters_names = list(model_params_dict.keys())
#     a = list(model_params_dict.values())
#     all_combination_hyper_parameters = list(itertools.product(*a))
#     random.shuffle(all_combination_hyper_parameters)
#     d1 = date.today().strftime("%d_%m_%Y_%H_%M_%S")
#     hyper_params_file_name = f"{model}_fc_hyper_parameters_results_train_test_{d1}_normalize_adj_{normalize_adj}_" \
#                              f"corr_values_{corr_values}.csv"
#     results_file = open(hyper_params_file_name, "w")
#     headers = list(model_params_dict.keys()) + ["train_metric_mean", "test_metric_mean", "train_metric_std",
#                                                 "test_metric_std"]
#     headers_str = ",".join(headers)
#     results_file.write(f"{headers_str}\n")
#     RECEIVED_PARAMS = {}
#     for hyper_parameters_set in tqdm(all_combination_hyper_parameters):
#         try:
#             for i, hyper_parameter in enumerate(hyper_parameters_set):
#                 RECEIVED_PARAMS[hyper_parameters_names[i]] = hyper_parameter
#             print(RECEIVED_PARAMS)
#
#             file_name_correlated = run_backbone(model, RECEIVED_PARAMS, hyper_parameters_dict, device,
#                                      normalize_adj=normalize_adj, corr_values=True)
#             file_name_uncorrelated = run_backbone(model, RECEIVED_PARAMS, hyper_parameters_dict, device,
#                                      normalize_adj=normalize_adj, corr_values=False)
#
#             train_mean_auc, train_std_auc, test_mean_auc, test_std_auc = calc_mean_and_std(file_name_correlated,
#                                                                                            file_name_uncorrelated)
#             result_str = ""
#             for i, (k, v) in enumerate(RECEIVED_PARAMS.items()):
#                 result_str += str(v) + ","
#             result_str += str(train_mean_auc) + ","
#             result_str += str(test_mean_auc) + ","
#             result_str += str(train_std_auc) + ","
#             result_str += str(test_std_auc)
#             result_str += "\n"
#             print(result_str)
#
#             results_file.write(result_str)
#             results_file.flush()
#             os.fsync(results_file.fileno())
#         except:
#             raise
#             # print("Error occured")
#
#     results_file.close()
#     return hyper_params_file_name

