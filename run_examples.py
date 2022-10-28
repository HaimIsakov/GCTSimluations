from main import *

config_params_file_path = "config.json"
config_parameters = json.load(open(config_params_file_path, 'r'))
print("Config parameters", config_parameters)
# For a single set of graphs simulations
params_file_path = os.path.join("GridSearchResults", f"best_hp_{config_parameters['model']}.json")
RECEIVED_PARAMS = json.load(open(params_file_path, 'r'))
device = f"cuda:{config_parameters['cuda_number']}" if torch.cuda.is_available() else "cpu"
toy_models_params = {"m": 400,
                     'n': 50,
                     'p': 0.1,
                     'features_dim': 1,
                     'mu_0': 0,
                     'sigma_0': 1,
                     'sigma_1': 1,
                     'sigma_values': 0.5,
                     'epsilon': 0}
print(toy_models_params)
train_loss, val_loss, train_acc, val_acc, alpha_list =\
    run_trial(RECEIVED_PARAMS,  toy_models_params, device, model_name=config_parameters['model'], epochs=40,
              corr_values=config_parameters['corr_values'], learnt_alpha=config_parameters['learnt_alpha'],
              normalize_adj=config_parameters['normalize_adj'])


# For hyper-parameters of graphs simulations
hyper_parameters_dict = {"sigma_values": list(np.linspace(0, 0.5, num=10)),
                         "epsilon": [0, 0.02, 0.04, 0.07, 0.09],
                         "m": [400],
                         "features_dim": [1],
                         "n": [50],
                         "p": [0.1],
                         "mu_0": [0],
                         "sigma_0": [1],
                         "sigma_1": [1]}
run_grid(hyper_parameters_dict, model_name=config_parameters['model'], normalize_adj=config_parameters['normalize_adj'],
         corr_values=config_parameters['corr_values'], trials=config_parameters['trials'],
         learnt_alpha=config_parameters['learnt_alpha'])
