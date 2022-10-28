# GCTSimluations

Code for the GCT (=Graph Classification Task) models and simulation described in the attached file. 

To run the models run the command:

python run_examples.py

But first set the required configuration in config.json file.

{
  "cuda_number": Number of cuda card, >
  "model": Name of the required model, one of these ["VM", "GM", "GVM", "GCN1", "GCN2"], >
  "normalize_adj": Whether to normalize the adjacency matrix, >
  "corr_values": Whether to add correlation between features with multiplication of random matrix, >
  "trials": How many times each simulation for single σ and ε runs, >
  "learnt_alpha": Whether alpha is learnt >
}
