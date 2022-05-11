"""Script for running an MLP classifier."""


import numpy as np
from models.sklearn_MLP import MLP
from models.run_model import run_model


# path to data folder:
data_folder = "../../data/"


print()
# create MLP classifier (with arbitrary initial hyperparameters):
model = MLP(1, 5)
# fraction of total data to use for test set:
test_fract = 0.2
# feature generation parameters:
feature_type = "harmonics"
freq_range = (55.0, 1760.0)
n_harmonics = 9
peak_height_fract = 0.05
peak_sep = 10.0
N_fft = np.power(2, 16)
feature_gen_params = {
    "freq_range": freq_range,
    "n_harmonics": n_harmonics,
    "peak_height_fract": peak_height_fract,
    "peak_sep": peak_sep,
    "N_fft": N_fft,
    "norm": True
}
# feature normalization type:
norm_type = "standard"
# number of folds (K) to use in stratified K-fold cross validation:
n_folds = 3


# hyperparameter search method and scoring metric:
search_type = "random"
n_iters = 5
metric = "f1"
# hyperparameter values to search over:
n_hidden_layers = [1, 2]
n_hidden_units = np.arange(5, 20+1, step=1, dtype=int)

hyperparams = {"model__num_hidden_units": n_hidden_layers,
               "model__hidden_size": n_hidden_units}
print("Hyperparameter search values:\n{}".format(hyperparams))

# run model:
run_model(data_folder, model, test_fract, feature_type, feature_gen_params, norm_type=norm_type,
          hyperparams=hyperparams, search_type=search_type, metric=metric, n_iters=n_iters, n_folds=n_folds, verbose=2)

