"""Script for running a k-nearest neighbors classifier."""


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from models.run_model import run_model


# path to data folder:
data_folder = "../../data/"


print()
# create K-nearest neighbors classifier:
model = KNeighborsClassifier()
# fraction of total data to use for test set:
test_fract = 0.2
# feature generation parameters:
feature_type = "fft_bins"
freq_range = (55.0, 1760.0)
N_fft = np.power(2, 16)
n_bins = 100
feature_gen_params = {
    "freq_range": freq_range,
    "n_bins": n_bins,
    "N_fft": N_fft,
    "norm": True
}
# feature normalization type:
norm_type = None
# number of folds (K) to use in stratified K-fold cross validation:
n_folds = 5


# hyperparameter search method and scoring metric:
search_type = "grid"
metric = "accuracy"
# hyperparameter values to search over:
# type of weight function for KNN:
weight_types = ["uniform", "distance"]
# number of nearest neighbors for KNN:
K_values = np.arange(start=1, stop=31, step=1)

hyperparams = {"model__weights": weight_types,
               "model__n_neighbors": K_values}
print("Hyperparameter search values:\n{}".format(hyperparams))

# run model:
run_model(data_folder, model, test_fract, feature_type, feature_gen_params, norm_type=norm_type,
          hyperparams=hyperparams, search_type=search_type, metric=metric, n_folds=n_folds, verbose=2)

