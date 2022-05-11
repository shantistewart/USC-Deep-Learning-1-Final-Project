"""Script for running a random forest classifier."""


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models.run_model import run_model


# path to data folder:
data_folder = "../../data/"


print()
# create random forest classifier:
model = RandomForestClassifier()
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
n_folds = 5


# hyperparameter search method and scoring metric:
search_type = "random"
n_iters = 200
metric = "f1"
# hyperparameter values to search over:
num_trees = [100]
min_samples_leaf = np.power(10, np.arange(-4, -2+0.1, step=0.1))
max_features = np.arange(0.1, 1.0+0.1, step=0.1)

hyperparams = {"model__n_estimators": num_trees,
               "model__min_samples_leaf": min_samples_leaf,
               "model__max_features": max_features}
print("Hyperparameter search values:\n{}".format(hyperparams))

# run model:
run_model(data_folder, model, test_fract, feature_type, feature_gen_params, norm_type=norm_type,
          hyperparams=hyperparams, search_type=search_type, metric=metric, n_iters=n_iters, n_folds=n_folds, verbose=2)

