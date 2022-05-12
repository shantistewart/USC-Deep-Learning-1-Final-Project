"""File for testing MLP class."""

# path to data folder:
data_folder = "../../data/"

import numpy as np
import matplotlib.pyplot as plt
from modeling.sklearn_MLP import MLP
from preprocessing.load_data_class import DataLoader
from preprocessing.generate_features_class import FeatureGenerator


# import data
# split into training and validation sets
data_loader = DataLoader()
X_train, y_train, X_test, y_test = data_loader.load_and_split_data(data_folder)
Fs = data_loader.Fs

# pass fft_bins into MLP model
feature_gen = FeatureGenerator()
freq_range = (55.0, 1760.0)
n_bins = 50
N_fft = np.power(2, 16)
feature_gen_params = {
    "Fs": Fs,
    "freq_range": freq_range,
    "n_bins": n_bins,
    "N_fft": N_fft,
    "norm": True
}
X_fft_bin_train = feature_gen.generate_features("fft_bins", X_train, feature_gen_params)
X_fft_bin_test = feature_gen.generate_features("fft_bins", X_test, feature_gen_params)
print(X_fft_bin_train.shape)

net = MLP(num_hidden_units=10, hidden_size=25)
net.fit(X_fft_bin_train, y_train)
y_pred = net.predict(X_fft_bin_test)

accuracy = (y_pred == y_test).sum()
print(accuracy/len(y_test))
