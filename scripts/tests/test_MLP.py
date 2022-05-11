"""File for testing MLP class."""

# path to data folder:
data_folder = "../../data/"

import numpy as np
import matplotlib.pyplot as plt
from models.mlp_class import MLP
from preprocessing.load_data_class import DataLoader
from preprocessing.generate_features_class import FeatureGenerator


# import data
# split into training and validation sets
data_loader = DataLoader()
X_train, y_train, X_test, y_test = data_loader.load_and_split_data(data_folder)
Fs = data_loader.Fs

# turn to frequency domain with FeatureGenerator
feature_gen = FeatureGenerator()
freq_range = (50.0, 1000.0)
X_train_fft = feature_gen.compute_fft(X_train, Fs)
# feature_gen_params = {
#     "Fs": Fs,
#     "freq_range": freq_range,
#     "n_peaks": n_peaks,
#     "N_fft": N_fft,
#     "norm": True
# }
# peaks = feature_gen.generate_features('fft_peaks', X_train, feature_gen_params)

# select which chord to look at
example = 5

# now that FFT peaks works properly, test MLP
N = X_train_fft.shape[1]
net = MLP(input_dim=N, output_dim=2, hidden_layer_dims=[5, 5], num_epochs=10)

net.fit(X_train_fft, y_train)
X_test_fft = feature_gen.compute_fft(X_test, Fs)
net.score(X_test_fft, y_test)

acc = net.history['accuracy']
plt.plot(acc)
plt.show()
