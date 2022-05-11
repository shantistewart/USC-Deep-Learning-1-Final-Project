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

# pass fft_bins into MLP model
feature_gen = FeatureGenerator()
freq_range = (55.0, 1760.0)
n_bins = 100
N_fft = np.power(2, 16)
feature_gen_params = {
    "Fs": Fs,
    "freq_range": freq_range,
    "n_bins": n_bins,
    "N_fft": N_fft,
    "norm": True
}
X_fft_bin_train = feature_gen.generate_features("fft_bins", X_train, feature_gen_params)
print(X_fft_bin_train.shape)

# now that FFT peaks works properly, test MLP
N = X_fft_bin_train.shape[1]
net = MLP(input_dim=N, output_dim=2, hidden_layer_dims=(5, 5), num_epochs=50)

net.fit(X_fft_bin_train, y_train)
X_fft_bin_test = feature_gen.generate_features("fft_bins", X_test, feature_gen_params)
valid_acc = net.score(X_fft_bin_test, y_test)
print()
print(valid_acc)

acc = net.history['accuracy']
plt.plot(acc)
plt.show()
