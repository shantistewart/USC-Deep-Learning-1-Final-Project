"""File for testing intervals() method of FeatureGenerator class."""


import numpy as np
from preprocessing.load_data_class import DataLoader
from preprocessing.generate_features_class import FeatureGenerator


# path to data folder:
data_folder = "../../data/"

# number of FFT points:
N_fft = np.power(2, 16)
# number of data examples to use:
num_examples = 100


print()
# load data:
data_loader = DataLoader()
X_raw, y, _, _ = data_loader.load_and_split_data(data_folder)
Fs = data_loader.Fs
X_raw = X_raw[0:num_examples]
y = y[0:num_examples]


# test intervals() method of FeatureGenerator class:
feature_generator = FeatureGenerator()
freq_range = (55.0, 1760.0)
n_harmonics = 9
peak_height_fract = 0.05
peak_sep = 10.0
feature_gen_params = {
    "Fs": Fs,
    "freq_range": freq_range,
    "n_harmonics": n_harmonics,
    "peak_height_fract": 0.05,
    "peak_sep": peak_sep,
    "N_fft": N_fft,
    "norm": True
}
X = feature_generator.generate_features("intervals", X_raw, feature_gen_params)
print(X.shape)

