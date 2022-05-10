"""File for testing DataLoader class."""


import matplotlib.pyplot as plt
from preprocessing.load_data_class import DataLoader
from preprocessing.generate_features import FeatureGenerator


# path to data folder:
data_folder = "../../data/"


# type of feature to generate:
feature_type = "fft_peaks"


print()
# load data:
data_loader = DataLoader()
X_raw, y, _, _ = data_loader.load_and_split_data(data_folder)
Fs = data_loader.Fs

# test FeatureGenerator:
example = 20
feature_generator = FeatureGenerator(feature_type=feature_type)
# plot raw audio data in time domain:
feature_generator.plot_time(X_raw, Fs, example, fig_num=1)
# compute and plot FFT:
X_fft = feature_generator.compute_fft(X_raw, Fs, N_fft=None)
feature_generator.plot_freq(X_fft, example, fig_num=2)


# show plots:
plt.show()

