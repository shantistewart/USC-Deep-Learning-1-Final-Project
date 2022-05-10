"""File for testing DataLoader class."""


import matplotlib.pyplot as plt
from preprocessing.load_data_class import DataLoader
from preprocessing.engineer_features import FeatureGenerator


# path to data folder:
data_folder = "../../data/"


print()
# load data:
data_loader = DataLoader()
X_raw, y, Fs = data_loader.load_data(data_folder)

# test FeatureGenerator:
example = 20
feature_generator = FeatureGenerator(feature_type="fft")
# plot raw audio data in time domain:
feature_generator.plot_time(X_raw, Fs, example, fig_num=1)
# compute and FFT:
X_fft = feature_generator.generate_features(X_raw, Fs, N_fft=len(X_raw[example]))
feature_generator.plot_freq(X_fft, example, fig_num=2)


# show plots:
plt.show()

