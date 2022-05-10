"""File for testing DataLoader class."""


import numpy as np
import matplotlib.pyplot as plt
from preprocessing.engineer_features import FeatureGenerator


print()
# test FeatureGenerator:
feature_generator = FeatureGenerator(feature_type="fft")

# test data:
N = 10
Fs = 100
Ts = 1.0 / Fs
temp = np.arange(0.0, 4.0, Ts)
n_samples = temp.shape[0]

t = np.arange(0.0, 4.0, Ts)
X = []
cos_freq = 10.0
for i in range(N):
    X.append(1.0 * np.cos(2 * np.pi * cos_freq * t))

# plot in time-domain:
n = 5
plt.figure(1)
plt.plot(t, X[n])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Cosine with Frequency " + str(cos_freq) + " Hz")

# compute FFT:
X_fft, freq = feature_generator.generate_features(X, Fs, N_fft=1024)

# plot in time-domain:
n = 5
plt.figure(2)
plt.plot(freq, X_fft[n])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT of Cosine with Frequency " + str(cos_freq) + " Hz")


# show plots:
plt.show()

