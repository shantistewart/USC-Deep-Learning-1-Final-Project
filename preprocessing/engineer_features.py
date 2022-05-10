"""File containing class for engineering features."""


import numpy as np
from numpy import fft


class FeatureGenerator:
    """Class for generating features.

    Attributes:
        feature_type: Type of feature to generate.
            allowed values: "fft"
    """

    def __init__(self, feature_type="fft"):
        # validate type of feature:
        if feature_type is not None and feature_type != "fft":
            raise Exception("Invalid feature type")

        self.feature_type = feature_type

    def generate_features(self, X_raw, Fs, N_fft=128):
        """Generates features from audio data.

        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N
            Fs: Sampling frequency (Hz)
            N_fft: Number of FFT points to use.

        Returns:
            X: Generated features.
                dim: (N, D)
            freq: Corresponding frequencies:
                dim: (D, )
        """

        N = len(X_raw)
        D = int(N_fft/2)
        X = np.zeros((N, D))

        if self.feature_type == "fft":
            for i in range(N):
                X_fft_orig = np.abs(fft.fft(X_raw[i], n=N_fft, norm="ortho"))
                # extract non-negative part of PSD:
                X_fft = X_fft_orig[0:D]
                X[i] = X_fft
            # frequencies (same for all N examples):
            freq_orig = fft.fftfreq(N_fft, d=1 / Fs)
            # n_freq = X_fft.shape[-1]
            freq = freq_orig[0:D]

        return X, freq

