"""File containing class for generating features."""


import numpy as np
from numpy import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class FeatureGenerator:
    """Class for generating features.

    Attributes:
        feature_type: Type of feature to generate.
            allowed values: "fft_bins", "fft_peaks"
        Fs: Sampling frequency (Hz).
        freq: Corresponding frequencies:
            dim: (D, N_fft/2)
    """

    def __init__(self, feature_type):
        # validate type of feature:
        if feature_type != "fft_bins" and feature_type != "fft_peaks":
            raise Exception("Invalid feature type")

        self.feature_type = feature_type
        self.Fs = None
        self.freq = None

    def generate_features(self, X_raw, Fs, N_fft=None):
        """Generates features from audio data.

        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N
            Fs: Sampling frequency (Hz)
            N_fft: Number of FFT points to use.
                If None, a default number of FFT points is used.

        Returns:
            X: Generated features.
                dim: (N, D)
        """

        # generate features:
        if self.feature_type == "fft_bins":
            X = None
        elif self.feature_type == "fft_peaks":
            X = self.fft_peaks(X_raw, Fs, N_fft=N_fft)

        return X

    def fft_peaks(self, X_raw, Fs, N_fft=None):
        """
        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N
            Fs: sampling frequency (Hz)

        Returns:
            peaks: List of peaks in raw audio numpy arrays.
                dim: (N, D)
        """

        N = len(X_raw)
        # choose the number of important peaks
        D = 3

        # compute FFT to analyze frequencies
        X_fft = self.compute_fft(X_raw, Fs, N_fft=N_fft)

        # parameters for finding peaks
        dist = 10
        h = 50
        prom = 1

        peaks = np.zeros((N, D))
        for i in range(N):
            curr = X_fft[i]
            peak_indices, _ = find_peaks(curr, distance=dist, height=h, prominence=prom)
            # remove DC component
            indices_over_50 = np.abs(peak_indices - 50).argmin()
            peak_indices = peak_indices[peak_indices > indices_over_50]
            # find three highest peaks
            peak_indices = peak_indices[0:D-1]
            peaks[i] = curr[peak_indices]

        return peaks

    def compute_fft(self, X_raw, Fs, N_fft=None):
        """Computes FFT of raw audio data.

        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N
            Fs: Sampling frequency (Hz)
            N_fft: Number of FFT points to use.
                If None, default number of FFT points is used.

        Returns:
            X_fft: FFT values.
                dim: (N, N_fft/2)
        """

        N = len(X_raw)

        # set default value for N_fft, if no value provided:
        if N_fft is None:
            N_fft_max = 0
            for i in range(N):
                if len(X_raw[i]) > N_fft_max:
                    N_fft_max = len(X_raw[i])
            N_fft = N_fft_max

        # compute FFTs:
        X_fft = np.zeros((N, int(N_fft/2)))
        for i in range(N):
            X_fft_orig = np.abs(fft.fft(X_raw[i], n=N_fft, norm="ortho"))
            # extract non-negative part of PSD:
            X_fft_pos = X_fft_orig[0:int(N_fft/2)]
            X_fft[i] = X_fft_pos
        # frequencies (same for all N examples):
        freq_orig = fft.fftfreq(N_fft, d=1 / Fs)
        # n_freq = X_fft.shape[-1]
        freq = freq_orig[0:int(N_fft/2)]

        self.Fs = Fs
        self.freq = freq

        return X_fft

    def plot_time(self, X_raw, Fs, example, fig_num=1):
        """Plots raw audio data in time domain.

        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N
            Fs: Sampling frequency (Hz)
            example: Index of data example.
            fig_num: matplotlib figur number.

        Returns:
        """

        n_samples = len(X_raw[example])
        t = (1/Fs) * np.arange(0, n_samples)

        plt.figure(fig_num)
        plt.plot(t, X_raw[example])
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Raw Audio Data of Example " + str(example))

    def plot_freq(self, X_fft, example, fig_num=1):
        """Plots FFT of audio data in frequency domain.

        Args:
            X_fft: FFT values.
                dim: (N, N_fft/2)
            example: Index of data example.
            fig_num: matplotlib figur number.

        Returns:
        """

        plt.figure(fig_num)
        plt.plot(self.freq, X_fft[example])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("FFT of Audio Data of Example " + str(example))

