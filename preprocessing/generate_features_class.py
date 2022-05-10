"""File containing class for generating features."""


import numpy as np
from numpy import fft
from scipy.stats import binned_statistic
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class FeatureGenerator:
    """Class for generating features.

    Attributes:
        Fs: Sampling frequency (Hz).
        freq: Corresponding frequencies:
            dim: (D, N_fft/2)
    """

    def __init__(self):
        self.Fs = None
        self.freq = None

    def generate_features(self, feature_type, X_raw, Fs, N_fft=None, norm=True, freq_range=None, n_bins=None):
        """Generates features from audio data.

        Args:
            feature_type: Type of feature to generate.
                allowed values: "fft_bins", "fft_peaks"
            X_raw: List of raw audio data numpy arrays.
                length: N
            Fs: Sampling frequency (Hz)
            N_fft: Number of FFT points to use.
                If None, a default number of FFT points is used.
            norm: Selects whether to normalize raw audio data.
            freq_range: (min_freq, max_freq) (in Hz) range of frequencies to include for binning (ignored if
                feature_type != "fft_bins").
            n_bins: Number of bins to use (ignored if feature_type != "fft_bins").

        Returns:
            X: Generated features.
                dim: (N, D)
        """

        # validate feature type:
        if feature_type != "fft_bins" and feature_type != "fft_peaks":
            raise Exception("Invalid feature type.")

        # generate features:
        if feature_type == "fft_bins":
            if freq_range is None:
                raise Exception("freq_range parameter is None.")
            if n_bins is None:
                raise Exception("n_bins parameter is None.")
            X, _ = self.fft_bins(X_raw, Fs, freq_range, n_bins, N_fft=None, norm=norm)
        elif feature_type == "fft_peaks":
            X = self.fft_peaks(X_raw, Fs, N_fft=N_fft)

        return X

    def fft_bins(self, X_raw, Fs, freq_range, n_bins, N_fft=None, norm=True):
        """Computes binned FFTs of raw audio data.

        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N
            Fs: sampling frequency (Hz)
            freq_range: (min_freq, max_freq) range of frequencies to include for binning (Hz).
            n_bins: Number of bins to use.
            N_fft: Number of FFT points to use.
                If None, a default number of FFT points is used.
            norm: Selects whether to normalize raw audio data.

        Returns:
            X_fft_bin: Binned FFTs of raw audio data.
                dim: (N, n_bins)
            bin_edges: Edges of frequency bins (Hz).
                dim: (n_bins+1, )
        """

        # compute FFTs of audio data:
        X_fft = self.compute_fft(X_raw, Fs, N_fft=N_fft, norm=norm)

        # compute binned means of FFTs:
        X_fft_bin, bin_edges, bin_indices = binned_statistic(self.freq, X_fft, statistic="mean", bins=n_bins,
                                                             range=freq_range)

        return X_fft_bin, bin_edges

    def fft_peaks(self, X_raw, Fs, N_fft=None, norm=True):
        """
        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N
            Fs: sampling frequency (Hz)
            N_fft: Number of FFT points to use.
                If None, a default number of FFT points is used.
            norm: Selects whether to normalize raw audio data.

        Returns:
            peaks: List of peaks in raw audio numpy arrays.
                dim: (N, D)
        """

        N = len(X_raw)
        # choose the number of important peaks
        D = 3

        # compute FFT to analyze frequencies
        X_fft = self.compute_fft(X_raw, Fs, N_fft=N_fft, norm=norm)

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

    def compute_fft(self, X_raw, Fs, N_fft=None, norm=True):
        """Computes FFTs of raw audio data.

        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N
            Fs: Sampling frequency (Hz)
            N_fft: Number of FFT points to use.
                If None, default number of FFT points is used.
            norm: Selects whether to normalize raw audio data.

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
        # ensure N_fft is even:
        if N_fft % 2 == 1:
            N_fft += 1

        # normalize audio data, if selected:
        if norm:
            X = self.normalize_data(X_raw)
        else:
            X = X_raw

        # compute FFTs of audio data:
        X_fft = np.zeros((N, int(N_fft/2)))
        for i in range(N):
            X_fft_orig = np.abs(fft.fft(X[i], n=N_fft, norm="forward"))
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

    def plot_time(self, X_raw, Fs, example, norm=True, fig_num=1):
        """Plots raw audio data in time domain.

        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N
            Fs: Sampling frequency (Hz)
            example: Index of data example.
            norm: Selects whether to normalize raw audio data.
            fig_num: matplotlib figur number.

        Returns: None
        """

        x_raw = X_raw[example]
        n_samples = len(x_raw)

        # normalize audio data, if selected:
        if norm:
            x = self.normalize_data([x_raw])[0]
        else:
            x = x_raw

        # plot audio data in time domain:
        t = (1/Fs) * np.arange(0, n_samples)
        plt.figure(fig_num)
        plt.plot(t, x)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Audio Waveform of Example " + str(example))

    def plot_freq(self, X_fft, example, fig_num=1):
        """Plots FFT of audio data in frequency domain.

        Args:
            X_fft: FFT values.
                dim: (N, N_fft/2)
            example: Index of data example.
            fig_num: matplotlib figur number.

        Returns: None
        """

        plt.figure(fig_num)
        plt.plot(self.freq, X_fft[example])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("FFT of Audio Data of Example " + str(example))

    @staticmethod
    def normalize_data(X_raw):
        """Normalizes each audio file so that its maximum magnitude is 1.

        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N

        Returns:
            X_norm: List of normalized audio data numpy arrays.
                length: N
        """

        N = len(X_raw)
        # normalize all audio files
        X_norm = []
        for i in range(N):
            max_val = np.amax(np.abs(X_raw[i]))
            X_norm.append(X_raw[i] / max_val)

        return X_norm

