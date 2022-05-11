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
        freq: Frequencies of FFTs.
            dim: (D, N_fft/2)
    """

    def __init__(self):
        self.Fs = None
        self.freq = None

    def generate_features(self, feature_type, X_raw, feature_gen_params):
        """Generates features from audio data.

        Args:
            feature_type: Type of feature to generate.
                allowed values: "harmonics", "fft_bins"
            X_raw: List of raw audio data numpy arrays.
                length: N
            feature_gen_params: Dictionary of parameters for feature generation, with keys/values:
                Fs: Sampling frequency (Hz).
                freq_range: (min_freq, max_freq) (in Hz) range of frequencies to include for binning/peak finding.
                n_bins: Number of bins to use in binning (ignored if feature_type != "fft_bins").
                n_peaks: Number of peaks to find in peak finding (ignored if feature_type != "fft_peaks").
                N_fft: Number of FFT points to use.
                    If None, a default number of FFT points is used.
                norm: Selects whether to normalize raw audio data.
                    If None, set to True.

        Returns:
            X: Generated features.
                dim: (N, D)
        """

        # validate feature type:
        if feature_type != "harmonics" and feature_type != "fft_bins":
            raise Exception("Invalid feature type.")

        # extract general feature generation parameters:
        Fs = feature_gen_params.get("Fs")
        freq_range = feature_gen_params.get("freq_range")
        N_fft = feature_gen_params.get("N_fft")
        norm = feature_gen_params.get("norm")
        if Fs is None or freq_range is None:
            raise Exception("Missing general feature generation parameters.")
        # set default value for norm parameter:
        if norm is None:
            norm = True

        # generate features:
        if feature_type == "harmonics":
            # extract specific feature generation parameters:
            n_peaks = feature_gen_params.get("n_peaks")
            if n_peaks is None:
                raise Exception("Missing specific feature generation parameters.")
            # generate features:
            X = self.harmonics(X_raw, Fs, freq_range, n_peaks, N_fft=N_fft, norm=norm)
        elif feature_type == "fft_bins":
            # extract specific feature generation parameters:
            n_bins = feature_gen_params.get("n_bins")
            if n_bins is None:
                raise Exception("Missing specific feature generation parameters.")
            # generate features:
            X, _ = self.fft_bins(X_raw, Fs, freq_range, n_bins, N_fft=N_fft, norm=norm)

        return X

    def harmonics(self, X_raw, Fs, freq_range, n_peaks, N_fft=None, norm=True):
        """
        Finds n_peaks largest peaks (and their locations) in FFTs of audio data.

        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N
            Fs: Sampling frequency (Hz).
            freq_range: (min_freq, max_freq) (in Hz) range of frequencies to include for finding peaks.
            n_peaks: Number of peaks to find.
            N_fft: Number of FFT points to use.
                If None, a default number of FFT points is used.
            norm: Selects whether to normalize raw audio data.

        Returns:
            peak_freq: Frequencies (locations) of peaks of FFTs of audio data.
                dim: (N, n_peaks)
        """

        N = len(X_raw)
        # choose the number of important peaks
        D = n_peaks

        # compute FFT to analyze frequencies
        X_fft = self.compute_fft(X_raw, Fs, N_fft=N_fft, norm=norm)

        # parameters for finding peaks
        dist = 10
        prom = 0

        peak_list = np.zeros((N, D))
        for i in range(N):
            curr = X_fft[i]
            h = curr.max() * 5/100
            peak_indices, _ = find_peaks(curr, height=h)
            # remove DC component
            indices_over_50 = np.abs(peak_indices - 50)
            peak_indices = peak_indices[peak_indices > indices_over_50]
            # find D highest peaks
            peak_indices = peak_indices[0:D]
            peak_list[i] = self.freq[peak_indices]

        return peak_list

    def fft_bins(self, X_raw, Fs, freq_range, n_bins, N_fft=None, norm=True):
        """Computes binned FFTs of audio data.

        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N
            Fs: Sampling frequency (Hz).
            freq_range: (min_freq, max_freq) (in Hz) range of frequencies to include for binning.
            n_bins: Number of bins to use.
            N_fft: Number of FFT points to use.
                If None, a default number of FFT points is used.
            norm: Selects whether to normalize raw audio data.

        Returns:
            X_fft_bin: Binned FFTs of audio data.
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
            X_fft_orig = np.abs(fft.fft(X[i], n=N_fft, norm="ortho"))
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

    def plot_signal(self, X_raw, Fs, example, norm=True, fig_num=1):
        """Plots audio signal in time domain.

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

        # normalize audio signal, if selected:
        if norm:
            x = self.normalize_data([x_raw])[0]
        else:
            x = x_raw

        # plot audio signal in time domain:
        t = (1/Fs) * np.arange(0, n_samples)
        plt.figure(fig_num)
        plt.plot(t, x)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Audio Waveform of Example " + str(example))

    def plot_fft(self, X_fft, example, max_freq=None, fig_num=1):
        """Plots FFT of audio signal.

        Args:
            X_fft: FFT values.
                dim: (N, N_fft/2)
            example: Index of data example.
            max_freq: Maximum frequency to plot (in Hz).
                If None, all frequencies are plotted.
            fig_num: matplotlib figur number.

        Returns: None
        """

        # plot all frequencies, if max_freq is None:
        if max_freq is None:
            max_freq = self.freq[len(self.freq)-1]

        # plot FFT of audio signal:
        x_fft = X_fft[example]
        plt.figure(fig_num)
        plt.plot(self.freq[self.freq <= max_freq], x_fft[self.freq <= max_freq])
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

