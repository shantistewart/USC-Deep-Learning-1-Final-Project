"""File containing class for generating features."""


import pandas as pd
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
                n_harmonics: Number of harmonics to find in harmonics estimation (ignored if
                    feature_type != "harmonics").
                peak_height_fract: Minimum height of FFT peaks = min_height_fract * max value of FFT.
                peak_sep: Minimum separation (distance) between neighboring FFT peaks (in Hz).
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
            n_harmonics = feature_gen_params.get("n_harmonics")
            peak_height_fract = feature_gen_params.get("peak_height_fract")
            peak_sep = feature_gen_params.get("peak_sep")
            if n_harmonics is None:
                raise Exception("Missing specific feature generation parameters.")
            # generate features:
            X = self.harmonics(X_raw, Fs, freq_range, n_harmonics, peak_height_fract=peak_height_fract,
                               peak_sep=peak_sep, N_fft=None, norm=True)
            X = X.to_numpy()

        elif feature_type == "fft_bins":
            # extract specific feature generation parameters:
            n_bins = feature_gen_params.get("n_bins")
            if n_bins is None:
                raise Exception("Missing specific feature generation parameters.")
            # generate features:
            X, _ = self.fft_bins(X_raw, Fs, freq_range, n_bins, N_fft=N_fft, norm=norm)

        return X

    def harmonics(self, X_raw, Fs, freq_range, n_harmonics, peak_height_fract=0.05, peak_sep=10.0, N_fft=None,
                  norm=True):
        """
        Finds n_harmonics strongest harmonics of audio data.

        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N
            Fs: Sampling frequency (Hz).
            freq_range: (min_freq, max_freq) (in Hz) range of frequencies to include for finding peaks.
            n_harmonics: Number of harmonics to find.
            peak_height_fract: Minimum height of FFT peaks = min_height_fract * max value of FFT.
            peak_sep: Minimum separation (distance) between neighboring FFT peaks (in Hz).
            N_fft: Number of FFT points to use.
                If None, a default number of FFT points is used.
            norm: Selects whether to normalize raw audio data.

        Returns:
            harmonics: Frequencies of harmonics.
                dim: (N, n_harmonics)
        """

        # set default values to certain parameters, if they are None:
        if peak_height_fract is None:
            peak_height_fract = 0.05
        if peak_sep is None:
            peak_sep = 10.0
        N = len(X_raw)

        # compute FFTs of audio data:
        X_fft = self.compute_fft(X_raw, Fs, N_fft=N_fft, norm=norm)

        # convert freq_range and peak_sep from Hz to samples:
        freq_space = self.freq[1] - self.freq[0]
        peak_sep_samples = int(np.round(peak_sep / freq_space))

        # estimate harmonics by finding largest peaks of FFTs:
        harmonics_data = []
        for i in range(N):
            # minimum peak height:
            peak_height = peak_height_fract * np.amax(X_fft[i])
            # find locations (sample indices) of peaks:
            peak_inds_all, peak_properties = find_peaks(X_fft[i], height=peak_height, distance=peak_sep_samples)
            # convert to Hz:
            peaks_all = self.freq[peak_inds_all]

            # only keep peaks in specified frequency range:
            peaks = []
            peak_heights = []
            for k in range(len(peaks_all)):
                peak = peaks_all[k]
                if freq_range[0] <= peak <= freq_range[1]:
                    peaks.append(peak)
                    peak_heights.append(peak_properties["peak_heights"][k])
            peaks = np.array(peaks)
            peak_heights = np.array(peak_heights)

            # keep only first n_harmonics highest peaks:
            n_peaks = len(peaks)
            if n_peaks > n_harmonics:
                # sort peaks in descending order of height:
                peak_sort_inds = np.flip(np.argsort(peak_heights))
                peaks_sort = peaks[peak_sort_inds]
                # keep only first n_harmonics peaks:
                peaks = peaks_sort[0:n_harmonics]
                # "unsort" resulting peaks:
                peaks = np.sort(peaks)
            n_peaks = len(peaks)

            # save peaks:
            harmonics_data.append(list(peaks))

        # construct dataframe:
        cols = []
        for h in range(n_harmonics):
            cols.append("Harmonic " + str(h+1))
        harmonics_df = pd.DataFrame(harmonics_data, columns=cols)
        # impute missing values with feature mean:
        miss_vals_series = harmonics_df.isnull().sum()
        miss_vals_series = miss_vals_series[miss_vals_series > 0]
        miss_vals_cols = list(miss_vals_series.index)

        """
        # TEMP:
        print()
        print(harmonics_df["# of Harmonics"].describe())
        print("\nNumbers of Missing values:")
        print(miss_vals_series)
        """

        for col in miss_vals_cols:
            harmonics_df[col] = harmonics_df[col].fillna(value=harmonics_df[col].mean())

        return harmonics_df

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

