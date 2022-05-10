"""File containing class for loading data."""


import os
from scipy.io.wavfile import read

# major and minor chord class labels:
MAJ = 0
MIN = 1


class DataLoader:
    """Class for loading data and splitting intro train/test sets.

    Attributes:
        Fs: Sampling frequency (Hz).
        test_fract: Fraction of total data to use for test set.
        N: Total number of data points.
        N_train: Number of data points in training set.
        N_test: Number of data points in test set.
    """

    def __init__(self, test_fract=0.2):
        self.Fs = None
        self.test_fract = test_fract
        self.N = None
        self.N_train = None
        self.N_test = None

    def load_data(self, data_folder):
        """Loads data.

        Args:
            data_folder: Path of data folder.

        Returns:
            X: Audio data.
                dim: (N, n_samples)
            y: Labels.
                dim: (N, )
        """

        X = []
        y = []
        for root, dirs, _ in os.walk(data_folder, topdown=True):
            for d in dirs:
                # set label:
                if d == 'major':
                    label = MAJ
                elif d == 'minor':
                    label = MIN
                path = os.path.join(root, d)
                for _, _, files in os.walk(path, topdown=True):
                    for f in files:
                        Fs, audio_data = read(os.path.join(path, f))
                        X.append(audio_data)
                        y.append(label)
        self.Fs = Fs
        self.N = len(y)

        return X, y, Fs

