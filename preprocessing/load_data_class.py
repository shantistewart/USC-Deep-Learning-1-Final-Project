"""File containing class for loading data."""
import numpy as np
import pandas as pd
import os
from scipy.io.wavfile import read


# major and minor class labels:
MAJ = 0
MIN = 1

class DataLoader:
    """Class for loading and splitting data into features and labels.

    Attributes:
        N: Total number of data points.
    """

    def __init__(self):
        self.N = None

    def load_data(self, data_folder):
        """Loads data.

        Args:
            data_folder: Path of data folder.

        Returns:
            X: Data.
                dim: (N, n_samples)
            y: Labels.
                dim: (N, )
        """
        X = pd.Dataframe()
        y = pd.Dataframe()
        for root, dirs, files in os.walk(data_folder, topdown = True):
            # label data
            if dirs == 'major':
                label = MAJ
            elif dirs == 'minor':
                label = MIN
            # import data
            files = sorted(files)
            for f in files:
                wav_file = read(os.path.join(root, f))
                wav_label = label

                X.append(wav_file)
                y.append(wav_label)

            return X, y
        # return X, y

