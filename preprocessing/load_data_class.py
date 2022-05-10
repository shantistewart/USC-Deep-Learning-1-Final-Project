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

        for root, dirs, files in os.walk(data_folder, topdown = True):
            

            files = sorted(files)
            for dir in dirs:
            for f in files:
                X = read(os.path.join(root, f))
        # return X, y

