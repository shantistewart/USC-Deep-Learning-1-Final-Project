"""File containing class for loading data."""


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

        # return X, y

