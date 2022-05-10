"""File containing class for MLP model."""

import numpy as np
import torch
from torch.autograd import Variable
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class MLP(ClassifierMixin, BaseEstimator):
    """MLP class.

    Attributes:
        Other attributes required to be a valid sklearn classifier.
    """

    def __init__(self):
        # GPU flag
        self._gpu = use_gpu and torch.cuda.is_available()
        pass

    def fit(self, X, y, **kwargs):
        """Trains MLP.

        Args:
            X: Features.
                dim: (N, D)
            y: Labels.
                dim: (N, )

        Returns: self
        """

        # things required to be a valid sklearn classifier:
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True

        # train MLP:
        # CODE HERE

        return self

    def predict(self, X):
        """Predicts class labels of data.

        Args:
            X: Features.
                dim: (N, D)

        Returns:
            y_pred: Class label predictions.
                dim: (N, )
        """

        # things required to be a valid sklearn classifier:
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])
        check_array(X)

        # make predictions:
        # determine length of each batch size
        batch_length = np.ceil(len(X) / self.batch_size)

        results = []
        # split X accordingly per batch
        for batch in np.array_split(X, batch_length):
            x_pred = Variable(torch.from_numpy(batch).float())
            # need to define model
            y_pred = self.model(x_pred.cuda() if self._gpu else x_pred)
            results.append(y_pred)

        return results

    def score(self, X, y):
        """Scores the data using PyTorch model

        Args:
            X: Features.
                dim: (N, D)
            y: true labels of each chord
                dim: (N, )

        Returns:
            accuracy score: float
        """
        # Run prediction model
        y_pred = self.predict(X)
        N = len(y)
        # total number of correct labels
        correct = 0
        for i in range(N):
            correct += (y_pred[i] == y[i]).sum().numpy()

        # return accuracy score
        accuracy = correct / N
        return accuracy

