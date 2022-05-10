"""File containing class for MLP model."""


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class MLP(ClassifierMixin, BaseEstimator):
    """MLP class.

    Attributes:
        Other attributes required to be a valid sklearn classifier.
    """

    def __init__(self):
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
        # TEMP:
        y_pred = None
        # CODE HERE

        return y_pred

