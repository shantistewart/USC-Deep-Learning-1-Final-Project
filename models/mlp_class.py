"""File containing class for MLP model."""

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from sklearn.utils.estimator_checks import check_estimator
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import mean_absolute_error

import inspect


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MLP(torch.nn.Module, ClassifierMixin, BaseEstimator):
    """MLP class.

    Attributes:
        Other attributes required to be a valid sklearn classifier.
    """

    def __init__(
            self,
            use_gpu=False,
            output_dim=2,
            input_dim=100,
            hidden_layer_dims=(100, 100),
            learning_rate=0.01,
            num_epochs=30,
            batch_size=30
    ):
        super(MLP, self).__init__()

        self._history = None
        self._model = None
        # GPU flag
        self._gpu = use_gpu and torch.cuda.is_available()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_dims = hidden_layer_dims
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        # define forward model
        self.hidden = torch.nn.Linear(input_dim, 10)
        self.hidden2 = torch.nn.Linear(10, 10)
        self.output = torch.nn.Linear(10, 2)

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    # def build_model(self):
    #     self.layer_dims = [self.input_dim] + self.hidden_layer_dims + [self.output_dim]
    #     self._model = torch.nn.Sequential()
    #     for idx, dim in enumerate(self.layer_dims):
    #         if idx < len(self.layer_dims) - 1:
    #             module = torch.nn.Linear(dim, self.layer_dims[idx + 1])
    #             torch.nn.init.xavier_uniform_(module.weight)
    #             self._model.add_module("linear" + str(idx), module)
    #
    #         if idx < len(self.layer_dims) - 2:
    #             self._model.add_module("relu" + str(idx), torch.nn.ReLU())
    #
    #         if self._gpu:
    #             self._model = self._model.cuda()

    def forward(self, x):
        """Define forward model of MLP

        Args:
            x:

        Returns:

        """
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.softmax(self.output(x), dim=1)
        return x

    def fit(self, X, y, **kwargs):
        """Trains MLP.

        Args:
            X: Features.
                dim: (N, D)
            y: Labels.
                dim: (N, )

        Returns: self
        """
        self._model = MLP()

        # things required to be a valid sklearn classifier:
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')
        # checks if dimensions agree
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True

        # train MLP:
        torch_x = torch.from_numpy(X).float()
        torch_y = torch.from_numpy(y).float()
        if self._gpu:
            torch_x = torch_x.cuda()
            torch_y = torch_y.cuda()

        train = torch.utils.data.TensorDataset(torch_x, torch_y)
        # not shuffling data here
        train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True)

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        self.history = {'accuracy':[], 'loss': []}
        for epoch in range(self.num_epochs):
            correct = 0
            for idx, (minibatch, target) in enumerate(train_loader):
                y_pred = self._model(Variable(minibatch))
                loss = loss_func(y_pred.float(), Variable(target.cuda().long()) if self._gpu else Variable(target.long()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                y_labels = target.cuda().numpy() if self._gpu else target.numpy()
            # y_pred_results = y_pred.cuda().data.numpy() if self._gpu else y_pred.data.numpy()

            predictions = torch.argmax(y_pred.data, dim=1).numpy()
            correct += (predictions == y_labels).sum()

            error = mean_absolute_error(predictions, y_labels)

            self._history["accuracy"].append(correct/len(train_loader))
            self._history["loss"].append(error)
            print("Results for epoch {0}, accuracy {1}, MSE_loss {2}".format(epoch + 1,
                                                                             correct/len(train_loader), error))

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
        self._model.eval()
        # determine length of each batch size
        batch_length = np.ceil(len(X) / self.batch_size)

        results = []
        # split X accordingly per batch
        for batch in np.array_split(X, batch_length):
            x_pred = Variable(torch.from_numpy(batch).float())
            y_pred = self._model(x_pred.cuda() if self._gpu else x_pred)

            predictions = torch.argmax(y_pred.data, dim=1).numpy()
            results = np.append(results, predictions)

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
        # total number of data
        N = len(y)
        y = np.array(y)
        # total number of correct labels
        correct = (y_pred == y).sum()

        # return accuracy score
        accuracy = correct / N
        return accuracy

check_estimator(MLP())