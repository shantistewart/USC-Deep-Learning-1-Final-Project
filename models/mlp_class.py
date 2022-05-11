"""File containing class for MLP model."""

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MLP(torch.nn.Module, ClassifierMixin, BaseEstimator):
    """MLP class.

    Attributes:
        Other attributes required to be a valid sklearn classifier.
    """

    def __init__(self, use_gpu=False, output_dim=2, input_dim=50715, hidden_layer_dims=[100, 100],
                 learning_rate=0.01, num_epochs=30, batch_size=30):
        super(MLP, self).__init__()
        self.history = None
        self.model = None
        # GPU flag
        self.gpu = use_gpu and torch.cuda.is_available()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_dims = hidden_layer_dims
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        # define forward model
        self.hidden = torch.nn.Linear(input_dim, 10)
        self.output = torch.nn.Linear(10, 2)

    # def build_model(self):
    #     self.layer_dims = [self.input_dim] + self.hidden_layer_dims + [self.output_dim]
    #     self.model = torch.nn.Sequential()
    #     for idx, dim in enumerate(self.layer_dims):
    #         if idx < len(self.layer_dims) - 1:
    #             module = torch.nn.Linear(dim, self.layer_dims[idx + 1])
    #             torch.nn.init.xavier_uniform_(module.weight)
    #             self.model.add_module("linear" + str(idx), module)
    #
    #         if idx < len(self.layer_dims) - 2:
    #             self.model.add_module("relu" + str(idx), torch.nn.ReLU())
    #
    #         if self.gpu:
    #             self.model = self.model.cuda()

    def forward(self, x):
        """Define forward model of MLP

        Args:
            x:

        Returns:

        """
        x = F.relu(self.hidden(x))
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
        self.model = MLP()

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
        if self.gpu:
            torch_x = torch_x.cuda()
            torch_y = torch_y.cuda()

        train = torch.utils.data.TensorDataset(torch_x, torch_y)
        # not shuffling data here
        train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size)

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.history = []
        for epoch in range(self.num_epochs):
            correct = 0
            for idx, (minibatch, target) in enumerate(train_loader):
                y_pred = self.model(Variable(minibatch))
                loss = loss_func(y_pred.float(), Variable(target.cuda().long()) if self.gpu else Variable(target.long()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                y_labels = target.cuda().numpy() if self.gpu else target.numpy()
                y_pred_results = y_pred.cuda().data.numpy() if self.gpu else y_pred.data.numpy()

                predictions = torch.argmax(y_pred, dim=1).numpy()
                correct += (predictions == y_labels).sum()
                self.history.append(correct/len(train_loader))
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

