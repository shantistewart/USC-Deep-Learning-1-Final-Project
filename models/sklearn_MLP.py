import warnings
import inspect
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import utils as sk_utils
from sklearn.utils.validation import check_X_y
from sklearn.utils.estimator_checks import check_estimator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from models.pytorch_MLP import _MLP


class MLP(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            num_hidden_units,
            hidden_size,
            num_epochs=50,
            learning_rate=0.05,
            batch_size=50,
            scheduler_step=20,
            scheduler_gamma=0.01
    ):

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        # tuning MLP model
        self.num_hidden_units = num_hidden_units
        self.hidden_size = hidden_size

        self.model = None

        # tuning batch and epoch
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # tuning step size and LR
        self.learning_rate = learning_rate
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma



    def fit(self, X, y=None, use_cuda=False):
        """Trains MLP.

        Args:
            X: Features.
                dim: (N, D)
            y: Labels.
                dim: (N, )

        Returns: self
        """
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')
        X, y = check_X_y(X, y)
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                warnings.warn("cuda not available", UserWarning)
        else:
            self.device = torch.device("cpu")

        self.model = self._train_classifier(X, y)

        return self

    def transform(self, X):
        sk_utils.validation.check_is_fitted(self, ['_model'])
        X = sk_utils.check_array(X)
        X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            self.model.eval()
            output = self.model.forward(X)
            return output.cpu().numpy()

    def _train_classifier(self, x_train, y_train):
        """Choose training classifier: minibatch vs batch

        Args:
            X: Features.
                dim: (N, D)
            y: Labels.
                dim: (N, )

        Returns: model
        """
        x_train = torch.from_numpy(x_train.astype(np.float32)).to(self.device)
        y_train = torch.from_numpy(y_train.astype(np.float32)).to(self.device)

        if self.batch_size is None:
            input_dim = x_train.shape[-1]
            output_dim = y_train.shape[-1]
        else:
            input_dim = self.batch_size
            output_dim = self.batch_size

        # call pytorch model
        self.model = _MLP(self.num_hidden_units, self.hidden_size, input_dim, output_dim).to(self.device)

        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
        if self.batch_size is None:
            return self._batch_train(x_train, y_train, self.model, loss_function, optimizer, scheduler)
        else:
            return self._minibatch_train(x_train, y_train, self.model, loss_function, optimizer, scheduler)

    def _batch_train(self, x_train, y_train, model, loss_function, optimizer, scheduler):
        """batch training classifier

        Args:
            X: Features.
                dim: (N, D)
            y: Labels.
                dim: (N, )
            model: type of model
            loss_function: type of loss function
            optimizer: type of optimizer

        Returns: model
        """
        for epoch in range(self.num_epochs):
            model.train()

            # forward pass
            y_pred = model.forward(x_train)
            loss = loss_function(y_pred, y_train)

            # back prop
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            scheduler.step()

            print("Train Epoch: {0}, Loss: {1},".format(
                epoch,
                loss.item()
            ))

        return model

    def _minibatch_train(self, x_train, y_train, model, loss_function, optimizer, scheduler):
        """minibatch training classifier

        Args:
            X: Features.
                dim: (N, D)
            y: Labels.
                dim: (N, )
            model: type of model
            loss_function: type of loss function
            optimizer: type of optimizer

        Returns: model
        """

        train = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size)
        for epoch in range(self.num_epochs):
            for idx, (minibatch, target) in enumerate(train_loader):
                model.train()

                # forward pass
                y_pred = model.forward(Variable(minibatch))
                y_pred = y_pred.reshape((-1,))
                loss = loss_function(y_pred, Variable(target.float()))

                # back prop
                optimizer.zero_grad()
                loss.backward()

                # optimize
                optimizer.step()
                scheduler.step()

            print("Train Epoch: {0} Loss: {1}".format(
                epoch,
                loss.item()
            ))

        return model

    def predict(self, X):
        """
        Makes a prediction using the trained pytorch model

        Args:
            X: Features.
                dim: (N, D)
            y: Labels.
                dim: (N, )
            model: model
        Returns:
            results: predictions
                dim: (N, )
        """

        results = []
        batches = np.ceil(len(X) / self.batch_size)

        for batch in np.array_split(X, batches):
            self.model.eval()
            x_pred = Variable(torch.from_numpy(batch).float())
            y_pred = self.model.forward(x_pred)
            y_pred = y_pred.reshape((-1,))

            predictions = np.round(y_pred.data.numpy()).astype(int)
            results = np.append(results, predictions)

        return np.array(results)

# check_estimator(MLP())