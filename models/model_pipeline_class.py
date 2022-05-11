"""File containing class for training, tuning, and evaluating a model."""


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from preprocessing.generate_features_class import FeatureGenerator


class ModelPipeline:
    """Class for training, tuning, and evaluating a model.

    Attributes:
        feature_type: Type of feature to generate.
            allowed values: "harmonics", "fft_bins"
        feature_gen: FeatureGenerator object
        norm_type: Type of feature normalization to use.
            allowed values: "standard", None
        feature_select: Method of feature selection.
            allowed values: "KBest", "SFS", None
        pca: Selects whether to use PCA.
        model_pipe: sklearn Pipeline object for model.
            Updated to best model (i.e., model with best hyperparameters) when tune_hyperparams() method is called.
        best_hyperparams: Best hyperparameters (dictionary).
    """

    def __init__(self, model, feature_type, norm_type=None, feature_select=None, pca=False):
        """Initializes ModelPipeline object.

        Args:
            model: sklearn model (estimator) object.
            feature_type: Type of feature to generate.
                allowed values: "harmonics", "fft_bins"
            norm_type: Type of normalization to use.
                allowed values: "standard", None
            feature_select: Method of feature selection.
                allowed values: "KBest", "SFS", None
            pca: Selects whether to use PCA.

        Returns: None
        """

        # validate parameters:
        if norm_type is not None and norm_type != "standard":
            raise Exception("Invalid normalization type.")
        if feature_select is not None and feature_select != "KBest" and feature_select != "SFS":
            raise Exception("Invalid feature selection method.")

        self.feature_type = feature_type
        self.feature_gen = FeatureGenerator()
        self.norm_type = norm_type
        self.feature_select = feature_select
        self.pca = pca
        self.best_hyperparams = None

        # initialize sklearn pipeline object:
        self.model_pipe = None
        self.make_pipeline(model)

    def train(self, X_raw_train, y_train, feature_gen_params):
        """Trains model.

        Args:
            X_raw_train: List of raw audio data numpy arrays (of training set).
                length: N
            y_train: Labels (of training set).
                dim: (N, )
            feature_gen_params: Dictionary of parameters for feature generation.

        Returns: None
        """

        # generate features:
        X_train = self.feature_gen.generate_features(self.feature_type, X_raw_train, feature_gen_params)

        # train model:
        self.model_pipe.fit(X_train, y_train)

    def eval(self, eval_type, X_raw, y, feature_gen_params, n_folds=5, verbose=1):
        """Evaluates model on data.

        Args:
            eval_type: Type of evaluation to run.
                allowed values: "train", "cross_val", "test"
            X_raw: List of raw audio data numpy arrays.
                length: N
            y: Labels.
                dim: (N, )
            feature_gen_params: Dictionary of parameters for feature generation.
            n_folds: Number of folds (K) to use in stratified K-fold cross validation (ignored if eval_type !=
                "cross_val").
            verbose: Nothing printed (0), accuracy and F1-score printed (1), all metrics printed (2).

        Returns:
            metrics: Dictionary of performance metrics.
                metrics["accuracy"] = accuracy
                metrics["f1"] = F1-score
                metrics["conf_matrix"] = confusion matrix
        """

        # validate evaluation type:
        if eval_type != "train" and eval_type != "cross_val" and eval_type != "test":
            raise Exception("Invalid evaluation type.")

        # generate features:
        X = self.feature_gen.generate_features(self.feature_type, X_raw, feature_gen_params)

        # perform cross validation if selected:
        if eval_type == "cross_val":
            scores = cross_validate(self.model_pipe, X, y, cv=n_folds, scoring=["accuracy", "f1"])
            accuracy = np.mean(scores["test_accuracy"])
            f1 = np.mean(scores["test_f1"])
        # otherwise, evaluate directly on provided dataset:
        else:
            # predict on data:
            y_pred = self.model_pipe.predict(X)
            # compute metrics (accuracy, F1-score, confusion matrix):
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            conf_matrix = confusion_matrix(y, y_pred, normalize=None)
        # print metrics:
        if verbose != 0:
            print("accuracy = {} %".format(100*accuracy))
            print("F1-score = {}".format(f1))
        if eval_type != "cross_val" and verbose == 2:
            print("confusion matrix = \n{}".format(conf_matrix))

        # save metrics to dictionary:
        metrics = {"accuracy": accuracy,
                   "f1": f1}
        if eval_type != "cross_val":
            metrics["conf_matrix"] = conf_matrix

        return metrics

    def tune_hyperparams(self, X_raw, y, feature_gen_params, hyperparams, search_type="grid", metric="accuracy",
                         n_iters=None, n_folds=10, verbose=1):
        """Tunes hyperparameters (i.e., model selection).

        Args:
            X_raw: List of raw audio data numpy arrays.
                length: N
            y: Labels.
                dim: (N, )
            feature_gen_params: Dictionary of parameters for feature generation.
            hyperparams: Dictionary of hyperparameter values to search over.
            search_type: Hyperparameter search type.
                allowed values: "grid", "random"
            metric: Type of metric to use for model evaluation.
            n_iters: Number of hyperparameter combinations that are tried in random search (ignored if
                search_type != "random")
            n_folds: Number of folds (K) to use in stratified K-fold cross validation.
            verbose: Nothing printed (0), best hyperparameters printed (not 0).

        Returns:
            model_pipe: sklearn Pipeline object for best model (i.e., model with best hyperparameters).
            best_hyperparams: Best hyperparameters (dictionary).
            best_cv_score: Cross-validation score of best model.
        """

        # validate hyperparameter search type:
        if search_type != "grid" and search_type != "random":
            raise Exception("Invalid hyperparameter search type.")

        # generate features:
        X = self.feature_gen.generate_features(self.feature_type, X_raw, feature_gen_params)

        # tune hyperparameters:
        if search_type == "grid":
            search = GridSearchCV(self.model_pipe, hyperparams, cv=n_folds, scoring=metric)
        elif search_type == "random":
            search = RandomizedSearchCV(self.model_pipe, hyperparams, n_iter=n_iters, cv=n_folds, scoring=metric)
        # run hyperparameter search:
        search.fit(X, y)

        # save best model, best hyperparameters, and best cross-validation score:
        self.model_pipe = search.best_estimator_
        self.best_hyperparams = search.best_params_
        best_cv_score = search.best_score_
        # print hyperparameter tuning results:
        if verbose != 0:
            print("Best hyperparameters: {}".format(self.best_hyperparams))

        return self.model_pipe, self.best_hyperparams, best_cv_score

    def make_pipeline(self, model):
        """Creates a sklearn Pipeline object for model.

        Args:
            model: sklearn model (estimator) object.

        Returns: None
        """

        # create sklearn normalization transformer:
        normalizer = None
        if self.norm_type == "standard":
            normalizer = StandardScaler()

        # create sklearn feature selection transformer:
        feature_selector = None
        if self.feature_select == "KBest":
            feature_selector = SelectKBest()
        elif self.feature_select == "SFS":
            feature_selector = SequentialFeatureSelector(model)

        # create sklearn PCA transformer:
        pca = None
        if self.pca:
            pca = PCA()

        # create sklearn pipeline:
        steps = []
        if normalizer is not None:
            steps.append(("normalizer", normalizer))
        if feature_selector is not None:
            steps.append(("selector", feature_selector))
        if pca is not None:
            steps.append(("pca", pca))
        steps.append(("model", model))
        self.model_pipe = Pipeline(steps=steps)

