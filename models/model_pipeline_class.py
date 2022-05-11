"""File containing class for training, tuning, and evaluating a model."""


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from preprocessing.load_data_class import DataLoader
from preprocessing.generate_features_class import FeatureGenerator


class ModelPipeline:
    """Class for training, tuning, and evaluating a model.

    Attributes:
        feature_type: Type of feature to generate.
            allowed values: "fft_bins", "fft_peaks"
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
                allowed values: "fft_bins", "fft_peaks"
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

