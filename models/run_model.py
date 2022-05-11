"""File containing function for running models (training, hyperparameter tuning, evaluation)."""


from preprocessing.load_data_class import DataLoader
from models.model_pipeline_class import ModelPipeline


def run_model(data_folder, model, test_fract, feature_type, feature_gen_params, norm_type=None, feature_select=None,
              pca=False, tune_model=True, hyperparams=None, search_type="grid", metric="accuracy", n_iters=None,
              n_folds=5, final_eval=False, verbose=1):
    """Trains, tunes, and evaluates model.

    Args:
        data_folder: Path of data folder.
        model: sklearn model (estimator) object, with some initial hyperparameters.
        test_fract: Fraction of total data to use for test set.
        feature_type: Type of feature to generate.
            allowed values: "harmonics", "intervals, "fft_bins"
        feature_gen_params: Dictionary of parameters for feature generation.
        norm_type: Type of feature normalization to use.
            allowed values: "standard", None
        feature_select: Method of feature selection.
            allowed values: "KBest", "SFS", None
        pca: Selects whether to use PCA.
        tune_model: Selects whether to tune hyperparameters of model.
        hyperparams: Dictionary of hyperparameter values to search over (ignored if tune_model = False).
        search_type: Hyperparameter search type (ignored if tune_model = False).
            allowed values: "grid", "random"
        metric: Type of metric to use for model evaluation (ignored if tune_model = False).
        n_iters: Number of hyperparameter combinations that are tried in random search (ignored if tune_model = False
            or if search_type != "random")
        n_folds: Number of folds (K) to use in stratified K-fold cross validation.
        final_eval: Selects whether to evaluate final model on test set.
        verbose: Nothing printed (0), some things printed (1), everything printed (2).

    Returns:
        metrics: Nested dictionary of training/test set metrics.
            metrics["train"]["metric_name"] = training set metric_name metric
            metrics["val"]["metric_name"] = cross-validation metric_name metric
            metrics["test"]["metric_name"] = test set metric_name metric
        best_models: Dictionary of best model information after hyperparameter tuning.
            best_models["hyperparams"] = best model hyperparameters
            best_models["cv_score"] = best model cross-validation score.
    """

    metrics = {}
    best_models = {}

    # load all data:
    data_loader = DataLoader(test_fract=test_fract)
    X_raw_train, y_train, X_raw_test, y_test = data_loader.load_and_split_data(data_folder)
    Fs = data_loader.Fs
    feature_gen_params["Fs"] = Fs

    # create initial ModelPipeline object:
    model_pipe = ModelPipeline(model, feature_type=feature_type, norm_type=norm_type, feature_select=feature_select,
                               pca=pca)

    # if tuning not desired, train model using given hyperparameters
    #   (note: feature selection and PCA transformer are created with default parameters):
    if not tune_model:
        if verbose != 0:
            print("Training model...")
        model_pipe.train(X_raw_train, y_train, feature_gen_params)
    # otherwise, tune hyperparameters:
    else:
        if verbose != 0:
            print("Tuning model hyperparameters...")
        if hyperparams is None:
            raise Exception("Hyperparameter search values argument is none.")
        best_model, best_hyperparams, best_cv_score = model_pipe.tune_hyperparams(X_raw_train, y_train,
                                                                                  feature_gen_params, hyperparams,
                                                                                  search_type=search_type,
                                                                                  metric=metric, n_iters=n_iters,
                                                                                  n_folds=n_folds, verbose=verbose)
        # save best model information to dictionary:
        best_models["hyperparams"] = best_hyperparams
        best_models["cv_score"] = best_cv_score

    # evaluate model on training set:
    if verbose != 0:
        print("\nEvaluating model on training set...")
    train_metrics = model_pipe.eval("train", X_raw_train, y_train, feature_gen_params, n_folds=n_folds, verbose=verbose)

    # evaluate model using cross validation:
    if verbose != 0:
        print("\nEvaluating model using cross validation...")
    val_metrics = model_pipe.eval("cross_val", X_raw_train, y_train, feature_gen_params, n_folds=n_folds,
                                  verbose=verbose)

    # evaluate model on test set, if selected:
    if final_eval:
        if verbose != 0:
            print("\nEvaluating model on test set...")
        # first, (re)train model on full training set (since cross validation alters the sklearn Pipeline object):
        model_pipe.train(X_raw_train, y_train, feature_gen_params)
        test_metrics = model_pipe.eval("test", X_raw_train, y_train, feature_gen_params, n_folds=n_folds,
                                       verbose=verbose)
    if verbose != 0:
        print("")

    # save training/cross-validation/test set metrics to nested dictionary:
    metrics["train"] = train_metrics
    metrics["val"] = val_metrics
    if final_eval:
        metrics["test"] = test_metrics

    return metrics, best_models

