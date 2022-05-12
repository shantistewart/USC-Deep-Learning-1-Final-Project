

# USC-Deep-Learning-Final-Project

The goal of this project was to develop several machine learning models to classify a chord as major or minor, given its
raw audio waveform.



## Required Python Libraries

* Pandas
* NumPy
* SciPy
* Scikit-learn
* PyTorch
* Matplotlib


## Repository Structure

* **data:** _Audio data files (in .wav format)._
   * **major:** _Audio data files of major chords._
   * **minor:** _Audio data files of minor chords._
* **modeling:** _Classes and functions for running and implementing models._
   * mlp_pytorch_class.py
   * mlp_sklearn_class.py
   * model_pipeline_class.py
   * run_model.py
* **preprocessing:** _Classes for loading and preprocessing data and generating features._
   * generate_features_class.py
   * load_data_class.py
* **scripts:** _Scripts (files) that can be run._
   * **models:** _Scripts to run different models._
      * kNN.py
      * mlp.py
      * nearest_means.py
      * random_forest.py
   * **tests:** _Scripts to run various unit tests._
      * test_DataLoader.py
      * test_FeatureGenerator.py
      * test_MLP_sklearn.py
      * test_harmonics.py
      * test_intervals.py


## Model Scripts

* **kNN.py:** _trains/tunes/evaluates a k-Nearest Neighbors classifier._
* **mlp.py:** _trains/evaluates a MLP (multi-layer perceptron) classifier._
* **nearest_means.py:** _trains/evaluates a Nearest Means (a.k.a. Nearest Centroids) classifier._
* **random_forest.py:** _trains/tunes/evaluates a Random Forest classifier._


## Authors

* **Shanti Stewart** -- M.S. student in Machine Learning & Data Science (Electrical & Computer Engineering) at the
University of Southern California.
   * [LinkedIn](https://www.linkedin.com/in/shanti-stewart/)
   * [GitHub](https://github.com/shantistewart)
* **Jeffery Wong** -- M.S. student in Electrical & Computer Engineering at the  University of Southern California.
   * [GitHub](https://github.com/jefferytwong)

