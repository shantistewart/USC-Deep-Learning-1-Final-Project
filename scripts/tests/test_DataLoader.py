"""File for testing DataLoader class."""


from preprocessing.load_data_class import DataLoader


# path to data folder:
data_folder = "../../data/"


# test DataLoader class:
print()
data_loader = DataLoader()
X, y, Fs = data_loader.load_data(data_folder)
print(len(X))
print(len(y))
print(Fs)

