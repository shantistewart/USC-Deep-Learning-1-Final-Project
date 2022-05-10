"""File for testing DataLoader class."""


from preprocessing.load_data_class import DataLoader


# path to data folder:
data_folder = "../../data/"


# test DataLoader class:
print()
data_loader = DataLoader(test_fract=0.1)
X_train, y_train, X_test, y_test = data_loader.load_and_split_data(data_folder)

print(data_loader.Fs)
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

