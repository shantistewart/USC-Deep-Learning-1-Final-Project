"""File for testing DataLoader class."""


from preprocessing.load_data_class import DataLoader


# path to data folder:
data_folder = "../../data/"


data_loader = DataLoader()
X, y = data_loader.load_data(data_folder)

