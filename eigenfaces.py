import numpy as np

class EigenfacesInterface:
    
    def fit(self):
        pass

    def predict(self):
        pass

class Clustering(EigenfacesInterface):

    def __init__(self, n_clusters:int):
        pass
    
    def fit(self, X:np.array):
        pass

    def predict(self, x:np.array) -> np.array:
        pass

class Classifier(EigenfacesInterface):

    def __init__(self):
        pass

    def fit(self, X:np.array, y:np.array):
        pass

    def predict(self, x:np.array) -> np.array:
        pass