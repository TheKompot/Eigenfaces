import numpy as np

class NotFittedError(Exception):
    ''' Raised when model is asked to do something(e.g. predict) before it was fitted '''
    pass

class EigenfacesInterface:
    
    def fit(self):
        pass

    def transform(self):
        pass

    def predict(self):
        pass

class DimensionalityReduction(EigenfacesInterface):
    
    def __init__(self):
        self.e_vec = None

    def fit(self,X:np.array, k:int):
        ''' finds k biggests eigenvectors of matrix X for further calculations (e.g. transform)
        
        INPUT:
        ------
        X : np.array
            input matrix with dimensions number_of_pictures x number_of_pixels
        k : int
            final number of dimensions
        '''
        X = X.T # transpose so one column is one image

        self.avg_vector = np.mean(X,axis=1).reshape(1,-1) # average vector 

        X = X - self.avg_vector.T # normalized images to the center

        #eigenvalues and eigenvectors for m x m cov matrix
        e_val, e_vec = np.linalg.eig(X.T@X)

        #eigenvectors for n^2 x n^2 cov matrix
        e_vec = X@e_vec

        # selecting k biggest eigenvectors
        e_vec = e_vec[:,:k]

        self.e_vec = e_vec
    
    def transform(self,x:np.array) -> np.array:
        ''' applies dimesionality reduction on input matrix
        
        INPUT:
        -----
        x : np.array
            input matrix with dimensions number_of_pictures x number_of_pixels
        
        RETURNS
        -------
        np.array
            transformed matrix with dimensions number_of_pictures x k, where 'k' is defined while fitting
        '''

        if self.e_vec is None:
            raise NotFittedError("Needs to be first fitted to the trainning data")

        # centering the data
        x = x - self.avg_vector

        # applying the eigenmatrix transformation
        return x@self.e_vec
    
    def fit_transform(self, X:np.array, k:int) -> np.array:
        '''Combination of methods fit and transform'''
        self.fit(X,k)
        return self.transform(X)

    def predict(self):
        '''Predict has no use for dimensionality reduction'''
        pass


class ClusteringEig(DimensionalityReduction):

    def __init__(self, n_clusters:int):
        self.n_clusters = n_clusters
        self._centroids = None
    
    def fit(self, X:np.array):
        super().fit(X,self.n_clusters)

        self._centroids = self.e_vec

    def predict(self, x:np.array) -> np.array:
        if self._centroids is None:
            raise NotFittedError('Needs to be first fitted to the trainning data')
        
        output = []

        for i in range(x.shape[0]):
            vec = x[i,:]
            label = None
            smallest_dist = None

            for j in range(self.n_clusters):
                cluster = self._centroids[:,j]
                
                dist = np.linalg.norm(vec - cluster)

                if label is None or dist < smallest_dist:
                    label = j
                    smallest_dist = dist
            output.append(label)
        return np.array(output)
        

class Classifier(DimensionalityReduction):

    def __init__(self):
        pass

    def fit(self, X:np.array, y:np.array):
        pass

    def predict(self, x:np.array) -> np.array:
        pass