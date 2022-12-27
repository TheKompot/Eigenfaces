import numpy as np
from sklearn.cluster import KMeans

class NotFittedError(Exception):
    ''' Raised when model is asked to do something(e.g. predict) before it was fitted '''
    pass

class DimensionalityReduction:
    
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

        # finding the weights that define the linear combination of eigenvectors that create vector x
        return np.linalg.lstsq(self.e_vec,x.T,rcond=None)[0].T
    
    def fit_transform(self, X:np.array, k:int) -> np.array:
        '''Combination of methods fit and transform'''
        self.fit(X,k)
        return self.transform(X)


class Predictor:
    '''Abstract class for further models'''
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

    def fit(self, X:np.array):
        pass

class ClusteringEig(Predictor):

    def __init__(self, n_clusters:int):
        self.n_clusters = n_clusters
        self._centroids = None
    
    def fit(self, X:np.array):
        self.pca = DimensionalityReduction()
        self.pca.fit(X,self.n_clusters)

        self._centroids = self.pca.e_vec
        
class ClusteringKMeans(Predictor):
    
    def __init__(self, n_clusters:int, k_dimensions:int):
        self.n_clusters = n_clusters
        self.k_dimensions = k_dimensions
        self._centroids = None
        self._inertia = None
    
    def fit(self, X:np.array):
        self.pca = DimensionalityReduction()
        x = self.pca.fit_transform(X,self.k_dimensions)
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0,n_init=10)
        kmeans.fit(x)
        self._inertia = kmeans.inertia_
        self._centroids = kmeans.cluster_centers_.T
    
    def predict(self,x:np.array)->np.array:
        x = self.pca.transform(x)
        return super().predict(x)

class Classifier(Predictor):

    def __init__(self):
        pass

    def fit(self, X:np.array, y:np.array):
        pass