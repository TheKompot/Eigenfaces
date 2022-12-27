import numpy as np
from sklearn.cluster import KMeans

class NotFittedError(Exception):
    ''' Raised when model is asked to do something(e.g. predict) before it was fitted '''
    pass

class DimensionalityReduction:
    '''reduces dimension of a matrix by finding its eigenvectors.
     Ouput dimension can be from range [1, MIN(M,N)], where M and N are 
     the dimensions of input matrix'''

    def __init__(self):
        self.e_vec = None

    def fit(self,X:np.array, k:int):
        ''' finds k biggests eigenvectors of matrix X for further calculations (e.g. transform)
        
        INPUT:
        ------
        X : np.array
            input matrix with dimensions number_of_pictures x number_of_pixels
        k : int
            final number of dimensions, from range [1, MIN(number_of_pictures, number_of_pixels)]
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
        return np.linalg.lstsq(self.e_vec,x.T,rcond=None)[0].T # shape is number_of_pictures x k
    
    def fit_transform(self, X:np.array, k:int) -> np.array:
        '''Combination of methods fit and transform'''
        self.fit(X,k)
        return self.transform(X)


class Predictor:
    '''Abstract class for further models, needs attribute '_centroids' 
    to be defines and of shape number_of_rows_in_x x number_of_centroids'''

    def predict(self, x:np.array) -> np.array:
        '''To all rows of 'x' the method finds the closest centroid and
         returns a vector with the indexes of the closest cetroid'''
        if self._centroids is None:
            raise NotFittedError('Needs to be first fitted to the trainning data')
        
        output = []

        for i in range(x.shape[0]): # itter through all rows/pictures
            vec = x[i,:]

            # finding the closest cetroid and its labels
            label = None
            smallest_dist = None

            for j in range(self._centroids.shape[1]):  # itter through columns/centroids
                cluster = self._centroids[:,j]
                
                dist = np.linalg.norm(vec - cluster)    # calculatiing euclidean distance FREE FOR EXPERIMENTATION

                if label is None or dist < smallest_dist: # if no label set or this centorid is the closes -> set new label
                    label = j
                    smallest_dist = dist
            output.append(label)                        # after cheching all centroids, mark the label of the best one
        return np.array(output)                         # shape (number_of_rows_in_x,)

    def fit(self, X:np.array):
        '''For children to implement'''
        pass

class ClusteringEig(Predictor):
    '''Clustering model that uses eigenvectors from train data as centroids'''
    def __init__(self, n_clusters:int):
        self.n_clusters = n_clusters    # number of clusters
        self._centroids = None          # must be defined for parent class
    
    def fit(self, X:np.array):
        '''Fit the model using PCA'''
        self.pca = DimensionalityReduction()
        self.pca.fit(X,self.n_clusters)

        self._centroids = self.pca.e_vec # eigenvectors are the centroids
        
class ClusteringKMeans(Predictor):
    '''Class for clustering that first puts data in to a lower dimension k
    then uses kmeans algorithm for computing centroids
    
    Attributes:
    ----------
    _centroids : np.array
        matrix where the columns represent the centroids
    _inertia: float
        error of KMeans algorithm
    '''

    def __init__(self, n_clusters:int, k_dimensions:int):
        '''Constructor
        INPUT:
        ------
        n_clusters : int
            number of clusters for KMeans
        k_dimensions : int
            dimension in which KMeans will calculate centroids
            and in which prediction will occur
        '''
        self.n_clusters = n_clusters
        self.k_dimensions = k_dimensions
        self._centroids = None
        self._inertia = None        
    
    def fit(self, X:np.array):
        '''fit the model using PCA and KMeans'''
        self.pca = DimensionalityReduction()
        x = self.pca.fit_transform(X,self.k_dimensions) # getting data from lower dimension
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0,n_init=10) # applying KMeans in loweer dimension
        kmeans.fit(x)                            # fitting model, euclidean distance used FREE FOR EXPERIMENTATION
        self._inertia = kmeans.inertia_          # saving the kmeans error
        self._centroids = kmeans.cluster_centers_.T # saving the centroids calculated by kmeans
    
    def predict(self,x:np.array) -> np.array:
        ''' Transformes data in to lower dimension that predicts its label, returns vector of labels'''
        x = self.pca.transform(x)
        return super().predict(x)

class MemoryClassifier(Predictor):
    '''Classifier that transformes input data in to lower dimensions and remembers every point in training data and
    classifies a new data point according to what point from the training set is the closest'''

    def __init__(self,k_dimensions:int):
        self.k_dimensions = k_dimensions

    def fit(self, X:np.array, y:np.array):
        '''Fits model using PCA'''
        self.pca = DimensionalityReduction()                            
        self._centroids = self.pca.fit_transform(X,self.k_dimensions).T # saves centroid (all data points)
        self.y = y.copy() # saves labels for predict method

    def predict(self, x:np.array) -> np.array:
        '''Predicts label using input data, PCA and remembered training data'''
        x = self.pca.transform(x)
        x = super().predict(x) # parent method returns the index of the centorid

        return np.array([self.y[i] for i in x]) # transforming index to label

class CentroidClassifier(Predictor):
    '''Classifier puts data in to lower dimension, groups the data points according to labels and finds
    the average data points. These average data points are used as centroids '''

    def __init__(self,k_dimensions:int):
        self.k_dimensions = k_dimensions

    def fit(self, X:np.array, y:np.array):
        '''Fits model using PCA and by calculating the centers of data points with same label'''

        self.pca = DimensionalityReduction()
        X = self.pca.fit_transform(X,self.k_dimensions)
        
        vec_in_labels = {}      # dict, key is label, value is a list of data points (numpy arays)
        self.labels = []        # list of labels

        for i in range(X.shape[0]):  # itter through rows
            if y[i] not in vec_in_labels:   # add label to dict and list
                vec_in_labels[y[i]] = []
                self.labels.append(y[i])
            vec_in_labels[y[i]].append(X[i,:]) # append data point to its label
        
        centroids = []          # temp list for calculating centroids
        for i in self.labels:   # itter through all labels
            arr = np.array(vec_in_labels[i])    # change list of vectors to matrix
            centroids.append(np.mean(arr,axis=0)) # append the mean vector
        self._centroids = np.array(centroids).T # save them as matrix, where columns are centroids
    
    def predict(self, x:np.array) -> np.array:
        '''Uses PCA, input data and list of existing labels to predict specific labels'''
        x = self.pca.transform(x)
        x = super().predict(x)

        return np.array([self.labels[i] for i in x])