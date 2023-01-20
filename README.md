# Eigenfaces
Semester project for class **Principles of Data Science**

## Overview
This project is focused on using Principal Component Analysis (PCA) on photos. We are implementing PCA algorithm using `numpy` library in Python and comparing it with  build-in [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) algorithm in `scikit.learn`. Using PCA algorithm and three different clustering methods ([K-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) from `scikit.learn`, our own implemantation of K-means and [Gaussian Mixture algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) also from `scikit.learn`) we are trying to find groups of teachers at our faculty who look alike. We are also curious about which of our teachers look alike famous mathematicians (e.g. Gauss, Neumann, Turing, Einstein, ...). 

## Data
*   Testing PCA and ML algorithms is done on synthetic face images acquired from: https://github.com/microsoft/DigiFace1M
*   Faces for analysis and classification were scraped from personal websites from individual departments here: https://fmph.uniba.sk/pracoviska/

Web scraping was carried out using Python libraries `requests` (https://requests.readthedocs.io/en/latest/) and `beautifulsoup4` (https://pypi.org/project/beautifulsoup4/)

##
## Theory
First let's look at PCA algorithm.

## PCA
PCA is technique used for dimensionality reduction. When we have a dataset of $m$ face images of size $N$ x $N$ and we want to convert them into vectors and put them together, we get a matrix $X$ with the shape $m$ x $N^2$ (number_of_pictures x width_of_picture^2). The variable $N^2$ can be easily in the tens of thousands, which would be inefficient and troublesome to compute.

To make an efficient face recognition algorithm, we want to reduce the dimension of our dataset. We will use the $k$ largest eigenvectors (meaning eigenvectors with the largest eigenvalues) from covariance matrix of our dataset to encode the dataset into a matrix $\Omega$ with the shape of $m$ x $k$.

 First we center our dataset by subtracking the mean:
 $a_i =  x_i - \mu$ , where  
 ```math
 \mu = \frac{1}{m} \sum_{i=1}^m x_i $
 ```
 From that we get a new matrix $A$: 
```math
A = \begin{bmatrix}a_1 & a_2 & ... & a_m\end{bmatrix}
```
 with the shape of $N^2$ x $m$. The covariance matrix of $A$ is $AA^T$, which has $N^2$ eigenvectors of size $N^2$. To reduce the computation power needed, we will compute the eigenvectors of the smaller covariance matrix $A^TA$, which has only $m$ eigenvectors of size $m$ and then convert them into eigenvectors of matrix $AA^T$, using:
 ```math
 A^TAv_i = \lambda v_i
 ```
```math
 AA^TAv_i = \lambda A v_i
 ```
```math
 Cu_i = \lambda u_i
 ```
 where $C = AA^T$ (larger covariance matrix) and $u_i = Av_i$, which is the relationship between the eigenvectors of $AA^T$ and $A^TA$.

 Now we compute the eigenvectors of $A^TA$, take those with the $k$ largest eigenvalues, convert them into the eigenvectors of $AA^T$ and create a matrix $U$:
  ```math
 U = \begin{bmatrix}
    u_1 & u_2 & ... & u_k
 \end{bmatrix}
 ```
 which has the shape of $N^2$ x $k$.

 To get our matrix $\Omega$ we need to calculate the coefficients $w_1$ ... $w_k$ of the linear combination of eigenvectors that comprise vector $a_i$. To do that, we compute the following system of equations using the least squares method:
```math
 a_{i1} = w_{i1} u_{11} + w_{i2} u_{21} + ... + w_{ik} u_{k1}
 ```
  ```math
 a_{i2} = w_{i1} u_{12} + w_{i2} u_{22} + ... + w_{ik} u_{k2}
 ```
  ```math
 ...
 ```
  ```math
 a_{in^2} = w_{i1} u_{1n^2} + w_{i2} u_{2n^2} + ... + w_{ik} u_{kn^2}
 ```

for $i = {1,2 ... , m}$. 

We will not show you how to solve these equations, to stop boring you with basic algebra. When we get the vectors $w_1, w_2, ... , w_m$, we get the final matrix $\Omega$:
 ```math
\Omega =
\begin{bmatrix}
\_\_\_ &w_1^T & \_\_\_ \\
\_\_\_ & w_2^T & \_\_\_ \\
 & ... & \\
\_\_\_ & w_m^T & \_\_\_
\end{bmatrix}
 ```
 with the shape of $m$ x $k$. Now everytime we want to transform pictures we only need the mean vector $\mu$ and the eigenmatrix $U$ to get the transformed matrix of $\Omega_{new}$ with only $k$ dimensions.

Reference: https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm/

##
In order for the PCA algorithm to work properly, we need to pre-process the input data into a uniform format.

## Image pre-processing
All input images need to:
* contain only one face in a specified *position*
* have the same *size*

### Face positioning
The face which we wish to input into the algorithm can be rotated and positioned anywhere in the original image. 
Sadly, without advanced image reconstruction algorithms we cannot change a profile view into a full face view. The information is simply just not in the image and any attempt at its reconstruction would be only guesswork. As the images we are using seem to have been taken by a professional photographer rather than random images uploaded by each person, the images should contain no profile views. 

Some captured faces, however, are almost in a so called three quarter view (not completely frontal faces). The influence of this fact has been documented in the [results section](#results).
Though we cannot rotate the person post-capture around their vertical axis, we can extract the smaller area containing their face from the image and rotate it, so that the main features of the face are uniformly positioned in all our images.

#### Face and eye detection
Face or eye detection is a common computer vision task. For this purpose, we used the [Haar feature-based cascade classifiers](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html) with cascade files contained in the `OpenCV` library.

#### Face alignment
To deal with the rotation of the face, we use the position of the eyes in the picture to correct the rotation. To align the face, we find the eyes in the face and rotate the image, so that the eyes are on a horizontal line.
First, we locate the centres of the eyes with the use of Haar cascade classifiers. 

![kollareyes3](https://user-images.githubusercontent.com/96919296/210486600-6d4e22d3-c897-4cf5-b6db-f96a69a09dc7.jpg)

Then we find the angle by which we need to rotate the image and then rotate the image around the middle point between the eyes. 

![Untitled1](https://user-images.githubusercontent.com/96919296/210490253-8011d2e7-6ac8-4d20-921b-fc66e173ada5.png)

So far, we have just aligned the face. The only thing left is to extract the face. This is, again, done with the help of a Haar cascade classifier. Here is shown an example of a result image we get after this procedure:

![kollareyesFinal](https://user-images.githubusercontent.com/96919296/210490239-282d08f6-97ec-4fab-9d1b-76318b826810.jpg)

#### Size
Resizing is a standard task. We just needed to select a reasonable size. Based on the images we are using, we chose size 100x100 pixels.

#### Brightness 
Brightness seems to have a great impact on the results. To mitigate this effect, we tried using the equalisation of the histogram technique. In theory, this helps with unifying the brightness of the images as well enhances the image's contrast. Such alteration yealds:

![kollareyesFinal2](https://user-images.githubusercontent.com/96919296/210489796-d224f7b3-c85b-470c-bdae-aeaf1a631e4b.jpg)

Sadly, the overall results did not benefit from this. The various backgrounds and other differences such as lighting influenced the final image. This technique highlited them instead of removing them.

In the end, we deciced not to change the brightness at all.


## 
After we scrape and pre-process our desired image set, we need to perfom dimensionality reduction. PCA reduces the complexity of high dimensional data, while preserving trends and patterns.
PCA projects data into lower dimensions, by maximizing the variance of projected points (all must be uncorrelated). We achieved 95% variance using 75 components, however, the clustering seemed better
when using just the first 15, at least for human perception of the faces.
When comparing our own PCA algorithm with sklearn-imported one, the average face from both is completely identical, however, the feature faces (principal components) seem to have lightning swapped,
but other than that, are identical as well.

##
## Applying the theory (Clustering and Classfication)

### Resemblance 
One ot the questions we investigated was finding people from our image dataset, that resemble famous mathematicians, namely Gauss, Neumann, Turing and Einstein.
First, we performed the dimensionality reduction on our main dataset and transformed mathematician images using our trained PCA.
There was no need to use any clustering algorithm, we simply calculated the euclidean distance between the first 15 decomposed features between our main dataset and the mathematicians and selected the closest images from our dataset.

The restults are as follows:

![Mathematicians](https://user-images.githubusercontent.com/93282067/213715362-900767de-1303-462a-a9dd-b5a3fb0923e7.png)

### Clustering Algorithms and Tuning
The second question we investigated, the main goal of creating a clustering was to find groups of people from the Faculty of Mathematics, Physics and Informatics of the Comenius University that looked the most alike.

We tried 2 unsupervised algorithms, K-Means and Gaussian Mixture. We did not use any supervised methods, since we did not work with any labeled data. We basically had only two parameters for tuning:
* Number of components
* Number of clusters

We can easily determine the optimal number of clusters using either the silhouette score, or elbow method. Silhouette score is easier to use, since it basically just boils down to getting the index of a maximum number from a list of numbers and we had no clear "elbow" in the elbow method.
Elbow method plots the explained variation and we just have to look at a point where diminishing returns are no longer worth the additional cost.
Silhouette method measures how similar object is to its own cluster compared to other clusters (using Euclidean distance).

We tried different numbers of components, ranging from 2 to 75, but the best results seem to be produced when using the first 15, at least from general observation (eye method). Its hard to determine, whether there actually is an optimal amount of components, but some are definitely better than others.

When using both algorithms on a trivial, easily plottable (2D) cases, they often perform very differently. K-Means seems to produce more reasonable clusterings than Gaussian Mixture, at least in our case.

![km vs gm](https://user-images.githubusercontent.com/93282067/213715627-21b6255a-398c-41ae-8635-9086b9654b3f.png)

Its difficult to objectively assess which algorithm is better, they both seem to produce qualitatively almost identical reuslts when clustering our image dataset.

Example of K-Means Cluster:


![K-Means Cluster](https://user-images.githubusercontent.com/93282067/213715220-bb32fb24-ec22-487a-a5a5-d761c56531f1.png)

Example of Gaussian Mixture Cluster:


![Gaussian Mixture Cluster](https://user-images.githubusercontent.com/93282067/213714912-aae55dd6-f350-4141-8ca4-755ce7e6eecc.png)


Both methods were adequate at producing reasonable clusters. 

#### Results and observations
There were several, perhaps unwanted features, namely lightning (too dark/too light), glasses, and rotation of the face that affected the result clustering. 
Faces that were rotated about three quarters to either left or right were often grouped together, even though they did not look very similar to each other. This, however, did not occur very often, so even though the algorithms sometimes clustered based on these features, we still received satisfactory results. There were not any groups that were clustered based solely on their rotation of face. Same goes with glasses and lightning.


