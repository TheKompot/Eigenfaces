# Eigenfaces
Semester project for class **Principles of Data Science**

## Dimension reduction
 When we convert our images to vectors and put them together, we get a matrix $X$ with the shape $m$ x $N^2$ (number_of_pictures x width_of_picture^2). The variable $N^2$ can be easily in the tens of thousands, which would be ineffective and troublesome to compute.

 To make an effective face recognition algorithm, we want to reduce the dimension of our dataset. We will use the $k$ largest eigenvectors (meaning eigenvectors with the largest eigenvalues) from our datasets covariance matrix to encode our dataset in to a matrix $\Omega$ with the shape of $m$ x $k$.

 First we center our dataset by subtracking the mean:
 $a_i =  x_i - \mu$ , where  
 ```math
 \mu = \frac{1}{m} \sum_{i=1}^m x_i $
 ```
 From that we get a new matrix $A$: 
```math
A = \begin{bmatrix}a_1 & a_2 & ... & a_m\end{bmatrix}
```
 with the shape of $N^2$ x $m$. The covariance matrix of $A$ is $AA^T$, which has $N^2$ eigenvectors of size $N^2$. To reduce the computation power needed, we will compute the eigenvectors of the smaller covariance matrix $A^TA$, which has only $m$ eigenvectors of size $m$ and then convert them in to eigenvectors of matrix $AA^T$, using:
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

 Now we compute the eigenvectors of $A^TA$, take those with the $k$ largest eigenvalues, convert them in to the eigenvectors of $AA^T$ and create a matrix $U$:
  ```math
 U = \begin{bmatrix}
    u_1 & u_2 & ... & u_k
 \end{bmatrix}
 ```
 which has the shape of $N^2$ x $k$.

 To get our matrix $\Omega$ we need to calculate the coeficients $w_1$ ... $w_k$ of the linear combination of eigenvectors that comprise vector $a_i$. To do that we compute with the method of least squares the system of equations:
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

 For $i = {1,2 ... , m}$. We will not show you how to solve these equations, to stop boring you with basic algebra. When we get the vectors $w_1, w_2, ... , w_m$ we get the final matrix $\Omega$:
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

This algorithm is broadly known as **PCA**(Principle Component Analyses).
Reference: https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm/

## Data
*   Data for testing PCA and ML algorithms acquired from: https://github.com/microsoft/DigiFace1M
## Clustering and Classfication
talk about:
*   in which process of clustering and classfication do we use PCA
* what supervised and unsupervised algorithms did we try
* compare their stats 
