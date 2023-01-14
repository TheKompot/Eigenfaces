# Eigenfaces
Semester project for class **Principles of Data Science**

## Overview
This project is focused on using Principal Component Analysis (PCA) on photos. We are comparing build-in PCA algorithm in Python library scikit.learn with our own implementation. Using PCA algorithm and three different clustering methods we are trying to find groups of teachers at our faculty who look alike. We are also curious about which of our teachers look alike famous mathematicians (e.g. Gauss, Neumann, Turing, Einstein,...). 

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

This algorithm is broadly known as **PCA**(Principle Component Analysis).
Reference: https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm/

## Data
*   Data for testing PCA and ML algorithms acquired from: https://github.com/microsoft/DigiFace1M
*   Faces for analysis and classification were scraped from personal websites from individual departments here: https://fmph.uniba.sk/pracoviska/
*   Web scraping was carried out using Python libraries requests (https://requests.readthedocs.io/en/latest/) and beautifulsoup4 (https://pypi.org/project/beautifulsoup4/)
## Clustering and Classfication
talk about:
*   in which process of clustering and classfication do we use PCA
* what supervised and unsupervised algorithms did we try
* compare their stats 

## Image pre-processing
In order for the PCA algorithm to work properly, we need to pre-process the input data into a uniform format. All input images need to:
* contain only one face in a specified *position*
* have the same *size*
* have their *brightness* adjusted

### Face positioning
The face which we wish to input into the algorithm can be rotated and positioned anywhere in the original image. 
Sadly, without advanced image reconstruction algorithms we cannot change a profile view into a full face view. The information is simply just not in the image and any attempt at its reconstruction would be only guesswork. As the images we are using seem to have been taken by a professional photographer rather than random images uploaded by each person, the images should contain no profile views. 
- [ ] ❓talk about three quarter view in the results (not completely frontal faces), if they have been grouped together

Some captured faces, however, are almost in a so called three quarter view. The influence of this fact has been documented in the results section. **❗TODO reference**
Though we cannot rotate the person post-capture around their vertical axis, we can extract the smaller area containing their face from the image and rotate it, so that the main features of the face are uniformly positioned in all our images.
#### Face and eye detection
Face or eye detection is a common computer vision task. For this purpose, we used the [Haar feature-based cascade classifiers](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html) with cascade files contained in the OpenCV library.
#### Face alignment
To deal with the rotation of the face, we use the position of the eyes in the picture to correct the rotation. To align the face, we find the eyes in the face and rotate the image, so that the eyes are on a horizontal line.
First, we locate the centres of the eyes with the use of Haar cascade classifiers. 

![kollareyes3](https://user-images.githubusercontent.com/96919296/210486600-6d4e22d3-c897-4cf5-b6db-f96a69a09dc7.jpg)

Then we find the angle by which we need to rotate the image and then rotate the image around the middle point between the eyes. 

![Untitled1](https://user-images.githubusercontent.com/96919296/210490253-8011d2e7-6ac8-4d20-921b-fc66e173ada5.png)

So far, we have just aligned the face. The only thing left is to extract the face. This is, again, done with the help of a Haar cascade classifier. Here is shown an example of a result image we get after this procedure:

![kollareyesFinal](https://user-images.githubusercontent.com/96919296/210490239-282d08f6-97ec-4fab-9d1b-76318b826810.jpg)

### Size and brightness
These two parameters are quite easy to unify. Resizing is a standard task. We just needed to select a reasonable size. Based on the images we are using, we chose size 100x100 pixels.
As for brightness, we decided to use equalisation of the histogram technique. This helps with unifying the brightness of the images, as well enhances the images contrast. Such alteration yealds:

![kollareyesFinal2](https://user-images.githubusercontent.com/96919296/210489796-d224f7b3-c85b-470c-bdae-aeaf1a631e4b.jpg)

(this image is 200x200 for viewing purposes)
