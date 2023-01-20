import numpy as np
import matplotlib.pyplot as plt
import random

# Takes 1d array of images and reshapes them to w x h matrix
def reconstruct_images(images, w, h):
    return [np.reshape(images[i], (w,h)) for i in range(len(images))]

# Gets the closest point of a vector from matrix using euclidean distance
def get_closest_point(matrix, vector):
    min_dist = None
    idx = None
    for i, v in enumerate(matrix):
        dist = np.linalg.norm(v - vector)
        if not min_dist or dist < min_dist:
            min_dist = dist
            idx = i

    return idx

# Plots images, row_size is number of images in one row, column_size is number of images in one column
def plot_images(images, row_size, column_size, fig_size = (8, 6)):
    fig = plt.figure(figsize=fig_size)
    for i in range(row_size * column_size):
        try:
            ax = fig.add_subplot(row_size, column_size, i + 1, xticks=[], yticks=[])
            ax.imshow(images[i], cmap=plt.cm.bone)
        except IndexError:
            return

# Creates color list from labels
def create_colors(labels, no_of_groups):
    colors = [(random.randint(0, 100) / 100, random.randint(0, 100) / 100, random.randint(0, 100) / 100) for i in range(no_of_groups)]
    color_list = []
    
    for i in range(len(labels)):
        color_list.append(colors[labels[i]])
    return color_list



def get_difference_image(image_1, image_2):
    return image_1 - image_2
    