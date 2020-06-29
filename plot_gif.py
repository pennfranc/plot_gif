import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import imageio
from sklearn.decomposition import PCA

COLORS = ['red', 'blue', 'yellow', 'green', 'orange', 'pink', 'grey']

def gif_creation(data, classes=None, name=None, fps=25, steps=180):
    """
    This function takes a matrix `data` where the rows represent individual data points
    that are at least 3-dimensional and creates a gif of a rotating scatter plot of
    these (dimensionally reduced) data points. Optionally, a vector of classes can be passed
    to color the data points differently.

    Parameters
    ----------
    data
        A matrix with shape (number of data points, dimensionality) containing the data points to be plotted .
    classes
        Optionally, a vector with length 'number of data points' containing integer class labels.
    fps
        The number of of frames displayed per second in the gif. Given a number of steps, this determines the speed
        of rotation.
    steps
        The number of frames that the gif will consist of. This also corresponds to the granularity of the rotation.
        A higher number of steps will result in much smaller rotations between frames.
    """

    # handle case of less than 3 dimensions
    if (data.shape[1] < 3):
        raise ValueError("Data needs to be at least 3 dimensional")

    # reduce dimensions if necessary
    print("Performing dimensionality reduction...")
    if (data.shape[1] > 3):
        pca = PCA(n_components=3)
        data = pd.DataFrame(pca.fit_transform(data))

    # handle case of no classes
    if (classes is None):
        classes = np.zeros(data.shape[0])

    # get number of classes
    num_classes = len(np.unique(classes))

    # create plot
    data = pd.DataFrame(data)
    data['y'] = classes

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for class_nr in range(-1, num_classes):
        color_nr = class_nr % len(COLORS)
        if (class_nr == -1): color = 'black'
        else: color = COLORS[color_nr]
        ax.scatter(data[data.y==class_nr][0], data[data.y==class_nr][1], 
        data[data.y==class_nr][2], label=('class ' + str(class_nr)), c=color)
    
    plt.title(name)
    plt.axis('off')

    # create gif
    tmp_dir = 'gif_tmp_dir'
    os.mkdir(tmp_dir)
    images = []
    print("Creating individual images...")
    for i in tqdm(range(0, steps)):
        ax.view_init(30, i * (360 / steps))
        name = tmp_dir + '/' + str(i) + '.png'
        fig.savefig(name)
        images.append(imageio.imread(name))

    print("Creating gif...")
    imageio.mimsave('creation.gif', images, fps=fps)
    shutil.rmtree(tmp_dir)







