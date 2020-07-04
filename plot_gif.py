import os
import shutil
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import imageio
from sklearn.decomposition import PCA
from sklearn.base import ClusterMixin
from sklearn import cluster

COLORS = ['red', 'blue', 'yellow', 'green', 'orange', 'pink', 'grey']

def gif_creation(
    data: np.ndarray,
    classes: Optional[np.ndarray] = None,
    name: Optional[str] = None,
    fps: int = 25,
    steps: int = 180,
    gif_path: str = 'creation.gif',
    cluster_estimator: Optional[str] = None,
    **clustering_kwargs
):
    """
    This function takes a matrix `data` where the rows represent individual data points
    that are at least 3-dimensional and creates a gif of a rotating scatter plot of
    these (dimensionally reduced) data points. Optionally, a vector of classes can be passed
    to color the data points differently.

    Optionally, if no classes are passed and if a clustering algorithm is provided,
    class labels of the given data will be computed.

    Note that during this process, a temporary directory called `gif_tmp_dir` is created and
    subsequently deleted.

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
    gif_path
        A string representing the file path of the gif to be created.
    cluster_estimator
        A string value representing an sklearn.cluster clustering algorithm to be used to determine the classes,
        if no explicit `classes` argument is provided.
    clustering_kwargs
        Keyword arguments to be passed on to the clustering estimator class.
    """

    # check that the dimensionality of the data is high enough
    if (data.shape[1] < 3):
        raise ValueError("Data needs to be at least 3 dimensional")

    # reduce dimensions if necessary
    print("Performing dimensionality reduction...")
    if (data.shape[1] > 3):
        pca = PCA(n_components=3)
        data = pd.DataFrame(pca.fit_transform(data))

    # put default classes or compute clustering if no explicit classes are provided
    if (classes is None and cluster_estimator is None):
        classes = np.zeros(data.shape[0])
    elif (classes is None):
        print("Computing clustering...")
        classes = _compute_clustering(data, cluster_estimator, **clustering_kwargs)

    # get number of classes
    num_classes = len(np.unique(classes))

    # create plot
    fig, ax = _create_figure(data, classes, num_classes, name)

    # create gif
    _create_gif(ax, fig, fps, steps, gif_path)


def _compute_clustering(data, cluster_estimator, **clustering_kwargs):
    if not hasattr(cluster, cluster_estimator):
        raise ValueError("Provided 'cluster_estimator' {} does not represent a 'sklearn.cluster' attribute.")
    cluster_model_class = getattr(cluster, cluster_estimator)
    cluster_model_instance = cluster_model_class(**clustering_kwargs)
    if not isinstance(cluster_model_instance, ClusterMixin):
        raise ValueError("'cluster_estimator' must correspond to a 'sklearn.base.ClusterMixin' class.")
    return cluster_model_instance.fit_predict(data)


def _create_figure(data, classes, num_classes, name):
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
    return fig, ax


def _create_gif(ax, fig, fps, steps, gif_path):
    tmp_dir = 'gif_tmp_dir'
    os.mkdir(tmp_dir)
    try:
        images = []
        print("Creating individual images...")
        for i in tqdm(range(0, steps)):
            ax.view_init(30, i * (360 / steps))
            name = tmp_dir + '/' + str(i) + '.png'
            fig.savefig(name)
            images.append(imageio.imread(name))
        print("Creating gif...")
        imageio.mimsave(gif_path, images, fps=fps)
    finally:
        shutil.rmtree(tmp_dir)
