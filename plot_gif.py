import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from sklearn.decomposition import PCA



    

def gif_creation(data, classes, num_classes, name = None):

    # handle case of less than 3 dimensions
    if (data.shape[1] < 3):
        print("Data has less than 3 dimensions, aborting")
        return

    # reduce dimensions if necessary
    if (data.shape[1] > 3):
        pca = PCA(n_components=3)
        data = pd.DataFrame(pca.fit_transform(data))

    # handle case of no classes
    if (classes is None):
        classes = np.zeros(data.shape[0])

    # create plot
    COLORS = ['red', 'blue', 'yellow', 'green', 'orange', 'pink', 'grey']
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

    # create gif
    tmp_dir = 'gif_tmp_dir'
    os.mkdir(tmp_dir)
    images = []
    for i in range(0, 72):
        ax.view_init(30, i * 5)
        name = tmp_dir + '/' + str(i) + '.png'
        fig.savefig(name)
        images.append(imageio.imread(name))

    imageio.mimsave('creation.gif', images)
    shutil.rmtree(tmp_dir)







