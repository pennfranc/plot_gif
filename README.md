-----------
Description
-----------

Takes matrix of datapoints (more than 2 dimesnional in column space) and creates
a rotating gif showing the 3d plot. If classes are provided, datapoints have the same color
if they belong to the same class. Datapoints with class -1 are colored black.

-------------
Dependencies
-------------

pandas, numpy, matplotlib, mpl_toolkits, sklearn, imageio

------
Usage
------

import plot_gif as pg
pg.gif_creation(data, classes, num_classes, name)

-----------
Parameters
-----------

data: datapoints to be plotted
classes: integer values corresponding to cluster/class of each datapoint
num_classes: total number of distinct classes
name: title of plot (optional)

-------------
Preconditions
-------------

classes != None => data.shape[0] == len(classes)