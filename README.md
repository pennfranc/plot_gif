# plot_gif

## Example

```
from sklearn import datasets
import plot_gif as pg

X1, y1 = datasets.make_blobs(n_samples=1000, centers=4, n_features=4)
pg.gif_creation(X1, y1, fps=25, steps=180)
```
Result:
<div style="text-align:center;">
<img src="https://github.com/pennfranc/plot_gif/blob/master/creation.gif" alt="plot gif creation example" />
</div>

## Description

Takes matrix of datapoints (more than 2 dimensional in column space) and creates
a rotating gif showing a 3D scatter plot of a PCA projection of the data.
If classes are provided, datapoints have the same color if they belong to the same class.
Datapoints with class -1 are colored black.

## Dependencies

- pandas
- numpy
- matplotlib
- mpl_toolkits
- imageio
- tqdm
- sklearn
