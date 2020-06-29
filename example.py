from sklearn import datasets
import plot_gif as pg

X1, y1 = datasets.make_blobs(n_samples=1000, centers=4, n_features=4)
pg.gif_creation(X1, y1, fps=25, steps=180)