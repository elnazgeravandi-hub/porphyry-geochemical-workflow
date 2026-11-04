"""Fixed DBSCAN baseline and consistent candidate fitting."""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def fit(z,radius,minimum):
    return DBSCAN(eps=radius,min_samples=minimum,metric='euclidean').fit_predict(z)

def baseline(z):
    distances=NearestNeighbors(n_neighbors=8,metric='euclidean').fit(z).kneighbors(z)[0]
    radius=float(np.quantile(distances[:,7],.95))
    return fit(z,radius,8),(radius,8)
