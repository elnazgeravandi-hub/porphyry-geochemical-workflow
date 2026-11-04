"""OPTICS ordering and fixed DBSCAN-style extraction baseline."""
import numpy as np
from sklearn.cluster import OPTICS,cluster_optics_dbscan
from sklearn.neighbors import NearestNeighbors

def ordering(z,minimum):
    model=OPTICS(min_samples=minimum,max_eps=np.inf,metric='euclidean',n_jobs=1).fit(z)
    return model.reachability_,model.core_distances_,model.ordering_

def extract(graph,radius):
    reach,core,order=graph
    return cluster_optics_dbscan(reachability=reach,core_distances=core,ordering=order,eps=radius)

def baseline(z):
    distances=NearestNeighbors(n_neighbors=8,metric='euclidean').fit(z).kneighbors(z)[0]
    radius=float(np.quantile(distances[:,7],.95))
    return extract(ordering(z,8),radius),(radius,8)
