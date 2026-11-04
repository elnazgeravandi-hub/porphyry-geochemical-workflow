"""K-means fitting shared by its baseline and centroid-search candidates."""
import numpy as np
from sklearn.cluster import KMeans,kmeans_plusplus

def refine(z,initial):
    initial=np.asarray(initial).reshape(-1,6)
    return KMeans(n_clusters=len(initial),init=initial,n_init=1,max_iter=300,
                  tol=1e-4,algorithm='lloyd',random_state=42).fit(z)

def baseline(z,k,seed,starts=30):
    rng=np.random.RandomState(seed);best=None;selected_initial=None
    for _ in range(starts):
        initial=kmeans_plusplus(z,n_clusters=k,random_state=rng)[0]
        fitted=refine(z,initial)
        if best is None or fitted.inertia_<best.inertia_:
            best=fitted;selected_initial=initial.copy()
    return best.labels_,tuple(selected_initial.ravel())
