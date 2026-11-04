import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from common.partition_evaluation import sampling_stability
from validation_interpretation.stability.run import make_refit


def test_twenty_fixed_parameter_subsamples():
    x,labels=make_blobs(n_samples=150,n_features=6,centers=3,cluster_std=.1,random_state=12)
    model=KMeans(n_clusters=3,n_init=5,random_state=42).fit(x)
    refit=make_refit('kmeans',{'decoded':model.cluster_centers_.ravel().tolist()})
    rows=sampling_stability(x,model.labels_,refit)
    assert len(rows)==20 and rows[0]['seed']==42 and rows[-1]['seed']==61
    assert all(r['ARI_all']==1 for r in rows)
    assert all(r['assigned_count']==120 for r in rows)
