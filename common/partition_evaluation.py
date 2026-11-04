"""Partition agreement and whole-group cumulative target evaluation."""
import numpy as np
from sklearn.metrics import adjusted_rand_score
from common.core import describe


def agreement(a,b):
    a=np.asarray(a); b=np.asarray(b)
    if a.shape!=b.shape: raise ValueError('Align sample IDs before comparing labels.')
    common=(a>=0)&(b>=0)
    def ari(mask):
        if len(np.unique(a[mask]))<2 or len(np.unique(b[mask]))<2: return None
        return float(adjusted_rand_score(a[mask],b[mask]))
    return {'ARI_all':ari(np.ones(len(a),dtype=bool)),'ARI_common_assigned':ari(common),
            'common_assigned_count':int(common.sum()),'common_assigned_fraction':float(common.mean())}


def sampling_stability(z,full_labels,refit):
    """refit(X_subset) must preserve full-data preprocessing and selected hyperparameters."""
    rows=[]
    for seed in range(42,62):
        indices=np.sort(np.random.default_rng(seed).choice(len(z),int(.8*len(z)),replace=False))
        labels=refit(z[indices])
        rows.append({'seed':seed,**agreement(np.asarray(full_labels)[indices],labels),**describe(labels)})
    return rows


from exploration_evaluation.areas import capture
