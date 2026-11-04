"""Common preprocessing, fitting and validation. No raw-data imputation is implicit."""
import hashlib
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, OPTICS, DBSCAN, cluster_optics_dbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

from common.schema import ELEMENTS


from data_preparation.transforms import prepare


def partition_key(labels):
    mapping={-1:-1}; next_label=0; canonical=[]
    for value in labels:
        value=int(value)
        if value not in mapping:
            mapping[value]=next_label; next_label+=1
        canonical.append(mapping[value])
    return hashlib.sha256(np.array(canonical,dtype='<i4').tobytes()).hexdigest()


def describe(labels):
    assigned=labels>=0
    _, counts=np.unique(labels[assigned],return_counts=True)
    return {'K':len(counts),'minimum_cluster_size':int(counts.min()) if len(counts) else 0,
            'coverage':float(assigned.mean()),'noise_fraction':float((~assigned).mean()),'noise_count':int((~assigned).sum()),
            'assigned_count':int(assigned.sum())}


def violation(info, density):
    k=info['K']; size=info['minimum_cluster_size']; c=info['coverage']
    if k == 0: return float('inf')
    return max(0,(3-k)/3)+max(0,(k-8)/8)+max(0,(5-size)/5)+(max(0,(.95-c)/.95) if density else 0)


def metrics(z, labels, subset=None, complementary=False):
    assigned=labels>=0; indices=np.flatnonzero(assigned)
    if subset is not None: indices=subset[assigned[subset]]
    k=len(np.unique(labels[indices])); result={}
    if not 2 <= k < len(indices): raise ValueError('Invalid silhouette evaluation partition.')
    result['SI']=float(silhouette_score(z[indices],labels[indices],metric='euclidean'))
    if not 2 <= len(np.unique(labels[assigned])) < assigned.sum():
        raise ValueError('Invalid full assigned partition.')
    result['DBI']=float(davies_bouldin_score(z[assigned],labels[assigned]))
    if complementary:
        result['CHI']=float(calinski_harabasz_score(z[assigned],labels[assigned]))
    if not np.isfinite(list(result.values())).all(): raise ValueError('Nonfinite metric.')
    return result


class Evaluator:
    def __init__(self,z,method,k=None,subset_size=600):
        self.z=z; self.method=method; self.k=k
        if method not in {'kmeans','dbscan','optics'}: raise ValueError('Unknown clustering method.')
        self.subset=np.random.default_rng(2026).choice(len(z),min(subset_size,len(z)),replace=False)
        self.cache={}; self.partitions={}; self.orderings={}; self.requests=0; self.fits=0
        self.fit_seconds=0.; self.logs=[]
        if method=='kmeans':
            self.lo=np.tile(z.min(axis=0),k); self.hi=np.tile(z.max(axis=0),k)
        else:
            low_m,high_m=(5,30) if method=='dbscan' else (8,20)
            nn=NearestNeighbors(n_neighbors=high_m,metric='euclidean').fit(z)
            d=nn.kneighbors(z,return_distance=True)[0]  # query explicitly includes self
            low=np.quantile(d[:,low_m-1],.5); high=np.quantile(d[:,high_m-1],.995)
            if not 0 < low < high: raise ValueError('Degenerate radius domain.')
            self.lo=np.array([np.log(low),5 if method=='dbscan' else 0.])
            self.hi=np.array([np.log(high),30 if method=='dbscan' else 4.])
            self.baseline_radius=float(np.quantile(d[:,7],.95))

    def decode(self,theta):
        theta=np.clip(theta,self.lo,self.hi)
        if self.method=='kmeans': return tuple(theta.tolist())
        m=int(np.rint(theta[1]))
        if self.method=='optics': m=[8,10,12,15,20][m]
        return (float(np.exp(theta[0])),m)

    def fit(self,decoded):
        start=time.perf_counter(); self.fits+=1
        if self.method=='kmeans':
            from baseline_models.kmeans.model import refine
            labels=refine(self.z,np.asarray(decoded).reshape(self.k,6)).labels_
        elif self.method=='dbscan':
            from baseline_models.dbscan.model import fit
            labels=fit(self.z,decoded[0],decoded[1])
        else:
            from baseline_models.optics.model import ordering,extract
            radius,m=decoded
            if m not in self.orderings:self.orderings[m]=ordering(self.z,m)
            labels=extract(self.orderings[m],radius)
        self.fit_seconds+=time.perf_counter()-start
        return labels

    def evaluate(self,theta):
        self.requests+=1; decoded=self.decode(theta); cached=decoded in self.cache
        if not cached:
            labels=self.fit(decoded); info=describe(labels); v=violation(info,self.method!='kmeans')
            try:
                scores=metrics(self.z,labels,self.subset)
                f=np.array([-scores['SI'],scores['DBI']])
            except ValueError:
                f=np.array([np.inf,np.inf]); v=np.inf
            rec={'theta':np.array(theta).copy(),'decoded':decoded,'f':f,'v':v,
                 'info':info,'key':partition_key(labels)}
            self.cache[decoded]=rec
            if v==0 and rec['key'] not in self.partitions:
                self.partitions[rec['key']]={'labels':labels.copy(),**rec}
        rec=self.cache[decoded]
        self.logs.append({'request':self.requests,'cached':cached,'violation':rec['v'],
                          'negative_search_SI':rec['f'][0],'DBI':rec['f'][1],
                          'partition':rec['key'],'decoded':decoded})
        return {**rec,'theta':np.array(theta).copy()}
