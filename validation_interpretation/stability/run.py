"""Manuscript 20 x 80% refit assessment with fixed full-input transformation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN,OPTICS,cluster_optics_dbscan
from common.io import read_indexed,load_labels
from common.integrity import verify_model_input
from common.core import ELEMENTS
from common.partition_evaluation import sampling_stability


def make_refit(method,selected):
    decoded=selected['decoded']
    if method=='kmeans':
        initial=np.asarray(decoded).reshape(-1,6)
        return lambda x:KMeans(n_clusters=len(initial),init=initial,n_init=1,max_iter=300,tol=1e-4,algorithm='lloyd',random_state=42).fit_predict(x)
    radius,neighborhood=decoded
    if method=='dbscan':return lambda x:DBSCAN(eps=radius,min_samples=neighborhood,metric='euclidean').fit_predict(x)
    def optics(x):
        ordering=OPTICS(min_samples=neighborhood,max_eps=np.inf,metric='euclidean',n_jobs=1).fit(x)
        return cluster_optics_dbscan(reachability=ordering.reachability_,core_distances=ordering.core_distances_,ordering=ordering.ordering_,eps=radius)
    return optics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True)
    parser.add_argument('--models',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    verify_model_input(args.input,args.models)
    raw=read_indexed(args.input)[ELEMENTS]
    models=Path(args.models);parameters=json.loads((models/'preprocessing.json').read_text())
    if parameters['elements']!=ELEMENTS:raise ValueError('Feature-order mismatch.')
    x=raw.to_numpy(float).copy();x[:,1]/=1000
    if not np.isfinite(x).all() or (x<=0).any():raise ValueError('Need positive processed concentrations.')
    logs=np.log(x);clr=logs-logs.mean(axis=1,keepdims=True)
    z=(clr-np.asarray(parameters['median']))/np.asarray(parameters['iqr'])
    labels=load_labels(models,raw.index);rows=[]
    for name,full_labels in labels.items():
        selected=json.loads((models/f'{name}_selected.json').read_text())
        method=name.split('_')[0]
        for row in sampling_stability(z,full_labels,make_refit(method,selected)):
            rows.append({'model':name,**row})
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    table=pd.DataFrame(rows);table.to_csv(out/'stability_runs.csv',index=False)
    metrics=['ARI_all','ARI_common_assigned','common_assigned_fraction','K','minimum_cluster_size','coverage']
    summary=table.groupby('model')[metrics].agg(['median',lambda x:x.quantile(.25),lambda x:x.quantile(.75)])
    summary.to_csv(out/'stability_summary.csv')

if __name__=='__main__':main()
