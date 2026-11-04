import numpy as np
from common.core import metrics
from common.optimizers import objective_front

def choose(pool,bounds):
    feasible=[p for p in pool if p['v']==0]
    if not feasible: return None
    nondominated=[feasible[i] for i in objective_front(feasible)]
    lo,hi=bounds; width=hi-lo; active=width>0
    def key(p):
        dist=float(np.sum(((p['f'][active]-lo[active])/width[active])**2))
        return (dist,-p['info']['coverage'],-p['info']['minimum_cluster_size'],tuple(p['decoded']))
    winner=min(nondominated,key=key).copy(); winner['distance_squared']=key(winner)[0]
    return winner

def report_metrics(z,labels,method,include_dbcv):
    try:
        score=metrics(z,labels,complementary=True)
    except ValueError as error:
        score={'SI':None,'DBI':None,'CHI':None,'metric_status':str(error)}
    if include_dbcv and method!='kmeans':
        from hdbscan.validity import validity_index
        assigned=labels>=0
        _,relabeled=np.unique(labels[assigned],return_inverse=True)
        if len(np.unique(relabeled))<2:
            score.update(DBCV=None,DBCV_status='undefined: fewer than two assigned clusters')
        else:
            try:
                value=float(validity_index(z[assigned].astype(np.float64),relabeled,metric='euclidean'))
                score.update(DBCV=value if np.isfinite(value) else None,
                    DBCV_status='valid' if np.isfinite(value) else 'nonfinite numerical result')
            except (ValueError,ZeroDivisionError,FloatingPointError) as error:
                score.update(DBCV=None,DBCV_status=str(error))
    return score
