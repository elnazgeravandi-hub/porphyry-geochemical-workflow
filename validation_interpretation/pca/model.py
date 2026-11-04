import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

def pca_reference(z,enrichment_score):
    model=PCA(svd_solver='full').fit(z); scores=model.transform(z)
    candidates=[]
    for j,var in enumerate(model.explained_variance_):
        if var<=np.finfo(float).eps*max(model.explained_variance_[0],1): continue
        correlation=float(spearmanr(scores[:,j],enrichment_score).statistic)
        if np.isfinite(correlation): candidates.append((abs(correlation),model.explained_variance_ratio_[j],-j,correlation))
    if not candidates: raise ValueError('No informative PCA component.')
    _,_,negative_index,correlation=max(candidates); index=-negative_index
    sign=1 if correlation>=0 else -1
    return {'component':index+1,'scores':sign*scores[:,index], 'coefficients':sign*model.components_[index],
            'explained_variance_ratio':model.explained_variance_ratio_[index], 'spearman':abs(correlation)}
