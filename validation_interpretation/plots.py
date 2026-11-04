import numpy as np
from common.schema import ELEMENTS

def plot_references(scores,pca,cn,out):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].bar(ELEMENTS,pca['coefficients']);axes[0].axhline(0,color='black',lw=.6)
    axes[0].set_title(f'Selected PC{pca["component"]} coefficients');axes[0].set_ylabel('Oriented coefficient')
    thresholds=cn['distinct_thresholds']; counts=cn['cumulative_counts']; model=cn['selected']; x=np.log10(thresholds)
    design=np.column_stack([np.ones(len(x)),x]+[np.maximum(x-b,0) for b in model['breakpoints_log10']])
    axes[1].loglog(thresholds,counts,'.',ms=2,label='Observed cumulative counts')
    axes[1].loglog(thresholds,10**(design@model['coefficients']),label='Selected continuous fit')
    for threshold in model['thresholds']:axes[1].axvline(threshold,color='gray',ls='--')
    axes[1].set_xlabel('Cu (ppm)');axes[1].set_ylabel('N(>= concentration)');axes[1].legend(fontsize=7)
    fig.savefig(out/'pca_cn_reference.png',dpi=300);fig.savefig(out/'pca_cn_reference.pdf');plt.close(fig)
