import numpy as np
from common.schema import ELEMENTS

def plot_profiles(profiles,out):
    import matplotlib.pyplot as plt
    algorithms=['kmeans','dbscan','optics']; methods=['baseline','mopso','nsga2']
    fig,axes=plt.subplots(3,3,figsize=(14,11),sharex=True,sharey=True,layout='constrained')
    for r,algorithm in enumerate(algorithms):
        for c,method in enumerate(methods):
            ax=axes[r,c]; table=profiles[profiles.model==f'{algorithm}_{method}']
            for j,(cluster,group) in enumerate(table.groupby('cluster',sort=True)):
                g=group.set_index('element').loc[ELEMENTS]
                color='gray' if cluster==-1 else f'C{j%10}'
                ax.errorbar(np.arange(6)+(j-(table.cluster.nunique()-1)/2)*.025,g.relative_median,
                    yerr=np.vstack([g.relative_median-g.relative_q25,g.relative_q75-g.relative_median]),
                    marker='s' if cluster==-1 else 'o',linestyle='none',capsize=2,color=color,
                    label='Noise' if cluster==-1 else f'Cluster {cluster}')
            ax.axhline(1,color='black',ls=':',lw=.8); ax.set_yscale('log')
            ax.set_xticks(range(6),ELEMENTS); ax.set_title(f'{algorithm} / {method}')
            if c==0: ax.set_ylabel('Concentration / common median')
            ax.legend(fontsize=7)
    fig.savefig(out/'elemental_profiles.png',dpi=300); fig.savefig(out/'elemental_profiles.pdf'); plt.close(fig)
