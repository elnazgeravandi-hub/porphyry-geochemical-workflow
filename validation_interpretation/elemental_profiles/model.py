import numpy as np
import pandas as pd
from common.schema import ELEMENTS
from data_preparation.run import UNITS

def elemental_profiles(concentrations,models):
    common=concentrations[ELEMENTS].median(); rows=[]
    for name,labels in models.items():
        for cluster in sorted(np.unique(labels)):
            part=concentrations.loc[labels==cluster,ELEMENTS]
            for element in ELEMENTS:
                values=part[element]; q25,median,q75=values.quantile([.25,.5,.75])
                rows.append({'model':name,'cluster':int(cluster),'group_type':'Noise' if cluster==-1 else 'Cluster',
                    'n':len(part),'element':element,'unit':UNITS[element],'q25':q25,'median':median,'q75':q75,
                    'common_median':common[element],'relative_q25':q25/common[element],
                    'relative_median':median/common[element],'relative_q75':q75/common[element]})
    return pd.DataFrame(rows)
