import json
from pathlib import Path
import numpy as np

def serial(value):
    if isinstance(value,np.ndarray): return serial(value.tolist())
    if isinstance(value,np.generic): return serial(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    if isinstance(value,dict): return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serial(v) for v in value]
    return value

def save(path,value):
    Path(path).write_text(json.dumps(serial(value),indent=2,allow_nan=False),encoding='utf-8')

import pandas as pd

def read_indexed(path):
    data=pd.read_csv(path,dtype={'sample_id':str},float_precision='round_trip')
    if 'sample_id' not in data or data.sample_id.isna().any() or data.sample_id.str.strip().eq('').any() or data.sample_id.duplicated().any():
        raise ValueError('Unique nonempty sample_id column required.')
    return data.set_index('sample_id')

def load_labels(directory,index):
    models={}
    for algorithm in ['kmeans','dbscan','optics']:
        for method in ['baseline','mopso','nsga2']:
            name=f'{algorithm}_{method}'; path=Path(directory)/f'{name}_labels.csv'
            if not path.is_file(): raise FileNotFoundError(f'Required model labels missing: {path.name}')
            data=read_indexed(path)
            if set(data.index)!=set(index): raise ValueError(f'Label IDs do not match concentrations: {name}')
            values=pd.to_numeric(data.loc[index,'cluster'],errors='raise').to_numpy(float)
            if not np.isfinite(values).all() or not (values==np.floor(values)).all() or (values< -1).any():
                raise ValueError(f'Invalid cluster labels: {name}')
            models[name]=values.astype(int)
    return models
