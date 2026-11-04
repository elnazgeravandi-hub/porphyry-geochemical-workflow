import numpy as np
from scipy.stats import rankdata

def enrichment(concentrations):
    n=len(concentrations)
    return np.column_stack([(rankdata(concentrations[e],method='average')-.5)/n for e in ['Cu','Au','Mo']]).mean(axis=1)
