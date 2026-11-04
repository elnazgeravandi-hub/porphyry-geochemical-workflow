"""Independent mopso search branch for dbscan."""
from common.optimizers import mopso

def search(evaluator,initial_population,seed,generations):
    if evaluator.method!='dbscan':raise ValueError('Clustering-method mismatch.')
    return mopso(evaluator,initial_population.copy(),seed,generations=generations)
