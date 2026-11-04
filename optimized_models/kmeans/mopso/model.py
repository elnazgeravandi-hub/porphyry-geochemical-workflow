"""Independent mopso search branch for kmeans."""
from common.optimizers import mopso

def search(evaluator,initial_population,seed,generations):
    if evaluator.method!='kmeans':raise ValueError('Clustering-method mismatch.')
    return mopso(evaluator,initial_population.copy(),seed,generations=generations)
