"""Independent nsga_ii search branch for optics."""
from common.optimizers import nsga2

def search(evaluator,initial_population,seed,generations):
    if evaluator.method!='optics':raise ValueError('Clustering-method mismatch.')
    return nsga2(evaluator,initial_population.copy(),seed,generations=generations)
