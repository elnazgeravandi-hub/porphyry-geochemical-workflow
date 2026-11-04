import numpy as np

def capture(groups,domain,occurrences,budgets=(.05,.10,.20)):
    """Groups: iterable (rank, geometries), highest ranks first; equal ranks merged.

    Inputs must already share one appropriate projected CRS. Domain and eligibility
    are caller-verified. Geometries are clipped and unioned; points counted once.
    """
    from shapely import union_all, GeometryCollection
    if domain.is_empty or not domain.is_valid or domain.area<=0: raise ValueError('Invalid evaluation domain.')
    points={(float(point.x),float(point.y)):point for point in occurrences if domain.covers(point)}
    if not points: raise ValueError('No eligible unique occurrences inside domain.')
    combined={}
    for rank,geometries in groups:
        if not np.isfinite(rank): raise ValueError('Invalid target rank.')
        combined.setdefault(rank,[]).extend(geometries)
    cumulative=GeometryCollection(); steps=[{'area_fraction':0.,'capture_percent':0.}]
    for rank in sorted(combined,reverse=True):
        geoms=combined[rank]
        if any(not g.is_valid for g in geoms): raise ValueError('Resolve invalid target geometries explicitly.')
        cumulative=union_all([cumulative,*geoms]).intersection(domain)
        steps.append({'rank':float(rank),'area_fraction':cumulative.area/domain.area,
            'capture_percent':100*sum(cumulative.covers(p) for p in points.values())/len(points)})
    return {'eligible_occurrences':len(points),'steps':steps,'budgets':[
        {'budget':budget,**max((s for s in steps if s['area_fraction']<=budget),key=lambda s:s['area_fraction'])}
        for budget in budgets]}
