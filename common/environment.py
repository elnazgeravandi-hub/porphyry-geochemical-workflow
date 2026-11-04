"""Record actual installed library versions, including optional packages."""
import importlib.metadata
import platform
import sys


def environment_manifest():
    versions={}
    for name in ['numpy','pandas','scipy','scikit-learn','threadpoolctl','matplotlib','geopandas','shapely','pyproj','pyogrio','hdbscan','pytest']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'python':sys.version,'platform':platform.platform(),'packages':versions}
