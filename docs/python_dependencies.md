# Python environment

Python 3.12. Core pinned versions: NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0, threadpoolctl 3.6.0.

| Package | Purpose |
|---|---|
| numpy | Arrays, transforms, distances, quantiles and seeded random generators |
| pandas | Data tables, exact ID joins, profiles and CSV I/O |
| scipy | Stable normal-tail functions, ranks, C-N numerical search, Procrustes |
| scikit-learn | RF/OOB, PCA, K-means/DBSCAN/OPTICS, neighbors, metrics and splits |
| threadpoolctl | Control native numerical threads in model CLI |
| matplotlib 3.10.8 | Scientific figures |
| geopandas 1.2.0 | Read, reproject and link spatial data |
| shapely 2.2.0 | Unions, intersections, area and point containment |
| hdbscan 0.8.44 | DBCV validation |
| pytest 9.2.0 | Automated code verification |
| pyproj / pyogrio | GeoPandas CRS/file-I/O dependencies; runtime versions recorded |

These are the new implementation's tested versions, not presumed historical versions. Standard Python modules require no separate installation. No notebook execution, pymoo or XGBoost is needed. ArcMap/DEM processing is external.

```bash
python -m pip install -e ".[report,spatial,dbcv,test]"
```

Record an environment freeze locally after installation. Actual installed package versions are included in execution manifests, including optional dependencies. Core-only requirements.txt does not install the plotting/spatial/test extras.
