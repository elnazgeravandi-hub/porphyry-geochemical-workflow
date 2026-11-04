import json
from pathlib import Path
import numpy as np
import pandas as pd
from exploration_evaluation.areas import capture

def read_spatial(config_path,index):
    import geopandas as gpd
    from pyproj import CRS
    from shapely import union_all
    config=json.loads(Path(config_path).read_text())
    crs=CRS.from_user_input(config['area_crs'])
    if not crs.is_projected or not crs.axis_info or not all(a.unit_name.lower() in ['metre','meter'] for a in crs.axis_info[:2]):
        raise ValueError('Area CRS must be a documented projected CRS in metres.')
    def load(path):
        layer=gpd.read_file(path)
        if layer.crs is None: raise ValueError('Missing spatial CRS.')
        if layer.empty or layer.geometry.isna().any() or layer.geometry.is_empty.any() or not layer.geometry.is_valid.all():
            raise ValueError('Resolve empty, missing or invalid geometries explicitly.')
        return layer.to_crs(crs)
    basins=load(config['catchments_path']); boundary=load(config['study_boundary_path']); points=load(config['occurrences_path'])
    if not basins.geom_type.isin(['Polygon','MultiPolygon']).all() or not boundary.geom_type.isin(['Polygon','MultiPolygon']).all():
        raise ValueError('Catchments and boundary must be polygons.')
    if not points.geom_type.eq('Point').all():raise ValueError('Occurrences must be point locations.')
    if not config.get('occurrence_inventory_preverified',False) or not str(config.get('eligibility_description','')).strip():
        raise ValueError('Supply a preverified eligible occurrence inventory and document eligibility criteria.')
    id_field=config['catchment_id_field']
    if basins[id_field].isna().any():raise ValueError('Missing catchment sample links.')
    basins[id_field]=basins[id_field].astype(str)
    if not set(index).issubset(set(basins[id_field])):raise ValueError('Some samples lack a catchment link.')
    basins=basins[basins[id_field].isin(index)].copy()
    # Multiple features per sample are dissolved, not duplicated in area/counts.
    basins=basins.dissolve(by=id_field).loc[index]
    roi=union_all(boundary.geometry.to_numpy())
    basins.geometry=basins.geometry.intersection(roi)
    if basins.geometry.is_empty.any() or not basins.geometry.is_valid.all() or (basins.geometry.area<=0).any():raise ValueError('Invalid or empty clipped linked catchment.')
    domain=union_all(basins.geometry.to_numpy())
    return basins,domain,list(points.geometry),config

def spatial_assessment(scores,models,basins,domain,points):
    if not scores.index.equals(basins.index):raise ValueError('Score and catchment ID order differs; align explicitly before evaluation.')
    if any(len(labels)!=len(scores) for labels in models.values()):raise ValueError('Label count differs from spatial sample count.')
    rows=[]; all_results={}
    for name in list(models)+['pca','cn']:
        if name in models:
            labels=models[name]; ranks=pd.Series(scores.enrichment_score.to_numpy(),index=labels).groupby(level=0).median()
            sample_ranks=np.array([ranks[label] if label>=0 else np.nan for label in labels])
        elif name=='pca':sample_ranks=scores.selected_pc_score.to_numpy()
        else:
            ranks=scores.groupby('cn_class').Cu.median()
            sample_ranks=scores.cn_class.map(ranks).to_numpy()
        groups=[]
        for rank in sorted(set(sample_ranks[np.isfinite(sample_ranks)]),reverse=True):
            groups.append((float(rank),list(basins.geometry[sample_ranks==rank])))
        result=capture(groups,domain,points);all_results[name]=result
        for item in result['budgets']:
            rows.append({'model':name,'nominal_budget_percent':100*item['budget'],
                'actual_area_percent':100*item['area_fraction'],'capture_percent':item['capture_percent'],
                'eligible_occurrences':result['eligible_occurrences']})
    return pd.DataFrame(rows),all_results

def plot_spatial(scores,basins,domain,points,captures,out):
    import matplotlib.pyplot as plt
    import geopandas as gpd
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    layer=basins.copy(); layer['PCA']=scores.selected_pc_score.to_numpy();layer['CN']=scores.cn_class.to_numpy()
    for ax,column in zip(axes,['PCA','CN']):
        layer.plot(column=column,ax=ax,legend=True,cmap='viridis',edgecolor='none',alpha=.8)
        gpd.GeoSeries([domain],crs=layer.crs).boundary.plot(ax=ax,color='black',linewidth=.5)
        visible=[p for p in points if domain.covers(p)]
        if visible:gpd.GeoSeries(visible,crs=layer.crs).plot(ax=ax,color='red',markersize=7)
        ax.set_title('Oriented PCA score' if column=='PCA' else 'Cu C-N classes');ax.set_axis_off()
    fig.savefig(out/'reference_maps.png',dpi=300);fig.savefig(out/'reference_maps.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True,layout='constrained')
    for ax,algorithm in zip(axes,['kmeans','dbscan','optics']):
        for method in ['baseline','mopso','nsga2']:
            table=captures[captures.model==f'{algorithm}_{method}'].sort_values('nominal_budget_percent')
            ax.plot(table.nominal_budget_percent,table.capture_percent,marker='o',label=method)
        ax.set_title(algorithm);ax.set_xlabel('Nominal target-area budget (%)');ax.set_xticks([5,10,20]);ax.legend(fontsize=8)
    axes[0].set_ylabel('Eligible occurrence capture (%)');axes[0].set_ylim(0,100)
    fig.savefig(out/'occurrence_capture.png',dpi=300);fig.savefig(out/'occurrence_capture.pdf');plt.close(fig)
