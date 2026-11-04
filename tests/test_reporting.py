import numpy as np
import pandas as pd
import pytest
from validation_interpretation.elemental_profiles.model import elemental_profiles
from exploration_evaluation.spatial import spatial_assessment
from common.core import ELEMENTS


def test_profiles_use_common_median_and_include_noise_separately():
    data=pd.DataFrame({e:[1.,2.,3.,10.] for e in ELEMENTS})
    table=elemental_profiles(data,{'test':np.array([0,0,1,-1])})
    row=table[(table.cluster==0)&(table.element=='Cu')].iloc[0]
    assert row.relative_median==1.5/2.5
    assert row.relative_q25==1.25/2.5
    assert set(table.group_type)=={'Cluster','Noise'}


def test_spatial_excludes_noise_and_does_not_split_cluster_groups():
    pytest.importorskip('geopandas')
    import geopandas as gpd
    from shapely.geometry import box,Point
    basins=gpd.GeoDataFrame(geometry=[box(0,0,3,2),box(3,0,6,2),box(6,0,10,10)],crs=32638,index=['a','b','c'])
    scores=pd.DataFrame({'enrichment_score':[.9,.8,1.],'selected_pc_score':[1.,2.,3.],
        'cn_class':[1,1,2],'Cu':[1.,2.,3.]},index=basins.index)
    table,_=spatial_assessment(scores,{'dbscan_baseline':np.array([0,0,-1])},basins,box(0,0,10,10),[Point(1,1),Point(7,1)])
    selected=table[table.model=='dbscan_baseline'].set_index('nominal_budget_percent')
    assert selected.loc[5.,'capture_percent']==0
    assert selected.loc[10.,'capture_percent']==0 # whole 12% cluster cannot be split
    assert selected.loc[20.,'capture_percent']==50 # noise target cannot be selected


def test_reporting_cli_exports_computed_figures(tmp_path,monkeypatch):
    import sys
    pytest.importorskip('matplotlib')
    from workflow.reporting import main
    rng=np.random.default_rng(41)
    data=pd.DataFrame(np.exp(rng.normal(size=(100,6))),columns=ELEMENTS)
    data['sample_id']=[f'ARTIFICIAL_{i}' for i in range(100)]
    path=tmp_path/'processed.csv';data.to_csv(path,index=False)
    folder=tmp_path/'labels';folder.mkdir()
    for algorithm in ['kmeans','dbscan','optics']:
        for method in ['baseline','mopso','nsga2']:
            pd.DataFrame({'sample_id':data.sample_id,'cluster':np.arange(100)%3}).to_csv(folder/f'{algorithm}_{method}_labels.csv',index=False)
    import hashlib,json
    (folder/'run_manifest.json').write_text(json.dumps({'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    out=tmp_path/'report'
    monkeypatch.setattr(sys,'argv',['reporting','--input',str(path),'--labels-dir',str(folder),'--output',str(out)])
    main()
    assert (out/'elemental_profiles.png').is_file()
    assert (out/'pca_cn_reference.png').is_file()
    assert pd.read_csv(out/'reference_scores.csv').shape[0]==100


def test_spatial_file_loading_and_computed_maps(tmp_path):
    pytest.importorskip('geopandas')
    import geopandas as gpd
    from shapely.geometry import box,Point
    from exploration_evaluation.spatial import read_spatial,plot_spatial
    import json
    basins=gpd.GeoDataFrame({'sid':['a','b','c']},geometry=[box(0,0,3,2),box(3,0,6,2),box(6,0,10,10)],crs=32638)
    boundary=gpd.GeoDataFrame(geometry=[box(0,0,10,10)],crs=32638)
    points=gpd.GeoDataFrame(geometry=[Point(1,1),Point(1,1),Point(7,1)],crs=32638)
    for name,layer in [('basins',basins),('boundary',boundary),('points',points)]:layer.to_file(tmp_path/f'{name}.gpkg',driver='GPKG')
    config={'catchments_path':str(tmp_path/'basins.gpkg'),'catchment_id_field':'sid',
        'study_boundary_path':str(tmp_path/'boundary.gpkg'),'occurrences_path':str(tmp_path/'points.gpkg'),
        'area_crs':'EPSG:32638','occurrence_inventory_preverified':True,'eligibility_description':'Synthetic test inventory only'}
    path=tmp_path/'spatial.json';path.write_text(json.dumps(config))
    linked,domain,occurrences,_=read_spatial(path,pd.Index(['a','b','c']))
    scores=pd.DataFrame({'enrichment_score':[.9,.8,1.],'selected_pc_score':[1.,2.,3.],
        'cn_class':[1,1,2],'Cu':[1.,2.,3.]},index=linked.index)
    models={f'{a}_{m}':np.array([0,0,-1]) for a in ['kmeans','dbscan','optics'] for m in ['baseline','mopso','nsga2']}
    table,detail=spatial_assessment(scores,models,linked,domain,occurrences)
    assert detail['kmeans_baseline']['eligible_occurrences']==2
    plot_spatial(scores,linked,domain,occurrences,table,tmp_path)
    assert (tmp_path/'reference_maps.png').stat().st_size>0
    assert (tmp_path/'occurrence_capture.png').stat().st_size>0
