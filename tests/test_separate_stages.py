import hashlib
import importlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from workflow.jobs import read_bundle
from workflow.run import run
from model_comparison.run import compare


def inputs(tmp_path):
    z,_=make_blobs(n_samples=120,n_features=6,centers=3,cluster_std=.1,random_state=4)
    frame=pd.DataFrame(np.exp(z/8),columns=['Cu','Au','Mo','Bi','Pb','Zn'])
    frame['sample_id']=[f'ARTIFICIAL_{i}' for i in range(len(frame))]
    path=tmp_path/'input.csv';frame.to_csv(path,index=False)
    config={'units':{'Cu':'ppm','Au':'ppb','Mo':'ppm','Bi':'ppm','Pb':'ppm','Zn':'ppm'},
        'methods':['kmeans','dbscan','optics'],'k_values':[3],'seeds':[42],
        'population':4,'generations':1,'silhouette_subset_size':100,'baseline_n_init':3,
        'expected_samples':120,'dbcv':False}
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config))
    return path,cfg,frame


@pytest.mark.parametrize('method',['kmeans','dbscan','optics'])
@pytest.mark.parametrize('solver',['baseline','mopso','nsga_ii'])
def test_each_branch_has_its_own_runnable_module(tmp_path,monkeypatch,method,solver):
    path,cfg,_=inputs(tmp_path);out=tmp_path/'branch'
    if solver=='baseline':module=importlib.import_module(f'baseline_models.{method}.run')
    else:module=importlib.import_module(f'optimized_models.{method}.{solver}.run')
    monkeypatch.setattr(sys,'argv',['run','--input',str(path),'--config',str(cfg),'--output',str(out)])
    # Entrypoints delegate only the branch's identity, never fit the other methods.
    configuration='nsga2' if solver=='nsga_ii' else solver
    module.branch_main(method,configuration)
    manifest,records=read_bundle(out)
    assert manifest['method']==method and manifest['configuration']==configuration
    assert (out/'candidate_labels.npz').is_file()
    if solver!='baseline':
        audits=list(out.glob('k*_seed*_audit.json'))
        assert audits and all(json.loads(p.read_text())['requests']==8 for p in audits)


def test_joint_comparison_rejects_unequal_optimizer_starts(tmp_path):
    path,cfg,_=inputs(tmp_path);root=tmp_path/'models';run(path,cfg,root)
    audit=root/'optimized_models'/'kmeans'/'nsga_ii'/'k3_seed42_audit.json'
    data=json.loads(audit.read_text());data['initial_population'][0][0]+=.1;audit.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='initial_population'):
        compare(path,cfg,root,tmp_path/'recompare')


def test_reference_stages_and_exploration_run_separately(tmp_path,monkeypatch):
    pytest.importorskip('geopandas')
    import geopandas as gpd
    from shapely.geometry import box,Point
    path,cfg,frame=inputs(tmp_path);models=tmp_path/'model_comparison';models.mkdir()
    (models/'run_manifest.json').write_text(json.dumps({'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    for algorithm in ['kmeans','dbscan','optics']:
        for method in ['baseline','mopso','nsga2']:
            pd.DataFrame({'sample_id':frame.sample_id,'cluster':np.arange(len(frame))%3}).to_csv(models/f'{algorithm}_{method}_labels.csv',index=False)
    for stage in ['elemental_profiles','pca','cn']:
        module=importlib.import_module(f'validation_interpretation.{stage}.run')
        monkeypatch.setattr(sys,'argv',['run','--input',str(path),'--models',str(models),'--output',str(tmp_path/stage)])
        module.main()
    basins=gpd.GeoDataFrame({'sid':frame.sample_id},geometry=[box(i,0,i+1,1) for i in range(len(frame))],crs=32638)
    boundary=gpd.GeoDataFrame(geometry=[box(0,0,len(frame),1)],crs=32638)
    points=gpd.GeoDataFrame(geometry=[Point(.5,.5),Point(30.5,.5),Point(100.5,.5)],crs=32638)
    for name,layer in [('basins',basins),('boundary',boundary),('points',points)]:layer.to_file(tmp_path/f'{name}.gpkg',driver='GPKG')
    spec={'catchments_path':str(tmp_path/'basins.gpkg'),'catchment_id_field':'sid',
        'study_boundary_path':str(tmp_path/'boundary.gpkg'),'occurrences_path':str(tmp_path/'points.gpkg'),
        'area_crs':'EPSG:32638','occurrence_inventory_preverified':True,'eligibility_description':'Artificial fixture only.'}
    spatial=tmp_path/'spatial.json';spatial.write_text(json.dumps(spec))
    from exploration_evaluation.run import main
    monkeypatch.setattr(sys,'argv',['run','--input',str(path),'--models',str(models),
        '--pca-results',str(tmp_path/'pca'),'--cn-results',str(tmp_path/'cn'),
        '--spatial-config',str(spatial),'--output',str(tmp_path/'exploration')])
    main()
    result=pd.read_csv(tmp_path/'exploration'/'occurrence_capture.csv')
    assert len(result)==33 # nine clustering configurations and two references, three budgets
    assert (result.actual_area_percent<=result.nominal_budget_percent).all()
    assert (tmp_path/'exploration'/'reference_maps.png').is_file()
