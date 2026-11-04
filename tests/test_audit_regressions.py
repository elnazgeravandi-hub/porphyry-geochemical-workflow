import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from workflow.run import run
from common.core import Evaluator
from common.integrity import verify_model_input
from common.optimizers import initial_population,mopso,nsga2


@pytest.mark.parametrize('method',['kmeans','dbscan','optics'])
@pytest.mark.parametrize('optimizer',[mopso,nsga2])
def test_actual_416_request_budget(method,optimizer):
    z,_=make_blobs(n_samples=90,n_features=6,centers=3,cluster_std=.12,random_state=5)
    ev=Evaluator(z,method,3 if method=='kmeans' else None)
    population=initial_population(ev,42,16)
    optimizer(ev,population,42,generations=25)
    assert ev.requests==416
    assert len(ev.logs)==416


def test_exact_input_provenance_detects_reordering(tmp_path):
    path=tmp_path/'input.csv';path.write_text('sample_id,Cu\na,1\nb,2\n')
    manifest=tmp_path/'run_manifest.json'
    manifest.write_text(json.dumps({'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    verify_model_input(path,tmp_path)
    path.write_text('sample_id,Cu\nb,2\na,1\n')
    with pytest.raises(ValueError,match='differs'):verify_model_input(path,tmp_path)


def test_all_noise_baseline_is_reported_when_no_optimizer_solution(tmp_path,monkeypatch):
    class NoSolution:
        def __init__(self,z,method,k=None,subset_size=600):
            self.z=z;self.method=method;self.lo=np.array([-2.,5.]);self.hi=np.array([2.,30.])
            self.baseline_radius=1.;self.requests=0;self.fits=0;self.fit_seconds=0.
            self.partitions={};self.logs=[];self.orderings={};self.subset=np.arange(len(z))
        def fit(self,decoded):return np.full(len(self.z),-1)
        def evaluate(self,theta):
            self.requests+=1
            return {'theta':theta.copy(),'f':np.array([np.inf,np.inf]),'v':np.inf}
    monkeypatch.setattr('workflow.jobs.Evaluator',NoSolution)
    monkeypatch.setattr('baseline_models.dbscan.model.baseline',lambda z:(np.full(len(z),-1),(1.,8)))
    monkeypatch.setattr('baseline_models.optics.model.baseline',lambda z:(np.full(len(z),-1),(1.,8)))
    rng=np.random.default_rng(2)
    frame=pd.DataFrame(np.exp(rng.normal(size=(40,6))),columns=['Cu','Au','Mo','Bi','Pb','Zn'])
    frame['sample_id']=[f'TEST_{i}' for i in range(40)]
    path=tmp_path/'input.csv';frame.to_csv(path,index=False)
    cfg={'units':{'Cu':'ppm','Au':'ppb','Mo':'ppm','Bi':'ppm','Pb':'ppm','Zn':'ppm'},
        'methods':['dbscan','optics'],'seeds':[42],'population':4,'generations':0,
        'silhouette_subset_size':40,'expected_samples':40,'dbcv':False}
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg));out=tmp_path/'run'
    run(path,config,out)
    table=pd.read_csv(out/'model_comparison'/'validation_summary.csv')
    assert len(table)==6
    baseline=table[table.configuration=='baseline']
    assert baseline.coverage.eq(0).all() and baseline.noise_count.eq(40).all()
    assert baseline.status.eq('baseline_constraint_violation').all()
    assert table[table.configuration!='baseline'].status.eq('infeasible').all()
    assert (out/'model_comparison'/'dbscan_baseline_labels.csv').is_file()


def test_spatial_misaligned_ids_are_rejected():
    pytest.importorskip('geopandas')
    import geopandas as gpd
    from shapely.geometry import box,Point
    from exploration_evaluation.spatial import spatial_assessment
    basins=gpd.GeoDataFrame(geometry=[box(0,0,1,1),box(1,0,2,1)],index=['b','a'],crs=32638)
    scores=pd.DataFrame({'enrichment_score':[.2,.8],'selected_pc_score':[1.,2.],
        'cn_class':[1,2],'Cu':[1.,2.]},index=['a','b'])
    with pytest.raises(ValueError,match='order differs'):
        spatial_assessment(scores,{'kmeans_baseline':np.array([0,1])},basins,box(0,0,2,1),[Point(.5,.5)])


def test_fast_final_pareto_matches_general_sorting_with_ties():
    from common.optimizers import objective_front,fronts
    rng=np.random.default_rng(31)
    values=rng.integers(0,10,size=(200,2))
    records=[{'v':0.,'f':row.astype(float)} for row in values]
    assert set(objective_front(records))==set(fronts(records)[0])
