import json
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from common.core import prepare, partition_key, Evaluator, metrics
from common.optimizers import initial_population, mopso, nsga2
from common.partition_evaluation import agreement
from workflow.run import run


def test_clr_unit_conversion_cancels_after_centering():
    rng=np.random.default_rng(19); x=np.exp(rng.normal(size=(90,6)))
    frame=pd.DataFrame(x,columns=['Cu','Au','Mo','Bi','Pb','Zn']); frame['sample_id']=range(90)
    units={'Cu':'ppm','Au':'ppb','Mo':'ppm','Bi':'ppm','Pb':'ppm','Zn':'ppm'}
    z,_=prepare(frame,units)
    raw=np.log(x)-np.log(x).mean(axis=1,keepdims=True)
    expected=(raw-np.median(raw,axis=0))/(np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0))
    np.testing.assert_allclose(z,expected,atol=1e-12)
    frame.loc[0,'Cu']=0
    with pytest.raises(ValueError): prepare(frame,units)


def test_partition_relabeling_and_noise():
    assert partition_key(np.array([7,7,-1,3]))==partition_key(np.array([1,1,-1,0]))
    assert partition_key(np.array([7,7,-1,3]))!=partition_key(np.array([7,7,3,-1]))
    assert agreement([0,0,1,1,-1],[5,5,8,8,-1])['ARI_all']==1


@pytest.mark.parametrize('optimizer',[mopso,nsga2])
def test_budget_and_feasibility(optimizer):
    z,_=make_blobs(n_samples=90,n_features=6,centers=3,cluster_std=.15,random_state=2)
    ev=Evaluator(z,'kmeans',3); initial=initial_population(ev,42,4)
    optimizer(ev,initial,42,generations=2)
    assert ev.requests==12
    assert ev.partitions
    assert all(p['info']['K']==3 for p in ev.partitions.values())


def test_end_to_end_nine_configurations(tmp_path):
    z,_=make_blobs(n_samples=150,n_features=6,centers=3,cluster_std=.1,random_state=4)
    frame=pd.DataFrame(np.exp(z/8),columns=['Cu','Au','Mo','Bi','Pb','Zn'])
    frame['sample_id']=[f'ARTIFICIAL_{i}' for i in range(len(frame))]
    input_path=tmp_path/'local_test.csv'; frame.to_csv(input_path,index=False)
    cfg={'units':{'Cu':'ppm','Au':'ppb','Mo':'ppm','Bi':'ppm','Pb':'ppm','Zn':'ppm'},
         'methods':['kmeans','dbscan','optics'],'k_values':[3],'seeds':[42],
         'population':4,'generations':1,'silhouette_subset_size':100,'baseline_n_init':3,
         'expected_samples':150,'dbcv':True}
    config=tmp_path/'test.json'; config.write_text(json.dumps(cfg))
    run(input_path,config,tmp_path/'out')
    summary=pd.read_csv(tmp_path/'out'/'model_comparison'/'validation_summary.csv')
    assert set(summary.algorithm)=={'kmeans','dbscan','optics'}
    assert len(summary)==9
    assert summary.SI.dropna().between(-1,1).all()
    assert all(json.loads(p.read_text())['requests']==8 for p in (tmp_path/'out').rglob('k*_seed*_audit.json'))
    # Selected initial centroids and density parameters must reconstruct each export.
    from validation_interpretation.stability.run import make_refit
    z,_=prepare(frame,cfg['units'])
    for selected_path in (tmp_path/'out'/'model_comparison').glob('*_selected.json'):
        name=selected_path.name.removesuffix('_selected.json')
        selected=json.loads(selected_path.read_text())
        reproduced=make_refit(name.split('_')[0],selected)(z)
        exported=pd.read_csv(tmp_path/'out'/'model_comparison'/f'{name}_labels.csv').cluster.to_numpy()
        assert partition_key(reproduced)==partition_key(exported)



def test_selection_uses_shared_bounds_and_coverage_tie():
    from model_comparison.selection import choose
    a={'v':0.,'f':np.array([0.,1.]),'info':{'coverage':.95,'minimum_cluster_size':5},'decoded':(1.,)}
    b={'v':0.,'f':np.array([1.,0.]),'info':{'coverage':1.,'minimum_cluster_size':5},'decoded':(2.,)}
    assert choose([a,b],(np.zeros(2),np.ones(2)))['decoded']==(2.,)


def test_pca_orientation_and_enrichment_ties():
    from common.ranking import enrichment
    from validation_interpretation.pca.model import pca_reference
    rng=np.random.default_rng(9); z=rng.normal(size=(80,6))
    score=z[:,2]
    result=pca_reference(z,score)
    from scipy.stats import spearmanr
    assert spearmanr(result['scores'],score).statistic>=0
    frame=pd.DataFrame({'Cu':[1,1,3],'Au':[1,1,3],'Mo':[1,1,3]})
    e=enrichment(frame)
    assert e[0]==e[1] and e[2]>e[0]


def test_cn_single_power_law():
    from validation_interpretation.cn.model import cn_reference
    # Exactly a power law in the distinct threshold/cumulative count representation.
    x=1/np.arange(100,0,-1,dtype=float)
    result=cn_reference(x,min_segment=20)
    assert result['selected']['segments']==1


def test_capture_whole_ties_and_union():
    shapely=pytest.importorskip('shapely')
    from shapely.geometry import box,Point
    from common.partition_evaluation import capture
    domain=box(0,0,10,10)
    result=capture([(2,[box(0,0,2,2)]),(2,[box(1,0,3,2)]),(1,[box(0,0,4,4)])],
        domain,[Point(1,1),Point(1,1),Point(9,9)],budgets=(.05,.1,.2))
    assert result['eligible_occurrences']==2
    assert result['budgets'][0]['capture_percent']==0 # top whole tied group is 6%, exceeds 5%
    assert result['budgets'][1]['capture_percent']==50
    assert result['steps'][1]['area_fraction']==.06
