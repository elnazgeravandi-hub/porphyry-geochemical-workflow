from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from scipy.stats import truncnorm
from data_preparation.censoring import Settings,conditional_log_mean,treat,artificial_validation,structural_diagnostics,ConvergenceError
from data_preparation.run import prepare_files,UNITS


def example(n=160):
    rng=np.random.default_rng(37)
    data=pd.DataFrame(np.exp(rng.normal(size=(n,6))),columns=['Cu','Au','Mo','Bi','Pb','Zn'],index=[f'ARTIFICIAL_{i}' for i in range(n)])
    flags=pd.DataFrame(False,index=data.index,columns=data.columns)
    limits=pd.DataFrame(np.nan,index=data.index,columns=data.columns)
    return data,flags,limits


def test_conditional_mean_and_tail():
    mu=np.array([-2.,0.,2.,10.,100.]);sigma=.7;ceiling=np.zeros(5)
    expected=truncnorm.mean(-np.inf,(ceiling-mu)/sigma,loc=mu,scale=sigma)
    actual=conditional_log_mean(mu,sigma,ceiling)
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-9)
    assert (actual<ceiling).all()


def test_hidden_truth_never_enters_training_or_reference_selection():
    data,flags,limits=example();flags.iloc[:12,1]=True;limits.iloc[:12,1]=.5
    changed=data.copy();changed.iloc[:12,1]=np.nan
    a,audit=treat(data,flags,limits);b,_=treat(changed,flags,limits)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a.to_numpy()[~flags.to_numpy()],data.to_numpy()[~flags.to_numpy()])
    assert audit['converged'] and audit['reference']!='Au'
    assert (a.iloc[:12,1]<.5).all()


def test_unknown_codes_fail_and_identity_without_censoring():
    data,flags,limits=example();completed,audit=treat(data,flags,limits)
    np.testing.assert_array_equal(data,completed);assert audit['converged']
    data.iloc[0,0]=-999
    with pytest.raises(ValueError):treat(data,flags,limits)


def test_nonconvergence_recorded_without_releasing_results():
    data,flags,limits=example();flags.iloc[:12,1]=True;limits.iloc[:12,1]=.5
    with pytest.raises(ConvergenceError) as caught:treat(data,flags,limits,replace(Settings(),maximum_iterations=1,tolerance=1e-20))
    assert caught.value.diagnostics['converged'] is False
    assert len(caught.value.diagnostics['history'])==1


def test_validation_matched_masks_and_structure_identity():
    data,flags,limits=example(200)
    detail,summary,skipped=artificial_validation(data,flags,seeds=(42,),quantiles=(.2,))
    assert set(detail.method)=={'alr_rf_reimplementation','DL/2','DL/sqrt(2)'}
    assert (detail.groupby(['element','seed','quantile']).masked_test_n.nunique()==1).all()
    assert np.isfinite(detail.log_rmse).all()
    structural=structural_diagnostics(data.to_numpy(),data.to_numpy())
    assert structural['normalized_pca_procrustes_residual']<1e-12


def test_primary_bounds_and_exact_native_measurement_preservation(tmp_path):
    raw,_,_=example();raw['Au']*=1000.;raw.index.name='sample_id'
    path=tmp_path/'raw.csv';raw.to_csv(path)
    manifest={'units':UNITS,'documentation_basis':'Artificial test flags only; not laboratory data.',
        'censored_cells':[{'sample_id':sid,'element':'Au','upper_bound':5.} for sid in raw.index[:10]]}
    mask=tmp_path/'flags.json';mask.write_text(json.dumps(manifest))
    completed=prepare_files(path,mask,tmp_path/'run')
    native=pd.read_csv(path,dtype={'sample_id':str},float_precision='round_trip').set_index('sample_id')
    np.testing.assert_array_equal(completed.iloc[10:].to_numpy(),native.iloc[10:].to_numpy())
    assert (completed.iloc[:10].Au<5.).all()
    manifest['censored_cells'][0]['upper_bound']=600
    mask.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):prepare_files(path,mask,tmp_path/'invalid')
