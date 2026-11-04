"""Independent implementation of the manuscript's ALR/RF censoring specification.

This is a method reimplementation, not a vendored or certified upstream ReX-CoDA
implementation. Choices not specified by the manuscript are explicit in Settings.
"""
from dataclasses import dataclass,asdict
import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Settings:
    trees:int=100
    depth:int=8
    seed:int=42
    tolerance:float=1e-5
    maximum_iterations:int=100
    # Extra implementation decisions, not historical settings claimed by the paper:
    initialization_fraction:float=1.0
    residual_scale:str='out_of_bag_rmse'
    minimum_scale:float=1e-8
    minimum_observed:int=30


class ConvergenceError(RuntimeError):
    def __init__(self,diagnostics):
        super().__init__('ALR censoring procedure did not converge; no final concentrations released.')
        self.diagnostics=diagnostics


def conditional_log_mean(prediction,scale,ceiling):
    """Mean of a normal variable conditional on being below ceiling."""
    prediction,scale,ceiling=np.broadcast_arrays(np.asarray(prediction,float),np.asarray(scale,float),np.asarray(ceiling,float))
    if not all(np.isfinite(a).all() for a in (prediction,scale,ceiling)) or (scale<=0).any():
        raise ValueError('Finite arguments and strictly positive scales required.')
    standardized=(ceiling-prediction)/scale
    result=np.empty_like(prediction)
    # Mills-ratio tail expansion avoids catastrophic subtraction of near equals.
    far=standardized < -12
    t=-1/standardized[far]
    result[far]=ceiling[far]-scale[far]*t*(1-2*t*t+10*t**4-74*t**6)
    regular=~far
    z=standardized[regular]
    log_mills=-.5*z*z-.5*np.log(2*np.pi)-log_ndtr(z)
    result[regular]=prediction[regular]-scale[regular]*np.exp(log_mills)
    result=np.minimum(result,np.nextafter(ceiling,-np.inf))
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite conditional log mean.')
    return result


def validate_inputs(values,mask,limits):
    if len(values.columns)!=6 or values.columns.duplicated().any() or not values.index.is_unique:
        raise ValueError('Six unique elemental columns and unique sample IDs required.')
    for table in (mask,limits):
        if not table.index.equals(values.index) or not table.columns.equals(values.columns):
            raise ValueError('Concentrations, flags and limits must align exactly.')
    if not all(t==bool for t in mask.dtypes) or mask.isna().to_numpy().any():
        raise ValueError('Explicit nonmissing boolean censoring flags required.')
    x=values.to_numpy(float); m=mask.to_numpy(bool); b=limits.to_numpy(float)
    if not np.isfinite(x[~m]).all() or (x[~m]<=0).any():
        raise ValueError('Unflagged measurements must be finite and positive; unknown codes are not inferred.')
    if not np.isfinite(b[m]).all() or (b[m]<=0).any():
        raise ValueError('Every censored cell requires a documented positive finite bound.')


def select_reference(values,mask):
    choices=[e for e in values if not mask[e].any()]
    if not choices:raise ValueError('No fully positive, uncensored reference component exists.')
    logs=np.log(values[choices].astype(float))
    widths=logs.quantile(.75)-logs.quantile(.25)
    return min(choices,key=lambda e:(float(widths[e]),e))


def fit_response(predictors,response,settings):
    if len(response)<settings.minimum_observed:raise ValueError('Too few uncensored training responses.')
    model=RandomForestRegressor(n_estimators=settings.trees,max_depth=settings.depth,
        random_state=settings.seed,n_jobs=1,bootstrap=True,oob_score=True,max_features=1.0,
        min_samples_split=2,min_samples_leaf=1,criterion='squared_error')
    model.fit(predictors,response)
    residual=response-model.oob_prediction_
    if not np.isfinite(residual).all():raise FloatingPointError('Nonfinite OOB residuals.')
    scale=max(float(np.sqrt(np.mean(residual**2))),settings.minimum_scale)
    return model,scale


def treat(values,mask,limits,settings=Settings()):
    """All concentrations and limits are in ppm. Return only converged results."""
    validate_inputs(values,mask,limits)
    if settings.residual_scale!='out_of_bag_rmse' or not 0<settings.initialization_fraction<=1 or settings.minimum_scale<=0 or settings.maximum_iterations<1 or settings.tolerance<=0:
        raise ValueError('Unsupported/invalid explicitly declared implementation settings.')
    raw=values.astype(float).copy();reference=select_reference(raw,mask)
    denominator=np.log(raw[reference])
    columns=[e for e in raw if e!=reference]
    initialized=raw.copy();initialized[mask]=limits[mask]*settings.initialization_fraction
    current=np.log(initialized[columns]).sub(denominator,axis=0)
    targets=[e for e in columns if mask[e].any()]
    audit={'method':'Manuscript-guided ALR-RF upper-truncation reimplementation',
        'settings':asdict(settings),'reference':reference,'reference_rule':'lowest log-IQR; alphabetical ties',
        'converged':not targets,'history':[],'scale_diagnostics':[],
        'censored_counts':{e:int(mask[e].sum()) for e in mask},
        'extra_choices':'OOB RMSE scale; initialization at DL; synchronous updates. Not specified by manuscript.'}
    for iteration in range(1,settings.maximum_iterations+1) if targets else []:
        previous=current.copy();following=previous.copy()
        for element in targets:
            observed=~mask[element].to_numpy();flagged=~observed
            predictor_columns=[e for e in columns if e!=element]
            x=previous[predictor_columns].to_numpy();response=previous[element].to_numpy()
            forest,scale=fit_response(x[observed],response[observed],settings)
            ceiling=np.log(limits.loc[flagged,element].to_numpy())-denominator.to_numpy()[flagged]
            following.loc[flagged,element]=conditional_log_mean(forest.predict(x[flagged]),scale,ceiling)
            audit['scale_diagnostics'].append({'iteration':iteration,'element':element,'oob_rmse_alr':scale,'training_n':int(observed.sum())})
        change=float(np.linalg.norm(following.to_numpy()-previous.to_numpy())/max(np.linalg.norm(previous.to_numpy()),np.finfo(float).eps))
        if not np.isfinite(change):raise FloatingPointError('Nonfinite ALR convergence change.')
        audit['history'].append({'iteration':iteration,'relative_frobenius_change':change});current=following
        if change<settings.tolerance:audit['converged']=True;break
    if not audit['converged']:raise ConvergenceError(audit)
    completed=raw.copy()
    for element in targets:
        flagged=mask[element].to_numpy()
        log_estimate=current.loc[flagged,element].to_numpy()+denominator.to_numpy()[flagged]
        if (log_estimate<=np.log(np.finfo(float).tiny)).any() or (log_estimate>=np.log(np.finfo(float).max)-2).any():
            raise FloatingPointError('Concentration back-transformation exceeds floating-point range.')
        estimates=np.exp(log_estimate)
        completed.loc[flagged,element]=np.minimum(estimates,np.nextafter(limits.loc[flagged,element].to_numpy(),0.))
    m=mask.to_numpy()
    if not np.array_equal(completed.to_numpy()[~m],raw.to_numpy()[~m]):raise AssertionError('Unflagged measured values changed.')
    if not np.isfinite(completed.to_numpy()).all() or not (completed.to_numpy()>0).all():raise FloatingPointError('Invalid final concentration.')
    if not (completed.to_numpy()[m]<limits.to_numpy()[m]).all():raise AssertionError('Censoring limit exceeded.')
    audit['observed_values_preserved']=True
    return completed,audit


def clr_matrix(values):
    logs=np.log(values);return logs-logs.mean(axis=1,keepdims=True)


def structural_diagnostics(truth,estimated):
    """Known-truth diagnostic; unscaled clr, correlations and full-rank PCA Procrustes.

    PCA dimension and normalization are explicit additional choices. A pure rotation
    preserves equivalence; normalized full-rank configurations are compared here.
    """
    a=clr_matrix(np.asarray(truth,float));b=clr_matrix(np.asarray(estimated,float))
    ca=np.corrcoef(a,rowvar=False);cb=np.corrcoef(b,rowvar=False)
    if not np.isfinite(ca).all() or not np.isfinite(cb).all():raise ValueError('Undefined clr correlations.')
    dim=min(5,len(a)-1)
    sa=PCA(n_components=dim,svd_solver='full').fit_transform(a)
    sb=PCA(n_components=dim,svd_solver='full').fit_transform(b)
    sa-=sa.mean(axis=0);sb-=sb.mean(axis=0)
    na=np.linalg.norm(sa);nb=np.linalg.norm(sb)
    if na==0 or nb==0:raise ValueError('Degenerate PCA configuration.')
    sa/=na;sb/=nb;rotation,_=orthogonal_procrustes(sb,sa)
    return {'mean_absolute_clr_correlation_change':float(np.abs(ca-cb)[np.triu_indices(6,1)].mean()),
        'normalized_pca_procrustes_residual':float(np.linalg.norm(sa-sb@rotation)),
        'pca_components':dim,'procrustes_rule':'unit Frobenius norm, orthogonal alignment; no additional fitted scale'}


def artificial_validation(values,mask,seeds=(42,43,44,45,46),quantiles=(.1,.2,.3),settings=Settings()):
    """Training-only fitting/reference/thresholds; truth revealed only to score."""
    if not values.index.equals(mask.index) or not values.columns.equals(mask.columns):raise ValueError('Flags do not align.')
    data=values.loc[~mask.any(axis=1)].astype(float)
    if len(data)<100 or not np.isfinite(data.to_numpy()).all() or not (data.to_numpy()>0).all():raise ValueError('Need 100 complete confirmed observed rows.')
    rows=[];skipped=[]
    for element in data:
        for seed in seeds:
            train,test=train_test_split(np.arange(len(data)),test_size=.2,random_state=seed)
            train_data=data.iloc[train]
            reference_mask=pd.DataFrame(False,index=train_data.index,columns=train_data.columns)
            reference_mask[element]=True
            reference=select_reference(train_data,reference_mask)
            predictors=[e for e in data if e not in (element,reference)]
            x=np.log(data[predictors]).sub(np.log(data[reference]),axis=0).to_numpy()
            denominator=np.log(data[reference].to_numpy());response_values=data[element].to_numpy()
            for q in quantiles:
                if not 0<q<1:raise ValueError('Quantile must be in (0,1).')
                threshold=float(np.quantile(response_values[train],q))
                hidden_train=response_values[train]<threshold;hidden_test=response_values[test]<threshold
                fitting=train[~hidden_train];held=test[hidden_test]
                if len(fitting)<settings.minimum_observed or not len(held):
                    skipped.append({'element':element,'seed':seed,'quantile':q,'reason':'insufficient fit/score observations'});continue
                response=np.log(response_values[fitting])-denominator[fitting]
                forest,scale=fit_response(x[fitting],response,settings)
                ceiling=np.log(threshold)-denominator[held]
                prediction=conditional_log_mean(forest.predict(x[held]),scale,ceiling)+denominator[held]
                truth=np.log(response_values[held])
                for method,predicted in [('alr_rf_reimplementation',prediction),('DL/2',np.full(len(held),np.log(threshold/2))),('DL/sqrt(2)',np.full(len(held),np.log(threshold/np.sqrt(2))))]:
                    error=predicted-truth
                    # Reconstruct the complete test set only for known-truth structural diagnostics.
                    reconstructed=data.iloc[test].to_numpy().copy();column=list(data.columns).index(element)
                    reconstructed[hidden_test,column]=np.exp(predicted)
                    structure=structural_diagnostics(data.iloc[test].to_numpy(),reconstructed)
                    rows.append({'element':element,'seed':seed,'quantile':q,'method':method,'reference':reference,
                        'threshold_ppm':threshold,'training_fit_n':len(fitting),'masked_test_n':len(held),
                        'actual_training_censor_fraction':float(hidden_train.mean()),'actual_test_censor_fraction':float(hidden_test.mean()),
                        'log_rmse':float(np.sqrt(np.mean(error**2))),'log_signed_error':float(error.mean()),**structure})
    if not rows:raise RuntimeError('No completed artificial-censoring evaluations.')
    detail=pd.DataFrame(rows)
    summary=detail.groupby(['element','quantile','method'],sort=False).agg(splits=('seed','nunique'),
        log_rmse_mean=('log_rmse','mean'),log_rmse_sd=('log_rmse','std'),
        log_signed_error_mean=('log_signed_error','mean'),
        clr_correlation_change_mean=('mean_absolute_clr_correlation_change','mean'),
        pca_procrustes_residual_mean=('normalized_pca_procrustes_residual','mean')).reset_index()
    return detail,summary,skipped
