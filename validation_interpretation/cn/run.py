"""Fit C-N references and breakpoint sensitivity without fitting clustering models."""
import numpy as np
import pandas as pd
from common.stages import stage_arguments,stage_input
from common.io import save
from data_preparation.run import UNITS
from validation_interpretation.cn.model import cn_reference


def main():
    args=stage_arguments('Continuous C-N regression and threshold sensitivity.')
    values,_,out=stage_input(args);results={};sensitivity=[]
    for element in ['Cu','Au','Mo']:
        try:results[element]={'unit':UNITS[element],**cn_reference(values[element].to_numpy(),20)}
        except ValueError as error:
            if element=='Cu':raise
            results[element]={'status':str(error)}
    model=results['Cu']['selected'];classes=np.searchsorted(model['thresholds'],values.Cu,side='right')+1
    pd.DataFrame({'cn_class':classes,'Cu':values.Cu},index=values.index).to_csv(out/'cn_scores.csv')
    for minimum in [10,20,30]:
        try:sensitivity.append({'minimum_thresholds':minimum,**cn_reference(values.Cu.to_numpy(),minimum)['selected']})
        except ValueError as error:sensitivity.append({'minimum_thresholds':minimum,'status':str(error)})
    save(out/'cn_metadata.json',results);save(out/'threshold_sensitivity.json',sensitivity)
    import matplotlib.pyplot as plt
    x=np.log10(results['Cu']['distinct_thresholds']);design=np.column_stack([np.ones(len(x)),x]+[np.maximum(x-b,0) for b in model['breakpoints_log10']])
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    ax.loglog(10**x,results['Cu']['cumulative_counts'],'.',label='Cumulative counts');ax.loglog(10**x,10**(design@model['coefficients']),label='Selected fit')
    for threshold in model['thresholds']:ax.axvline(threshold,color='gray',ls='--')
    ax.set_xlabel('Cu (ppm)');ax.set_ylabel('N(>= concentration)');ax.legend()
    fig.savefig(out/'cu_cn.png',dpi=300);fig.savefig(out/'cu_cn.pdf');plt.close(fig)

if __name__=='__main__':main()
