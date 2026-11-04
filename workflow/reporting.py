import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common.schema import ELEMENTS
from common.core import prepare
from common.environment import environment_manifest
from common.integrity import verify_model_input
from common.io import save,read_indexed,load_labels
from common.ranking import enrichment
from data_preparation.run import UNITS
from validation_interpretation.pca.model import pca_reference
from validation_interpretation.cn.model import cn_reference
from validation_interpretation.elemental_profiles.model import elemental_profiles
from validation_interpretation.elemental_profiles.plot import plot_profiles
from validation_interpretation.plots import plot_references
from exploration_evaluation.spatial import read_spatial,spatial_assessment,plot_spatial

def calculate_references(concentrations,z):
    score=enrichment(concentrations); pca=pca_reference(z,score)
    cn=cn_reference(concentrations.Cu.to_numpy(),20)
    # Values equal to a breakpoint enter the upper concentration class.
    classes=np.searchsorted(cn['selected']['thresholds'],concentrations.Cu,side='right')+1
    scores=pd.DataFrame({'enrichment_score':score,'selected_pc_score':pca['scores'],
        'cn_class':classes,'Cu':concentrations.Cu},index=concentrations.index)
    return scores,pca,cn

def main():
    parser=argparse.ArgumentParser(description='Local computed profiles/reference plots and optional spatial capture.')
    parser.add_argument('--input',required=True);parser.add_argument('--labels-dir',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--spatial-config')
    args=parser.parse_args()
    verify_model_input(args.input,args.labels_dir)
    concentrations=read_indexed(args.input)[ELEMENTS]
    z,scaler=prepare(concentrations.reset_index(),UNITS)
    models=load_labels(args.labels_dir,concentrations.index)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    save(out/'environment.json',environment_manifest())
    profiles=elemental_profiles(concentrations,models);profiles.to_csv(out/'elemental_profiles.csv',index=False)
    scores,pca,cn=calculate_references(concentrations,z);scores.to_csv(out/'reference_scores.csv')
    save(out/'reference_metadata.json',{'pca':{k:v for k,v in pca.items() if k!='scores'},'cn':cn,'scaler':scaler,
        'note':'Computed reimplementation outputs; agreement with manuscript results not established.'})
    sensitivity=[]
    for minimum in [10,20,30]:
        try:
            fit=cn_reference(concentrations.Cu.to_numpy(),minimum)['selected']
            sensitivity.append({'minimum_distinct_thresholds':minimum,**fit,'status':'fitted'})
        except ValueError as exc:sensitivity.append({'minimum_distinct_thresholds':minimum,'status':str(exc)})
    save(out/'cn_threshold_sensitivity.json',sensitivity)
    supporting={}
    for element in ['Au','Mo']:
        try:supporting[element]={'unit':UNITS[element],'fit':cn_reference(concentrations[element].to_numpy(),20)}
        except ValueError as exc:supporting[element]={'unit':UNITS[element],'status':str(exc)}
    save(out/'supporting_cn_references.json',supporting)

    plot_profiles(profiles,out);plot_references(scores,pca,cn,out)
    if args.spatial_config:
        basins,domain,points,config=read_spatial(args.spatial_config,concentrations.index)
        captures,detail=spatial_assessment(scores,models,basins,domain,points)
        captures.to_csv(out/'occurrence_capture.csv',index=False)
        save(out/'spatial_audit.json',{'configuration':config,'domain_area_m2':domain.area,'evaluations':detail,
            'ranking':'Whole cluster or C-N class groups; exact tied ranks merged. PCA ranked by oriented scores.',
            'noise':'Included in common domain; not independently selected as clustering targets.'})
        plot_spatial(scores,basins,domain,points,captures,out)

if __name__=='__main__':main()
