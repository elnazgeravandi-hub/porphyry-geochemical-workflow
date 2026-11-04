"""Catchment union, unique occurrence capture and GIS reference maps."""
import argparse
import hashlib
import json
from pathlib import Path
from common.integrity import verify_model_input
from common.io import read_indexed,load_labels,save
from exploration_evaluation.spatial import read_spatial,spatial_assessment,plot_spatial


def main():
    parser=argparse.ArgumentParser(description='Evaluate exploration correspondence using saved PCA/C-N outputs.')
    parser.add_argument('--input',required=True);parser.add_argument('--models',required=True)
    parser.add_argument('--pca-results',required=True);parser.add_argument('--cn-results',required=True)
    parser.add_argument('--spatial-config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();verify_model_input(args.input,args.models)
    expected=hashlib.sha256(Path(args.input).read_bytes()).hexdigest();values=read_indexed(args.input)
    for directory in [args.pca_results,args.cn_results]:
        if json.loads((Path(directory)/'stage_manifest.json').read_text())['input_sha256']!=expected:
            raise ValueError('Reference stages used a different input artifact.')
    pca=read_indexed(Path(args.pca_results)/'pca_scores.csv');cn=read_indexed(Path(args.cn_results)/'cn_scores.csv')
    if set(pca.index)!=set(values.index) or set(cn.index)!=set(values.index):raise ValueError('Reference sample IDs differ.')
    scores=pca.loc[values.index].join(cn.loc[values.index],validate='one_to_one')
    labels=load_labels(args.models,values.index)
    basins,domain,points,config=read_spatial(args.spatial_config,values.index)
    table,details=spatial_assessment(scores,labels,basins,domain,points)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    table.to_csv(out/'occurrence_capture.csv',index=False)
    save(out/'spatial_audit.json',{'input_sha256':expected,'configuration':config,'domain_area_m2':domain.area,'evaluations':details})
    plot_spatial(scores,basins,domain,points,table,out)

if __name__=='__main__':main()
