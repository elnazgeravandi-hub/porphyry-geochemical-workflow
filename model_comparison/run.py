"""Joint selection and comparison after all independent model searches."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common.io import save
from common.optimizers import objective_front
from common.environment import environment_manifest
from model_comparison.selection import choose,report_metrics
from workflow.jobs import context,branch_directory,read_bundle,CONFIGURATIONS,implementation_fingerprint


def compare(input_path,config_path,branches_root,output):
    frame,z,config,scaler=context(input_path,config_path)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    expected_hash=hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    summary=[];run_selections=[];fairness=[]
    for method in config['methods']:
        pools={};manifests={}
        for name in CONFIGURATIONS:
            manifest,records=read_bundle(branch_directory(branches_root,method,name))
            if manifest['method']!=method or manifest['configuration']!=name:raise ValueError('Branch identity mismatch.')
            if manifest['input_sha256']!=expected_hash or manifest['configuration_sha256']!=config_hash:
                raise ValueError('Independent branches did not use identical input/configuration.')
            if manifest['sample_ids']!=frame.sample_id.tolist() or manifest['preprocessing']!=scaler:
                raise ValueError('Sample order or preprocessing differs between branches.')
            if manifest.get('implementation_sha256')!=implementation_fingerprint():
                raise ValueError('Branch source implementation differs from the current comparison code.')
            packages=manifest['environment']['packages']
            current_packages=environment_manifest()['packages']
            if any(packages.get(p)!=current_packages.get(p) for p in ['numpy','scipy','pandas','scikit-learn','threadpoolctl']):
                raise ValueError('Scientific package versions differ between branch and comparison.')
            manifests[name]=manifest
            pools[name]=records if method!='kmeans' or name!='baseline' else [p for p in records if p['v']==0]
        # Inspect the exact starts, bounds, subset and request count of paired searches.
        ks=config['k_values'] if method=='kmeans' else [None]
        for k in ks:
            for seed in config['seeds']:
                paired=[]
                for name in ['mopso','nsga2']:
                    folder=branch_directory(branches_root,method,name)
                    paired.append(json.loads((folder/f'k{k}_seed{seed}_audit.json').read_text()))
                for field in ['requests','bounds_lower','bounds_upper','silhouette_subset_indices','initial_population']:
                    if paired[0][field]!=paired[1][field]:raise ValueError(f'Optimizer comparison differs in {field}.')
                expected=config['population']*(1+config['generations'])
                if any(p['requests']!=expected for p in paired):raise ValueError('Incorrect branch evaluation budget.')
                fairness.append({'method':method,'K_search':k,'seed':seed,'equal_input':True,'equal_preprocessing':True,
                    'equal_implementation':True,'equal_runtime_versions':True,'equal_domains':True,'equal_initial_population':True,'equal_silhouette_subset':True,
                    'requests_per_optimizer':expected})
        normal_pool=pools['mopso']+pools['nsga2']+(pools['baseline'] if method=='kmeans' else [])
        unique={item['key']:item for item in normal_pool if item['v']==0}
        bounds=None
        if unique:
            objectives=np.array([p['f'] for p in unique.values()]);bounds=(objectives.min(axis=0),objectives.max(axis=0))
            save(out/f'{method}_normalization.json',{'min':bounds[0],'max':bounds[1],'unique_partition_count':len(unique)})
        for name,pool in pools.items():
            if method!='kmeans' and name=='baseline' and pool:winner={**pool[0],'distance_squared':None}
            else:winner=choose(pool,bounds) if bounds is not None else None
            if winner is None:
                summary.append({'algorithm':method,'configuration':name,'status':'infeasible'});continue
            labels=winner['labels'];scores=report_metrics(z,labels,method,config.get('dbcv',False))
            summary.append({'algorithm':method,'configuration':name,**winner['info'],**scores,
                'distance_squared':winner['distance_squared'],'constraint_violation':winner['v'],
                'status':'feasible' if winner['v']==0 else 'baseline_constraint_violation'})
            save(out/f'{method}_{name}_selected.json',{k:v for k,v in winner.items() if k!='labels'})
            pd.DataFrame({'sample_id':frame.sample_id,'cluster':labels}).to_csv(out/f'{method}_{name}_labels.csv',index=False)
            save(out/f'{method}_{name}_pareto.json',[{k:pool[i][k] for k in ['decoded','f','info','key','K_search','seed']} for i in objective_front(pool)])
            if name!='baseline':
                for k in ks:
                    for seed in config['seeds']:
                        local=[p for p in pool if p['K_search']==k and p['seed']==seed]
                        selected=choose(local,bounds) if bounds is not None else None
                        run_selections.append({'method':method,'configuration':name,'K_search':k,'seed':seed,
                            'selected':None if selected is None else {key:selected[key] for key in ['decoded','info','f','key']}})
    pd.DataFrame(summary).to_csv(out/'validation_summary.csv',index=False)
    save(out/'preprocessing.json',scaler)
    save(out/'run_manifest.json',{'input_sha256':expected_hash,'configuration':config,'environment':environment_manifest(),
        'branch_configuration_sha256':config_hash,'status':'computed; manuscript numerical agreement not established'})
    save(out/'fairness_audit.json',fairness);save(out/'run_selections.json',run_selections)
    return pd.DataFrame(summary)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--branches',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();compare(args.input,args.config,args.branches,args.output)

if __name__=='__main__':main()
