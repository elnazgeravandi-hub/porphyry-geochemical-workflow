"""Independent branch execution with shared preprocessing and comparison rules."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common.core import prepare,Evaluator,metrics,describe,violation,partition_key
from common.environment import environment_manifest
from common.io import save
from common.optimizers import initial_population

METHODS=('kmeans','dbscan','optics')
CONFIGURATIONS=('baseline','mopso','nsga2')


def implementation_fingerprint():
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for package in ['common','data_preparation','baseline_models','optimized_models','model_comparison','validation_interpretation','exploration_evaluation','workflow']:
        for path in sorted((root/package).rglob('*.py')):
            digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def context(input_path,config_path):
    config=json.loads(Path(config_path).read_text())
    frame=pd.read_csv(input_path,dtype={'sample_id':str},float_precision='round_trip')
    if config.get('expected_samples') is not None and len(frame)!=config['expected_samples']:
        raise ValueError('Sample count differs from the declared configuration.')
    z,scaler=prepare(frame,config['units'])
    return frame,z,config,scaler


def branch_directory(root,method,configuration):
    if configuration=='baseline':return Path(root)/'baseline_models'/method
    solver='nsga_ii' if configuration=='nsga2' else configuration
    return Path(root)/'optimized_models'/method/solver


def record(labels,decoded,method,k,seed,z):
    info=describe(labels);v=violation(info,method!='kmeans')
    try:score=metrics(z,labels);objective=np.array([-score['SI'],score['DBI']])
    except ValueError:objective=np.array([np.inf,np.inf]);v=np.inf
    return {'labels':np.asarray(labels),'decoded':decoded,'f':objective,'v':v,
        'info':info,'key':partition_key(labels),'K_search':k,'seed':seed}


def write_bundle(output,records,frame,config,scaler,input_path,method,configuration):
    output=Path(output)
    payload=[{k:v for k,v in item.items() if k!='labels'} for item in records]
    labels=np.stack([item['labels'] for item in records]) if records else np.empty((0,len(frame)),dtype=int)
    np.savez_compressed(output/'candidate_labels.npz',labels=labels)
    save(output/'candidates.json',payload)
    save(output/'branch_manifest.json',{'method':method,'configuration':configuration,
        'input_sha256':hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        'configuration_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        'sample_ids':frame.sample_id.tolist(),'configuration_parameters':config,
        'preprocessing':scaler,'environment':environment_manifest(),
        'candidate_count':len(records),'implementation_sha256':implementation_fingerprint(),'status':'completed'})


def read_bundle(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'branch_manifest.json').read_text())
    entries=json.loads((directory/'candidates.json').read_text())
    with np.load(directory/'candidate_labels.npz',allow_pickle=False) as data:labels=data['labels']
    if len(entries)!=len(labels) or len(entries)!=manifest['candidate_count']:raise ValueError('Candidate bundle count mismatch.')
    records=[]
    for item,label in zip(entries,labels):
        item['labels']=label
        item['decoded']=tuple(item['decoded'])
        item['f']=np.array([np.inf if v is None else v for v in item['f']])
        item['v']=np.inf if item['v'] is None else item['v']
        if partition_key(label)!=item['key']:raise ValueError('Candidate label checksum mismatch.')
        records.append(item)
    return manifest,records


def execute_branch(method,configuration,input_path,config_path,output,prepared_context=None):
    if method not in METHODS or configuration not in CONFIGURATIONS:raise ValueError('Unknown model branch.')
    frame,z,config,scaler=prepared_context if prepared_context is not None else context(input_path,config_path)
    if method not in config['methods']:raise ValueError('Method is not enabled in the supplied configuration.')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    records=[];run_records=[];full_scores={}
    if configuration=='baseline':
        module=importlib.import_module(f'baseline_models.{method}.model')
        if method=='kmeans':
            for k in config['k_values']:
                for seed in config['seeds']:
                    labels,decoded=module.baseline(z,k,seed,config['baseline_n_init'])
                    records.append(record(labels,decoded,method,k,seed,z))
        else:
            labels,decoded=module.baseline(z)
            records.append(record(labels,decoded,method,None,config['seeds'][0],z))
    else:
        solver='nsga_ii' if configuration=='nsga2' else configuration
        module=importlib.import_module(f'optimized_models.{method}.{solver}.model')
        for k in config['k_values'] if method=='kmeans' else [None]:
            for seed in config['seeds']:
                evaluator=Evaluator(z,method,k,config['silhouette_subset_size'])
                initial=initial_population(evaluator,seed,config['population'])
                module.search(evaluator,initial,seed,config['generations'])
                expected=config['population']*(config['generations']+1)
                if evaluator.requests!=expected:raise AssertionError('Evaluation budget mismatch.')
                prefix=f'k{k}_seed{seed}'
                save(out/f'{prefix}_requests.json',evaluator.logs)
                save(out/f'{prefix}_audit.json',{'requests':evaluator.requests,'decoded_fits_or_extractions':evaluator.fits,
                    'fit_seconds':evaluator.fit_seconds,'unique_feasible_partitions':len(evaluator.partitions),
                    'optics_ordering_fits':len(evaluator.orderings),'bounds_lower':evaluator.lo,'bounds_upper':evaluator.hi,
                    'silhouette_subset_indices':evaluator.subset,'initial_population':initial})
                keys=[]
                for candidate in evaluator.partitions.values():
                    key=candidate['key']
                    if key not in full_scores:full_scores[key]=metrics(z,candidate['labels'])
                    score=full_scores[key]
                    records.append({**candidate,'f':np.array([-score['SI'],score['DBI']]),'K_search':k,'seed':seed})
                    keys.append(key)
                run_records.append({'K_search':k,'seed':seed,'partitions':keys,'requests':evaluator.requests})
    save(out/'runs.json',run_records)
    write_bundle(out,records,frame,config,scaler,input_path,method,configuration)
    return records


def branch_main(method,configuration):
    parser=argparse.ArgumentParser(description=f'Run only {method}/{configuration}; final selection is performed jointly afterwards.')
    parser.add_argument('--input',required=True);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):execute_branch(method,configuration,args.input,args.config,args.output)
