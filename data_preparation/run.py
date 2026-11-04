"""Manuscript-guided data preparation with explicit laboratory censoring records."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from common.core import ELEMENTS,prepare
from data_preparation.censoring import Settings,treat,artificial_validation,ConvergenceError
from common.environment import environment_manifest
from common.io import save

UNITS={'Cu':'ppm','Au':'ppb','Mo':'ppm','Bi':'ppm','Pb':'ppm','Zn':'ppm'}


def load_inputs(path,manifest_path):
    raw=pd.read_csv(path,dtype={'sample_id':str},float_precision='round_trip')
    if 'sample_id' not in raw or raw.sample_id.isna().any() or raw.sample_id.str.strip().eq('').any() or raw.sample_id.duplicated().any():
        raise ValueError('Nonempty, unique sample_id required.')
    native=raw.set_index('sample_id')[ELEMENTS].apply(pd.to_numeric,errors='raise')
    spec=json.loads(Path(manifest_path).read_text())
    if spec.get('units')!=UNITS:raise ValueError('Declare Au ppb, all other elements ppm.')
    if not str(spec.get('documentation_basis','')).strip():raise ValueError('Document the source of flags and detection limits.')
    scenario=spec.get('scenario','primary_documented_bounds')
    if scenario not in ['primary_documented_bounds','assumption_based_sensitivity']:raise ValueError('Unknown censoring scenario.')
    mask=pd.DataFrame(False,index=native.index,columns=ELEMENTS)
    limits=pd.DataFrame(np.nan,index=native.index,columns=ELEMENTS)
    seen=set()
    for item in spec['censored_cells']:
        sid=item['sample_id'];element=item['element'];bound=float(item['upper_bound'])
        if (sid,element) in seen or sid not in native.index or element not in ELEMENTS:raise ValueError('Invalid/duplicate flagged cell.')
        if not np.isfinite(bound) or bound<=0:raise ValueError('Invalid censoring upper bound.')
        if scenario=='primary_documented_bounds' and (element not in ['Au','Bi'] or bound!={'Au':5.,'Bi':.1}.get(element)):
            raise ValueError('Primary manuscript bounds are Au 5 ppb and Bi 0.1 ppm. Other assumptions must be explicitly sensitivity scenarios.')
        mask.loc[sid,element]=True;limits.loc[sid,element]=bound;seen.add((sid,element))
    common=native.copy();common['Au']/=1000.;limits['Au']/=1000.
    return native,common,mask,limits,spec


def prepare_files(input_path,censoring_path,output,settings=Settings(),validate=False):
    native,common,mask,limits,spec=load_inputs(input_path,censoring_path)
    out=Path(output)
    if out.exists():raise FileExistsError('Output directory exists; preserve previous run artifacts.')
    out.mkdir(parents=True)
    provenance={'input_sha256':hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        'censoring_sha256':hashlib.sha256(Path(censoring_path).read_bytes()).hexdigest(),
        'source':'Manuscript Methods sections 2, 3.4–3.5 and Tables 1–2/5, supplied 2026-11-04.',
        'environment':environment_manifest(),'scenario':spec['scenario'] if 'scenario' in spec else 'primary_documented_bounds',
        'documentation_basis':spec['documentation_basis'],'sample_count':len(common)}
    try:processed,audit=treat(common,mask,limits,settings)
    except ConvergenceError as error:
        save(out/'censoring_failure.json',{**provenance,**error.diagnostics})
        raise
    result=processed.copy();result['Au']*=1000.
    # Preserve exact native-unit observed measurements as well as common-unit values.
    result[~mask]=native[~mask]
    observed=~mask.to_numpy()
    if not np.array_equal(native.to_numpy()[observed],result.to_numpy()[observed]):raise AssertionError('Native observed measurements changed.')
    result.to_csv(out/'processed_concentrations.csv',index_label='sample_id')
    z,scaler=prepare(result.reset_index(),UNITS)
    pd.DataFrame(z,index=result.index,columns=ELEMENTS).to_csv(out/'scaled_clr.csv',index_label='sample_id')
    mask.to_csv(out/'censoring_flags.csv',index_label='sample_id')
    save(out/'censoring_audit.json',{**provenance,**audit,'settings':asdict(settings),'native_observed_values_preserved':True})
    save(out/'scaling.json',scaler)
    if validate:
        detail,summary,skipped=artificial_validation(common,mask,settings=settings)
        detail.to_csv(out/'artificial_validation.csv',index=False)
        summary.to_csv(out/'artificial_validation_summary.csv',index=False)
        save(out/'artificial_validation_scope.json',{'skipped':skipped,'scope':'Known truth; complete observed rows; one artificially censored target at a time.',
            'seeds':[42,43,44,45,46],'training_quantiles':[.1,.2,.3],
            'note':'No ground-truth claim for naturally censored cells. See documentation for explicit structural-diagnostic definition.'})
    return result


def main():
    parser=argparse.ArgumentParser(description='Independent manuscript-guided ALR-RF censoring preparation.')
    parser.add_argument('--input',required=True);parser.add_argument('--censoring',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--settings',help='Explicit extra numerical choices JSON.')
    parser.add_argument('--validate-artificial',action='store_true')
    args=parser.parse_args()
    settings=Settings(**json.loads(Path(args.settings).read_text())) if args.settings else Settings()
    prepare_files(args.input,args.censoring,args.output,settings,args.validate_artificial)

if __name__=='__main__':main()
