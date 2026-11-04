"""Validated local input loading for post-comparison stages."""
import argparse
import hashlib
from pathlib import Path
from common.io import read_indexed,save,load_labels
from common.core import prepare
from common.schema import ELEMENTS
from common.integrity import verify_model_input
from common.environment import environment_manifest
from data_preparation.run import UNITS


def stage_arguments(description):
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument('--input',required=True);parser.add_argument('--models',required=True);parser.add_argument('--output',required=True)
    return parser.parse_args()


def stage_input(args):
    verify_model_input(args.input,args.models)
    concentrations=read_indexed(args.input)[ELEMENTS]
    z,scaler=prepare(concentrations.reset_index(),UNITS)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    save(out/'stage_manifest.json',{'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        'environment':environment_manifest(),'preprocessing':scaler})
    return concentrations,z,out
