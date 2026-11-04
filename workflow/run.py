"""Execute all independent baseline/search branches, then compare jointly."""
import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from workflow.jobs import context,execute_branch,branch_directory,CONFIGURATIONS
from model_comparison.run import compare


def run(input_path,config_path,output):
    prepared=context(input_path,config_path)
    _,_,config,_=prepared
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    for method in config['methods']:
        for name in CONFIGURATIONS:
            execute_branch(method,name,input_path,config_path,branch_directory(root,method,name),prepared_context=prepared)
    return compare(input_path,config_path,root,root/'model_comparison')


def main():
    parser=argparse.ArgumentParser(description='Run separate baseline/search stages followed by audited model comparison.')
    parser.add_argument('--input',required=True);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.input,args.config,args.output)

if __name__=='__main__':main()
