"""Calculate and plot elemental profiles only."""
from common.stages import stage_arguments,stage_input
from common.io import load_labels
from validation_interpretation.elemental_profiles.model import elemental_profiles
from validation_interpretation.elemental_profiles.plot import plot_profiles


def main():
    args=stage_arguments('Cluster-wise elemental medians and interquartile profiles.')
    values,_,out=stage_input(args)
    profiles=elemental_profiles(values,load_labels(args.models,values.index))
    profiles.to_csv(out/'elemental_profiles.csv',index=False);plot_profiles(profiles,out)

if __name__=='__main__':main()
