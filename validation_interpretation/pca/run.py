"""Select and orient PCA from the common modeling input."""
import pandas as pd
from common.stages import stage_arguments,stage_input
from common.io import save
from common.ranking import enrichment
from common.schema import ELEMENTS
from validation_interpretation.pca.model import pca_reference


def main():
    args=stage_arguments('PCA component selection and interpretation.')
    values,z,out=stage_input(args);score=enrichment(values);pca=pca_reference(z,score)
    pd.DataFrame({'enrichment_score':score,'selected_pc_score':pca['scores']},index=values.index).to_csv(out/'pca_scores.csv')
    save(out/'pca_metadata.json',{k:v for k,v in pca.items() if k!='scores'})
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained');ax.bar(ELEMENTS,pca['coefficients']);ax.axhline(0,color='black',lw=.7)
    ax.set_title(f'Oriented PC{pca["component"]}');ax.set_ylabel('Coefficient')
    fig.savefig(out/'pca_coefficients.png',dpi=300);fig.savefig(out/'pca_coefficients.pdf');plt.close(fig)

if __name__=='__main__':main()
