import numpy as np
from common.schema import ELEMENTS

def prepare(frame, units):
    if units != {'Cu':'ppm','Au':'ppb','Mo':'ppm','Bi':'ppm','Pb':'ppm','Zn':'ppm'}:
        raise ValueError('Expected declared units: Au ppb; other five elements ppm.')
    if 'sample_id' not in frame or frame.sample_id.isna().any() or frame.sample_id.astype(str).str.strip().eq('').any() or frame.sample_id.astype(str).duplicated().any():
        raise ValueError('Unique, nonmissing sample_id required.')
    if frame.columns.duplicated().any(): raise ValueError('Duplicate column names.')
    x = frame[ELEMENTS].to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError('Input must contain verified positive processed concentrations. No automatic replacement.')
    # Converting Au to ppm makes composition units explicit. Fixed column conversion
    # cancels under subsequent column median centering, but is recorded nevertheless.
    x = x.copy(); x[:,1] /= 1000
    logx = np.log(x); clr = logx - logx.mean(axis=1, keepdims=True)
    median = np.median(clr, axis=0)
    iqr = np.quantile(clr,.75,axis=0)-np.quantile(clr,.25,axis=0)
    if (iqr <= 0).any():
        raise ValueError('Zero-IQR clr feature: resolve explicitly before analysis.')
    z = (clr-median)/iqr
    assert np.allclose(clr.sum(axis=1),0,atol=1e-10)
    return z, {'elements':ELEMENTS, 'median':median.tolist(),'iqr':iqr.tolist(),
               'units':units,'au_ppb_to_ppm':True}
