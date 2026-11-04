"""Generate independent synthetic inputs and execute the research workflow."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ELEMENTS = ['Cu', 'Au', 'Mo', 'Bi', 'Pb', 'Zn']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def generate(out):
    """No study records, coordinates, fitted parameters or result values are used."""
    import geopandas as gpd
    from shapely.geometry import box, Point
    rng = np.random.default_rng(91703)
    n = 180
    groups = np.repeat(np.arange(3), n // 3)
    means = np.array([[12, 6, .5, .13, 12, 55],
                      [180, 32, 12, .6, 28, 95],
                      [35, 9, 2, .16, 140, 220]])
    x = means[groups] * np.exp(rng.normal(0, .12, (n, 6)))
    ids = [f'SYN_{i:04d}' for i in range(n)]
    truth = pd.DataFrame(x, columns=ELEMENTS)
    truth.insert(0, 'sample_id', ids)
    truth.to_csv(out / 'synthetic_truth.csv', index=False)
    raw = truth.copy()
    cells = []
    for element, bound in [('Au', 5.), ('Bi', .1)]:
        for i in np.flatnonzero(raw[element].to_numpy() < bound):
            cells.append({'sample_id': ids[i], 'element': element, 'upper_bound': bound})
            raw.loc[i, element] = bound / 2  # Placeholder only; explicitly flagged.
    raw.to_csv(out / 'synthetic_raw.csv', index=False)
    write_json(out / 'censoring.json', {
        'units': dict(zip(ELEMENTS, ['ppm', 'ppb', 'ppm', 'ppm', 'ppm', 'ppm'])),
        'documentation_basis': 'Artificial flags generated from known synthetic truth; no laboratory records.',
        'scenario': 'primary_documented_bounds', 'censored_cells': cells})
    # Abstract overlapping squares exercise union logic; these are not DEM catchments.
    origin_x, origin_y = 500000., 3000000.
    basins = [box(origin_x + (i % 18) * 100, origin_y + (i // 18) * 100,
                  origin_x + (i % 18) * 100 + 120, origin_y + (i // 18) * 100 + 120)
              for i in range(n)]
    crs = 'EPSG:32639'  # Numerical metre-based test frame; no real study location.
    for filename, frame in [
        ('synthetic_catchments.geojson', gpd.GeoDataFrame({'sample_id': ids}, geometry=basins, crs=crs)),
        ('synthetic_boundary.geojson', gpd.GeoDataFrame({'synthetic': [True]}, geometry=[box(origin_x, origin_y, origin_x+1820, origin_y+1020)], crs=crs))]:
        frame.to_file(out / filename, driver='GeoJSON')
    points = [Point(origin_x + rng.uniform(0, 1820), origin_y + rng.uniform(0, 1020)) for _ in range(40)]
    points.append(points[0])  # Intentional duplicate exercises deduplication.
    gpd.GeoDataFrame({'synthetic': [True] * len(points)}, geometry=points, crs=crs).to_file(out / 'synthetic_occurrences.geojson', driver='GeoJSON')
    write_json(out / 'spatial.json', {
        'catchments_path': str(out / 'synthetic_catchments.geojson'),
        'catchment_id_field': 'sample_id', 'study_boundary_path': str(out / 'synthetic_boundary.geojson'),
        'occurrences_path': str(out / 'synthetic_occurrences.geojson'), 'area_crs': crs,
        'occurrence_inventory_preverified': True,
        'eligibility_description': 'Artificial points only; all eligible for this numerical demonstration. Exact duplicate intentionally retained for deduplication.'})
    config = json.loads((ROOT / 'configs/manuscript.json').read_text())
    config['expected_samples'] = n
    config['status'] = 'SYNTHETIC_DEMONSTRATION_ONLY'
    write_json(out / 'config.json', config)
    write_json(out / 'synthetic_provenance.json', {
        'synthetic': True, 'generator_seed': 91703, 'samples': n,
        'source': 'Independent parametric generator; no original study data used.',
        'spatial_scope': 'Abstract overlapping square test supports; not hydrological catchments or real occurrence locations.',
        'comparison_settings': 'Manuscript search budgets, seeds, objectives and constraints retained; sample count changed to 180.',
        'scope': 'Execution demonstration, not reproduction or validation of manuscript results.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/synthetic_demo')
    parser.add_argument('--artificial-validation', action='store_true', help='Also execute the five-split artificial-censoring assessment.')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / 'inputs'; inputs.mkdir()
    generate(inputs)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    def run(module, *arguments):
        print(f'Running {module}', flush=True)
        subprocess.run([sys.executable, '-m', module, *map(str, arguments)], cwd=ROOT, env=env, check=True)
    prepared = out / 'prepared'
    run('data_preparation.run', '--input', inputs/'synthetic_raw.csv', '--censoring', inputs/'censoring.json',
        '--settings', ROOT/'configs/censoring_numerics.json', '--output', prepared,
        *(['--validate-artificial'] if args.artificial_validation else []))
    processed = prepared/'processed_concentrations.csv'
    models = out/'model_runs/model_comparison'
    run('workflow.run', '--input', processed, '--config', inputs/'config.json', '--output', out/'model_runs')
    for stage in ['elemental_profiles', 'pca', 'cn', 'stability']:
        run(f'validation_interpretation.{stage}.run', '--input', processed, '--models', models, '--output', out/'interpretation'/stage)
    primary = pd.read_csv(processed)
    alternative = primary.copy()
    alternative.loc[primary.sample_id.isin([c['sample_id'] for c in json.loads((inputs/'censoring.json').read_text())['censored_cells'] if c['element']=='Au']), 'Au'] = 2.5
    alternative.loc[primary.sample_id.isin([c['sample_id'] for c in json.loads((inputs/'censoring.json').read_text())['censored_cells'] if c['element']=='Bi']), 'Bi'] = .05
    alternative.to_csv(inputs/'dl_half.csv', index=False)
    write_json(inputs/'scenarios.json', {'scenarios': [{'name': 'synthetic_dl_half', 'processed_input': str(inputs/'dl_half.csv'),
        'scientific_basis': 'DL/2 substitution of known artificial censored cells only; demonstration of fixed-configuration sensitivity.', 'scaling_mode': 'fixed_primary'}]})
    run('validation_interpretation.sensitivity.run', '--primary-input', processed, '--models', models,
        '--scenarios', inputs/'scenarios.json', '--output', out/'interpretation/sensitivity')
    run('exploration_evaluation.run', '--input', processed, '--models', models,
        '--pca-results', out/'interpretation/pca', '--cn-results', out/'interpretation/cn',
        '--spatial-config', inputs/'spatial.json', '--output', out/'exploration')
    summary = pd.read_csv(models/'validation_summary.csv')
    assert len(summary) == 9, 'Expected three baselines and six optimizer branches.'
    assert not summary.status.eq('infeasible').any(), 'Synthetic example did not produce all selected models.'
    audit = json.loads((models/'fairness_audit.json').read_text())
    assert len(audit) == 24 and all(item['requests_per_optimizer'] == 416 for item in audit)
    capture = pd.read_csv(out/'exploration/occurrence_capture.csv')
    assert len(capture) == 33 and capture.capture_percent.between(0, 100).all()
    write_json(out/'verification.json', {'synthetic': True, 'model_rows': len(summary), 'paired_search_audits': len(audit),
        'requests_per_optimizer_run': 416, 'capture_rows': len(capture),
        'processed_input_sha256': hashlib.sha256(processed.read_bytes()).hexdigest(),
        'artificial_validation_executed': args.artificial_validation,
        'scope': 'Successful workflow execution on synthetic inputs; no claim about manuscript numerical results.'})
    print(f'Synthetic demonstration completed: {out}', flush=True)


if __name__ == '__main__':
    main()
