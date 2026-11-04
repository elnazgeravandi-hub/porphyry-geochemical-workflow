# Private input specifications

## Concentration CSV

Unique nonempty sample_id, then Cu, Au, Mo, Bi, Pb, Zn. Au in ppb; other elements ppm. Measured cells must be positive and finite. A flagged censored cell's stored numeric placeholder is not used as a response. Unclassified missing/sentinel values cause an error, not an undocumented substitution or row filter. Sample order is preserved.

## Censoring JSON

Private JSON fields:

- units: {"Cu":"ppm","Au":"ppb","Mo":"ppm","Bi":"ppm","Pb":"ppm","Zn":"ppm"}
- documentation_basis: actual source/laboratory documentation supporting the flags.
- scenario: primary_documented_bounds or assumption_based_sensitivity.
- censored_cells: array of objects, each with sample_id, element, upper_bound.

Bounds use native units. In the primary manuscript scenario, only Au=5 ppb and Bi=0.1 ppm bounds are accepted. This check does not prove the supplied flags are laboratory-confirmed. Other thresholds must be labelled assumption-based sensitivity scenarios. No sentinel-code meaning or censoring flag is inferred from a repeated low number.

Internally all ALR inputs use ppm. Output processed concentrations return to native units and unflagged native values are copied exactly. The common-unit change introduces a constant clr-coordinate offset which cancels after median centering. Its equivalence is tested.

## Spatial JSON

Required fields: catchments_path, catchment_id_field, study_boundary_path, occurrences_path, area_crs, occurrence_inventory_preverified, eligibility_description. Paths refer to local datasets, never public input downloads. Choose an appropriate projected CRS in metres for the study. Eligibility must be verified before setting occurrence_inventory_preverified=true; explain the actual coordinate/mineralization-type criteria.

All sample IDs must link to catchments. Extra unlinked catchments do not enter the common domain. Multipart features for one sample are dissolved; clipped empty/invalid geometries fail explicitly. Occurrence points are deduplicated by exact XY coordinates in the evaluation CRS. Input records with no documented eligibility are not automatically assumed suitable.

Store populated manifests, original records, GIS inputs and outputs under private_data/ or results/ locally. Public upload through the browser bypasses git-ignore selection rules; choose code/docs files only.

Downstream reporting, stability and sensitivity require run_manifest.json and the exact processed-input artifact used by the model run (SHA-256 match). Changing numeric contents, row order or CSV serialization requires an explicit new model run; do not substitute a similarly named file.
