# Software Verification

The flowchart-organized implementation passed 41 automated tests with no failures or skipped tests in the recorded run (23.09 seconds in the test environment). Fixtures are artificial and contain no original sample or occurrence records.

## Verified behavior

| Area | Verification |
|---|---|
| Preparation | Units, clr centering, positive scales, censoring bounds, preserved measured values and explicit nonconvergence |
| Artificial censoring | Training-only selection/fitting, identical evaluation masks and structural-diagnostic identity |
| Optimizers | Six complete 416-request tests, covering both optimizers for all three clustering algorithms |
| Independent model folders | Each of the three baselines and six optimized branches executes separately and writes its own candidate bundle |
| Joint comparison | Nine-configuration integration, shared final selection, selected-label reconstruction and explicit infeasibility reporting |
| Controlled comparison | A branch with a deliberately changed starting population is rejected |
| Provenance | Changed input artifacts and reordered spatial identities are rejected |
| Interpretation | Profile normalization, oriented PCA, C-N model selection and scientific plots |
| Separate postprocessing | Independent profiles/PCA/C-N stages feed a separate exploration evaluator |
| Spatial evaluation | Projected file I/O, verified sample links, whole-group ceilings, geometric unions and unique occurrence capture |
| Stability | Twenty fixed-parameter 80% subsamples |

The installable wheel also builds successfully. Installed command entry points are checked separately from source-tree imports. Run metadata records Python/package versions, input/configuration hashes and source fingerprints. Independent branches retain candidate labels, decoded parameters, objectives and detailed request audits. Paired searches are compared using their actual recorded starting populations, domains, subsets and request counts before final selection.

## Interpretation

Software verification establishes the tested behavior of the implementation, not numerical replication of the restricted original manuscript dataset. Laboratory flag meanings, occurrence eligibility and agreement with the manuscript's exact results require authorized original inputs and scientific verification. Additional censoring numerical conventions remain explicitly identified in the implementation specification. Equal optimizer conditions within a clustering algorithm do not equate fixed-baseline and optimized-search budgets, nor remove the need to interpret differences in K or coverage.
