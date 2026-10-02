# Scenario-Based LCA of Colombian Beef: Monte Carlo model

This repository contains the reproducible Monte Carlo model used in:

**Scenario-Based Life Cycle Assessment of Beef Production in Colombia: Intensive Silvopastoral Transition Scenarios under Uncertainty**

## Functional unit

1 kg live-weight gain (LWG) at the farm gate.

## Main scenarios

- BAU-F: extensive reference configuration
- SSPi-M: moderate silvopastoral configuration
- SSPi-I: intensive silvopastoral configuration

## Reproducible manuscript run

The manuscript results use:

- 10,000 accepted Monte Carlo iterations per scenario
- random seed = 42
- Python 3.10+
- parameter values documented in Supplementary Table S1

Run:

```bash
python LCA_Beef_SSPi_MonteCarlo.py
```

To also export all accepted Monte Carlo draws:

```bash
python LCA_Beef_SSPi_MonteCarlo.py --save-draws
```

To run without generating figures:

```bash
python LCA_Beef_SSPi_MonteCarlo.py --no-figures
```

## Principal expected results

With the default settings, mean primary greenhouse-gas intensities are:

| Scenario | Mean kg CO2eq/kg LWG |
|---|---:|
| BAU-F | 13.879362 |
| SSPi-M | 8.488936 |
| SSPi-I | 7.103373 |

These correspond to modeled reductions of approximately 38.84% and 48.82% for SSPi-M and SSPi-I relative to BAU-F.

## Output files

The script writes:

- `primary_summary.csv`
- `component_summary.csv`
- `sensitivity_spearman.csv`
- `shapley_decomposition.csv`
- `structural_sensitivity.csv`
- `convergence.csv`
- `climate_sensitivity.csv`
- `conservative_productivity.csv`
- optional `accepted_monte_carlo_draws.csv`
- figures corresponding to the manuscript analyses

## Methodological note

FAOSTAT is **not** used to calibrate the BAU-F Monte Carlo parameter vector. It is retained only as a national high-emission contextual benchmark. Colombian farm-level evidence is used for external plausibility assessment.

Land-use change and carbon sequestration are evaluated as separate structural accounting configurations rather than intrinsic properties of the production scenarios.

## Citation

After archiving a release in Zenodo, replace this section with the Zenodo citation and DOI for the archived release.

## Revision: harmonized 450-kg livestock basis

This version adopts the explicit 450-kg Colombian UGG reference-animal basis. It replaces the previous 300-kg mass approximation for VS and N excretion with 450 kg; all affected results have been recomputed. Stocking rate is expressed as UGG/ha. The model makes the simplifying assumption that 1 UGG is represented by one 450-kg reference animal, so that per-head gain and per-head IPCC enteric factors can be consistently applied. This is not a literal farm head count, and heterogeneous herd weights are not resolved.

The SSPi-I upper productivity limit (1,098 kg ha-1 yr-1) is supported by Mahecha et al. (2012), Carta Fedegán 129, Figure 1, where 3.5 UGG/ha and approximately 0.856 kg daily live-weight gain per animal yield approximately 1,094 kg/ha/yr. The source labels output 'meat production'; it is interpreted here as an approximate live-weight-gain proxy based on the paired values, not carcass mass.

This reviewer-revised release replaces the published results of the earlier v1.0.0. Archive this code as a **new** GitHub release (e.g. v1.1.0) and create a **new version DOI** in Zenodo. Do not retroactively reuse the older version-specific DOI as if it documented these results.

## Code check

The default run includes a strict assertion of manuscript means, with tolerance 5e-10. Run `python LCA_Beef_SSPi_MonteCarlo.py --save-draws` to export summary tables, diagnostic statistics, and accepted simulation draws.
