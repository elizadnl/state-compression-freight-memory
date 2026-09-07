# Derived results

This directory contains fixed **derived** outputs used to document the analysis. It does not contain raw BDI index levels.

- `results.json` - primary empirical estimates, robustness checks and OOS forecast metrics.
- `identification_checks.json` - path controls, OOS path vs age comparisons, forecast start date sensitivity and BDI methodology change checks.
- `hazard_by_age.csv` - reversal frequencies by regime age bin.
- `frozen_sim_weekly.json` - AR(1) and AR(4) weekly regime Monte Carlo summaries.
- `frozen_sim_rolling.json` - overlapping 4, 8, 13 and 26 week regime Monte Carlo summaries.

The slower simulation outputs are frozen so the reported paper results can be checked without re-running every Monte Carlo replication. Detailed expanding window prediction files are generated locally by `src/analyze_public_bdi.py`.
