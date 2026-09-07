# Monte Carlo note

The paper reports the frozen full-run Monte Carlo summaries in `frozen_sim_weekly.json` and `frozen_sim_rolling.json`.

- Weekly-direction nulls: 1,000 residual-bootstrap paths under AR(1) and 1,000 under AR(4).
- Overlapping trend nulls: 500 target replications per horizon; a small number of fits can fail numerically, so the realised valid counts are recorded in the JSON.
- Random seed: 20260904.

The end-to-end script exposes `--simulations --b-week 1000 --b-roll 500` to regenerate the full battery. Core empirical estimates and expanding-window forecasts are reproduced by the default fast run.
