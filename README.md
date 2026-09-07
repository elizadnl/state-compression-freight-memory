# State Compression and Apparent Memory in Freight Markets

Reproducible code and derived results for **State Compression and Apparent Memory in Freight Markets**.

The project studies whether apparent duration dependence in freight regimes reflects a genuine time effect or information already contained in the market path and the way regimes are constructed. The empirical application uses weekly Baltic Dry Index (BDI) observations from 2009-2025 and combines logistic regime-exit models, leakage safe expanding window forecasts, path controls and autoregressive simulation nulls.

> **Paper status:** preprint submitted to SSRN 
## Main findings

| Result | Estimate |
|---|---:|
| Weekly regime-age coefficient | 0.0841 |
| Cluster robust p-value | 0.0267 |
| 2020-2025 Brier improvement from adding age to a current return model | 1.23% |
| HAC(4) p-value for that Brier improvement | 0.0231 |
| Correlation: regime age vs prior cumulative absolute return | 0.836 |
| Age coefficient after controlling for prior path information | -0.0044 |
| p-value after path control | 0.950 |
| Brier change from adding age to the path model | -0.68% |
| HAC(4) p-value for that deterioration | 0.0400 |
| False significance under overlapping AR(1) regime construction | 69-84% |

The central result is therefore not that freight has an autonomous duration clock. Regime age can look predictive in a simple discrete model, but the effect disappears after conditioning on prior path information, and overlapping regime construction can generate strong spurious duration effects even under standard autoregressive persistence.

## Repository structure

```text
.
├── src/
│   ├── prepare_public_bdi.py       # fetch + verify the public source data
│   ├── analyze_public_bdi.py       # primary models, OOS forecasts, simulations
│   └── identification_checks.py    # path controls, stability and methodology checks
├── data/
│   └── README.md                   # data provenance / redistribution note
├── results/
│   ├── results.json
│   ├── identification_checks.json
│   ├── hazard_by_age.csv
│   ├── frozen_sim_weekly.json
│   └── frozen_sim_rolling.json
├── DATA.md
├── requirements.txt
├── Makefile
└── CITATION.cff
```

## Reproduce

Create an environment and install dependencies:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Fetch and verify the public source series:

```bash
python src/prepare_public_bdi.py
```

Run the primary analysis:

```bash
python src/analyze_public_bdi.py
```

Run the identification and stability checks:

```bash
python src/identification_checks.py
```

Re-run the slower Monte Carlo battery:

```bash
python src/analyze_public_bdi.py --simulations --b-week 1000 --b-roll 500
```

The random seed is fixed at `20260904`.

## Data and rights

The repository **does not redistribute raw BDI index levels**. The preparation script downloads a publicly accessible research file from Zenodo, extracts the BDI series used in the analysis and verifies the source and prepared file checksums. See [`DATA.md`](DATA.md) for the full provenance chain and rights note.

Frozen summary results are included for inspection. Detailed expanding window forecast outputs are generated locally when the analysis is run.

## Citation

If you use the code or results, please cite the accompanying paper. A permanent SSRN link and DOI are to be added. 

## Licence

The source code in this repository is released under the MIT License. This licence does **not** apply to third party data, including Baltic Data or to the manuscript itself.
