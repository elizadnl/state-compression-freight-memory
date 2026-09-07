# Data provenance and redistribution note

## Empirical input

The analysis uses 808 consecutive weekly Baltic Dry Index observations from **6 November 2009 to 25 April 2025**.

The observations are independently accessible through the following public research data chain:

1. **Chen, Y., Feng, A. & Tang, C. (2026)**, *Identifying the significant drivers of containerized freight rates: From the perspective of dynamic multiscale dependence*, PLOS ONE 21(4), e0344386. Supporting-data DOI: `10.1371/journal.pone.0344386.s001`.
2. **Battal, T. (2026)**, *Replication package: A Regime Identification Framework for Container Shipping Freight Markets*, Zenodo, DOI `10.5281/zenodo.22035531`. The public record states that `predictors_FINAL.csv` contains Baltic Dry Index observations extracted from the Chen–Feng–Tang supporting dataset.

The source field used here is `bdry`. `src/prepare_public_bdi.py` keeps only `date` and `bdry`, renames `bdry` to `bdi`, sorts by date and verifies:

- 808 observations;
- sample endpoints;
- unique dates and no missing BDI values;
- seven day spacing;
- the published Zenodo MD5 for the source file; and
- the SHA256 checksum of the prepared local analysis file.

## BDI methodology change

The sample spans the **1 March 2018** BDI composition change. The Baltic Exchange stated that the index was reweighted to 40% Capesize, 30% Panamax and 30% Supramax and no longer included Handysize. The analysis therefore reports separate pre and post change estimates; the 2020-2025 out-of-sample period lies entirely under the post change definition.

Official notice: https://www.balticexchange.com/en/news-and-events/news/member-news/2018/bdi-changes-1-march.html

## Rights and redistribution

The Baltic Exchange asserts intellectual property rights over Baltic Data and restricts redistribution under its data policy. This repository therefore **does not bundle or redistribute the raw BDI index levels**.

To reproduce the analysis, run:

```bash
python src/prepare_public_bdi.py
```

The script downloads the cited public Zenodo source and verifies its integrity before creating the local analysis file.

Useful source pages:

- PLOS article: https://doi.org/10.1371/journal.pone.0344386
- PLOS supporting data: https://doi.org/10.1371/journal.pone.0344386.s001
- Zenodo record: https://doi.org/10.5281/zenodo.22035531
- Baltic Exchange Data Policy: https://www.balticexchange.com/en/site-services/data-policy.html
