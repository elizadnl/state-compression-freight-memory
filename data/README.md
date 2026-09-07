# Data directory

Raw Baltic Dry Index levels are not redistributed in this repository.

Run from the repository root:

```bash
python src/prepare_public_bdi.py
```

This creates `data/public_bdi_2009_2025.csv` locally after verifying the public source file and the expected prepared file checksum.

See [`../DATA.md`](../DATA.md) for provenance and rights information.
