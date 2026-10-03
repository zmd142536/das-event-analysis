# Full event catalogs

The full catalogs are supplied in the Zenodo archive under this directory. Copy them here if using the GitHub package for waveform-based event-activity calculations.

For each experiment S1, S2 and S4:

- `*_screening_catalog_04B.csv.gz`: supplied four-reference screening output.
- `*_activity_catalog_04B.csv.gz`: coordinate-remapped input paired with `data/coordinates` for event-activity calculation.

Gzip preserves every catalog row. `pandas.read_csv` reads these files directly. Raw HDF5 waveforms are supplied separately by the authors; this deposit contains processed supporting data.
