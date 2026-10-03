# DAS event analysis: data and code

Python code and processed data for distributed acoustic sensing (DAS) event screening, activity accumulation and near-surface spatial-organization analysis in experiments S1, S2 and S4.

## Reproduction

Use Python 3.11 and install `requirements-tested.txt` to reproduce the verified environment. `requirements.txt` describes the supported dependency ranges; other version combinations have not been verified.

```sh
python -m pip install -r requirements-tested.txt
python -m unittest discover -s tests -v
python scripts/reproduce_cumulative_activity.py
python scripts/reproduce_archived_scan.py
python scripts/scan_original_figure.py --no-plots
```

The cumulative tables can be reconstructed from the supplied event-activity tables. The original comparison-figure scan can be reconstructed from the archived selected source records, node graph and saved permutations. These commands require no raw HDF5 waveforms.

## Analysis datasets

| Dataset | Contents | Reproduction command |
|---|---|---|
| `data/cumulative` | One-minute event count, squared strain-rate integral increments and cumulative activity; effective events, four-reference screening | `reproduce_cumulative_activity.py` |
| `data/events` | Event-level activity for effective-only and effective-plus-uncertain policies | Input to cumulative reconstruction and the configured 04B scan |
| `data/spatial_scan_original_figure` | Selected source records, merged local events, coordinates, graph, windows and 5000 archived permutations underlying the original comparison figure | `reproduce_archived_scan.py` |
| `data/spatial_scan_04B` | Validation results for the configured 04B scan with the deposited coordinate files | `scan_spatial_organization.py --no-plots` |
| `data/coordinates` | Coordinate layouts paired with the event-activity calculation inputs | Input to activity and the configured 04B scan |
| `data/coordinates/supplied_layout` | The coordinate layouts originally supplied for packaging, retained separately | Source-layout reference |
| `data/hydrology` | Rainfall, water content, matric suction, pore-water pressure and earth-pressure workbooks | Observational supporting data |
| `data/screening_inputs` | Monitoring network, stage blocks, background windows, nuisance intervals and configuration | Input to raw-waveform screening |

Run `python scripts/scan_original_figure.py --no-plots` to recompute the original comparison-figure scan. Its S1 event input is the full `data/events/S1_event_activity_original_figure_effective_plus_uncertain.csv`; S2 and S4 use their archived `05_selected_source_records.csv` tables in `data/spatial_scan_original_figure`. Coordinates for all three experiments are read from `data/coordinates/supplied_layout`, whose node/channel mappings match the archived original-figure nodes. The S2/S4 source tables are the saved ROI/time-selected records, rather than the complete upstream event-activity tables. Their event-selection audit row totals therefore refer to those deposited subsets. For the separate all-04B analysis, run `python scripts/scan_spatial_organization.py --no-plots`.

The original-figure scan and the configured 04B scan are separate analysis datasets. Their S1 event selections differ. See `docs/analysis_datasets.md`. The cumulative curves use effective events; the spatial scans use effective-plus-uncertain events.

## Raw-waveform processing

Raw DAS HDF5 waveforms are not included in this deposit. To run quality control, preprocessing, screening or waveform-based event-activity integration, supply the raw/preprocessed HDF5 files under `data/raw/S1`, `data/raw/S2`, `data/raw/S4` and `data/preprocessed/S1`, `data/preprocessed/S2`, `data/preprocessed/S4`, as appropriate. The HDF5 dataset and sample-rate contracts are specified in each script. Source filenames in the public catalogs are neutral experiment/timestamp identifiers; waveform filenames must match those identifiers when running waveform-based integration. Their timestamps and HDF5 metadata must be preserved.

Full candidate catalogs are distributed in the accompanying Zenodo data archive. Extract its `data/catalogs` directory into this repository when running `calculate_event_activity.py`. The `*_screening_catalog_04B.csv.gz` files retain the supplied screening outputs. The `*_activity_catalog_04B.csv.gz` files are the coordinate-remapped catalogs actually used for event-activity calculation, paired with `data/coordinates`. The original screening coordinates are not substituted for the activity inputs.

| Script | Function |
|---|---|
| `das_quality_control.py` | Streaming acquisition and channel quality checks |
| `preprocess_das.py` | Centering, linear detrending and 20–2000 Hz band-pass filtering |
| `screen_events_s1.py` | S1 multiscale detection and three/four-reference screening |
| `screen_events_s2.py` | S2 stage-conditioned detection and screening |
| `screen_events_s4.py` | S4 detection and balanced nuisance-reference screening |
| `screening_features.py` | Shared feature extraction and reference-domain calculations |
| `calculate_event_activity.py` | Gauge-median RMS and squared strain-rate integration |
| `scan_original_figure.py` | Original-figure event inputs with the same multiscale scan and Holm correction |
| `scan_spatial_organization.py` | 04B event inputs with the multiscale scan and Holm correction |
| `organization_plots.py` | Scan-result plotting |

```sh
python scripts/screen_events_s1.py --mode smoke
python scripts/calculate_event_activity.py --experiment S1 --event-policy auto_effective --cutoff-time "2025-07-07 22:38:39"
python scripts/scan_spatial_organization.py --no-plots
```

These raw-waveform commands require the external HDF5 files and are not a substitute for the processed-data reproduction commands above. The preprocessing entry point retains S4 as its example experiment; its input/output configuration can be set for another experiment.

The screening adapters retain their embedded module loading structure. Readable mirrors of the sanitized embedded source are provided in `scripts/embedded_sources`; they are reference files, not separate entry points. Plotting uses Times New Roman, which must be available in the plotting environment.

## Measurement definitions

Event activity is the median over the representative channel's gauge neighborhood of `sum(strain_rate**2)/sampling_rate`. With strain rate in microstrain per second, the activity unit is microstrain squared per second. RMS accumulation is retained only as a diagnostic. The spatial scan counts merged local event records with unit weight and does not sum their activity.

The spatial reference permutes entire node histories within the specified geometry, stratum and gauge-opportunity groups. Scan windows are 1, 2, 5, 10, 20, 30 and 60 minutes, with 60-second steps and a 35-cm binary three-dimensional radius graph. Three experiments receive Holm correction after within-experiment maximum-statistic correction. The exchangeability condition is part of this test. Shared DAS measurement support is audited rather than treated as independent. The analysis is retrospective; statistical support intervals do not by themselves identify a physical transition, propagation, damage coalescence or predictive lead time.

## Integrity and citation

`SHA256SUMS.txt` lists public-file checksums. `docs/data_integrity.json` records row counts, fields and privacy sanitation checks. `docs/code_integrity.json` records the calculation-function comparison against the supplied code. `docs/validation_summary.json` records the checks performed. Original analysis fingerprints are retained separately from the checksums of the public files.

The creators, repository URL, DOI, publication reference and licenses are supplied by the authors when publishing the release. Cite the final version-specific Zenodo DOI. License selection for code and data is recorded at publication; this prepared package does not assign an unconfirmed license.
