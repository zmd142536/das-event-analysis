# Data dictionary

The accompanying `data_integrity.json` lists the column names and row counts of all 96 CSV datasets under `data/`, including the six gzip-compressed candidate catalogs and both spatial scan datasets. `docs/file_manifest.csv` is a release inventory rather than a scientific dataset and is excluded from this dataset count. File contents are verified using `SHA256SUMS.txt` and `CATALOG_SHA256SUMS.txt`. Workbook cells and existing scientific sheet labels are retained as provided. Local directory metadata, creator metadata and local file references have been sanitized.

| Field | Definition |
|---|---|
| `physical_event_id` | Unique local event identifier within an experiment/catalog |
| `start_time`, `end_time`, `peak_time` | Event interval endpoints and representative peak timestamp |
| `minute_time` | Calendar-minute bin label |
| `file_name` | Neutral source-waveform identifier preserving the acquisition timestamp |
| `automatic_screening_label` | Automatic screening category; not a manually confirmed physical mechanism |
| `manual_label`, `manual_confidence`, `manual_notes` | Existing manual annotation fields; not generated or strengthened for release |
| `manual_reviewer` | Pseudonymous reviewer identifier where populated |
| `representative_data_channel` | Waveform channel index for the event's representative channel |
| `layout_channel` | Coordinate-layout channel; data-channel offset is 3000 for these experiments |
| `gauge_data_channels` | Semicolon-separated waveform channels included in the event gauge neighborhood |
| `event_rms_gauge_median` | Median of per-channel event RMS values; microstrain per second |
| `event_squared_strain_rate_integral_gauge_median` | Median per-channel integral of squared strain rate; microstrain squared per second |
| `event_count` | Number of retained events in the minute bin |
| `activity_increment` | Sum of retained event integrals in the minute bin |
| `cumulative_activity` | Running sum of activity increments over the retained analysis period |
| `excluded_from_analysis` | True for bins excluded from the DAS analysis |
| `count_trailing_10min_mean` | Trailing 10-bin mean event count, with available bins at the start |
| `x_cm`, `y_cm`, `z_cm`, `burial_depth_cm` | Geometry coordinates and burial depth in centimetres |
| `permutation_group` | Group within which complete node histories are exchanged |
| `node_i`, `node_j` | Endpoints of an unordered binary geometric edge |
| `observed_moran_i` | Spatial Moran statistic for node counts in a scan window |
| `null_mean`, `null_sd`, `null_q025`, `null_q975` | Randomized pointwise reference summaries |
| `scan_z` | Symmetrically standardized scan score |
| `p_scan_within_experiment` | p-value against the randomized maximum over all scanned windows |
| `p_scan_study` | Study correction associated with the experiment's Holm step |
| `support_union_start`, `support_union_end` | Descriptive union of overlapping significant windows |

## Array files

Each `11_reproducibility_arrays.npz` contains `permutations` (observed identity row followed by randomized target-to-source node indices), `window_merged_event_counts` (window by node integer counts) and `maximum_scan_z` (one maximum per observed/randomized row). Load with `numpy.load(..., allow_pickle=False)`.

## Candidate catalogs

The compressed candidate catalogs preserve every source row and scientific field. Their feature columns include detection scale, detection score, reference-domain distances, locality, gauge coherence and automatic/manual screening fields. They are read directly with `pandas.read_csv`; gzip decompression is automatic. Candidate status does not establish event mechanism or cross-node source independence.

## Workbook groups

Coordinate workbooks define channel geometry. Hydrology workbooks contain S1/S2/S4 observations with their supplied sensor labels, timestamps and units. Sensor H3/H4 labels and existing units are preserved. No calibration, interpolation or unit conversion is applied during packaging.
