# Analysis dataset definitions

## Original comparison figure

`data/spatial_scan_original_figure` contains the archived input records and randomizations for the supplied original three-experiment comparison figure. Its S1 input is the earlier effective-plus-uncertain event table, deposited in full as `data/events/S1_event_activity_original_figure_effective_plus_uncertain.csv` (7932 rows). Run `python scripts/scan_original_figure.py --no-plots` to recompute the scan from this full S1 table and the archived S2/S4 `05_selected_source_records.csv` tables, using the coordinate workbooks in `data/coordinates/supplied_layout` for all three experiments. These workbooks reproduce the archived node/channel mappings, gauge-opportunity profiles and permutation groups. The S2/S4 input tables are the original saved ROI/time-selected source records; their input audit row counts refer to these subsets, while the count matrices and statistics reproduce the original analysis. The workbooks paired with the 04B activity catalogs have different channel mappings and must not be substituted into the original-figure analysis. The original upstream screening configuration is not established by this event-activity table alone; it must not be attributed to the current 04B screening pipeline. Its exact ROI mapping is preserved in the archived node tables. Reproduction rebuilds merged records, the window count matrices, the geometric adjacency graph and the scan statistics using the saved 5000 permutations.

| Experiment | Within-experiment p | Holm-adjusted p |
|---|---:|---:|
| S1 | 0.028994201159768047 | 0.028994201159768047 |
| S2 | 0.002599480103979204 | 0.007798440311937612 |
| S4 | 0.006398720255948810 | 0.012797440511897620 |

## Configured 04B scan

`scan_spatial_organization.py` reads the deposited effective-plus-uncertain 04B event-activity tables and the coordinate workbooks paired with the activity inputs. Its results are stored in `data/spatial_scan_04B`. S1 uses the current four-reference event selection. The analysis cutoff, random seed and other scan parameters are recorded in each manifest.

| Experiment | Within-experiment p | Holm-adjusted p |
|---|---:|---:|
| S1 | 0.121975604879024200 | 0.121975604879024200 |
| S2 | 0.002599480103979204 | 0.007798440311937612 |
| S4 | 0.006398720255948810 | 0.012797440511897620 |

These datasets must be cited with their corresponding result tables; the original-figure S1 significance cannot be attributed to the configured 04B scan.

## Coordinate and catalog pairing

The supplied screening catalogs contain the detector's coordinate assignments. The event-activity calculation uses the separate coordinate-remapped catalogs. The original supplied workbooks are retained in `data/coordinates/supplied_layout`; the matched activity-coordinate workbooks are in `data/coordinates`. These source files are retained as separate inputs; no event measurement or screening label was changed during privacy sanitation. Mixing the supplied screening catalogs with the activity-coordinate files fails the explicit coordinate-consistency check in the calculation code.

## Cumulative activity

The deposited cumulative curves use 04B effective-only event selections and calendar-minute bins. They are independent of the near-surface ROI cropping and same-node interval merging used by the spatial scan.

| Experiment | Event peak cutoff, exclusive | Deposited minute axis | Additional exclusion |
|---|---|---|---|
| S1 | 2025-07-07 22:38:39 | 14:06–22:38 on 2025-07-07 | None |
| S2 | 2025-07-19 20:40:00 | 09:00–20:40 on 2025-07-19 | None |
| S4 | 2025-08-11 20:30:00 | 09:00–20:30 on 2025-08-11 | DAS before 09:30 |

Zeros marked `excluded_from_analysis` only support calculation and do not represent measured inactivity. The S1 cutoff must be applied before aggregation to avoid adding one event in its final calendar-minute bin.
