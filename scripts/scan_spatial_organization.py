"""Scan merged local event counts using constrained whole-history permutations and Holm correction."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
import sys
import time
import numpy as np
import pandas as pd
BASE_DIRECTORY = REPOSITORY_ROOT
OUTPUT_DIRECTORY = REPOSITORY_ROOT / 'outputs/spatial_scan_04B'
COORDINATE_DIRECTORY = REPOSITORY_ROOT / 'data/coordinates'
EVENT_DIRECTORY = REPOSITORY_ROOT / 'data/events'
EXPERIMENTS_TO_RUN = ('S1', 'S2', 'S4')
SIMULATIONS = 5000
RANDOM_SEED = 20260922
SCAN_STEP_SECONDS = 60
WINDOW_MINUTES = (1, 2, 5, 10, 20, 30, 60)
NEIGHBOUR_RADIUS_CM = 35.0
MIN_WINDOW_MERGED_EVENT_RECORDS = 5
MIN_WINDOW_ACTIVE_NODES = 3
ALPHA = 0.05
STUDY_FAMILY_SIZE = 3
PERMUTATION_BATCH_SIZE = 200
PNG_DPI = 600
SAVE_FIGURES = True
GAUGE_HALF_CHANNELS = 4
MERGE_TOUCHING_INTERVALS = True

@dataclass(frozen=True)
class Experiment:
    name: str
    coordinate_file: Path
    event_file: Path
    start: str
    end: str
    event_set: str = 'effective_plus_uncertain'
CONFIG = {'S1': Experiment('S1', COORDINATE_DIRECTORY / 'S1_coordinates.xlsx', EVENT_DIRECTORY / 'S1_event_activity_04B_effective_plus_uncertain.csv', '2025-07-07 14:06:54', '2025-07-07 22:38:39'), 'S2': Experiment('S2', COORDINATE_DIRECTORY / 'S2_S4_coordinates.xlsx', EVENT_DIRECTORY / 'S2_event_activity_04B_effective_plus_uncertain.csv', '2025-07-19 09:00:00', '2025-07-19 20:37:47'), 'S4': Experiment('S4', COORDINATE_DIRECTORY / 'S2_S4_coordinates.xlsx', EVENT_DIRECTORY / 'S4_event_activity_04B_effective_plus_uncertain.csv', '2025-08-11 09:00:00', '2025-08-11 20:29:07')}

@dataclass(frozen=True)
class Settings:
    simulations: int = SIMULATIONS
    seed: int = RANDOM_SEED
    scan_step_seconds: int = SCAN_STEP_SECONDS
    window_minutes: tuple = WINDOW_MINUTES
    radius_cm: float = NEIGHBOUR_RADIUS_CM
    min_merged_event_records: int = MIN_WINDOW_MERGED_EVENT_RECORDS
    min_active_nodes: int = MIN_WINDOW_ACTIVE_NODES
    alpha: float = ALPHA
    family_size: int = STUDY_FAMILY_SIZE
    batch_size: int = PERMUTATION_BATCH_SIZE

def log(message):
    print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, content):
    Path(path).write_text(json.dumps(content, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')

def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')

def region_name(depth, x, z):
    if np.isclose(depth, 2):
        return 'surface_2cm_top' if np.isclose(z, 98) else 'surface_2cm_slope'
    if np.isclose(depth, 20):
        return 'platform_20cm'
    if np.isclose(depth, 50) and np.isclose(x, 120):
        return 'near_face_50cm'
    if np.isclose(depth, 80) and np.isclose(x, 150):
        return 'near_face_80cm'
    return 'excluded_interior'

def load_nodes(spec):
    layout = pd.read_excel(spec.coordinate_file)
    required = {'channel', 'segment_id', 'segment_type', 'burial_depth', 'x_cm', 'y_cm', 'z_cm', 'slope_or_leadin'}
    if not required.issubset(layout.columns):
        raise ValueError(f'{spec.name}: coordinate columns missing: {required - set(layout.columns)}')
    if layout.channel.duplicated().any():
        raise ValueError('Coordinate channel must be unique')
    layout['channel'] = pd.to_numeric(layout.channel, errors='raise').astype(int)
    valid = layout.slope_or_leadin.eq('slope')
    if layout.loc[valid, ['x_cm', 'y_cm', 'z_cm', 'burial_depth']].isna().any().any():
        raise ValueError('A slope coordinate is missing; no coordinate interpolation is performed')
    if not set(layout.segment_type).issubset({'ring', 'line'}):
        raise ValueError('Expected ring/line segment types')
    channels = set(layout.channel)
    rows = []
    for key, group in layout[valid].groupby(['burial_depth', 'x_cm', 'y_cm', 'z_cm'], sort=True):
        depth, x, y, z = map(float, key)
        raw = sorted(group.channel.tolist())
        gc = [sum((c in channels for c in range(ch - GAUGE_HALF_CHANNELS, ch + GAUGE_HALF_CHANNELS + 1))) for ch in raw]
        profile = ';'.join((f'{g}:{gc.count(g)}' for g in sorted(set(gc))))
        geometry = 'ring' if 'ring' in set(group.segment_type) else 'line'
        region = region_name(depth, x, z)
        stratum = ('surface_2cm_top' if np.isclose(z, 98) else 'surface_2cm_slope') if np.isclose(depth, 2) else f'horizontal_{depth:g}cm'
        support = sorted({c for ch in raw for c in range(ch - GAUGE_HALF_CHANNELS, ch + GAUGE_HALF_CHANNELS + 1) if c in channels})
        rows.append(dict(node_id=f'{spec.name}_D{depth:g}_X{x:g}_Y{y:g}_Z{z:g}', burial_depth_cm=depth, x_cm=x, y_cm=y, z_cm=z, geometry_type=geometry, stratum=stratum, region=region, selected=region != 'excluded_interior', raw_channels=';'.join(map(str, raw)), raw_channel_count=len(raw), segment_ids=';'.join(sorted(set(group.segment_id))), gauge_opportunity_profile=profile, gauge_support_channels=';'.join(map(str, support)), permutation_group=f'{stratum}|{geometry}|{profile}'))
    all_nodes = pd.DataFrame(rows)
    all_nodes.insert(0, 'original_node_index', np.arange(len(all_nodes)))
    nodes = all_nodes[all_nodes.selected].copy().reset_index(drop=True)
    nodes.insert(0, 'node_index', np.arange(len(nodes)))
    expected = 112 if spec.name == 'S1' else 113
    if len(nodes) != expected:
        raise ValueError(f'{spec.name}: expected confirmed ROI {expected} nodes; got {len(nodes)}')
    return (layout, all_nodes, nodes)

def connected_sizes(n, left, right):
    neighbours = [set() for _ in range(n)]
    for i, j in zip(left, right):
        neighbours[int(i)].add(int(j))
        neighbours[int(j)].add(int(i))
    unseen = set(range(n))
    sizes = []
    while unseen:
        todo = [unseen.pop()]
        size = 0
        while todo:
            i = todo.pop()
            size += 1
            new = neighbours[i] & unseen
            unseen.difference_update(new)
            todo.extend(new)
        sizes.append(size)
    return sorted(sizes, reverse=True)

def make_graph(nodes, radius_cm):
    xyz = nodes[['x_cm', 'y_cm', 'z_cm']].to_numpy(float)
    distance = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    left, right = np.where(np.triu((distance > 1e-10) & (distance <= radius_cm + 1e-09), 1))
    if not len(left):
        raise ValueError('No edges under the fixed radius; do not silently force connectivity')
    supports = [set(map(int, v.split(';'))) for v in nodes.gauge_support_channels]
    edges = pd.DataFrame({'node_i': left, 'node_j': right, 'distance_cm': distance[left, right], 'weight': np.ones(len(left)), 'node_id_i': nodes.node_id.iloc[left].to_numpy(), 'node_id_j': nodes.node_id.iloc[right].to_numpy(), 'shared_geometric_gauge_channels': [len(supports[i] & supports[j]) for i, j in zip(left, right)]})
    degrees = np.bincount(np.r_[left, right], minlength=len(nodes))
    audit = dict(rule='symmetric binary 3-D radius graph; no added edges', radius_cm=radius_cm, nodes=len(nodes), edges=len(edges), isolated_nodes=int((degrees == 0).sum()), component_sizes=connected_sizes(len(nodes), left, right), edges_with_shared_geometric_gauge_support=int(edges.shared_geometric_gauge_channels.gt(0).sum()), interpretation='Statistical organization of node-level observations; shared DAS measurement support is audited, not asserted independent.')
    return (edges, audit)

def merge_overlapping(events):
    """Interval connected components within a node; retain one strongest record per component."""
    records = []
    membership = []
    for node, g in events.groupby('node_index', sort=True):
        ordered = g.sort_values(['start_time', 'end_time', 'physical_event_id'], kind='stable')
        cluster = []
        latest = None

        def finish(items):
            group = pd.DataFrame(items)
            strongest = group.sort_values(['activity', 'peak_time', 'physical_event_id'], ascending=[False, True, True], kind='stable').iloc[0]
            cid = f'N{int(node):03d}_C{len(records):07d}'
            records.append(dict(cluster_id=cid, node_index=int(node), start_time=group.start_time.min(), end_time=group.end_time.max(), peak_time=strongest.peak_time, representative_event_id=strongest.physical_event_id, source_records=len(group), representative_activity=float(strongest.activity), any_gauge_crosses_domain=bool(group.gauge_crosses_domain.any()), any_gauge_has_unlocated_channel=bool(group.gauge_has_unlocated_channel.any())))
            for v in group.physical_event_id:
                membership.append(dict(physical_event_id=v, cluster_id=cid, node_index=int(node)))
        for row in ordered.to_dict('records'):
            overlaps = latest is not None and (row['start_time'] <= latest if MERGE_TOUCHING_INTERVALS else row['start_time'] < latest)
            if cluster and (not overlaps):
                finish(cluster)
                cluster = []
            cluster.append(row)
            latest = row['end_time'] if latest is None or not overlaps else max(latest, row['end_time'])
        if cluster:
            finish(cluster)
    columns = ['cluster_id', 'node_index', 'start_time', 'end_time', 'peak_time', 'representative_event_id', 'source_records', 'representative_activity', 'any_gauge_crosses_domain', 'any_gauge_has_unlocated_channel']
    merged = pd.DataFrame(records, columns=columns).sort_values(['peak_time', 'node_index'], kind='stable').reset_index(drop=True)
    return (merged, pd.DataFrame(membership, columns=['physical_event_id', 'cluster_id', 'node_index']))

def load_events(spec, all_nodes, nodes, layout):
    events = pd.read_csv(spec.event_file)
    required = {'physical_event_id', 'start_time', 'end_time', 'peak_time', 'layout_channel', 'representative_data_channel', 'gauge_data_channels', 'gauge_channel_count', 'event_squared_strain_rate_integral_gauge_median'}
    if not required.issubset(events):
        raise ValueError(f'Missing event columns: {required - set(events.columns)}')
    if events.physical_event_id.isna().any() or events.physical_event_id.duplicated().any():
        raise ValueError('Input event ID must be nonmissing and unique')
    for c in ['start_time', 'end_time', 'peak_time']:
        events[c] = pd.to_datetime(events[c], format='mixed', errors='raise')
        if events[c].isna().any():
            raise ValueError(f'Missing event time: {c}')
    if ((events.start_time > events.peak_time) | (events.peak_time > events.end_time)).any():
        raise ValueError('Event interval must include its peak')
    events['activity'] = pd.to_numeric(events.event_squared_strain_rate_integral_gauge_median, errors='raise')
    if (~np.isfinite(events.activity) | (events.activity < 0)).any():
        raise ValueError('Invalid activity')
    mapping = {int(ch): int(r.node_index) for r in nodes.itertuples() for ch in r.raw_channels.split(';')}
    all_mapped = {int(ch) for s in all_nodes.raw_channels for ch in s.split(';')}
    start, end = (pd.Timestamp(spec.start), pd.Timestamp(spec.end))
    before = events.peak_time.lt(start)
    after = events.peak_time.gt(end)
    inside_time = events.loc[~before & ~after].copy()
    unknown = ~inside_time.layout_channel.isin(all_mapped)
    if unknown.any():
        raise ValueError(f'{int(unknown.sum())} in-time event representative channels lack a modeled coordinate')
    selected = inside_time[inside_time.layout_channel.isin(mapping)].copy()
    selected['node_index'] = selected.layout_channel.map(mapping).astype(int)
    offsets = (selected.layout_channel - selected.representative_data_channel).unique()
    if len(offsets) != 1:
        raise ValueError('Expected one explicit data-to-layout channel offset')
    offset = int(offsets[0])
    selected_channels = set(mapping)
    located_channels = set(all_mapped)
    crosses = []
    unlocated = []
    for r in selected.itertuples():
        footprint = {int(v) + offset for v in str(r.gauge_data_channels).split(';')}
        if len(footprint) != int(r.gauge_channel_count):
            raise ValueError('Gauge list/count mismatch')
        crosses.append(bool(footprint - selected_channels))
        unlocated.append(bool(footprint - located_channels))
    selected['gauge_crosses_domain'] = crosses
    selected['gauge_has_unlocated_channel'] = unlocated
    if selected.empty:
        raise ValueError('No event remains in the confirmed ROI and time range')
    merged, membership = merge_overlapping(selected)
    audit = dict(source_rows=len(events), excluded_before=int(before.sum()), excluded_after=int(after.sum()), in_time_rows=len(inside_time), excluded_outside_domain=len(inside_time) - len(selected), selected_raw_records=len(selected), merged_local_records=len(merged), records_removed_by_same_node_overlap=len(selected) - len(merged), merged_components_larger_than_one=int(merged.source_records.gt(1).sum()), largest_component_source_records=int(merged.source_records.max()), longest_merged_interval_seconds=float((merged.end_time - merged.start_time).dt.total_seconds().max()), selected_records_with_crossing_gauge=int(sum(crosses)), selected_records_with_unlocated_gauge=int(sum(unlocated)), layout_channel_minus_data_channel=offset, counting_rule='One local record per connected overlapping interval at the same coordinate node; count at the peak of its largest-activity source record. No amplitude summation.', remaining_scope='Cross-node duplicate/source equivalence and measurement-support independence are not assumed resolved.')
    return (selected, merged, membership, audit)

def candidate_windows(start, end, settings):
    start, end = (pd.Timestamp(start), pd.Timestamp(end))
    duration = (end - start).total_seconds()
    rows = []
    for minutes in settings.window_minutes:
        length = float(minutes) * 60
        if length > duration:
            continue
        starts = np.arange(0, duration - length + 1e-08, settings.scan_step_seconds, dtype=float)
        starts = np.unique(np.r_[starts, duration - length])
        for a in starts:
            rows.append(dict(start_time=start + pd.Timedelta(seconds=float(a)), end_time=start + pd.Timedelta(seconds=float(a + length)), duration_minutes=float(minutes), start_hours=float(a / 3600), end_hours=float((a + length) / 3600)))
    if not rows:
        raise ValueError('No requested scan duration fits the analysis period')
    frame = pd.DataFrame(rows).drop_duplicates(['start_time', 'end_time']).reset_index(drop=True)
    frame.insert(0, 'window_id', np.arange(len(frame)))
    return frame

def window_event_counts(events, windows, node_count, analysis_end):
    """Count merged local records in [start,end); the fixed analysis end is inclusive."""
    starts = windows.start_time.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    ends = windows.end_time.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    final_ns = pd.Timestamp(analysis_end).value
    counts = np.zeros((len(windows), node_count), dtype=np.int64)
    for node, group in events.groupby('node_index'):
        times = np.sort(group.peak_time.to_numpy(dtype='datetime64[ns]').astype(np.int64))
        upper = np.searchsorted(times, ends, side='left')
        at_final = ends == final_ns
        upper[at_final] = np.searchsorted(times, ends[at_final], side='right')
        counts[:, int(node)] = upper - np.searchsorted(times, starts, side='left')
    return counts

def permutation_groups(nodes, events):
    rows = []
    groups = []
    event_counts = events.groupby('node_index').size()
    for group_id, (key, g) in enumerate(nodes.groupby('permutation_group', sort=True)):
        idx = g.node_index.to_numpy(int)
        groups.append(idx)
        rows.append(dict(group_id=group_id, permutation_group=key, node_count=len(idx), exchangeable=len(idx) > 1, node_indices=';'.join(map(str, idx)), node_ids=';'.join(g.node_id), merged_local_records=int(event_counts.reindex(idx, fill_value=0).sum())))
    return (groups, pd.DataFrame(rows))

def draw_permutations(groups, node_count, count, rng):
    permutations = np.tile(np.arange(node_count), (count, 1))
    for k in range(count):
        for g in groups:
            if len(g) > 1:
                permutations[k, g] = rng.permutation(g)
    return permutations

def moran_direct(values, left, right):
    x = np.atleast_2d(np.asarray(values, float))
    centered = x - x.mean(axis=1, keepdims=True)
    denominator = np.sum(centered ** 2, axis=1)
    result = np.full(len(x), np.nan)
    valid = denominator > 0
    result[valid] = x.shape[1] / len(left) * np.sum(centered[valid][:, left] * centered[valid][:, right], axis=1) / denominator[valid]
    return result

def permutation_moran(values, left, right, permutations, batch_size=200, label=''):
    """Exact binary-graph Moran computation; each permutation maps target->source histories."""
    values = np.asarray(values, float)
    n = values.shape[1]
    u, v = np.triu_indices(n, 1)
    lookup = np.full((n, n), -1, dtype=int)
    lookup[u, v] = np.arange(len(u))
    lookup[v, u] = np.arange(len(u))
    centered = values - values.mean(axis=1, keepdims=True)
    denom = np.sum(centered ** 2, axis=1)
    if np.any(denom <= 0):
        raise ValueError('Only variable windows may enter Moran calculation')
    products = centered[:, u] * centered[:, v] * (n / len(left) / denom[:, None])
    products = np.ascontiguousarray(products.T)
    scores = np.empty((len(permutations), len(values)), float)
    for begin in range(0, len(permutations), batch_size):
        p = permutations[begin:begin + batch_size]
        columns = lookup[p[:, left], p[:, right]]
        weights = np.zeros((len(p), len(u)), float)
        weights[np.arange(len(p))[:, None], columns] = 1
        scores[begin:begin + len(p)] = weights @ products
        if label:
            log(f'{label}: permutations {min(begin + len(p), len(permutations))}/{len(permutations)}')
    return scores

def max_scan_inference(scores, alpha):
    """Studentize symmetrically using observed+all random rows, then max over all windows.

    Symmetric scaling avoids fitting a reference only to random rows and treating the observed
    row differently. Ties count against rejection.
    """
    center = scores.mean(axis=0)
    scale = scores.std(axis=0, ddof=1)
    informative = scale > 1e-12
    z = np.zeros_like(scores)
    z[:, informative] = (scores[:, informative] - center[informative]) / scale[informative]
    if informative.any():
        maxima = z[:, informative].max(axis=1)
        sorted_null = np.sort(maxima[1:])
        p = (1 + len(sorted_null) - np.searchsorted(sorted_null, z[0] - 1e-12, side='left')) / (len(sorted_null) + 1)
        p[~informative] = 1
        global_p = float((1 + np.sum(maxima[1:] >= maxima[0] - 1e-12)) / len(maxima))
        critical = float(np.quantile(maxima[1:], 1 - alpha, method='higher'))
    else:
        maxima = np.zeros(len(scores))
        p = np.ones(scores.shape[1])
        global_p = 1.0
        critical = 0.0
    return dict(center=center, scale=scale, informative=informative, z_observed=z[0], maxima=maxima, p_within=p, p_study=p.copy(), global_p=global_p, global_p_study=global_p, critical=critical)

def detected_episodes(windows, alpha):
    """Group overlapping corrected-significant windows; report the strongest representative.

    Union bounds are descriptive support ranges, not a claim every instant is organized.
    The representative window itself has a scan- and study-corrected p-value.
    """
    columns = ['episode_id', 'representative_window_id', 'start_time', 'end_time', 'duration_minutes', 'observed_moran_i', 'null_mean', 'delta_moran_i', 'scan_z', 'p_scan_within_experiment', 'p_scan_study', 'support_union_start', 'support_union_end', 'support_union_duration_minutes', 'significant_window_count']
    sig = windows[windows.p_scan_study.le(alpha)].sort_values(['start_time', 'end_time'])
    groups = []
    current = []
    current_end = None
    for row in sig.to_dict('records'):
        if current and row['start_time'] > current_end:
            groups.append(current)
            current = []
            current_end = None
        current.append(row)
        current_end = row['end_time'] if current_end is None else max(current_end, row['end_time'])
    if current:
        groups.append(current)
    output = []
    for k, group in enumerate(groups, 1):
        best = sorted(group, key=lambda r: (-r['scan_z'], r['duration_minutes'], r['start_time']))[0]
        a = min((r['start_time'] for r in group))
        b = max((r['end_time'] for r in group))
        result = {c: best[c] for c in columns if c in best}
        result.update(episode_id=k, representative_window_id=int(best['window_id']), support_union_start=a, support_union_end=b, support_union_duration_minutes=(b - a).total_seconds() / 60, significant_window_count=len(group))
        output.append(result)
    return pd.DataFrame(output, columns=columns)

def validate_settings(s):
    if s.simulations < 19 or s.scan_step_seconds <= 0 or s.radius_cm <= 0 or (s.batch_size < 1):
        raise ValueError('Invalid simulation, step, radius, or batch configuration')
    if not s.window_minutes or any((float(v) <= 0 for v in s.window_minutes)):
        raise ValueError('Durations must be positive')
    if len(set(s.window_minutes)) != len(s.window_minutes):
        raise ValueError('Duplicate durations')
    if s.min_merged_event_records < 1 or s.min_active_nodes < 2 or (not 0 < s.alpha < 1) or (s.family_size < 1):
        raise ValueError('Invalid test settings')

def run_experiment(spec, settings, run_dir, make_figures=True):
    output = run_dir / spec.name
    output.mkdir(parents=True, exist_ok=False)
    log(f'{spec.name}: reading coordinates and event catalogue')
    layout, all_nodes, nodes = load_nodes(spec)
    edges, graph_audit = make_graph(nodes, settings.radius_cm)
    raw, events, membership, event_audit = load_events(spec, all_nodes, nodes, layout)
    groups, pools = permutation_groups(nodes, events)
    fixed = int(pools.loc[~pools.exchangeable, 'merged_local_records'].sum())
    write_csv(all_nodes, output / '01_all_coordinate_nodes.csv')
    write_csv(nodes, output / '02_selected_nodes.csv')
    write_csv(edges, output / '03_geometric_edges.csv')
    write_csv(pools, output / '04_permutation_groups.csv')
    write_csv(raw, output / '05_selected_source_records.csv')
    write_csv(events, output / '06_merged_local_records.csv')
    write_csv(membership, output / '07_merge_membership.csv')
    windows = candidate_windows(spec.start, spec.end, settings)
    counts = window_event_counts(events, windows, len(nodes), spec.end)
    windows['merged_event_records'] = counts.sum(axis=1)
    windows['active_nodes'] = (counts > 0).sum(axis=1)
    eligible = (windows.merged_event_records >= settings.min_merged_event_records) & (windows.active_nodes >= settings.min_active_nodes) & (counts.var(axis=1) > 0)
    windows['eligible'] = eligible
    windows['information_status'] = np.where(eligible, 'eligible', 'insufficient_merged_records_nodes_or_variation')
    float_columns = ['observed_moran_i', 'null_mean', 'null_sd', 'null_q025', 'null_q975', 'delta_moran_i', 'scan_z', 'p_scan_within_experiment', 'p_scan_study', 'simultaneous_upper_study', 'symmetric_center', 'symmetric_sd']
    for column in float_columns:
        windows[column] = np.nan
    seed = settings.seed + {'S1': 1, 'S2': 2, 'S4': 4}[spec.name]
    rng = np.random.default_rng(seed)
    permutations = np.vstack([np.arange(len(nodes)), draw_permutations(groups, len(nodes), settings.simulations, rng)])
    global_p = global_study = None
    critical = None
    maxima = np.array([])
    best = None
    if eligible.any():
        matrix = counts[eligible]
        scores = permutation_moran(matrix, edges.node_i.to_numpy(), edges.node_j.to_numpy(), permutations, settings.batch_size, spec.name)
        np.testing.assert_allclose(scores[0], moran_direct(matrix, edges.node_i.to_numpy(), edges.node_j.to_numpy()), atol=2e-12)
        infer = max_scan_inference(scores, settings.alpha)
        mean = scores[1:].mean(axis=0)
        data = dict(observed_moran_i=scores[0], null_mean=mean, null_sd=scores[1:].std(axis=0, ddof=1), null_q025=np.quantile(scores[1:], 0.025, axis=0), null_q975=np.quantile(scores[1:], 0.975, axis=0), delta_moran_i=scores[0] - mean, scan_z=infer['z_observed'], p_scan_within_experiment=infer['p_within'], p_scan_study=infer['p_study'], simultaneous_upper_study=infer['center'] + infer['critical'] * infer['scale'], symmetric_center=infer['center'], symmetric_sd=infer['scale'])
        for column, value in data.items():
            windows.loc[eligible, column] = value
        eligible_indices = np.flatnonzero(eligible)
        windows.loc[eligible_indices[~infer['informative']], 'information_status'] = 'no_permutation_variation'
        global_p = infer['global_p']
        global_study = infer['global_p_study']
        critical = infer['critical']
        maxima = infer['maxima']
        informative_indices = eligible_indices[infer['informative']]
        if len(informative_indices):
            best = int(windows.loc[informative_indices, 'scan_z'].idxmax())
        del scores
    episodes = detected_episodes(windows, settings.alpha)
    windows['significant_study'] = windows.p_scan_study.le(settings.alpha)
    write_csv(windows, output / '08_all_scan_windows.csv')
    write_csv(episodes, output / '09_detected_episodes.csv')
    write_csv(pd.DataFrame({'permutation_id': np.arange(len(maxima)), 'is_observed': np.arange(len(maxima)) == 0, 'maximum_scan_z': maxima}), output / '10_maximum_null_distribution.csv')
    np.savez_compressed(output / '11_reproducibility_arrays.npz', permutations=permutations.astype(np.int16), window_merged_event_counts=counts.astype(np.int32), maximum_scan_z=maxima)
    summary = dict(experiment=spec.name, main_variable='window_count_of_merged_local_event_records', event_set=spec.event_set, selected_nodes=len(nodes), edges=len(edges), selected_raw_records=len(raw), merged_local_records=len(events), fixed_node_record_fraction=fixed / len(events), scan_candidates=len(windows), eligible_candidates=int(eligible.sum()), simulations=settings.simulations, global_p_scan=global_p, global_p_study=global_study, evidence_status='insufficient_test_information' if best is None else 'above_random_spatial_organization' if global_study <= settings.alpha else 'not_detected', detected_episode_groups=len(episodes), critical_max_z_study=critical, strongest_start=None, strongest_end=None, strongest_duration_minutes=None, strongest_moran_i=None, strongest_null_mean=None, strongest_delta_moran_i=None, strongest_scan_z=None)
    if best is not None:
        b = windows.loc[best]
        summary.update(strongest_start=str(b.start_time), strongest_end=str(b.end_time), strongest_duration_minutes=float(b.duration_minutes), strongest_moran_i=float(b.observed_moran_i), strongest_null_mean=float(b.null_mean), strongest_delta_moran_i=float(b.delta_moran_i), strongest_scan_z=float(b.scan_z))
        strongest = nodes.copy()
        strongest['merged_local_event_records'] = counts[best]
        write_csv(strongest, output / '12_strongest_window_node_event_counts.csv')
    manifest = dict(analysis='Near-surface multiscale merged-event-frequency spatial organization scan', settings=asdict(settings), experiment=asdict(spec), effective_seed=seed, script_sha256=sha256(__file__), inputs={'coordinates_sha256': sha256(spec.coordinate_file), 'events_sha256': sha256(spec.event_file)}, graph=graph_audit, event_audit=event_audit, summary=summary, assumptions=['Whole histories are exchangeable within geometry/stratum/gauge-opportunity groups under the null.', 'This conditional reference does not prove independent DAS measurement support or exclude all coupling/response biases.', 'The ROI and scan settings were specified after earlier exploratory analyses; this is a retrospective analysis.', 'Time-window support and aggregated spatial association do not establish physical transition times, propagation, damage coalescence or predictive lead time.'], definitions={'main_variable': 'Per-node count of merged local event records whose representative peak falls within the scan window; every record has weight 1 and activity amplitude is not summed.', 'window': '[start,end), except that an end equal to the fixed analysis cutoff includes records exactly at that cutoff.', 'eligibility': 'At least 5 merged local event records, at least 3 active nodes and nonzero spatial count variance.', 'graph_weight': '1 for 0<3D distance<=35 cm (or configured radius), else 0; no self edges', 'moran': 'n/E * sum_(unordered graph edges)(x_i-xbar)(x_j-xbar) / sum_i(x_i-xbar)^2', 'scan': 'Standardize each eligible window symmetrically across observed and randomized fields; maximum over every scanned duration and start time.', 'p': '(1 + number of randomized scan maxima >= observed score)/(B+1); Holm across planned 3 experiments.', 'episodes': 'Overlapping significant scan windows grouped; strongest corrected-significant window is the representative. Union bounds are descriptive, not true continuous-state boundaries.'})
    write_json(output / 'manifest.json', manifest)
    result = dict(spec=spec, nodes=nodes, edges=edges, windows=windows, episodes=episodes, maxima=maxima, critical=critical, best=best, counts=counts, summary=summary, output=output)
    if make_figures:
        from organization_plots import plot_experiment
        plot_experiment(result, settings, PNG_DPI)
    log(f'{spec.name}: global scan p={global_p}')
    return result

def holm_adjusted(p_values):
    p_values = np.asarray(p_values, float)
    order = np.argsort(p_values, kind='stable')
    adjusted = np.empty(len(order), float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[index]))
        adjusted[index] = running
    return (order, adjusted)

def apply_holm(results, settings):
    order, adjusted = holm_adjusted([1.0 if r['summary']['global_p_scan'] is None else r['summary']['global_p_scan'] for r in results])
    previous = 0.0
    for rank, index in enumerate(order):
        result = results[index]
        windows = result['windows']
        divisor = len(results) - rank
        p = windows['p_scan_within_experiment']
        valid = p.notna()
        windows.loc[valid, 'p_scan_study'] = np.minimum(1.0, np.maximum(previous, divisor * p[valid]))
        critical = None
        if len(result['maxima']) and previous <= settings.alpha:
            critical = float(np.quantile(result['maxima'][1:], 1 - settings.alpha / divisor, method='higher'))
            windows.loc[valid, 'simultaneous_upper_study'] = windows.loc[valid, 'symmetric_center'] + critical * windows.loc[valid, 'symmetric_sd']
        else:
            windows.loc[valid, 'simultaneous_upper_study'] = np.nan
        windows['significant_study'] = windows.p_scan_study.le(settings.alpha)
        episodes = detected_episodes(windows, settings.alpha)
        summary = result['summary']
        summary['global_p_study'] = float(adjusted[index])
        summary['critical_max_z_study'] = critical
        summary['detected_episode_groups'] = len(episodes)
        summary['evidence_status'] = 'insufficient_test_information' if result['best'] is None else 'above_random_spatial_organization' if adjusted[index] <= settings.alpha else 'not_detected'
        result['episodes'] = episodes
        result['critical'] = critical
        output = result['output']
        write_csv(windows, output / '08_all_scan_windows.csv')
        write_csv(episodes, output / '09_detected_episodes.csv')
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        manifest['summary'] = summary
        write_json(output / 'manifest.json', manifest)
        previous = float(adjusted[index])

def write_study_report(results, settings, run_dir):
    summary = pd.DataFrame([r['summary'] for r in results])
    write_csv(summary, run_dir / 'study_summary.csv')
    all_episodes = []
    for r in results:
        e = r['episodes'].copy()
        e.insert(0, 'experiment', r['spec'].name)
        all_episodes.append(e)
    write_csv(pd.concat(all_episodes, ignore_index=True), run_dir / 'study_detected_episodes.csv')
    lines = ['# 三组近表层事件频次空间组织扫描结果', '', '主变量：每个扫描窗口内各节点的合并局部事件记录数。每条记录权重为1，不使用事件activity作为权重或求和。', '同一节点上相交或接触的源事件区间先合并；窗口按代表峰时计数。', '核心问题：局部事件发生频次是否存在超出约束随机排列预期的空间组织；若存在，被支持的时间尺度和强度如何。', '', f'模拟数：{settings.simulations}；扫描步长：{settings.scan_step_seconds} s；片段长度：{settings.window_minutes} min。', f'主连接：{settings.radius_cm:g} cm三维半径内二元几何邻接。整体显著性水平{settings.alpha}，按计划{settings.family_size}组统一校正。', '', '结果均限定于当前事件筛选、合并规则、选区、测量尺度和可交换性假设。未检出不等于证明不存在。', '检测片段是聚合活动的统计支持区间，其边界不是精确物理转变时刻，也不是实时报警时间。', '事件频次的空间自相关不等于活动贯通、裂纹连接或滑面形成。', '相邻节点存在标距重叠；已审计但未宣称其独立或排除全部仪器响应偏差。', '']
    for r in results:
        s = r['summary']
        lines.extend([f"## {s['experiment']}", f"- 选区源记录 {s['selected_raw_records']}；同节点重叠合并后 {s['merged_local_records']}；通过信息条件的窗口 {s['eligible_candidates']} 个。", f"- 状态：{s['evidence_status']}；全扫描p={s['global_p_scan']}；三实验校正p={s['global_p_study']}。", f"- 得到支持的片段组数：{s['detected_episode_groups']}。"])
        if s['strongest_start'] is not None:
            lines.append(f"- 最强候选（显著性由上述p判断）：{s['strongest_start']} 至 {s['strongest_end']}；{s['strongest_duration_minutes']:g} min；I={s['strongest_moran_i']:.5f}，随机均值={s['strongest_null_mean']:.5f}，ΔI={s['strongest_delta_moran_i']:.5f}。")
        lines.append('')
    lines.extend(['## 图件排版', '', '单图84 mm，双图组合和宽图168 mm；PDF矢量输出，PNG为600 dpi。Times New Roman：轴标题、刻度、图标题10 pt，图例/拥挤标注8 pt。', '168 mm适合A4有效正文宽度至少168 mm的页面（例如左右页边距各20 mm）。若正文宽度更小，排版软件仍会缩放，需调整页边距或代码宽度。', '保存时不使用bbox_inches=tight，避免改变物理宽度。figure_quality.csv记录导出宽度和字体检查。'])
    (run_dir / '结果说明.md').write_text('\n'.join(lines), encoding='utf-8')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs='+', choices=sorted(CONFIG), default=list(EXPERIMENTS_TO_RUN))
    parser.add_argument('--simulations', type=int, default=SIMULATIONS)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    settings = Settings(simulations=args.simulations)
    validate_settings(settings)
    if len(set(args.experiments)) != len(args.experiments):
        parser.error('Duplicate experiment names')
    if set(args.experiments) != set(CONFIG):
        parser.error('Holm study correction requires S1, S2, and S4')
    run_dir = args.output_root / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    make_figures = SAVE_FIGURES and (not args.no_plots)
    if make_figures:
        from organization_plots import configure_font
        configure_font()
    started = time.perf_counter()
    results = []
    write_json(run_dir / 'run_status.json', dict(status='running', settings=asdict(settings), experiments=args.experiments))
    try:
        for name in args.experiments:
            results.append(run_experiment(CONFIG[name], settings, run_dir, False))
        apply_holm(results, settings)
        if make_figures:
            from organization_plots import plot_experiment, plot_study, write_figure_quality
            for result in results:
                plot_experiment(result, settings, PNG_DPI)
            plot_study(results, settings, run_dir, PNG_DPI)
            write_figure_quality(run_dir)
        write_study_report(results, settings, run_dir)
    except Exception as error:
        write_json(run_dir / 'run_status.json', dict(status='failed', error=repr(error), settings=asdict(settings)))
        raise
    write_json(run_dir / 'run_status.json', dict(status='complete', settings=asdict(settings), experiments=args.experiments, elapsed_seconds=time.perf_counter() - started))
    print(f'OUTPUT_DIRECTORY={run_dir}', flush=True)
    return run_dir
if __name__ == '__main__':
    main()
