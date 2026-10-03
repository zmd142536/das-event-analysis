"""DAS processing functions for screening refiner."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
from collections import defaultdict
import argparse
import hashlib
import json
import math
import re
import warnings
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import spectrogram
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
PROJECT_DIR = REPOSITORY_ROOT
H5_INPUT_DIR = REPOSITORY_ROOT / 'data/raw/S1'
LAYOUT_XLSX = REPOSITORY_ROOT / 'data/coordinates/S1_coordinates.xlsx'
BASE_RESULT_DIR = REPOSITORY_ROOT / 'data/screening_inputs/S1'
OUTPUT_DIR = REPOSITORY_ROOT / 'outputs/screening/S1'
FIGURE_DIR = OUTPUT_DIR / 'paper_figures_A4'
REVIEW_DIR = OUTPUT_DIR / 'manual_review_panels'
H5_DATASET = '/default'
SAMPLE_RATE_HZ = 5000.0
FILE_DURATION_S = 600.0
DATA_CHANNEL_COUNT = 500
LAYOUT_TO_DATA_OFFSET = 3000
GAUGE_LENGTH_M = 1.6
CHANNEL_SPACING_M = 0.2
GAUGE_INTERVALS = int(round(GAUGE_LENGTH_M / CHANNEL_SPACING_M))
GAUGE_HALF_CHANNELS = int(math.ceil(GAUGE_INTERVALS / 2.0))
GAUGE_EVENT_TIME_TOLERANCE_S = 0.6
ANALYSIS_SCALES_S = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
RAW_CONTEXT_S = 3.0
SEARCH_HALF_WIDTH_S = 0.5
BASELINE_EDGE_S = 0.45
SPECTRAL_MIN_HZ = 20.0
SPECTRAL_MAX_HZ = 2000.0
SPECTRAL_BAND_COUNT = 24
REFERENCE_COUNTS = {'stable_rain': 240, 'personnel': 180, 'pumping': 120}
NOISE_DISTANCE_QUANTILE = 0.995
MIN_NOVEL_SCALES = 2
MIN_CONSECUTIVE_NOVEL_SCALES = 2
MIN_REFERENCE_GROUP_SIZE = 24
K_NEIGHBORS = 10
TYPE_MIN_COUNT = 2
TYPE_MAX_COUNT = 6
TYPE_MIN_EVENTS = 25
REVIEW_PER_TYPE = 6
REVIEW_UNCERTAIN_COUNT = 8
RANDOM_SEED = 20260816
FORCE_REEXTRACT_RAW = False
FORCE_REBUILD_RESULTS = False
MM_TO_IN = 1.0 / 25.4
WIDE_WIDTH_IN = 168.0 * MM_TO_IN
NARROW_WIDTH_IN = 84.0 * MM_TO_IN
PNG_DPI = 600
KNOWN_NOISE_INTERVALS = [('2025-07-07 15:15:00', '2025-07-07 15:17:32', 'personnel'), ('2025-07-07 16:00:00', '2025-07-07 16:05:16', 'personnel'), ('2025-07-07 17:52:30', '2025-07-07 18:09:00', 'personnel'), ('2025-07-07 18:19:40', '2025-07-07 18:28:50', 'personnel'), ('2025-07-07 19:25:00', '2025-07-07 19:31:20', 'pumping'), ('2025-07-07 19:59:00', '2025-07-07 20:01:00', 'personnel'), ('2025-07-07 22:03:26', '2025-07-07 22:07:31', 'pumping')]
PERSONNEL_BUFFER_S = 15.0
PUMPING_BUFFER_S = 30.0
MACRO_OBSERVATION = pd.Timestamp('2025-07-07 22:38:39')
EXPERIMENT_START = pd.Timestamp('2025-07-07 14:06:54.427050')
COLORS = {'stable_rain': '#0072B2', 'personnel': '#D55E00', 'pumping': '#E69F00', 'effective_candidate': '#009E73', 'uncertain': '#777777', 'nuisance_like': '#CC79A7', 'macro': '#CC3311'}

def log(message: str) -> None:
    print(message, flush=True)

def setup_style() -> None:
    font_paths = [Path(mpl.font_manager.findfont('Times New Roman', fallback_to_default=False)), Path(mpl.font_manager.findfont('Times New Roman', fallback_to_default=False)), Path(mpl.font_manager.findfont('Times New Roman', fallback_to_default=False))]
    for path in font_paths:
        if path.exists():
            try:
                mpl.font_manager.fontManager.addfont(str(path))
            except Exception:
                pass
    mpl.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'figure.titlesize': 10, 'axes.linewidth': 0.8, 'lines.linewidth': 1.0, 'savefig.dpi': PNG_DPI, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontweight='bold', fontsize=10)

def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f'{stem}.png', dpi=PNG_DPI, bbox_inches='tight')
    fig.savefig(FIGURE_DIR / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)

def stable_hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()[:16]

def robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(values - center), axis=0)
    fallback = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-08), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-08), scale, 1.0)
    return (center, scale)

def spectral_edges() -> np.ndarray:
    return np.geomspace(SPECTRAL_MIN_HZ, SPECTRAL_MAX_HZ, SPECTRAL_BAND_COUNT + 1)

def band_centers() -> np.ndarray:
    edges = spectral_edges()
    return np.sqrt(edges[:-1] * edges[1:])

def operation_masks(times: pd.DatetimeIndex, buffered: bool=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    personnel = np.zeros(len(times), dtype=bool)
    pumping = np.zeros(len(times), dtype=bool)
    for start_text, end_text, kind in KNOWN_NOISE_INTERVALS:
        extra = 0.0
        if buffered:
            extra = PERSONNEL_BUFFER_S if kind == 'personnel' else PUMPING_BUFFER_S
        start = pd.Timestamp(start_text) - pd.Timedelta(seconds=extra)
        end = pd.Timestamp(end_text) + pd.Timedelta(seconds=extra)
        mask = (times >= start) & (times <= end)
        if kind == 'personnel':
            personnel |= mask
        else:
            pumping |= mask
    return (personnel, pumping, personnel | pumping)

def mask_single_time(time: pd.Timestamp, buffered: bool=True) -> tuple[bool, str]:
    for start_text, end_text, kind in KNOWN_NOISE_INTERVALS:
        extra = 0.0
        if buffered:
            extra = PERSONNEL_BUFFER_S if kind == 'personnel' else PUMPING_BUFFER_S
        if pd.Timestamp(start_text) - pd.Timedelta(seconds=extra) <= time <= pd.Timestamp(end_text) + pd.Timedelta(seconds=extra):
            return (True, kind)
    return (False, 'none')

def parse_semicolon_ints(value: object) -> list[int]:
    if pd.isna(value):
        return []
    result = []
    for token in str(value).split(';'):
        token = token.strip()
        if token:
            try:
                result.append(int(float(token)))
            except ValueError:
                continue
    return result

class UnionFind:

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=int)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = (self.find(a), self.find(b))
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = (rb, ra)
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

@dataclass(frozen=True)
class H5Record:
    path: Path
    start: pd.Timestamp

def list_h5_records() -> list[H5Record]:
    pattern = re.compile('(20\\d{6})-(\\d{6})\\.(\\d+)(?:\\+\\d{4})?_')
    records: list[H5Record] = []
    for path in sorted(H5_INPUT_DIR.glob('*.h5')):
        match = pattern.search(path.name)
        if not match:
            continue
        stamp = pd.to_datetime(match.group(1) + match.group(2), format='%Y%m%d%H%M%S')
        fraction = match.group(3)[:9].ljust(9, '0')
        stamp += pd.Timedelta(nanoseconds=int(fraction))
        records.append(H5Record(path=path, start=stamp))
    if not records:
        raise FileNotFoundError(f'未在 {H5_INPUT_DIR} 找到可解析时间的H5文件')
    records.sort(key=lambda item: item.start)
    return records

def locate_h5_record(time: pd.Timestamp, records: list[H5Record]) -> tuple[int, H5Record, float]:
    starts = np.asarray([item.start.value for item in records], dtype=np.int64)
    index = int(np.searchsorted(starts, time.value, side='right') - 1)
    if index < 0:
        raise ValueError(f'时间 {time} 早于首个H5文件')
    local_second = (time - records[index].start).total_seconds()
    if not 0.0 <= local_second < FILE_DURATION_S + 0.05:
        raise ValueError(f'时间 {time} 不在文件 {records[index].path.name} 的600 s范围内')
    return (index, records[index], local_second)

def stage_for_times(times: pd.Series, blocks: pd.DataFrame) -> np.ndarray:
    starts = pd.to_datetime(blocks['start_time']).astype('int64').to_numpy()
    ends = pd.to_datetime(blocks['end_time']).astype('int64').to_numpy()
    stage = blocks['stage_id'].astype(int).to_numpy()
    values = pd.to_datetime(times).astype('int64').to_numpy()
    result = np.full(len(values), -1, dtype=int)
    for index, value in enumerate(values):
        pos = int(np.searchsorted(starts, value, side='right') - 1)
        if pos >= 0 and value <= ends[pos] + int(1000000000.0):
            result[index] = stage[pos]
        elif pos >= 0:
            result[index] = stage[pos]
        else:
            result[index] = stage[0]
    return result

def validate_base_inputs() -> None:
    required = [BASE_RESULT_DIR / 'latest_open_set_event_catalog.csv', BASE_RESULT_DIR / 'latest_event_members.csv', BASE_RESULT_DIR / 'monitoring_channel_layout.csv', BASE_RESULT_DIR / 'rainfall_stage_blocks.csv', BASE_RESULT_DIR / 'all_shallow_representation_cache.h5']
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('缺少第二段代码的已有结果，请先运行 S1_noise_candidate_data_driven_source_analysis.py：\n' + '\n'.join(missing))

def load_base_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_base_inputs()
    events = pd.read_csv(BASE_RESULT_DIR / 'latest_open_set_event_catalog.csv')
    members = pd.read_csv(BASE_RESULT_DIR / 'latest_event_members.csv')
    metadata = pd.read_csv(BASE_RESULT_DIR / 'monitoring_channel_layout.csv')
    blocks = pd.read_csv(BASE_RESULT_DIR / 'rainfall_stage_blocks.csv')
    for column in ['start_time', 'end_time', 'peak_time']:
        events[column] = pd.to_datetime(events[column])
    members['time'] = pd.to_datetime(members['time'])
    for column in ['start_time', 'end_time', 'center_time']:
        blocks[column] = pd.to_datetime(blocks[column])
    metadata['data_channel'] = pd.to_numeric(metadata['data_channel'], errors='coerce').astype('Int64')
    metadata['layout_channel'] = pd.to_numeric(metadata['layout_channel'], errors='coerce').astype('Int64')
    metadata['burial_depth_cm'] = pd.to_numeric(metadata['burial_depth_cm'], errors='coerce')
    return (events, members, metadata, blocks)

def interval_gap_s(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> float:
    if a_end >= b_start and b_end >= a_start:
        return 0.0
    if a_end < b_start:
        return float((b_start - a_end).total_seconds())
    return float((a_start - b_end).total_seconds())

def gauge_merge_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把标距重叠且时间重叠的粗事件合并为一个物理候选，不把8通道当作8个独立事件。"""
    work = events.sort_values('peak_time').reset_index(drop=True).copy()
    n = len(work)
    uf = UnionFind(n)
    peak_ns = np.asarray([pd.Timestamp(value).value for value in work['peak_time']], dtype=np.int64)
    channels = pd.to_numeric(work['source_center_data_channel'], errors='coerce').to_numpy(float)
    starts = work['start_time'].tolist()
    ends = work['end_time'].tolist()
    tolerance_ns = int(GAUGE_EVENT_TIME_TOLERANCE_S * 1000000000.0)
    left = 0
    for i in range(n):
        while peak_ns[i] - peak_ns[left] > tolerance_ns:
            left += 1
        for j in range(left, i):
            if not (np.isfinite(channels[i]) and np.isfinite(channels[j])):
                continue
            if abs(channels[i] - channels[j]) > GAUGE_INTERVALS:
                continue
            peak_gap = abs(peak_ns[i] - peak_ns[j]) / 1000000000.0
            if peak_gap <= GAUGE_EVENT_TIME_TOLERANCE_S:
                uf.union(i, j)
    roots = np.asarray([uf.find(i) for i in range(n)], dtype=int)
    codes, _ = pd.factorize(roots)
    member_rows: list[dict] = []
    event_rows: list[dict] = []
    for event_number, code in enumerate(np.unique(codes), start=1):
        indices = np.where(codes == code)[0]
        group = work.iloc[indices]
        peak_row = group.loc[group['max_anomaly_score'].astype(float).idxmax()]
        source_data_channels: set[int] = set()
        source_layout_channels: set[int] = set()
        for value in group['source_channels']:
            source_layout_channels.update(parse_semicolon_ints(value))
        for value in source_layout_channels:
            source_data_channels.add(value - LAYOUT_TO_DATA_OFFSET)
        if not source_data_channels:
            source_data_channels.update(group['source_center_data_channel'].dropna().astype(int).tolist())
        physical_id = f'GME_{event_number:05d}'
        min_channel = min(source_data_channels) if source_data_channels else int(peak_row['source_center_data_channel'])
        max_channel = max(source_data_channels) if source_data_channels else int(peak_row['source_center_data_channel'])
        exact_overlap, exact_kind = mask_single_time(pd.Timestamp(peak_row['peak_time']), buffered=False)
        buffered_overlap, buffered_kind = mask_single_time(pd.Timestamp(peak_row['peak_time']), buffered=True)
        event_rows.append({'physical_event_id': physical_id, 'start_time': group['start_time'].min(), 'end_time': group['end_time'].max(), 'peak_time': pd.Timestamp(peak_row['peak_time']), 'representative_data_channel': int(peak_row['source_center_data_channel']), 'representative_layout_channel': int(peak_row['source_center_layout_channel']), 'representative_sensor_id': str(peak_row['source_center_sensor_id']), 'representative_segment_id': str(peak_row['source_center_segment_id']), 'geometry_type': str(peak_row['source_center_geometry']).lower(), 'burial_depth_cm': float(peak_row['peak_depth_cm']), 'x_cm': float(peak_row.get('source_center_x_cm', np.nan)), 'y_cm': float(peak_row.get('source_center_y_cm', np.nan)), 'stage_id': int(peak_row['stage_id']), 'max_anomaly_score_1s': float(group['max_anomaly_score'].max()), 'median_anomaly_score_1s': float(group['median_anomaly_score'].median()), 'strict_core_1s': bool(group['strict_core_event'].astype(bool).any()), 'base_known_human_like': bool(group['known_human_like'].astype(bool).any()), 'base_event_count_merged': int(len(group)), 'base_event_ids': ';'.join(group['event_id'].astype(str)), 'source_data_channels': ';'.join((str(value) for value in sorted(source_data_channels))), 'source_layout_channels': ';'.join((str(value) for value in sorted(source_layout_channels))), 'source_min_data_channel': int(min_channel), 'source_max_data_channel': int(max_channel), 'source_channel_span_intervals': int(max_channel - min_channel), 'gauge_equivalent_count': float(max(1.0, (max_channel - min_channel + 1) / max(GAUGE_INTERVALS, 1))), 'exact_operation_overlap': bool(exact_overlap), 'exact_operation_kind': exact_kind, 'buffered_operation_overlap': bool(buffered_overlap), 'buffered_operation_kind': buffered_kind, 'seconds_from_macro_observation': float((pd.Timestamp(peak_row['peak_time']) - MACRO_OBSERVATION).total_seconds())})
        for row in group.itertuples(index=False):
            member_rows.append({'physical_event_id': physical_id, 'base_event_id': row.event_id, 'base_peak_time': row.peak_time, 'base_source_center_data_channel': row.source_center_data_channel, 'base_source_channels': row.source_channels, 'base_max_anomaly_score': row.max_anomaly_score})
    result = pd.DataFrame(event_rows).sort_values('peak_time').reset_index(drop=True)
    return (result, pd.DataFrame(member_rows))

def read_cache_times() -> pd.DatetimeIndex:
    cache = BASE_RESULT_DIR / 'all_shallow_representation_cache.h5'
    with h5py.File(cache, 'r') as h5:
        time_ns = np.asarray(h5['time_ns'], dtype=np.int64)
    return pd.to_datetime(time_ns)

def stratified_sensor_sample(metadata: pd.DataFrame, count: int, rng: np.random.Generator) -> pd.DataFrame:
    target = metadata[metadata['analysis_role'].astype(str).str.contains('target', case=False, na=False) & metadata['geometry_type'].astype(str).str.lower().isin(['ring', 'line']) & metadata['data_channel'].notna()].copy()
    target['stratum'] = target['burial_depth_cm'].round(3).astype(str) + '|' + target['geometry_type'].astype(str).str.lower()
    groups = [group for _, group in target.groupby('stratum')]
    rows = []
    for i in range(count):
        group = groups[i % len(groups)]
        rows.append(group.iloc[int(rng.integers(0, len(group)))])
    return pd.DataFrame(rows).reset_index(drop=True)

def stratified_time_sample(times: pd.DatetimeIndex, blocks: pd.DataFrame, class_label: str, count: int, rng: np.random.Generator) -> pd.DatetimeIndex:
    personnel, pumping, known = operation_masks(times, buffered=False)
    _, _, buffered = operation_masks(times, buffered=True)
    early = times < times[0] + pd.Timedelta(minutes=30)
    if class_label == 'stable_rain':
        allowed = ~(buffered | early)
    elif class_label == 'personnel':
        allowed = personnel
    elif class_label == 'pumping':
        allowed = pumping
    else:
        raise ValueError(class_label)
    candidate = np.where(allowed)[0]
    if len(candidate) == 0:
        raise RuntimeError(f'{class_label} 没有可抽取时间')
    if class_label == 'stable_rain':
        stage = stage_for_times(pd.Series(times), blocks)
        selected = []
        stage_ids = sorted(np.unique(stage[candidate]))
        for i in range(count):
            local = candidate[stage[candidate] == stage_ids[i % len(stage_ids)]]
            selected.append(int(rng.choice(local)))
        return times[np.asarray(selected, dtype=int)]
    replace = len(candidate) < count
    return times[rng.choice(candidate, size=count, replace=replace)]

def build_feature_tasks(physical_events: pd.DataFrame, metadata: pd.DataFrame, blocks: pd.DataFrame, max_events: int | None) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    events = physical_events.copy()
    if max_events is not None and len(events) > max_events:
        strict = events[events['strict_core_1s']].sort_values('peak_time')
        other = events[~events['strict_core_1s']].sort_values('peak_time')
        keep_strict = min(len(strict), max_events // 2)

        def evenly(group: pd.DataFrame, number: int) -> pd.DataFrame:
            if number <= 0 or group.empty:
                return group.iloc[:0]
            index = np.linspace(0, len(group) - 1, min(number, len(group))).round().astype(int)
            return group.iloc[index]
        events = pd.concat([evenly(strict, keep_strict), evenly(other, max_events - keep_strict)]).drop_duplicates('physical_event_id')
    event_tasks = pd.DataFrame({'task_id': 'candidate:' + events['physical_event_id'].astype(str), 'task_class': 'candidate', 'physical_event_id': events['physical_event_id'].astype(str), 'time': pd.to_datetime(events['peak_time']), 'data_channel': events['representative_data_channel'].astype(int), 'geometry_type': events['geometry_type'].astype(str).str.lower(), 'burial_depth_cm': events['burial_depth_cm'].astype(float), 'segment_id': events['representative_segment_id'].astype(str), 'stage_id': events['stage_id'].astype(int)})
    cache_times = read_cache_times()
    reference_rows = []
    for label, count in REFERENCE_COUNTS.items():
        sampled_times = stratified_time_sample(cache_times, blocks, label, count, rng)
        sampled_sensors = stratified_sensor_sample(metadata, count, rng)
        stages = stage_for_times(pd.Series(sampled_times), blocks)
        for i in range(count):
            sensor = sampled_sensors.iloc[i]
            reference_rows.append({'task_id': f'{label}:{i:05d}', 'task_class': label, 'physical_event_id': '', 'time': pd.Timestamp(sampled_times[i]), 'data_channel': int(sensor['data_channel']), 'geometry_type': str(sensor['geometry_type']).lower(), 'burial_depth_cm': float(sensor['burial_depth_cm']), 'segment_id': str(sensor['segment_id']), 'stage_id': int(stages[i])})
    tasks = pd.concat([event_tasks, pd.DataFrame(reference_rows)], ignore_index=True)
    tasks['time'] = pd.to_datetime(tasks['time'])
    return tasks

def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.maximum(np.asarray(p, dtype=float), 1e-15)
    q = np.maximum(np.asarray(q, dtype=float), 1e-15)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    value = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(math.sqrt(max(value, 0.0)))

def power_band_fractions(x: np.ndarray) -> np.ndarray:
    """x: sample x channel。跨局部通道平均功率后得到24个对数频带的归一化能量。"""
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=np.finfo(np.float32).max, neginf=-np.finfo(np.float32).max)
    if x.ndim == 1:
        x = x[:, None]
    x = x - np.mean(x, axis=0, keepdims=True)
    window = np.hanning(len(x)).astype(np.float64)[:, None]
    fft = rfft(x * window, axis=0, workers=-1)
    power = np.mean(fft.real * fft.real + fft.imag * fft.imag, axis=1)
    frequency = rfftfreq(len(x), d=1.0 / SAMPLE_RATE_HZ)
    values = []
    for low, high in zip(spectral_edges()[:-1], spectral_edges()[1:]):
        mask = (frequency >= low) & (frequency < high)
        values.append(float(np.sum(power[mask])) if np.any(mask) else 0.0)
    values = np.asarray(values, dtype=float)
    total = float(np.sum(values))
    if total <= 0:
        return np.full(SPECTRAL_BAND_COUNT, 1.0 / SPECTRAL_BAND_COUNT)
    return values / total

def median_abs_correlation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or x.shape[0] < 8:
        return 0.0
    x = x - np.mean(x, axis=0, keepdims=True)
    scale = np.std(x, axis=0)
    valid = scale > 1e-12
    if np.sum(valid) < 2:
        return 0.0
    x = x[:, valid] / scale[valid]
    corr = np.corrcoef(x, rowvar=False)
    upper = np.abs(corr[np.triu_indices_from(corr, k=1)])
    return float(np.nanmedian(upper)) if len(upper) else 0.0

def extract_task_multiscale(block: np.ndarray, block_time_s: np.ndarray, task: pd.Series, metadata: pd.DataFrame) -> list[dict]:
    center_channel = int(task['data_channel'])
    gauge_channels = np.arange(max(0, center_channel - GAUGE_HALF_CHANNELS), min(block.shape[1] - 1, center_channel + GAUGE_HALF_CHANNELS) + 1, dtype=int)
    target_meta = metadata[metadata['data_channel'].notna() & metadata['analysis_role'].astype(str).str.contains('target', case=False, na=False)].copy()
    network_channels = target_meta['data_channel'].astype(int).to_numpy()
    same_group = target_meta[np.isclose(target_meta['burial_depth_cm'].astype(float), float(task['burial_depth_cm']), atol=0.1) & target_meta['geometry_type'].astype(str).str.lower().eq(str(task['geometry_type']).lower())]['data_channel'].astype(int).to_numpy()
    nonlocal_channels = same_group[np.abs(same_group - center_channel) > GAUGE_INTERVALS]
    if len(nonlocal_channels) < 8:
        nonlocal_channels = network_channels[np.abs(network_channels - center_channel) > 2 * GAUGE_INTERVALS]
    control_channels = metadata[metadata['analysis_role'].astype(str).str.contains('control', case=False, na=False) & metadata['data_channel'].notna()]['data_channel'].astype(int).to_numpy()
    selected_channels = np.unique(np.concatenate([gauge_channels, nonlocal_channels, control_channels]))
    selected_channels = selected_channels[(selected_channels >= 0) & (selected_channels < block.shape[1])]
    selected = np.asarray(block[:, selected_channels], dtype=np.float64)
    selected = np.nan_to_num(selected, nan=0.0, posinf=np.finfo(np.float32).max, neginf=-np.finfo(np.float32).max)
    selected -= np.median(selected, axis=0, keepdims=True)
    channel_to_local = {int(channel): index for index, channel in enumerate(selected_channels)}
    gauge_local = np.asarray([channel_to_local[int(channel)] for channel in gauge_channels if int(channel) in channel_to_local], dtype=int)
    nonlocal_local = np.asarray([channel_to_local[int(channel)] for channel in nonlocal_channels if int(channel) in channel_to_local], dtype=int)
    control_local = np.asarray([channel_to_local[int(channel)] for channel in control_channels if int(channel) in channel_to_local], dtype=int)
    baseline_mask = (block_time_s <= -RAW_CONTEXT_S / 2 + BASELINE_EDGE_S) | (block_time_s >= RAW_CONTEXT_S / 2 - BASELINE_EDGE_S)
    baseline = selected[baseline_mask]
    baseline_rms = np.sqrt(np.mean(baseline * baseline, axis=0) + 1e-20)
    baseline_psd = power_band_fractions(baseline[:, gauge_local])
    rows: list[dict] = []
    for scale in ANALYSIS_SCALES_S:
        half = scale / 2.0
        step = max(0.025, scale / 4.0)
        offsets = np.arange(-SEARCH_HALF_WIDTH_S, SEARCH_HALF_WIDTH_S + 0.5 * step, step)
        best: dict | None = None
        for offset in offsets:
            mask = (block_time_s >= offset - half) & (block_time_s < offset + half)
            if np.sum(mask) < max(16, int(round(scale * SAMPLE_RATE_HZ * 0.8))):
                continue
            window = selected[mask]
            rms = np.sqrt(np.mean(window * window, axis=0) + 1e-20)
            log_ratio = np.log(np.maximum(rms / baseline_rms, 1e-12))
            local_gain = float(np.nanmedian(log_ratio[gauge_local]))
            nonlocal_gain = float(np.nanmedian(log_ratio[nonlocal_local])) if len(nonlocal_local) else 0.0
            locality = local_gain - nonlocal_gain
            score = local_gain + 0.5 * max(locality, 0.0)
            if best is None or score > best['search_score']:
                best = {'offset': float(offset), 'mask': mask, 'window': window, 'log_ratio': log_ratio, 'local_gain': local_gain, 'nonlocal_gain': nonlocal_gain, 'locality': locality, 'search_score': score}
        if best is None:
            continue
        local_window = best['window'][:, gauge_local]
        psd = power_band_fractions(local_window)
        entropy = float(-np.sum(psd * np.log(np.maximum(psd, 1e-15))) / np.log(len(psd)))
        spectral_js = js_distance(psd, baseline_psd)
        local_log_ratio = best['log_ratio'][gauge_local]
        nonlocal_log_ratio = best['log_ratio'][nonlocal_local] if len(nonlocal_local) else np.asarray([0.0])
        nonlocal_center = float(np.nanmedian(nonlocal_log_ratio))
        nonlocal_scale = 1.4826 * float(np.nanmedian(np.abs(nonlocal_log_ratio - nonlocal_center)))
        support_threshold = nonlocal_center + max(nonlocal_scale, 0.05)
        support_fraction = float(np.mean(local_log_ratio > support_threshold))
        control_gain = float(np.nanmedian(best['log_ratio'][control_local])) if len(control_local) else np.nan
        frame_count = min(10, max(2, int(round(scale / 0.025))))
        usable = len(local_window) // frame_count * frame_count
        if usable >= frame_count:
            frames = local_window[:usable].reshape(frame_count, usable // frame_count, local_window.shape[1])
            frame_rms = np.sqrt(np.mean(frames * frames, axis=(1, 2)) + 1e-20)
            temporal_concentration = float(np.max(frame_rms) / max(np.mean(frame_rms), 1e-20))
        else:
            temporal_concentration = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            waveform_kurtosis = float(np.nanmedian(kurtosis(local_window, axis=0, fisher=False, bias=False, nan_policy='omit')))
        if not np.isfinite(waveform_kurtosis):
            waveform_kurtosis = 3.0
        row = {'task_id': task['task_id'], 'task_class': task['task_class'], 'physical_event_id': task['physical_event_id'], 'time': pd.Timestamp(task['time']), 'data_channel': center_channel, 'geometry_type': str(task['geometry_type']).lower(), 'burial_depth_cm': float(task['burial_depth_cm']), 'segment_id': str(task['segment_id']), 'stage_id': int(task['stage_id']), 'scale_s': float(scale), 'best_offset_s': best['offset'], 'local_log_rms_gain': best['local_gain'], 'nonlocal_log_rms_gain': best['nonlocal_gain'], 'locality_log_ratio': best['locality'], 'control_log_rms_gain': control_gain, 'local_to_control_log_ratio': best['local_gain'] - control_gain if np.isfinite(control_gain) else np.nan, 'gauge_support_fraction': support_fraction, 'gauge_abs_correlation': median_abs_correlation(local_window), 'spectral_entropy': entropy, 'spectral_js_to_flank': spectral_js, 'waveform_kurtosis': waveform_kurtosis, 'temporal_concentration': temporal_concentration}
        for band_index, value in enumerate(psd):
            row[f'psd_fraction_{band_index:02d}'] = float(value)
        rows.append(row)
    return rows

def read_context_block(record: H5Record, local_second: float) -> tuple[np.ndarray, np.ndarray]:
    half = RAW_CONTEXT_S / 2.0
    start_second = max(0.0, local_second - half)
    end_second = min(FILE_DURATION_S, local_second + half)
    start_sample = int(math.floor(start_second * SAMPLE_RATE_HZ))
    end_sample = int(math.ceil(end_second * SAMPLE_RATE_HZ))
    with h5py.File(record.path, 'r') as h5:
        dataset = h5[H5_DATASET]
        block = np.asarray(dataset[start_sample:end_sample, :], dtype=np.float32)
    absolute_seconds = np.arange(start_sample, end_sample, dtype=float) / SAMPLE_RATE_HZ
    relative = absolute_seconds - local_second
    return (block, relative)

def task_cache_signature(tasks: pd.DataFrame) -> str:
    payload = {'task_count': len(tasks), 'first': str(tasks['time'].min()), 'last': str(tasks['time'].max()), 'scales': ANALYSIS_SCALES_S, 'gauge': [GAUGE_LENGTH_M, CHANNEL_SPACING_M, GAUGE_INTERVALS], 'context': RAW_CONTEXT_S, 'spectral': [SPECTRAL_MIN_HZ, SPECTRAL_MAX_HZ, SPECTRAL_BAND_COUNT]}
    return stable_hash(payload)

def extract_all_multiscale_features(tasks: pd.DataFrame, metadata: pd.DataFrame, records: list[H5Record]) -> pd.DataFrame:
    feature_path = OUTPUT_DIR / 'multiscale_task_features.csv.gz'
    signature_path = OUTPUT_DIR / 'multiscale_task_features.signature.json'
    signature = task_cache_signature(tasks)
    if feature_path.exists() and signature_path.exists() and (not FORCE_REEXTRACT_RAW):
        old = json.loads(signature_path.read_text(encoding='utf-8'))
        if old.get('signature') == signature:
            log(f'[multiscale] 复用缓存：{feature_path}')
            return pd.read_csv(feature_path, parse_dates=['time'])
    work = tasks.copy().reset_index(drop=True)
    file_indices, local_seconds = ([], [])
    for time in pd.to_datetime(work['time']):
        index, _, local = locate_h5_record(pd.Timestamp(time), records)
        file_indices.append(index)
        local_seconds.append(local)
    work['file_index'] = file_indices
    work['local_second'] = local_seconds
    rows: list[dict] = []
    completed = 0
    for file_index, file_group in work.groupby('file_index', sort=True):
        record = records[int(file_index)]
        file_group = file_group.sort_values('local_second')
        groups: list[list[int]] = []
        current: list[int] = []
        anchor = None
        for row_index, row in file_group.iterrows():
            value = float(row['local_second'])
            if anchor is None or abs(value - anchor) <= 0.03:
                current.append(row_index)
                if anchor is None:
                    anchor = value
            else:
                groups.append(current)
                current = [row_index]
                anchor = value
        if current:
            groups.append(current)
        log(f'[multiscale] file {int(file_index) + 1}/{len(records)} {record.path.name}，读取组数={len(groups)}')
        for indices in groups:
            local_center = float(work.loc[indices, 'local_second'].median())
            block, relative = read_context_block(record, local_center)
            for row_index in indices:
                task = work.loc[row_index].copy()
                offset = float(task['local_second'] - local_center)
                task_relative = relative - offset
                rows.extend(extract_task_multiscale(block, task_relative, task, metadata))
                completed += 1
            if completed % 100 == 0:
                log(f'[multiscale] 完成 {completed}/{len(work)} 个任务')
    result = pd.DataFrame(rows)
    result.to_csv(feature_path, index=False, encoding='utf-8-sig', compression='gzip')
    signature_path.write_text(json.dumps({'signature': signature, 'task_count': len(tasks)}, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
MODEL_FEATURES = ['local_log_rms_gain', 'locality_log_ratio', 'local_to_control_log_ratio', 'gauge_support_fraction', 'gauge_abs_correlation', 'spectral_entropy', 'spectral_js_to_flank', 'waveform_kurtosis', 'temporal_concentration']

def prepare_model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[MODEL_FEATURES].copy()
    x['waveform_kurtosis'] = np.log1p(np.clip(x['waveform_kurtosis'], 0, None))
    for column in x.columns:
        values = pd.to_numeric(x[column], errors='coerce')
        fill = float(values.median()) if np.isfinite(values.median()) else 0.0
        x[column] = values.fillna(fill)
    return x

def fit_group_standardizers(features: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    reference = features[features['task_class'].isin(['stable_rain', 'personnel', 'pumping'])].copy()
    matrix = prepare_model_matrix(reference)
    reference = reference.reset_index(drop=True)
    matrix = matrix.reset_index(drop=True)
    models: dict = {}
    rows = []
    for scale, scale_group in reference.groupby('scale_s'):
        scale_indices = scale_group.index.to_numpy()
        center_global, scale_global = robust_center_scale(matrix.loc[scale_indices].to_numpy())
        models[float(scale), 'global'] = (center_global, scale_global)
        for key, group in scale_group.groupby(['stage_id', 'burial_depth_cm', 'geometry_type'], dropna=False):
            indices = group.index.to_numpy()
            if len(indices) >= MIN_REFERENCE_GROUP_SIZE:
                center, spread = robust_center_scale(matrix.loc[indices].to_numpy())
                model_key = (float(scale), int(key[0]), float(key[1]), str(key[2]))
                models[model_key] = (center, spread)
                source = 'stage-depth-geometry'
            else:
                center, spread = (center_global, scale_global)
                source = 'scale-global-fallback'
            for feature_name, c, s in zip(MODEL_FEATURES, center, spread):
                rows.append({'scale_s': float(scale), 'stage_id': int(key[0]), 'burial_depth_cm': float(key[1]), 'geometry_type': str(key[2]), 'feature': feature_name, 'center': float(c), 'scale': float(s), 'sample_count': int(len(indices)), 'standardizer_source': source})
    return (models, pd.DataFrame(rows))

def standardize_features(features: pd.DataFrame, models: dict) -> np.ndarray:
    matrix = prepare_model_matrix(features).to_numpy(float)
    z = np.empty_like(matrix)
    for row_index, row in enumerate(features.itertuples(index=False)):
        detailed = (float(row.scale_s), int(row.stage_id), float(row.burial_depth_cm), str(row.geometry_type))
        center, spread = models.get(detailed, models[float(row.scale_s), 'global'])
        z[row_index] = (matrix[row_index] - center) / spread
    return np.clip(z, -15.0, 15.0)

def add_noise_domain_distances(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    result = features.reset_index(drop=True).copy()
    models, standardizer_table = fit_group_standardizers(result)
    z_all = standardize_features(result, models)
    result['noise_distance'] = np.nan
    result['noise_distance_threshold'] = np.nan
    result['noise_distance_ratio'] = np.nan
    result['nearest_noise_class'] = ''
    calibration_rows = []
    neighbor_models = {}
    for scale in ANALYSIS_SCALES_S:
        scale_mask = np.isclose(result['scale_s'].to_numpy(float), scale)
        reference_mask = scale_mask & result['task_class'].isin(['stable_rain', 'personnel', 'pumping']).to_numpy()
        candidate_mask = scale_mask & result['task_class'].eq('candidate').to_numpy()
        ref_indices = np.where(reference_mask)[0]
        candidate_indices = np.where(candidate_mask)[0]
        ref_z = z_all[ref_indices]
        k = min(K_NEIGHBORS, max(2, len(ref_z) - 1))
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(ref_z)), metric='euclidean').fit(ref_z)
        self_distances = nn.kneighbors(ref_z, return_distance=True)[0]
        loo = self_distances[:, min(k, self_distances.shape[1] - 1)]
        threshold = float(np.quantile(loo, NOISE_DISTANCE_QUANTILE))
        threshold = max(threshold, 1e-09)
        result.loc[ref_indices, 'noise_distance'] = loo
        result.loc[ref_indices, 'noise_distance_threshold'] = threshold
        result.loc[ref_indices, 'noise_distance_ratio'] = loo / threshold
        if len(candidate_indices):
            distances = nn.kneighbors(z_all[candidate_indices], n_neighbors=min(k, len(ref_z)), return_distance=True)[0]
            candidate_distance = distances[:, -1]
            result.loc[candidate_indices, 'noise_distance'] = candidate_distance
            result.loc[candidate_indices, 'noise_distance_threshold'] = threshold
            result.loc[candidate_indices, 'noise_distance_ratio'] = candidate_distance / threshold
        class_models = {}
        for label in ['stable_rain', 'personnel', 'pumping']:
            class_indices = np.where(scale_mask & result['task_class'].eq(label).to_numpy())[0]
            class_k = min(max(1, K_NEIGHBORS // 2), len(class_indices))
            class_models[label] = NearestNeighbors(n_neighbors=class_k).fit(z_all[class_indices])
        for indices in [ref_indices, candidate_indices]:
            if not len(indices):
                continue
            class_distance = np.column_stack([class_models[label].kneighbors(z_all[indices], return_distance=True)[0][:, -1] for label in ['stable_rain', 'personnel', 'pumping']])
            labels = np.asarray(['stable_rain', 'personnel', 'pumping'])
            result.loc[indices, 'nearest_noise_class'] = labels[np.argmin(class_distance, axis=1)]
            for class_index, label in enumerate(labels):
                result.loc[indices, f'distance_to_{label}'] = class_distance[:, class_index]
        calibration_rows.extend(({'scale_s': scale, 'noise_distance_quantile': NOISE_DISTANCE_QUANTILE, 'noise_distance_threshold': threshold, 'reference_count': len(ref_indices), 'loo_distance_median': float(np.median(loo)), 'loo_distance_q95': float(np.quantile(loo, 0.95)), 'loo_distance_q99': float(np.quantile(loo, 0.99))} for _ in [0]))
        neighbor_models[float(scale)] = nn
    return (result, pd.DataFrame(calibration_rows), {'standardizers': models, 'neighbors': neighbor_models, 'z': z_all, 'standardizer_table': standardizer_table})

def max_consecutive_true(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best

def aggregate_candidate_screening(scored_features: pd.DataFrame, physical_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = scored_features[scored_features['task_class'].eq('candidate')].copy()
    reference = scored_features[scored_features['task_class'].isin(['stable_rain', 'personnel', 'pumping'])].copy()
    gauge_threshold = float(reference['gauge_abs_correlation'].quantile(0.75))
    locality_threshold = float(reference['locality_log_ratio'].quantile(0.9))
    rows = []
    scale_rows = []
    for event_id, group in candidates.groupby('physical_event_id'):
        group = group.sort_values('scale_s')
        novel = group['noise_distance_ratio'].to_numpy(float) > 1.0
        ratios = group['noise_distance_ratio'].to_numpy(float)
        best_pos = int(np.nanargmax(ratios))
        best = group.iloc[best_pos]
        votes = int(np.sum(novel))
        consecutive = max_consecutive_true(novel.tolist())
        gauge_ok = bool(group['gauge_abs_correlation'].max() >= gauge_threshold)
        locality_ok = bool(group['locality_log_ratio'].max() >= locality_threshold)
        rows.append({'physical_event_id': event_id, 'novel_scale_count': votes, 'max_consecutive_novel_scales': consecutive, 'max_noise_distance_ratio': float(np.nanmax(ratios)), 'median_noise_distance_ratio': float(np.nanmedian(ratios)), 'best_scale_s': float(best['scale_s']), 'best_scale_offset_s': float(best['best_offset_s']), 'best_nearest_noise_class': str(best['nearest_noise_class']), 'max_local_log_rms_gain': float(group['local_log_rms_gain'].max()), 'max_locality_log_ratio': float(group['locality_log_ratio'].max()), 'max_gauge_support_fraction': float(group['gauge_support_fraction'].max()), 'max_gauge_abs_correlation': float(group['gauge_abs_correlation'].max()), 'max_spectral_js_to_flank': float(group['spectral_js_to_flank'].max()), 'max_temporal_concentration': float(group['temporal_concentration'].max()), 'gauge_coherence_above_noise_q75': gauge_ok, 'locality_above_noise_q90': locality_ok, 'gauge_reference_q75': gauge_threshold, 'locality_reference_q90': locality_threshold})
        local = group[['physical_event_id', 'scale_s', 'noise_distance_ratio', 'nearest_noise_class']].copy()
        local['scale_novel'] = novel
        scale_rows.append(local)
    aggregate = pd.DataFrame(rows)
    result = physical_events.merge(aggregate, on='physical_event_id', how='inner', validate='one_to_one')
    high_confidence = result['novel_scale_count'].ge(MIN_NOVEL_SCALES) & result['max_consecutive_novel_scales'].ge(MIN_CONSECUTIVE_NOVEL_SCALES) & (result['gauge_coherence_above_noise_q75'] | result['locality_above_noise_q90']) & ~result['buffered_operation_overlap'] & ~result['base_known_human_like']
    nuisance = result['buffered_operation_overlap'] | result['base_known_human_like'] | result['novel_scale_count'].eq(0)
    result['automatic_screening_label'] = np.select([high_confidence, nuisance], ['effective_candidate', 'nuisance_like'], default='uncertain')
    result['screening_score'] = result['max_noise_distance_ratio'].clip(upper=10) + 0.5 * result['novel_scale_count'] + 0.4 * result['max_consecutive_novel_scales'] + 0.5 * result['gauge_coherence_above_noise_q75'].astype(int) + 0.5 * result['locality_above_noise_q90'].astype(int) + 0.5 * result['strict_core_1s'].astype(int)
    result['manual_label'] = 'pending'
    result['manual_confidence'] = ''
    result['manual_reviewer'] = ''
    result['manual_notes'] = ''
    result['include_in_future_catrms'] = False
    result['rainfall_elapsed_min'] = (pd.to_datetime(result['peak_time']) - EXPERIMENT_START).dt.total_seconds() / 60.0
    result['review_rank'] = result['screening_score'].rank(method='first', ascending=False).astype(int)
    return (result.sort_values('review_rank').reset_index(drop=True), pd.concat(scale_rows, ignore_index=True))
TYPE_FEATURES = ['max_noise_distance_ratio', 'novel_scale_count', 'best_scale_s', 'max_local_log_rms_gain', 'max_locality_log_ratio', 'max_gauge_support_fraction', 'max_gauge_abs_correlation', 'max_spectral_js_to_flank', 'max_temporal_concentration']

def add_preliminary_types(catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = catalog.copy()
    result['preliminary_signal_type'] = 'not_assigned'
    result['type_pc1'] = np.nan
    result['type_pc2'] = np.nan
    subset_mask = result['automatic_screening_label'].eq('effective_candidate')
    subset = result.loc[subset_mask].copy()
    if len(subset) < TYPE_MIN_EVENTS:
        return (result, pd.DataFrame(), pd.DataFrame())
    x = subset[TYPE_FEATURES].apply(pd.to_numeric, errors='coerce')
    x = x.fillna(x.median())
    center, spread = robust_center_scale(x.to_numpy(float))
    z = np.clip((x.to_numpy(float) - center) / spread, -12, 12)
    pca_count = min(5, z.shape[1], len(z) - 1)
    pca = PCA(n_components=pca_count, random_state=RANDOM_SEED).fit(z)
    reduced = pca.transform(z)
    max_count = min(TYPE_MAX_COUNT, max(TYPE_MIN_COUNT, len(subset) // 20))
    bic_rows = []
    models = []
    for count in range(TYPE_MIN_COUNT, max_count + 1):
        model = GaussianMixture(n_components=count, covariance_type='full', random_state=RANDOM_SEED, n_init=5)
        model.fit(reduced)
        bic_rows.append({'type_count': count, 'bic': float(model.bic(reduced))})
        models.append(model)
    best = models[int(np.argmin([row['bic'] for row in bic_rows]))]
    raw_labels = best.predict(reduced)
    order_table = pd.DataFrame({'raw': raw_labels, 'best_scale': subset['best_scale_s'].to_numpy(), 'js': subset['max_spectral_js_to_flank'].to_numpy()})
    order = order_table.groupby('raw').agg(best_scale=('best_scale', 'median'), js=('js', 'median')).sort_values(['best_scale', 'js']).index.tolist()
    mapping = {raw: f'Type {chr(65 + i)}' for i, raw in enumerate(order)}
    labels = np.asarray([mapping[value] for value in raw_labels])
    result.loc[subset.index, 'preliminary_signal_type'] = labels
    result.loc[subset.index, 'type_pc1'] = reduced[:, 0]
    result.loc[subset.index, 'type_pc2'] = reduced[:, 1] if reduced.shape[1] > 1 else 0.0
    summary = result.loc[subset.index].groupby('preliminary_signal_type').agg(event_count=('physical_event_id', 'size'), median_best_scale_s=('best_scale_s', 'median'), median_noise_distance_ratio=('max_noise_distance_ratio', 'median'), median_locality=('max_locality_log_ratio', 'median'), median_gauge_correlation=('max_gauge_abs_correlation', 'median'), median_spectral_js=('max_spectral_js_to_flank', 'median'), median_depth_cm=('burial_depth_cm', 'median')).reset_index()
    explained = pd.DataFrame({'component': np.arange(1, len(pca.explained_variance_ratio_) + 1), 'explained_variance_ratio': pca.explained_variance_ratio_})
    return (result, summary, pd.DataFrame(bic_rows).assign(**{'pca_explained_first_two': explained['explained_variance_ratio'].iloc[:2].sum()}))

def class_frame(scored: pd.DataFrame, catalog: pd.DataFrame, scale: float=1.0) -> pd.DataFrame:
    refs = scored[np.isclose(scored['scale_s'], scale) & scored['task_class'].isin(['stable_rain', 'personnel', 'pumping'])].copy()
    candidate = scored[np.isclose(scored['scale_s'], scale) & scored['task_class'].eq('candidate')].copy()
    candidate = candidate.merge(catalog[['physical_event_id', 'automatic_screening_label', 'preliminary_signal_type']], on='physical_event_id', how='left')
    candidate['plot_class'] = candidate['automatic_screening_label']
    refs['plot_class'] = refs['task_class']
    return pd.concat([refs, candidate], ignore_index=True, sort=False)

def figure_spectral_comparison(scored: pd.DataFrame, catalog: pd.DataFrame) -> None:
    data = class_frame(scored, catalog, scale=1.0)
    order = ['stable_rain', 'personnel', 'pumping', 'effective_candidate', 'uncertain']
    labels = {'stable_rain': 'Stable rain', 'personnel': 'Personnel', 'pumping': 'Pumping', 'effective_candidate': 'Effective candidate', 'uncertain': 'Uncertain'}
    fig, ax = plt.subplots(figsize=(WIDE_WIDTH_IN, 3.4), layout='constrained')
    centers = band_centers()
    psd_columns = [f'psd_fraction_{i:02d}' for i in range(SPECTRAL_BAND_COUNT)]
    for label in order:
        group = data[data['plot_class'].eq(label)]
        if group.empty:
            continue
        matrix = np.log(np.maximum(group[psd_columns].to_numpy(float), 1e-12))
        matrix -= np.median(matrix, axis=1, keepdims=True)
        median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        color = COLORS.get(label, '#777777')
        ax.plot(centers, median, color=color, label=f'{labels[label]} (n={len(group)})')
        ax.fill_between(centers, q25, q75, color=color, alpha=0.11)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Median-centred log spectral fraction')
    ax.set_title('Gauge-constrained multiscale spectral-shape comparison (1-s scale)')
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.18)
    save_figure(fig, 'Figure01_gauge_multiscale_spectral_comparison')

def figure_multiscale_screening(scored: pd.DataFrame, catalog: pd.DataFrame) -> None:
    candidate = scored[scored['task_class'].eq('candidate')].merge(catalog[['physical_event_id', 'automatic_screening_label']], on='physical_event_id', how='left')
    fig, axes = plt.subplots(1, 2, figsize=(WIDE_WIDTH_IN, 3.2), layout='constrained')
    classes = ['nuisance_like', 'uncertain', 'effective_candidate']
    positions = np.arange(len(ANALYSIS_SCALES_S), dtype=float)
    width = 0.22
    for class_index, label in enumerate(classes):
        medians, q25s, q75s = ([], [], [])
        for scale in ANALYSIS_SCALES_S:
            values = candidate[candidate['automatic_screening_label'].eq(label) & np.isclose(candidate['scale_s'], scale)]['noise_distance_ratio'].to_numpy(float)
            if len(values):
                medians.append(np.median(values))
                q25s.append(np.quantile(values, 0.25))
                q75s.append(np.quantile(values, 0.75))
            else:
                medians.append(np.nan)
                q25s.append(np.nan)
                q75s.append(np.nan)
        medians = np.asarray(medians)
        q25s = np.asarray(q25s)
        q75s = np.asarray(q75s)
        x = positions + (class_index - 1) * width
        axes[0].bar(x, medians, width=width, color=COLORS[label], alpha=0.82, label=label.replace('_', ' '))
        axes[0].errorbar(x, medians, yerr=[medians - q25s, q75s - medians], fmt='none', color='#333333', lw=0.7, capsize=2)
    axes[0].axhline(1.0, color='#333333', ls='--', lw=0.8, label='noise-domain threshold')
    axes[0].set_xticks(positions, [f'{value:g}' for value in ANALYSIS_SCALES_S])
    axes[0].set_xlabel('Window scale (s)')
    axes[0].set_ylabel('Noise-distance ratio')
    axes[0].set_title('Scale-conditioned distance from the noise domain')
    axes[0].legend(frameon=False, fontsize=7)
    panel_label(axes[0], 'a')
    vote_table = catalog.groupby(['automatic_screening_label', 'novel_scale_count']).size().unstack(fill_value=0)
    bottom = np.zeros(7)
    x = np.arange(7)
    for label in classes:
        values = vote_table.loc[label].reindex(x, fill_value=0).to_numpy() if label in vote_table.index else np.zeros(7)
        axes[1].bar(x, values, bottom=bottom, color=COLORS[label], label=label.replace('_', ' '))
        bottom += values
    axes[1].set_xlabel('Number of novel scales (out of 6)')
    axes[1].set_ylabel('Event count')
    axes[1].set_title('Multiscale screening votes')
    axes[1].legend(frameon=False, fontsize=7)
    panel_label(axes[1], 'b')
    save_figure(fig, 'Figure02_multiscale_noise_distance_and_votes')

def figure_feature_comparison(scored: pd.DataFrame, catalog: pd.DataFrame) -> None:
    data = class_frame(scored, catalog, scale=1.0)
    order = ['stable_rain', 'personnel', 'pumping', 'effective_candidate', 'uncertain']
    labels = ['Rain', 'Personnel', 'Pumping', 'Effective', 'Uncertain']
    panels = [('locality_log_ratio', 'Local-to-nonlocal log RMS ratio'), ('gauge_abs_correlation', 'Within-gauge absolute correlation'), ('spectral_js_to_flank', 'Spectral change from local flank'), ('temporal_concentration', 'Temporal concentration')]
    fig, axes = plt.subplots(2, 2, figsize=(WIDE_WIDTH_IN, 5.6), layout='constrained')
    for panel_index, (column, ylabel) in enumerate(panels):
        ax = axes.flat[panel_index]
        values = [data[data['plot_class'].eq(label)][column].dropna().to_numpy(float) for label in order]
        box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.65)
        for patch, label in zip(box['boxes'], order):
            patch.set_facecolor(COLORS.get(label, '#777777'))
            patch.set_alpha(0.55)
        ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=24, ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.18)
        panel_label(ax, chr(97 + panel_index))
    fig.suptitle('Post-detection time-frequency-space feature comparison')
    save_figure(fig, 'Figure03_effective_candidate_feature_comparison')

def figure_time_depth_geometry(catalog: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDE_WIDTH_IN, 3.4), layout='constrained')
    for label, marker, size, alpha in [('uncertain', 'o', 10, 0.35), ('effective_candidate', 'o', 14, 0.75)]:
        group = catalog[catalog['automatic_screening_label'].eq(label)]
        axes[0].scatter(pd.to_datetime(group['peak_time']), group['burial_depth_cm'], s=size, alpha=alpha, color=COLORS[label], label=label.replace('_', ' '), linewidths=0)
    axes[0].axvline(MACRO_OBSERVATION, color=COLORS['macro'], ls='--', lw=0.9, label='22:38:39 observation')
    axes[0].invert_yaxis()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[0].set_xlabel('Time on 7 July 2025')
    axes[0].set_ylabel('Burial depth (cm)')
    axes[0].set_title('Gauge-merged candidate evolution')
    axes[0].legend(frameon=False, fontsize=7)
    panel_label(axes[0], 'a')
    subset = catalog[catalog['automatic_screening_label'].isin(['effective_candidate', 'uncertain'])]
    table = subset.pivot_table(index='geometry_type', columns='automatic_screening_label', values='physical_event_id', aggfunc='count', fill_value=0)
    desired = ['uncertain', 'effective_candidate']
    bottom = np.zeros(len(table))
    x = np.arange(len(table))
    for label in desired:
        values = table[label].to_numpy() if label in table.columns else np.zeros(len(table))
        axes[1].bar(x, values, bottom=bottom, color=COLORS[label], label=label.replace('_', ' '))
        bottom += values
    axes[1].set_xticks(x, table.index.astype(str))
    axes[1].set_xlabel('Data-driven source-centre geometry')
    axes[1].set_ylabel('Gauge-merged event count')
    axes[1].set_title('Ring and connecting-line candidates')
    axes[1].legend(frameon=False, fontsize=7)
    panel_label(axes[1], 'b')
    save_figure(fig, 'Figure04_candidate_time_depth_and_geometry')

def figure_preliminary_types(catalog: pd.DataFrame) -> None:
    subset = catalog[catalog['automatic_screening_label'].eq('effective_candidate') & catalog['preliminary_signal_type'].ne('not_assigned')].copy()
    if subset.empty:
        return
    types = sorted(subset['preliminary_signal_type'].unique())
    palette = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 2, figsize=(WIDE_WIDTH_IN, 3.3), layout='constrained')
    for i, label in enumerate(types):
        group = subset[subset['preliminary_signal_type'].eq(label)]
        axes[0].scatter(group['type_pc1'], group['type_pc2'], s=12, alpha=0.55, color=palette(i), label=f'{label} (n={len(group)})', linewidths=0)
    axes[0].set_xlabel('Type representation PC1')
    axes[0].set_ylabel('Type representation PC2')
    axes[0].set_title('Data-driven internal types of effective candidates')
    axes[0].legend(frameon=False, fontsize=7)
    panel_label(axes[0], 'a')
    for i, label in enumerate(types):
        group = subset[subset['preliminary_signal_type'].eq(label)]
        axes[1].scatter(pd.to_datetime(group['peak_time']), group['burial_depth_cm'], s=12, alpha=0.6, color=palette(i), label=label, linewidths=0)
    axes[1].invert_yaxis()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[1].set_xlabel('Time on 7 July 2025')
    axes[1].set_ylabel('Burial depth (cm)')
    axes[1].set_title('Type-specific temporal-depth evolution')
    axes[1].legend(frameon=False, fontsize=7)
    panel_label(axes[1], 'b')
    save_figure(fig, 'Figure05_preliminary_effective_signal_types')

def read_centered_trace(time: pd.Timestamp, channel: int, records: list[H5Record], half_s: float=1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, record, local = locate_h5_record(time, records)
    start = max(0, int(round((local - half_s) * SAMPLE_RATE_HZ)))
    end = min(int(FILE_DURATION_S * SAMPLE_RATE_HZ), int(round((local + half_s) * SAMPLE_RATE_HZ)))
    gauge = np.arange(max(0, channel - GAUGE_HALF_CHANNELS), min(DATA_CHANNEL_COUNT, channel + GAUGE_HALF_CHANNELS) + 1)
    with h5py.File(record.path, 'r') as h5:
        block = np.asarray(h5[H5_DATASET][start:end, :], dtype=np.float32)
    trace = np.asarray(block[:, channel], dtype=float)
    heat = np.asarray(block[:, gauge], dtype=float).T
    trace -= np.median(trace)
    scale = max(1.4826 * np.median(np.abs(trace)), 1e-12)
    trace /= scale
    heat -= np.median(heat, axis=1, keepdims=True)
    heat_scale = 1.4826 * np.median(np.abs(heat), axis=1, keepdims=True)
    heat /= np.maximum(heat_scale, 1e-12)
    t = np.arange(len(trace)) / SAMPLE_RATE_HZ - half_s
    return (t, trace, heat)

def choose_representative_tasks(scored: pd.DataFrame, catalog: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    rows: list[tuple[str, pd.Series]] = []
    for label in ['stable_rain', 'personnel', 'pumping']:
        group = scored[scored['task_class'].eq(label) & np.isclose(scored['scale_s'], 1.0)]
        if len(group):
            target = group.iloc[(group['noise_distance_ratio'] - group['noise_distance_ratio'].median()).abs().argsort().iloc[0]]
            rows.append((label.replace('_', ' ').title(), target))
    effective = catalog[catalog['automatic_screening_label'].eq('effective_candidate')].sort_values('screening_score', ascending=False)
    if len(effective):
        event = effective.iloc[0]
        row = pd.Series({'time': event['peak_time'], 'data_channel': event['representative_data_channel']})
        rows.append(('Effective candidate', row))
    return rows

def figure_representative_signals(scored: pd.DataFrame, catalog: pd.DataFrame, records: list[H5Record]) -> None:
    rows = choose_representative_tasks(scored, catalog)
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(WIDE_WIDTH_IN, min(9.2, 1.75 * len(rows))), layout='constrained')
    if len(rows) == 1:
        axes = np.asarray([axes])
    for i, (label, row) in enumerate(rows):
        t, trace, heat = read_centered_trace(pd.Timestamp(row['time']), int(row['data_channel']), records)
        limit = max(5.0, float(np.quantile(np.abs(trace), 0.995)))
        axes[i, 0].plot(t, np.clip(trace, -limit, limit), color='#333333', lw=0.6)
        axes[i, 0].set_ylabel(label, fontsize=8)
        axes[i, 0].grid(alpha=0.15)
        freq, st, power = spectrogram(trace, fs=SAMPLE_RATE_HZ, nperseg=512, noverlap=448, scaling='density')
        mask = (freq >= SPECTRAL_MIN_HZ) & (freq <= SPECTRAL_MAX_HZ)
        image = axes[i, 1].pcolormesh(st - 1.0, freq[mask], 10 * np.log10(np.maximum(power[mask], 1e-18)), shading='auto', cmap='magma')
        axes[i, 1].set_ylabel('Frequency (Hz)', fontsize=8)
        axes[i, 1].set_ylim(SPECTRAL_MIN_HZ, SPECTRAL_MAX_HZ)
        if i == 0:
            axes[i, 0].set_title('Representative standardized waveform')
            axes[i, 1].set_title('Time-frequency representation')
    axes[-1, 0].set_xlabel('Time from window centre (s)')
    axes[-1, 1].set_xlabel('Time from window centre (s)')
    save_figure(fig, 'Figure06_representative_noise_and_effective_candidate_signals')

def review_selection(catalog: pd.DataFrame) -> pd.DataFrame:
    selected = []
    effective = catalog[catalog['automatic_screening_label'].eq('effective_candidate')]
    for _, group in effective.groupby('preliminary_signal_type'):
        selected.append(group.sort_values('screening_score', ascending=False).head(REVIEW_PER_TYPE))
    uncertain = catalog[catalog['automatic_screening_label'].eq('uncertain')].sort_values('screening_score', ascending=False).head(REVIEW_UNCERTAIN_COUNT)
    selected.append(uncertain)
    if not selected:
        return catalog.iloc[:0]
    return pd.concat(selected).drop_duplicates('physical_event_id').sort_values('review_rank')

def create_manual_review_panels(catalog: pd.DataFrame, records: list[H5Record]) -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    selected = review_selection(catalog)
    for row in selected.itertuples(index=False):
        output = REVIEW_DIR / f'{row.physical_event_id}.png'
        if output.exists() and (not FORCE_REBUILD_RESULTS):
            continue
        t, trace, heat = read_centered_trace(pd.Timestamp(row.peak_time), int(row.representative_data_channel), records)
        fig, axes = plt.subplots(3, 1, figsize=(NARROW_WIDTH_IN, 5.6), layout='constrained')
        limit = max(5.0, float(np.quantile(np.abs(trace), 0.995)))
        axes[0].plot(t, np.clip(trace, -limit, limit), color='#333333', lw=0.6)
        axes[0].set_ylabel('Robust z')
        axes[0].set_title(f'{row.physical_event_id} | {row.automatic_screening_label} | {row.preliminary_signal_type}\n{pd.Timestamp(row.peak_time)} | depth={row.burial_depth_cm:g} cm | ch={row.representative_data_channel}')
        axes[1].imshow(np.clip(heat, -8, 8), aspect='auto', origin='lower', extent=[t[0], t[-1], -GAUGE_HALF_CHANNELS, GAUGE_HALF_CHANNELS], cmap='RdBu_r', vmin=-8, vmax=8)
        axes[1].set_ylabel('Gauge offset\n(channel)')
        freq, st, power = spectrogram(trace, fs=SAMPLE_RATE_HZ, nperseg=512, noverlap=448, scaling='density')
        mask = (freq >= SPECTRAL_MIN_HZ) & (freq <= SPECTRAL_MAX_HZ)
        axes[2].pcolormesh(st - 1.0, freq[mask], 10 * np.log10(np.maximum(power[mask], 1e-18)), shading='auto', cmap='magma')
        axes[2].set_ylabel('Frequency (Hz)')
        axes[2].set_xlabel('Time from centre (s)')
        fig.savefig(output, dpi=PNG_DPI, bbox_inches='tight')
        plt.close(fig)

def effect_size_table(scored: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    data = class_frame(scored, catalog, scale=1.0)
    candidate = data[data['plot_class'].eq('effective_candidate')]
    rows = []
    for feature in MODEL_FEATURES:
        a = pd.to_numeric(candidate[feature], errors='coerce').dropna().to_numpy(float)
        for reference_label in ['stable_rain', 'personnel', 'pumping']:
            b = pd.to_numeric(data[data['plot_class'].eq(reference_label)][feature], errors='coerce').dropna().to_numpy(float)
            if not len(a) or not len(b):
                continue
            pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0) if len(a) > 1 and len(b) > 1 else np.nan
            cliffs = float(np.mean(a[:, None] > b[None, :]) - np.mean(a[:, None] < b[None, :]))
            rows.append({'feature': feature, 'candidate_vs': reference_label, 'candidate_median': float(np.median(a)), 'reference_median': float(np.median(b)), 'cohens_d': float((np.mean(a) - np.mean(b)) / pooled) if np.isfinite(pooled) and pooled > 0 else np.nan, 'cliffs_delta': cliffs, 'candidate_n': len(a), 'reference_n': len(b)})
    return pd.DataFrame(rows)

def write_outputs(physical_events: pd.DataFrame, gauge_members: pd.DataFrame, tasks: pd.DataFrame, scored: pd.DataFrame, catalog: pd.DataFrame, scale_votes: pd.DataFrame, calibration: pd.DataFrame, standardizers: pd.DataFrame, type_summary: pd.DataFrame, type_bic: pd.DataFrame, metadata: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    physical_events.to_csv(OUTPUT_DIR / '01_gauge_merged_coarse_events.csv', index=False, encoding='utf-8-sig')
    gauge_members.to_csv(OUTPUT_DIR / '02_gauge_merge_members.csv', index=False, encoding='utf-8-sig')
    tasks.to_csv(OUTPUT_DIR / '03_multiscale_feature_tasks.csv', index=False, encoding='utf-8-sig')
    scored.to_csv(OUTPUT_DIR / '04_multiscale_scored_features.csv.gz', index=False, encoding='utf-8-sig', compression='gzip')
    catalog.to_csv(OUTPUT_DIR / '05_effective_signal_candidate_catalog.csv', index=False, encoding='utf-8-sig')
    scale_votes.to_csv(OUTPUT_DIR / '06_event_scale_votes.csv', index=False, encoding='utf-8-sig')
    calibration.to_csv(OUTPUT_DIR / '07_noise_distance_calibration.csv', index=False, encoding='utf-8-sig')
    standardizers.to_csv(OUTPUT_DIR / '08_stage_depth_geometry_standardizers.csv', index=False, encoding='utf-8-sig')
    type_summary.to_csv(OUTPUT_DIR / '09_preliminary_type_summary.csv', index=False, encoding='utf-8-sig')
    type_bic.to_csv(OUTPUT_DIR / '10_preliminary_type_BIC.csv', index=False, encoding='utf-8-sig')
    metadata.to_csv(OUTPUT_DIR / '11_monitoring_network_used.csv', index=False, encoding='utf-8-sig')
    review_columns = ['physical_event_id', 'peak_time', 'representative_layout_channel', 'representative_data_channel', 'geometry_type', 'burial_depth_cm', 'x_cm', 'y_cm', 'automatic_screening_label', 'preliminary_signal_type', 'screening_score', 'review_rank', 'novel_scale_count', 'max_noise_distance_ratio', 'best_scale_s', 'manual_label', 'manual_confidence', 'manual_reviewer', 'manual_notes', 'include_in_future_catrms']
    review = catalog[review_columns].copy()
    review['allowed_manual_labels'] = 'effective|noise|uncertain'
    review['review_panel'] = review['physical_event_id'].map(lambda value: str(REVIEW_DIR / f'{value}.png'))
    review.to_csv(OUTPUT_DIR / 'manual_review_label_template.csv', index=False, encoding='utf-8-sig')
    effect_size_table(scored, catalog).to_csv(OUTPUT_DIR / 'paper_effect_sizes_effective_vs_noise.csv', index=False, encoding='utf-8-sig')
    summary = catalog['automatic_screening_label'].value_counts().rename_axis('label').reset_index(name='event_count')
    summary.to_csv(OUTPUT_DIR / 'paper_results_summary.csv', index=False, encoding='utf-8-sig')

def write_readme(catalog: pd.DataFrame, type_summary: pd.DataFrame, run_hash: str) -> None:
    counts = catalog['automatic_screening_label'].value_counts().to_dict()
    text = f"# S1标距约束＋多尺度有效信号候选筛选\n\n## 本版本完成的工作\n\n- 继承第二段代码的分阶段降雨背景、人员/抽水屏蔽、2 cm圆环和连接直线监测网络。\n- 1.6 m标距和0.2 m通道间隔对应约{GAUGE_INTERVALS}个采样间隔；标距重叠且同时出现的粗异常被合并为同一物理候选。\n- 原始H5二次分析尺度：{', '.join((f'{value:g} s' for value in ANALYSIS_SCALES_S))}。\n- 在各尺度提取幅值变化、局部性、标距内相关、频谱变化、频谱熵、峭度和时间集中度。\n- 稳定降雨、人员、抽水共同构成噪声参照域；标准化按降雨阶段、埋深、圆环/直线完成，小样本组合回退到同尺度总体噪声。\n- 只有同时通过多尺度噪声域距离、标距相干/空间局部性并且不在已记录操作时段内的事件，才标为 effective_candidate。\n- 本版本不计算CAT-RMS。\n\n## 自动筛选结果\n\n- 标距合并后的事件：{len(catalog)}\n- effective_candidate：{counts.get('effective_candidate', 0)}\n- uncertain：{counts.get('uncertain', 0)}\n- nuisance_like：{counts.get('nuisance_like', 0)}\n- 初步内部类型数：{len(type_summary)}\n\n## 最重要的科学边界\n\n`effective_candidate` 是“高置信度有效候选”，不是已经证实的土体颗粒摩擦或微滑移真值。\n请打开 `manual_review_label_template.csv`，结合人工复核图、视频和实验日志填写\n`manual_label=effective/noise/uncertain`。只有积累足够人工标签后，才训练最终监督二分类器。\n\n## 表格使用顺序\n\n1. `05_effective_signal_candidate_catalog.csv`：主候选目录。\n2. `manual_review_label_template.csv`：人工复核和最终标签入口。\n3. `04_multiscale_scored_features.csv.gz`：六尺度详细特征与噪声域距离。\n4. `paper_effect_sizes_effective_vs_noise.csv`：候选与三类噪声的效应量。\n5. `paper_figures_A4`：论文对比图。\n6. `manual_review_panels`：逐事件人工复核图。\n\n## 复现信息\n\n- run_hash: {run_hash}\n- H5: {H5_INPUT_DIR}\n- 布设表: {LAYOUT_XLSX}\n- 第二段代码结果: {BASE_RESULT_DIR}\n- 输出: {OUTPUT_DIR}\n"
    (OUTPUT_DIR / 'README_标距约束多尺度筛选.md').write_text(text, encoding='utf-8')

def run(max_events: int | None=None, skip_review_panels: bool=False) -> None:
    np.random.seed(RANDOM_SEED)
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 76)
    log('S1标距约束 + 多尺度有效信号候选筛选')
    log(f'标距={GAUGE_LENGTH_M:g} m；间隔={CHANNEL_SPACING_M:g} m；标距间隔数={GAUGE_INTERVALS}')
    log('effective_candidate仅表示高置信候选，不等同于滑移真值')
    log('=' * 76)
    base_events, base_members, metadata, blocks = load_base_tables()
    physical_events, gauge_members = gauge_merge_events(base_events)
    log(f'[gauge] 1 s粗事件 {len(base_events)} -> 标距约束物理事件 {len(physical_events)}')
    tasks = build_feature_tasks(physical_events, metadata, blocks, max_events=max_events)
    records = list_h5_records()
    raw_features = extract_all_multiscale_features(tasks, metadata, records)
    scored, calibration, model_bundle = add_noise_domain_distances(raw_features)
    catalog, scale_votes = aggregate_candidate_screening(scored, physical_events)
    catalog, type_summary, type_bic = add_preliminary_types(catalog)
    run_payload = {'gauge_length_m': GAUGE_LENGTH_M, 'spacing_m': CHANNEL_SPACING_M, 'scales_s': ANALYSIS_SCALES_S, 'noise_distance_quantile': NOISE_DISTANCE_QUANTILE, 'base_events': len(base_events), 'physical_events': len(physical_events), 'tasks': len(tasks), 'max_events': max_events}
    run_hash = stable_hash(run_payload)
    write_outputs(physical_events, gauge_members, tasks, scored, catalog, scale_votes, calibration, model_bundle['standardizer_table'], type_summary, type_bic, metadata)
    figure_spectral_comparison(scored, catalog)
    figure_multiscale_screening(scored, catalog)
    figure_feature_comparison(scored, catalog)
    figure_time_depth_geometry(catalog)
    figure_preliminary_types(catalog)
    figure_representative_signals(scored, catalog, records)
    if not skip_review_panels:
        create_manual_review_panels(catalog, records)
    write_readme(catalog, type_summary, run_hash)
    configuration = {**run_payload, 'run_hash': run_hash, 'paths': {'h5': str(H5_INPUT_DIR), 'layout': str(LAYOUT_XLSX), 'base_result': str(BASE_RESULT_DIR), 'output': str(OUTPUT_DIR)}, 'classification_boundary': 'data-driven high-confidence candidate; not ground-truth deformation', 'cat_rms_calculated': False}
    (OUTPUT_DIR / 'run_configuration.json').write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding='utf-8')
    log('-' * 76)
    log(catalog['automatic_screening_label'].value_counts().to_string())
    log(f'输出目录：{OUTPUT_DIR}')

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='S1标距约束＋多尺度有效信号候选筛选')
    parser.add_argument('--max-events', type=int, default=None, help='仅调试时限制候选事件数；正式运行不填写')
    parser.add_argument('--skip-review-panels', action='store_true', help='跳过逐事件人工复核图')
    args = parser.parse_args(argv)
    run(max_events=args.max_events, skip_review_panels=args.skip_review_panels)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
