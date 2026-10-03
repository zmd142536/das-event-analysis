"""Detect multiscale DAS events and compare three- and four-reference screening."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
import shutil
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, periodogram, spectrogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
PROJECT_DIR = REPOSITORY_ROOT
SECOND_CODE_DIR = REPOSITORY_ROOT / 'data/screening_inputs/S1'
BASE_RESULT_DIR = REPOSITORY_ROOT / 'data/screening_inputs/S1'
OLD_REFINER_DIR = REPOSITORY_ROOT / 'scripts'
H5_INPUT_DIR = REPOSITORY_ROOT / 'data/raw/S1'
H5_DATASET = '/default'
OUTPUT_DIR = REPOSITORY_ROOT / 'outputs/screening/S1'
FIGURE_DIR = OUTPUT_DIR / 'paper_figures_A4'
REVIEW_DIR = OUTPUT_DIR / 'manual_review_panels'
SAMPLE_RATE_HZ = 5000.0
FILE_DURATION_S = 600.0
DATA_CHANNEL_COUNT = 500
GAUGE_LENGTH_M = 1.6
CHANNEL_SPACING_M = 0.2
GAUGE_INTERVALS = int(round(GAUGE_LENGTH_M / CHANNEL_SPACING_M))
GAUGE_HALF_CHANNELS = int(math.ceil(GAUGE_INTERVALS / 2))
SCALES_S = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
CHUNK_DURATION_S = 20.0
CHUNK_OVERLAP_S = 2.2
CALIBRATION_TIMES_PER_STAGE = 48
MIN_UNIT_CALIBRATION = 20
MIN_GROUP_CALIBRATION = 24
EXPANDED_Z = 3.5
STRICT_Z = 5.0
SECONDARY_EVIDENCE_Z = 2.0
MAX_GLOBAL_ACTIVE_FRACTION = 0.25
COMMON_MODE_CONTROL_Z = 4.0
MERGE_MAX_PEAK_GAP_S = 0.6
MERGE_MAX_EVENT_SPAN_S = 3.0
ENVELOPE_CORRELATION_MIN = 0.4
SPECTRAL_JS_MAX = 0.28
ADJACENT_DISTANCE_CM = 45.0
NOISE_DISTANCE_QUANTILE = 0.995
MIN_NOVEL_SCALES = 2
MIN_CONSECUTIVE_NOVEL_SCALES = 2
EARLY_CAUTION_MIN = 120.0
NEGATIVE_CONTROL_MIN = 30.0
EARLY30_REFERENCE_CLASS = 'early30_background'
EARLY30_REFERENCE_MAX_EVENTS = 240
FOUR_REFERENCE_CLASSES = ('stable_rain', 'personnel', 'pumping', EARLY30_REFERENCE_CLASS)
BASELINE_REFERENCE_CLASSES = ('stable_rain', 'personnel', 'pumping')
RANDOM_SEED = 20260817
RUN_MODE = 'all'
SMOKE_SECONDS = 60.0
MAX_REVIEW_PANELS = 60
MERGE_BATCH_FILES = 2
MERGE_PROGRESS_EVERY = 10000
ADOPT_COMPLETE_LEGACY_FILE_CACHE = True
WRITE_COMBINED_PRIMITIVE_CATALOG = True
MAX_EVENTS_FOR_FEATURE_EXTRACTION = None
KNOWN_NOISE_INTERVALS = [('2025-07-07 15:15:00', '2025-07-07 15:17:32', 'personnel'), ('2025-07-07 16:00:00', '2025-07-07 16:05:16', 'personnel'), ('2025-07-07 17:52:30', '2025-07-07 18:09:00', 'personnel'), ('2025-07-07 18:19:40', '2025-07-07 18:28:50', 'personnel'), ('2025-07-07 19:25:00', '2025-07-07 19:31:20', 'pumping'), ('2025-07-07 19:59:00', '2025-07-07 20:01:00', 'personnel'), ('2025-07-07 22:03:26', '2025-07-07 22:07:31', 'pumping')]
PERSONNEL_BUFFER_S = 15.0
PUMPING_BUFFER_S = 30.0
EXPERIMENT_START = pd.Timestamp('2025-07-07 14:06:54.427050')
EARLY30_END = EXPERIMENT_START + pd.Timedelta(minutes=NEGATIVE_CONTROL_MIN)
MACRO_OBSERVATION = pd.Timestamp('2025-07-07 22:38:39')
ANALYSIS_END = pd.Timestamp('2025-07-07 22:40:00')
FEATURE_SAFE_END = ANALYSIS_END - pd.Timedelta(seconds=max(SCALES_S) / 2.0)
for folder in [str(OLD_REFINER_DIR), str(SECOND_CODE_DIR), str(PROJECT_DIR)]:
    if folder not in sys.path:
        sys.path.insert(0, folder)
try:
    import screening_features as refine
except ImportError as exc:
    raise ImportError('缺少 S1_gauge_multiscale_effective_signal_screening.py；请将本程序与它放在同一目录。') from exc

def configure_refiner() -> None:
    refine.H5_INPUT_DIR = H5_INPUT_DIR
    refine.BASE_RESULT_DIR = BASE_RESULT_DIR
    refine.OUTPUT_DIR = OUTPUT_DIR
    refine.FIGURE_DIR = FIGURE_DIR
    refine.REVIEW_DIR = REVIEW_DIR
    refine.ANALYSIS_SCALES_S = SCALES_S
    refine.GAUGE_LENGTH_M = GAUGE_LENGTH_M
    refine.CHANNEL_SPACING_M = CHANNEL_SPACING_M
    refine.GAUGE_INTERVALS = GAUGE_INTERVALS
    refine.GAUGE_HALF_CHANNELS = GAUGE_HALF_CHANNELS
    refine.NOISE_DISTANCE_QUANTILE = NOISE_DISTANCE_QUANTILE
    refine.MIN_NOVEL_SCALES = MIN_NOVEL_SCALES
    refine.MIN_CONSECUTIVE_NOVEL_SCALES = MIN_CONSECUTIVE_NOVEL_SCALES
    refine.FORCE_REEXTRACT_RAW = False

def log(message: str) -> None:
    print(message, flush=True)

@contextmanager
def timed_stage(name: str):
    """为缺少内部进度输出的步骤提供开始、结束和耗时日志。"""
    started = time.perf_counter()
    log(f'\n===== {name}：开始 =====')
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        log(f'===== {name}：结束；耗时={elapsed / 60.0:.2f} min =====')

def atomic_write_json(path: Path, payload: dict) -> None:
    """先写临时文件再替换，防止中断后留下半个JSON。"""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def signature_digest(signature: dict) -> str:
    encoded = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:16]

def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (0.0, 1.0)
    center = float(np.median(x))
    spread = 1.4826 * float(np.median(np.abs(x - center)))
    if not np.isfinite(spread) or spread < 1e-06:
        spread = max(float(np.std(x)), 1e-06)
    return (center, spread)

def operation_status(time: pd.Timestamp, buffered: bool=True) -> tuple[bool, str]:
    time = pd.Timestamp(time)
    for start, end, kind in KNOWN_NOISE_INTERVALS:
        left, right = (pd.Timestamp(start), pd.Timestamp(end))
        if buffered:
            pad = PERSONNEL_BUFFER_S if kind == 'personnel' else PUMPING_BUFFER_S
            left -= pd.Timedelta(seconds=pad)
            right += pd.Timedelta(seconds=pad)
        if left <= time <= right:
            return (True, kind)
    return (False, '')

def load_network_and_stages() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_path = BASE_RESULT_DIR / 'monitoring_channel_layout.csv'
    stage_path = BASE_RESULT_DIR / 'rainfall_stage_blocks.csv'
    if not metadata_path.exists() or not stage_path.exists():
        raise FileNotFoundError('缺少第二段代码输出的监测网络或降雨阶段表。请先运行一次\nS1_noise_candidate_data_driven_source_analysis.py。')
    metadata = pd.read_csv(metadata_path)
    stages = pd.read_csv(stage_path)
    for column in ['start_time', 'end_time', 'center_time']:
        if column in stages:
            stages[column] = pd.to_datetime(stages[column])
    metadata['data_channel'] = pd.to_numeric(metadata['data_channel'], errors='coerce')
    metadata['burial_depth_cm'] = pd.to_numeric(metadata['burial_depth_cm'], errors='coerce')
    metadata['geometry_type'] = metadata['geometry_type'].astype(str).str.lower()
    metadata['analysis_role'] = metadata['analysis_role'].astype(str)
    metadata = metadata[metadata['data_channel'].between(0, DATA_CHANNEL_COUNT)].copy()
    return (metadata, stages)

def target_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    role = metadata['analysis_role'].str.lower()
    target = metadata[metadata['geometry_type'].isin(['ring', 'line']) & ~role.str.contains('control', na=False) & metadata['burial_depth_cm'].isin([2.0, 20.0, 50.0, 80.0])].copy()
    target = target.sort_values(['data_channel', 'geometry_type']).drop_duplicates('data_channel')
    return target.reset_index(drop=True)

def control_channels(metadata: pd.DataFrame) -> np.ndarray:
    role = metadata['analysis_role'].str.lower()
    values = metadata.loc[role.str.contains('control', na=False), 'data_channel'].dropna().astype(int)
    return np.sort(values.unique())

def stage_ids(times: pd.DatetimeIndex, stages: pd.DataFrame) -> np.ndarray:
    return refine.stage_for_times(pd.Series(times), stages).astype(int)

@dataclass(frozen=True)
class DetectionRecord:
    path: Path
    start: pd.Timestamp
    samples: int

def records_from_h5() -> list[DetectionRecord]:
    base = refine.list_h5_records()
    records = []
    for record in base:
        with h5py.File(record.path, 'r') as h5:
            samples = int(h5[H5_DATASET].shape[0])
        records.append(DetectionRecord(record.path, pd.Timestamp(record.start), samples))
    return records

def required_dataset_channels(target: pd.DataFrame, controls: np.ndarray) -> np.ndarray:
    channels: set[int] = set((int(v) for v in controls))
    for center in target['data_channel'].astype(int):
        channels.update(range(max(0, center - GAUGE_HALF_CHANNELS), min(DATA_CHANNEL_COUNT, center + GAUGE_HALF_CHANNELS) + 1))
    return np.asarray(sorted(channels), dtype=int)

def window_rms(block: np.ndarray, window_samples: int, starts: np.ndarray) -> np.ndarray:
    """利用累计平方和一次计算多个窗口、全部读取通道的RMS。"""
    x = np.nan_to_num(np.asarray(block, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x -= np.median(x, axis=0, keepdims=True)
    cumulative = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x * x, axis=0)])
    sums = cumulative[starts + window_samples] - cumulative[starts]
    return np.sqrt(np.maximum(sums / max(window_samples, 1), 1e-24))

def sensor_arrays(rms: np.ndarray, selected_channels: np.ndarray, target: pd.DataFrame, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel_position = {int(channel): i for i, channel in enumerate(selected_channels)}
    target_channels = target['data_channel'].astype(int).to_numpy()
    local = np.empty((len(rms), len(target)), dtype=float)
    nonlocal_value = np.empty_like(local)
    support = np.empty_like(local)
    control_pos = [channel_position[int(v)] for v in controls if int(v) in channel_position]
    control_rms = np.median(rms[:, control_pos], axis=1) if control_pos else np.ones(len(rms))
    for j, center in enumerate(target_channels):
        gauge = [channel_position[v] for v in range(max(0, center - GAUGE_HALF_CHANNELS), min(DATA_CHANNEL_COUNT, center + GAUGE_HALF_CHANNELS) + 1) if v in channel_position]
        local[:, j] = np.median(rms[:, gauge], axis=1)
    group_key = list(zip(target['burial_depth_cm'].astype(float), target['geometry_type'].astype(str)))
    for j, center in enumerate(target_channels):
        gauge = [channel_position[v] for v in range(max(0, center - GAUGE_HALF_CHANNELS), min(DATA_CHANNEL_COUNT, center + GAUGE_HALF_CHANNELS) + 1) if v in channel_position]
        same = np.asarray([k for k, (depth, geometry) in enumerate(group_key) if depth == group_key[j][0] and geometry == group_key[j][1] and (abs(target_channels[k] - center) > GAUGE_INTERVALS)], dtype=int)
        if len(same) < 4:
            same = np.where(np.abs(target_channels - center) > 2 * GAUGE_INTERVALS)[0]
        reference = np.median(local[:, same], axis=1) if len(same) and np.all(np.isfinite(local[:, same])) else np.median(rms, axis=1)
        nonlocal_value[:, j] = reference
        threshold = reference[:, None] * 1.1
        support[:, j] = np.mean(rms[:, gauge] > threshold, axis=1)
    return (local, nonlocal_value, np.broadcast_to(control_rms[:, None], local.shape), support)

def calibration_times(records: list[DetectionRecord], stages: pd.DataFrame) -> pd.DatetimeIndex:
    candidates = []
    for record in records:
        seconds = np.arange(5.0, min(FILE_DURATION_S, record.samples / SAMPLE_RATE_HZ) - 5.0, 10.0)
        candidates.extend(record.start + pd.to_timedelta(seconds, unit='s'))
    times = pd.DatetimeIndex(candidates)
    allowed = np.asarray([not operation_status(t, buffered=True)[0] for t in times], dtype=bool)
    allowed &= np.asarray(times <= FEATURE_SAFE_END, dtype=bool)
    times = times[allowed]
    ids = stage_ids(times, stages)
    rng = np.random.default_rng(RANDOM_SEED)
    selected = []
    for stage in sorted(np.unique(ids)):
        local = np.where(ids == stage)[0]
        number = min(CALIBRATION_TIMES_PER_STAGE, len(local))
        if number:
            selected.extend(rng.choice(local, size=number, replace=False).tolist())
    return times[np.asarray(sorted(selected), dtype=int)]

def locate_record(time: pd.Timestamp, records: list[DetectionRecord]) -> tuple[DetectionRecord, float]:
    for record in records:
        local = (pd.Timestamp(time) - record.start).total_seconds()
        if 0 <= local < record.samples / SAMPLE_RATE_HZ:
            return (record, float(local))
    raise KeyError(f'时间不在H5范围内：{time}')

def build_calibration(records: list[DetectionRecord], metadata: pd.DataFrame, stages: pd.DataFrame, target: pd.DataFrame, controls: np.ndarray, selected: np.ndarray, force: bool=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_path = OUTPUT_DIR / 'calibration_samples.csv.gz'
    model_path = OUTPUT_DIR / 'calibration_models.csv'
    if sample_path.exists() and model_path.exists() and (not force):
        log('[calibration] 复用背景校准缓存')
        return (pd.read_csv(sample_path), pd.read_csv(model_path))
    rows: list[dict] = []
    times = calibration_times(records, stages)
    log(f'[calibration] 非人为背景时间样本：{len(times)}')
    for number, time in enumerate(times, start=1):
        record, local_second = locate_record(time, records)
        half = max(SCALES_S) / 2.0
        start = max(0, int(round((local_second - half) * SAMPLE_RATE_HZ)))
        stop = min(record.samples, int(round((local_second + half) * SAMPLE_RATE_HZ)))
        with h5py.File(record.path, 'r') as h5:
            dataset = h5[H5_DATASET]
            valid_channels = selected[selected < dataset.shape[1]]
            block = np.asarray(dataset[start:stop, valid_channels], dtype=np.float32)
        centre = len(block) // 2
        stage = int(stage_ids(pd.DatetimeIndex([time]), stages)[0])
        for scale in SCALES_S:
            width = int(round(scale * SAMPLE_RATE_HZ))
            left = max(0, centre - width // 2)
            if left + width > len(block):
                continue
            rms = window_rms(block, width, np.asarray([left]))
            local, distant, control, support = sensor_arrays(rms, valid_channels, target, controls)
            for j, sensor in target.iterrows():
                rows.append({'time': time, 'stage_id': stage, 'scale_s': scale, 'data_channel': int(sensor.data_channel), 'burial_depth_cm': float(sensor.burial_depth_cm), 'geometry_type': str(sensor.geometry_type), 'segment_id': str(sensor.segment_id), 'log_local_rms': float(np.log(max(local[0, j], 1e-20))), 'locality_log_ratio': float(np.log(max(local[0, j] / distant[0, j], 1e-12))), 'local_to_control_log_ratio': float(np.log(max(local[0, j] / control[0, j], 1e-12))), 'gauge_support_fraction': float(support[0, j])})
        if number % 20 == 0:
            log(f'[calibration] {number}/{len(times)}')
    samples = pd.DataFrame(rows)
    model_rows = []
    features = ['log_local_rms', 'locality_log_ratio', 'local_to_control_log_ratio']
    group_columns = ['stage_id', 'scale_s', 'burial_depth_cm', 'geometry_type']
    for key, group in samples.groupby(group_columns, dropna=False):
        group_stats = {feature: robust_location_scale(group[feature].to_numpy()) for feature in features}
        for channel, unit in group.groupby('data_channel'):
            row = dict(zip(group_columns, key))
            row['data_channel'] = int(channel)
            for feature in features:
                center, spread = robust_location_scale(unit[feature].to_numpy())
                group_center, group_spread = group_stats[feature]
                if len(unit) < MIN_UNIT_CALIBRATION:
                    center, spread = (group_center, group_spread)
                spread = max(spread, 0.25 * group_spread, 1e-06)
                row[f'{feature}_center'] = center
                row[f'{feature}_scale'] = spread
            model_rows.append(row)
    models = pd.DataFrame(model_rows)
    samples.to_csv(sample_path, index=False, compression='gzip')
    models.to_csv(model_path, index=False, encoding='utf-8-sig')
    return (samples, models)

def model_lookup(models: pd.DataFrame) -> dict[tuple, dict[str, float]]:
    result = {}
    for row in models.itertuples(index=False):
        key = (int(row.stage_id), float(row.scale_s), float(row.burial_depth_cm), str(row.geometry_type), int(row.data_channel))
        result[key] = row._asdict()
    return result

def normalized_signature(window: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """返回8点包络、12点归一化频谱和标距内绝对相关。"""
    x = np.asarray(window, dtype=float)
    x -= np.median(x, axis=0, keepdims=True)
    usable = len(x) // 8 * 8
    if usable >= 8:
        frames = x[:usable].reshape(8, usable // 8, x.shape[1])
        envelope = np.sqrt(np.mean(frames * frames, axis=(1, 2)) + 1e-20)
    else:
        envelope = np.ones(8)
    envelope /= max(float(np.linalg.norm(envelope)), 1e-20)
    mean_trace = np.median(x, axis=1)
    frequency, power = periodogram(mean_trace, fs=SAMPLE_RATE_HZ, scaling='spectrum')
    edges = np.geomspace(20.0, 2000.0, 13)
    bands = np.asarray([np.sum(power[(frequency >= edges[i]) & (frequency < edges[i + 1])]) for i in range(12)])
    bands = np.maximum(bands, 1e-20)
    bands /= bands.sum()
    correlation = refine.median_abs_correlation(x)
    return (envelope, bands, correlation)

def primitive_required_columns() -> list[str]:
    columns = ['primitive_id', 'file_name', 'file_index', 'start_time', 'end_time', 'peak_time', 'local_second', 'scale_s', 'stage_id', 'data_channel', 'layout_channel', 'segment_id', 'geometry_type', 'burial_depth_cm', 'x_cm', 'y_cm', 'expanded_candidate', 'strict_candidate', 'detection_score', 'secondary_z', 'z_log_local_rms', 'z_locality', 'z_local_to_control', 'gauge_support_fraction', 'gauge_abs_correlation', 'operation_overlap', 'operation_kind', 'early_caution', 'negative_control_period', 'common_mode_or_leadin', 'instrument_anomaly', 'global_active_fraction']
    columns.extend((f'env_{i:02d}' for i in range(8)))
    columns.extend((f'psd_{i:02d}' for i in range(12)))
    return columns

def cache_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {'name': path.name, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

def validate_file_cache(path: Path, record: DetectionRecord, file_index: int) -> tuple[bool, int, str]:
    """检查缓存结构、文件归属和时间范围，不把完整缓存驻留在内存中。"""
    try:
        header = pd.read_csv(path, nrows=0)
        missing = sorted(set(primitive_required_columns()) - set(header.columns))
        if missing:
            return (False, 0, f"缺少字段：{', '.join(missing[:5])}")
        check = pd.read_csv(path, usecols=['primitive_id', 'file_name', 'file_index', 'peak_time'])
        if check.empty:
            return (True, 0, '空候选缓存')
        if not check['file_name'].astype(str).eq(record.path.name).all():
            return (False, len(check), 'file_name与H5不一致')
        numeric_index = pd.to_numeric(check['file_index'], errors='coerce')
        if not numeric_index.eq(file_index).all():
            return (False, len(check), 'file_index与H5顺序不一致')
        prefix = f'P{file_index + 1:02d}_'
        if not check['primitive_id'].astype(str).str.startswith(prefix).all():
            return (False, len(check), 'primitive_id前缀不一致')
        peaks = pd.to_datetime(check['peak_time'], errors='coerce')
        duration = record.samples / SAMPLE_RATE_HZ
        lower = record.start - pd.Timedelta(milliseconds=1)
        upper = record.start + pd.Timedelta(seconds=duration + 0.001)
        if peaks.isna().any() or not peaks.between(lower, upper).all():
            return (False, len(check), '候选时间超出对应H5范围')
        return (True, len(check), '通过')
    except Exception as exc:
        return (False, 0, f'读取失败：{type(exc).__name__}: {exc}')

def complete_legacy_cache_exists(cache_dir: Path, records: list[DetectionRecord], smoke: bool) -> bool:
    count = 1 if smoke else len(records)
    return cache_dir.exists() and all(((cache_dir / f'file_{i + 1:03d}.csv.gz').exists() for i in range(count)))

def scan_all_h5(records: list[DetectionRecord], metadata: pd.DataFrame, stages: pd.DataFrame, target: pd.DataFrame, controls: np.ndarray, selected: np.ndarray, models: pd.DataFrame, file_cache_dir: Path, smoke: bool=False, reuse_existing: bool=True) -> tuple[list[Path], list[int]]:
    """逐文件扫描并返回缓存路径；全程不累积数百万个Python字典。"""
    lookup = model_lookup(models)
    scan_records = records[:1] if smoke else records
    file_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths: list[Path] = []
    row_counts: list[int] = []
    scan_started = time.perf_counter()
    for file_index, record in enumerate(scan_records):
        file_cache = file_cache_dir / f'file_{file_index + 1:03d}.csv.gz'
        if file_cache.exists() and reuse_existing:
            valid, row_count, reason = validate_file_cache(file_cache, record, file_index)
            if valid:
                cache_paths.append(file_cache)
                row_counts.append(row_count)
                log(f'[scan/cache] {file_index + 1}/{len(scan_records)} 复用；候选={row_count}；{record.path.name}')
                continue
            log(f'[scan/cache] {file_index + 1}/{len(scan_records)} 缓存无效，仅重扫该文件；原因={reason}')
        file_rows: list[dict] = []
        duration = min(record.samples / SAMPLE_RATE_HZ, SMOKE_SECONDS if smoke else FILE_DURATION_S)
        seconds_to_analysis_end = max(0.0, (ANALYSIS_END - record.start).total_seconds())
        duration = min(duration, seconds_to_analysis_end)
        if duration <= 0:
            log(f'[scan] {file_index + 1}/{len(scan_records)} 跳过22:40之后文件：{record.path.name}')
            continue
        log(f'[scan] {file_index + 1}/{len(scan_records)} {record.path.name}，时长={duration:.1f}s')
        with h5py.File(record.path, 'r') as h5:
            dataset = h5[H5_DATASET]
            valid_channels = selected[selected < dataset.shape[1]]
            channel_position = {int(channel): i for i, channel in enumerate(valid_channels)}
            for core_start_s in np.arange(0.0, duration, CHUNK_DURATION_S):
                core_end_s = min(duration, core_start_s + CHUNK_DURATION_S)
                read_start_s = max(0.0, core_start_s - CHUNK_OVERLAP_S)
                read_end_s = min(duration, core_end_s + CHUNK_OVERLAP_S)
                read_start = int(round(read_start_s * SAMPLE_RATE_HZ))
                read_end = int(round(read_end_s * SAMPLE_RATE_HZ))
                raw = np.asarray(dataset[read_start:read_end, valid_channels], dtype=np.float32)
                nonfinite_fraction = float(1.0 - np.isfinite(raw).mean())
                for scale in SCALES_S:
                    width = int(round(scale * SAMPLE_RATE_HZ))
                    step_s = max(0.025, scale / 4.0)
                    centers_s = np.arange(core_start_s + scale / 2.0, core_end_s - scale / 2.0 + 0.5 * step_s, step_s)
                    if len(centers_s) == 0:
                        continue
                    starts = np.round((centers_s - scale / 2.0 - read_start_s) * SAMPLE_RATE_HZ).astype(int)
                    valid = (starts >= 0) & (starts + width <= len(raw))
                    starts, centers_s = (starts[valid], centers_s[valid])
                    if len(starts) == 0:
                        continue
                    rms = window_rms(raw, width, starts)
                    local, distant, control, support = sensor_arrays(rms, valid_channels, target, controls)
                    absolute_times = pd.DatetimeIndex(record.start + pd.to_timedelta(centers_s, unit='s'))
                    stage_vector = stage_ids(absolute_times, stages)
                    log_local = np.log(np.maximum(local, 1e-20))
                    locality = np.log(np.maximum(local / distant, 1e-12))
                    local_control = np.log(np.maximum(local / control, 1e-12))
                    z = np.zeros((len(centers_s), len(target), 3), dtype=float)
                    for j, sensor in target.iterrows():
                        for i, stage in enumerate(stage_vector):
                            key = (int(stage), float(scale), float(sensor.burial_depth_cm), str(sensor.geometry_type), int(sensor.data_channel))
                            model = lookup.get(key)
                            if model is None:
                                continue
                            for k, (name, values) in enumerate([('log_local_rms', log_local), ('locality_log_ratio', locality), ('local_to_control_log_ratio', local_control)]):
                                z[i, j, k] = (values[i, j] - model[f'{name}_center']) / model[f'{name}_scale']
                    score = np.max(z, axis=2)
                    secondary = np.partition(z, -2, axis=2)[:, :, -2]
                    expanded = (score >= EXPANDED_Z) & ((z[:, :, 0] > 0) | (z[:, :, 1] > 0))
                    strict = (score >= STRICT_Z) & (secondary >= SECONDARY_EVIDENCE_Z)
                    active_fraction = np.mean(expanded, axis=1)
                    for j, sensor in target.iterrows():
                        candidate = expanded[:, j]
                        if not np.any(candidate):
                            continue
                        padded = np.r_[False, candidate, False]
                        changes = np.flatnonzero(padded[1:] != padded[:-1])
                        for left, right in changes.reshape(-1, 2):
                            local_index = left + int(np.argmax(score[left:right, j]))
                            center_s = float(centers_s[local_index])
                            peak_time = pd.Timestamp(absolute_times[local_index])
                            center_channel = int(sensor.data_channel)
                            gauge_channels = [v for v in range(max(0, center_channel - GAUGE_HALF_CHANNELS), min(DATA_CHANNEL_COUNT, center_channel + GAUGE_HALF_CHANNELS) + 1) if v in channel_position]
                            gauge_pos = [channel_position[v] for v in gauge_channels]
                            start_sample = int(starts[local_index])
                            stop_sample = start_sample + width
                            envelope, psd, gauge_corr = normalized_signature(raw[start_sample:stop_sample, gauge_pos])
                            operation, operation_kind = operation_status(peak_time, buffered=True)
                            common_mode = bool(active_fraction[local_index] > MAX_GLOBAL_ACTIVE_FRACTION or z[local_index, j, 2] < -COMMON_MODE_CONTROL_Z)
                            elapsed_min = (peak_time - EXPERIMENT_START).total_seconds() / 60.0
                            row = {'primitive_id': f'P{file_index + 1:02d}_{len(file_rows) + 1:08d}', 'file_name': record.path.name, 'file_index': file_index, 'start_time': peak_time - pd.Timedelta(seconds=scale / 2), 'end_time': peak_time + pd.Timedelta(seconds=scale / 2), 'peak_time': peak_time, 'local_second': center_s, 'scale_s': scale, 'stage_id': int(stage_vector[local_index]), 'data_channel': center_channel, 'layout_channel': int(getattr(sensor, 'layout_channel', center_channel + 3000)), 'segment_id': str(sensor.segment_id), 'geometry_type': str(sensor.geometry_type), 'burial_depth_cm': float(sensor.burial_depth_cm), 'x_cm': float(getattr(sensor, 'x_cm', np.nan)), 'y_cm': float(getattr(sensor, 'y_cm', np.nan)), 'expanded_candidate': True, 'strict_candidate': bool(strict[local_index, j]), 'detection_score': float(score[local_index, j]), 'secondary_z': float(secondary[local_index, j]), 'z_log_local_rms': float(z[local_index, j, 0]), 'z_locality': float(z[local_index, j, 1]), 'z_local_to_control': float(z[local_index, j, 2]), 'gauge_support_fraction': float(support[local_index, j]), 'gauge_abs_correlation': gauge_corr, 'operation_overlap': operation, 'operation_kind': operation_kind, 'early_caution': bool(elapsed_min < EARLY_CAUTION_MIN), 'negative_control_period': bool(elapsed_min < NEGATIVE_CONTROL_MIN), 'common_mode_or_leadin': common_mode, 'instrument_anomaly': bool(nonfinite_fraction > 0), 'global_active_fraction': float(active_fraction[local_index])}
                            row.update({f'env_{i:02d}': float(v) for i, v in enumerate(envelope)})
                            row.update({f'psd_{i:02d}': float(v) for i, v in enumerate(psd)})
                            file_rows.append(row)
        frame = pd.DataFrame.from_records(file_rows, columns=primitive_required_columns())
        temporary = file_cache.with_suffix(file_cache.suffix + '.tmp')
        frame.to_csv(temporary, index=False, compression='gzip')
        temporary.replace(file_cache)
        cache_paths.append(file_cache)
        row_counts.append(len(frame))
        elapsed = time.perf_counter() - scan_started
        log(f'[scan] 文件候选片段={len(frame)}；断点缓存已原子保存；总进度={file_index + 1}/{len(scan_records)}；累计耗时={elapsed / 60.0:.1f} min')
    log(f'[scan] 逐文件缓存准备完成：{len(cache_paths)}/{len(scan_records)}；候选总数={sum(row_counts):,}')
    return (cache_paths, row_counts)

def combine_primitive_caches(cache_paths: list[Path], destination: Path, digest: str) -> None:
    """流式生成唯一的正式primitive目录，避免两次保存和全量内存汇总。"""
    manifest_path = destination.with_suffix(destination.suffix + '.manifest.json')
    fingerprints = [cache_fingerprint(path) for path in cache_paths]
    expected_manifest = {'signature_digest': digest, 'files': fingerprints}
    if destination.exists() and manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding='utf-8'))
        except Exception:
            old = None
        if old == expected_manifest and destination.stat().st_size > 0:
            log(f'[combine] 复用已汇总的正式primitive目录：{destination.name}')
            return
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    first_header: str | None = None
    with gzip.open(temporary, 'wt', encoding='utf-8', newline='') as output:
        for number, source_path in enumerate(cache_paths, start=1):
            with gzip.open(source_path, 'rt', encoding='utf-8', newline='') as source:
                header = source.readline()
                if first_header is None:
                    first_header = header
                    output.write(header)
                elif header != first_header:
                    raise ValueError(f'逐文件缓存字段不一致：{source_path}')
                shutil.copyfileobj(source, output, length=1024 * 1024)
            log(f'[combine] {number}/{len(cache_paths)} 已写入：{source_path.name}')
    temporary.replace(destination)
    atomic_write_json(manifest_path, expected_manifest)
    log(f'[combine] 唯一正式primitive目录已保存：{destination}')

def signature_similarity(a: pd.Series, b: pd.Series) -> bool:
    env_a = a[[f'env_{i:02d}' for i in range(8)]].to_numpy(float)
    env_b = b[[f'env_{i:02d}' for i in range(8)]].to_numpy(float)
    if np.std(env_a) > 1e-08 and np.std(env_b) > 1e-08:
        envelope_correlation = float(np.corrcoef(env_a, env_b)[0, 1])
    else:
        envelope_correlation = 0.0
    psd_a = a[[f'psd_{i:02d}' for i in range(12)]].to_numpy(float)
    psd_b = b[[f'psd_{i:02d}' for i in range(12)]].to_numpy(float)
    spectral_distance = refine.js_distance(psd_a, psd_b)
    return envelope_correlation >= ENVELOPE_CORRELATION_MIN or spectral_distance <= SPECTRAL_JS_MAX

def physically_adjacent(a: pd.Series, b: pd.Series) -> bool:
    if float(a.burial_depth_cm) != float(b.burial_depth_cm):
        return False
    if str(a.segment_id) == str(b.segment_id):
        return True
    if abs(int(a.data_channel) - int(b.data_channel)) > GAUGE_INTERVALS:
        return False
    if np.all(np.isfinite([a.x_cm, a.y_cm, b.x_cm, b.y_cm])):
        distance = math.hypot(float(a.x_cm) - float(b.x_cm), float(a.y_cm) - float(b.y_cm))
        return distance <= ADJACENT_DISTANCE_CM
    return True

def _boolean_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)
    return series.astype(str).str.strip().str.lower().isin(['true', '1', 'yes']).to_numpy()

def _js_distance_many(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """向量化Jensen-Shannon距离，与逐行调用相比显著减少Python开销。"""
    eps = 1e-20
    p = np.maximum(np.asarray(candidates, dtype=float), eps)
    q = np.maximum(np.asarray(reference, dtype=float), eps)
    p /= np.sum(p, axis=1, keepdims=True)
    q /= np.sum(q)
    midpoint = 0.5 * (p + q[None, :])
    divergence = 0.5 * np.sum(p * np.log(p / midpoint), axis=1)
    divergence += 0.5 * np.sum(q[None, :] * np.log(q[None, :] / midpoint), axis=1)
    return np.sqrt(np.maximum(divergence, 0.0))

def merge_primitives(primitives: pd.DataFrame, event_prefix: str='RGM', progress_label: str='merge') -> tuple[pd.DataFrame, pd.DataFrame]:
    """在一个时间批次内归并；候选比较使用数组筛选，组切分为线性复杂度。"""
    if primitives.empty:
        return (pd.DataFrame(), pd.DataFrame())
    work = primitives.sort_values('peak_time').reset_index(drop=True).copy()
    count = len(work)
    uf = refine.UnionFind(count)
    peaks = pd.to_datetime(work['peak_time']).astype('int64').to_numpy()
    starts = pd.to_datetime(work['start_time']).astype('int64').to_numpy()
    ends = pd.to_datetime(work['end_time']).astype('int64').to_numpy()
    gap_ns = int(round(MERGE_MAX_PEAK_GAP_S * 1000000000.0))
    span_ns = int(round(MERGE_MAX_EVENT_SPAN_S * 1000000000.0))
    channels = pd.to_numeric(work['data_channel'], errors='coerce').to_numpy(int)
    depths = pd.to_numeric(work['burial_depth_cm'], errors='coerce').to_numpy(float)
    x_values = pd.to_numeric(work['x_cm'], errors='coerce').to_numpy(float)
    y_values = pd.to_numeric(work['y_cm'], errors='coerce').to_numpy(float)
    segments = work['segment_id'].astype(str).to_numpy()
    envelopes = work[[f'env_{i:02d}' for i in range(8)]].to_numpy(float)
    spectra = work[[f'psd_{i:02d}' for i in range(12)]].to_numpy(float)
    env_centered = envelopes - np.mean(envelopes, axis=1, keepdims=True)
    env_norm = np.linalg.norm(env_centered, axis=1)
    env_valid = env_norm > 1e-08
    env_unit = np.divide(env_centered, env_norm[:, None], out=np.zeros_like(env_centered), where=env_norm[:, None] > 0)
    left = 0
    merge_started = time.perf_counter()
    pair_candidates = 0
    accepted_pairs = 0
    for i in range(count):
        while left < i and peaks[i] - peaks[left] > gap_ns:
            left += 1
        if left < i:
            candidate_indices = np.arange(left, i, dtype=int)
            basic = (np.abs(channels[candidate_indices] - channels[i]) <= GAUGE_INTERVALS) & (depths[candidate_indices] == depths[i])
            candidate_indices = candidate_indices[basic]
            if len(candidate_indices):
                finite_coordinates = np.isfinite(x_values[candidate_indices]) & np.isfinite(y_values[candidate_indices]) & np.isfinite(x_values[i]) & np.isfinite(y_values[i])
                distances = np.hypot(x_values[candidate_indices] - x_values[i], y_values[candidate_indices] - y_values[i])
                physically_close = (segments[candidate_indices] == segments[i]) | ~finite_coordinates | (distances <= ADJACENT_DISTANCE_CM)
                candidate_indices = candidate_indices[physically_close]
            if len(candidate_indices):
                pair_candidates += len(candidate_indices)
                envelope_correlation = np.zeros(len(candidate_indices), dtype=float)
                valid_env = env_valid[candidate_indices] & env_valid[i]
                if np.any(valid_env):
                    envelope_correlation[valid_env] = env_unit[candidate_indices[valid_env]] @ env_unit[i]
                envelope_match = envelope_correlation >= ENVELOPE_CORRELATION_MIN
                spectral_match = np.zeros(len(candidate_indices), dtype=bool)
                need_spectrum = ~envelope_match
                if np.any(need_spectrum):
                    spectral_match[need_spectrum] = _js_distance_many(spectra[i], spectra[candidate_indices[need_spectrum]]) <= SPECTRAL_JS_MAX
                similarity = envelope_match | spectral_match
                combined_start = np.minimum(starts[candidate_indices], starts[i])
                combined_end = np.maximum(ends[candidate_indices], ends[i])
                accepted = candidate_indices[similarity & (combined_end - combined_start <= span_ns)]
                accepted_pairs += len(accepted)
                for j in accepted:
                    uf.union(i, int(j))
        if (i + 1) % MERGE_PROGRESS_EVERY == 0 or i + 1 == count:
            elapsed = time.perf_counter() - merge_started
            rate = (i + 1) / max(elapsed, 1e-09)
            remaining = (count - i - 1) / max(rate, 1e-09)
            log(f'[{progress_label}/pairs] {i + 1:,}/{count:,}；邻近比较={pair_candidates:,}；已连接={accepted_pairs:,}；预计剩余={remaining / 60.0:.1f} min')
    roots = np.fromiter((uf.find(i) for i in range(count)), dtype=np.int64, count=count)
    codes, _ = pd.factorize(roots, sort=False)
    group_order = np.lexsort((peaks, codes))
    ordered_codes = codes[group_order]
    boundaries = np.r_[0, np.flatnonzero(ordered_codes[1:] != ordered_codes[:-1]) + 1, count]
    final_index_groups: list[np.ndarray] = []
    for boundary_index in range(len(boundaries) - 1):
        indices = group_order[boundaries[boundary_index]:boundaries[boundary_index + 1]]
        current: list[int] = []
        current_start = current_end = 0
        current_min_channel = current_max_channel = 0
        for index in indices:
            index = int(index)
            if not current:
                current = [index]
                current_start, current_end = (starts[index], ends[index])
                current_min_channel = current_max_channel = channels[index]
                continue
            proposed_start = min(current_start, starts[index])
            proposed_end = max(current_end, ends[index])
            proposed_min_channel = min(current_min_channel, channels[index])
            proposed_max_channel = max(current_max_channel, channels[index])
            if proposed_end - proposed_start > span_ns or proposed_max_channel - proposed_min_channel > 2 * GAUGE_INTERVALS:
                final_index_groups.append(np.asarray(current, dtype=int))
                current = [index]
                current_start, current_end = (starts[index], ends[index])
                current_min_channel = current_max_channel = channels[index]
            else:
                current.append(index)
                current_start, current_end = (proposed_start, proposed_end)
                current_min_channel = proposed_min_channel
                current_max_channel = proposed_max_channel
        if current:
            final_index_groups.append(np.asarray(current, dtype=int))
    scores = pd.to_numeric(work['detection_score'], errors='coerce').to_numpy(float)
    scales = pd.to_numeric(work['scale_s'], errors='coerce').to_numpy(float)
    locality = pd.to_numeric(work['z_locality'], errors='coerce').to_numpy(float)
    strict = _boolean_array(work['strict_candidate'])
    early = _boolean_array(work['early_caution'])
    negative = _boolean_array(work['negative_control_period'])
    common = _boolean_array(work['common_mode_or_leadin'])
    anomaly = _boolean_array(work['instrument_anomaly'])
    primitive_ids = work['primitive_id'].astype(str).to_numpy()
    event_rows: list[dict] = []
    member_rows: list[dict] = []
    for number, indices in enumerate(final_index_groups, start=1):
        representative_index = int(indices[np.nanargmax(scores[indices])])
        representative = work.iloc[representative_index]
        event_id = f'{event_prefix}_{number:07d}'
        peak_time = pd.Timestamp(representative.peak_time)
        exact_operation, exact_kind = operation_status(peak_time, buffered=False)
        buffered_operation, buffered_kind = operation_status(peak_time, buffered=True)
        unique_channels = np.unique(channels[indices])
        unique_scales = np.unique(scales[indices])
        event_rows.append({'physical_event_id': event_id, 'start_time': pd.Timestamp(np.min(starts[indices])), 'end_time': pd.Timestamp(np.max(ends[indices])), 'peak_time': peak_time, 'representative_data_channel': int(representative.data_channel), 'representative_layout_channel': int(representative.layout_channel), 'representative_sensor_id': str(representative.segment_id), 'representative_segment_id': str(representative.segment_id), 'geometry_type': str(representative.geometry_type), 'burial_depth_cm': float(representative.burial_depth_cm), 'x_cm': float(representative.x_cm), 'y_cm': float(representative.y_cm), 'stage_id': int(representative.stage_id), 'file_name': str(representative.file_name), 'file_index': int(representative.file_index), 'source_data_channels': ';'.join((str(v) for v in unique_channels)), 'source_min_data_channel': int(np.min(unique_channels)), 'source_max_data_channel': int(np.max(unique_channels)), 'primitive_count': int(len(indices)), 'detected_scale_count': int(len(unique_scales)), 'detected_scales': ';'.join((f'{v:g}' for v in unique_scales)), 'max_detection_score': float(np.nanmax(scores[indices])), 'strict_core_1s': bool(np.any(strict[indices])), 'strict_candidate': bool(np.any(strict[indices])), 'expanded_candidate': True, 'local_priority': bool(np.nanmax(locality[indices]) >= SECONDARY_EVIDENCE_Z), 'early_caution': bool(np.all(early[indices])), 'negative_control_period': bool(np.all(negative[indices])), 'common_mode_or_leadin': bool(np.any(common[indices])), 'instrument_anomaly': bool(np.any(anomaly[indices])), 'base_known_human_like': bool(buffered_operation), 'exact_operation_overlap': bool(exact_operation), 'exact_operation_kind': exact_kind, 'buffered_operation_overlap': bool(buffered_operation), 'buffered_operation_kind': buffered_kind, 'seconds_from_macro_observation': float((peak_time - MACRO_OBSERVATION).total_seconds())})
        member_rows.extend(({'physical_event_id': event_id, 'primitive_id': primitive_ids[index]} for index in indices))
    events = pd.DataFrame(event_rows)
    if not events.empty:
        events = events.sort_values('peak_time').reset_index(drop=True)
    log(f'[{progress_label}] 输入primitive={count:,}；输出事件={len(events):,}；成员关系={len(member_rows):,}')
    return (events, pd.DataFrame(member_rows))

def _read_primitive_cache(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=['start_time', 'end_time', 'peak_time'], low_memory=False)

def merge_primitives_in_batches(cache_paths: list[Path], records: list[DetectionRecord], digest: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按连续文件批量归并，并用边界重叠保证跨文件事件可连接。"""
    batch_dir = OUTPUT_DIR / 'merge_batch_cache' / digest
    batch_dir.mkdir(parents=True, exist_ok=True)
    overlap_s = 2.0 * MERGE_MAX_EVENT_SPAN_S + MERGE_MAX_PEAK_GAP_S
    batch_event_paths: list[Path] = []
    batch_member_paths: list[Path] = []
    total_batches = int(math.ceil(len(cache_paths) / MERGE_BATCH_FILES))
    for batch_number, first in enumerate(range(0, len(cache_paths), MERGE_BATCH_FILES), start=1):
        stop = min(len(cache_paths), first + MERGE_BATCH_FILES)
        event_path = batch_dir / f'events_{batch_number:03d}.csv.gz'
        member_path = batch_dir / f'members_{batch_number:03d}.csv.gz'
        meta_path = batch_dir / f'batch_{batch_number:03d}.json'
        relevant_start = max(0, first - 1)
        relevant_stop = min(len(cache_paths), stop + 1)
        meta = {'merge_algorithm_version': 2, 'signature_digest': digest, 'first_file': first, 'stop_file': stop, 'overlap_s': overlap_s, 'merge_parameters': {'peak_gap_s': MERGE_MAX_PEAK_GAP_S, 'event_span_s': MERGE_MAX_EVENT_SPAN_S, 'envelope_correlation_min': ENVELOPE_CORRELATION_MIN, 'spectral_js_max': SPECTRAL_JS_MAX, 'adjacent_distance_cm': ADJACENT_DISTANCE_CM, 'gauge_intervals': GAUGE_INTERVALS}, 'source_files': [cache_fingerprint(path) for path in cache_paths[relevant_start:relevant_stop]]}
        reusable = event_path.exists() and member_path.exists() and meta_path.exists()
        if reusable:
            try:
                reusable = json.loads(meta_path.read_text(encoding='utf-8')) == meta
            except Exception:
                reusable = False
        if reusable:
            log(f'[merge/batch] {batch_number}/{total_batches} 复用批次归并缓存')
            batch_event_paths.append(event_path)
            batch_member_paths.append(member_path)
            continue
        core_start = records[first].start
        if stop < len(records):
            core_end = records[stop].start
        else:
            last_record = records[stop - 1]
            core_end = last_record.start + pd.Timedelta(seconds=last_record.samples / SAMPLE_RATE_HZ)
        frames: list[pd.DataFrame] = []
        if first > 0:
            previous = _read_primitive_cache(cache_paths[first - 1])
            frames.append(previous[previous.peak_time >= core_start - pd.Timedelta(seconds=overlap_s)])
        for path in cache_paths[first:stop]:
            frames.append(_read_primitive_cache(path))
        if stop < len(cache_paths):
            following = _read_primitive_cache(cache_paths[stop])
            frames.append(following[following.peak_time < core_end + pd.Timedelta(seconds=overlap_s)])
        working = pd.concat(frames, ignore_index=True)
        log(f'[merge/batch] {batch_number}/{total_batches}；核心文件={first + 1}-{stop}；含边界候选={len(working):,}')
        local_events, local_members = merge_primitives(working, event_prefix=f'B{batch_number:03d}', progress_label=f'merge {batch_number}/{total_batches}')
        if not local_events.empty:
            keep = local_events.peak_time.ge(core_start) & local_events.peak_time.lt(core_end)
            local_events = local_events[keep].copy()
            keep_ids = set(local_events.physical_event_id.astype(str))
            local_members = local_members[local_members.physical_event_id.astype(str).isin(keep_ids)].copy()
        event_tmp = event_path.with_suffix(event_path.suffix + '.tmp')
        member_tmp = member_path.with_suffix(member_path.suffix + '.tmp')
        local_events.to_csv(event_tmp, index=False, compression='gzip')
        local_members.to_csv(member_tmp, index=False, compression='gzip')
        event_tmp.replace(event_path)
        member_tmp.replace(member_path)
        atomic_write_json(meta_path, meta)
        batch_event_paths.append(event_path)
        batch_member_paths.append(member_path)
        del working, frames, local_events, local_members
        log(f'[merge/batch] {batch_number}/{total_batches} 已保存批次断点')
    log('[merge/consolidate] 正在汇总批次事件并统一编号')
    event_frames = [pd.read_csv(path, parse_dates=['start_time', 'end_time', 'peak_time']) for path in batch_event_paths]
    member_frames = [pd.read_csv(path) for path in batch_member_paths]
    nonempty_event_frames = [frame for frame in event_frames if not frame.empty]
    nonempty_member_frames = [frame for frame in member_frames if not frame.empty]
    events = pd.concat(nonempty_event_frames, ignore_index=True) if nonempty_event_frames else event_frames[0].iloc[0:0].copy()
    members = pd.concat(nonempty_member_frames, ignore_index=True) if nonempty_member_frames else member_frames[0].iloc[0:0].copy()
    if events.empty:
        return (events, members)
    events = events.sort_values('peak_time').reset_index(drop=True)
    old_ids = events['physical_event_id'].astype(str).tolist()
    new_ids = [f'RGM_{i + 1:07d}' for i in range(len(events))]
    id_mapping = dict(zip(old_ids, new_ids))
    events['physical_event_id'] = new_ids
    members['physical_event_id'] = members['physical_event_id'].astype(str).map(id_mapping)
    if members['physical_event_id'].isna().any():
        raise RuntimeError('批次成员关系中存在无法映射的事件编号。')
    log(f'[merge/consolidate] 完成：事件={len(events):,}；成员关系={len(members):,}')
    return (events, members)

def select_early30_reference_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    从0--30 min内的“已检测候选事件”中构建第四类非目标参考事件表。

    选择原则
    --------
    1. 时间必须位于 [EXPERIMENT_START, EARLY30_END)；
    2. 排除已知人员/抽水缓冲重叠、仪器异常、明显共模/引入段异常；
    3. 若事件数超过 EARLY30_REFERENCE_MAX_EVENTS，则按“埋深×几何”分层并覆盖完整时间轴
       进行确定性抽样，避免第四类因数量过大而在KNN域中获得不成比例的密度优势；
    4. 该表只用于Branch B。Branch A完全不使用这些事件作为噪声参考。
    """
    work = events.copy()
    work['peak_time'] = pd.to_datetime(work['peak_time'], errors='coerce')
    eligible = work['peak_time'].ge(EXPERIMENT_START) & work['peak_time'].lt(EARLY30_END)
    for col in ['buffered_operation_overlap', 'instrument_anomaly', 'common_mode_or_leadin']:
        if col in work.columns:
            eligible &= ~work[col].fillna(False).astype(bool)
    reference = work.loc[eligible].copy().sort_values('peak_time').reset_index(drop=True)
    reference['early30_reference_eligible'] = True
    if reference.empty:
        log('[four-ref] 0--30 min没有符合条件的第四类参考事件；Branch B将退化为Baseline。')
        return reference
    max_events = EARLY30_REFERENCE_MAX_EVENTS
    if max_events is None or len(reference) <= int(max_events):
        reference['early30_reference_selected'] = True
        reference['early30_reference_selection_reason'] = 'all_eligible'
        return reference
    max_events = int(max_events)
    depth_text = pd.to_numeric(reference.get('burial_depth_cm'), errors='coerce').round(3).astype(str)
    geom_text = reference.get('geometry_type', pd.Series('unknown', index=reference.index)).astype(str).str.lower()
    reference['_stratum'] = depth_text + '|' + geom_text
    selected_indices: list[int] = []
    strata = [g for _, g in reference.groupby('_stratum', sort=True)]
    quota = max(1, max_events // max(1, len(strata)))
    for group in strata:
        group = group.sort_values('peak_time')
        take_n = min(quota, len(group))
        if take_n <= 0:
            continue
        pos = np.linspace(0, len(group) - 1, take_n).round().astype(int)
        selected_indices.extend(group.index.to_numpy()[pos].tolist())
    selected_indices = list(dict.fromkeys(selected_indices))
    if len(selected_indices) < max_events:
        remaining = reference.loc[~reference.index.isin(selected_indices)].sort_values('peak_time')
        need = min(max_events - len(selected_indices), len(remaining))
        if need > 0:
            pos = np.linspace(0, len(remaining) - 1, need).round().astype(int)
            selected_indices.extend(remaining.index.to_numpy()[pos].tolist())
    if len(selected_indices) > max_events:
        chosen = reference.loc[selected_indices].sort_values('peak_time')
        pos = np.linspace(0, len(chosen) - 1, max_events).round().astype(int)
        selected_indices = chosen.index.to_numpy()[pos].tolist()
    reference['early30_reference_selected'] = reference.index.isin(selected_indices)
    reference['early30_reference_selection_reason'] = np.where(reference['early30_reference_selected'], 'stratified_time_coverage', 'not_selected_due_to_reference_cap')
    reference = reference.drop(columns=['_stratum'], errors='ignore')
    log(f"[four-ref] 0--30 min符合条件事件={len(reference):,}；选作第四类参考={int(reference['early30_reference_selected'].sum()):,}")
    return reference

def build_four_reference_scored(baseline_scored: pd.DataFrame, baseline_bundle: dict, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    在Baseline完全相同的稳健标准化z空间中，构建“四类非目标参考域”。

    Branch A reference classes:
        stable_rain + personnel + pumping

    Branch B reference classes:
        stable_rain + personnel + pumping + early30_background

    关键控制
    --------
    - 不重新提取H5特征；
    - 不重新拟合标准化器；直接复用Baseline bundle['z']；
    - K_NEIGHBORS、NOISE_DISTANCE_QUANTILE、MIN_NOVEL_SCALES、
      MIN_CONSECUTIVE_NOVEL_SCALES均与Baseline一致；
    - 第四类来自0--30 min已检测候选的同一六尺度特征副本，因此两分支唯一实质差异
      是KNN非目标参考点集合是否包含early30_background。
    """
    base = baseline_scored.reset_index(drop=True).copy()
    z_base = np.asarray(baseline_bundle.get('z'), dtype=float)
    if z_base.ndim != 2 or len(z_base) != len(base):
        raise RuntimeError("四类参考域无法复用Baseline标准化结果：bundle['z']与scored行数不一致。")
    reference_events = select_early30_reference_events(events)
    if reference_events.empty:
        selected_ids: set[str] = set()
    else:
        selected_ids = set(reference_events.loc[reference_events['early30_reference_selected'].fillna(False).astype(bool), 'physical_event_id'].astype(str))
    candidate_mask = base['task_class'].astype(str).eq('candidate')
    source_mask = candidate_mask & base['physical_event_id'].astype(str).isin(selected_ids)
    source_indices = np.where(source_mask.to_numpy(bool))[0]
    if len(source_indices):
        early_ref = base.iloc[source_indices].copy()
        early_ref['reference_source_event_id'] = early_ref['physical_event_id'].astype(str)
        early_ref['task_class'] = EARLY30_REFERENCE_CLASS
        if 'task_id' in early_ref.columns:
            early_ref['task_id'] = 'early30_background:' + early_ref['reference_source_event_id'].astype(str)
        early_ref['physical_event_id'] = 'E30REF:' + early_ref['reference_source_event_id'].astype(str)
        four = pd.concat([base, early_ref], ignore_index=True, sort=False)
        z_all = np.vstack([z_base, z_base[source_indices]])
    else:
        four = base.copy()
        z_all = z_base.copy()
        four['reference_source_event_id'] = np.nan
    for col in ['noise_distance', 'noise_distance_threshold', 'noise_distance_ratio', 'nearest_noise_class', 'distance_to_stable_rain', 'distance_to_personnel', 'distance_to_pumping', f'distance_to_{EARLY30_REFERENCE_CLASS}']:
        if col == 'nearest_noise_class':
            four[col] = ''
        else:
            four[col] = np.nan
    k_neighbors = int(getattr(refine, 'K_NEIGHBORS', 10))
    calibration_rows: list[dict] = []
    neighbor_models: dict[float, NearestNeighbors] = {}
    scale_values = pd.to_numeric(four['scale_s'], errors='coerce').to_numpy(float)
    task_class = four['task_class'].astype(str)
    four_candidate = task_class.eq('candidate').to_numpy(bool)
    for scale in SCALES_S:
        scale_mask = np.isclose(scale_values, float(scale))
        ref_mask = scale_mask & task_class.isin(FOUR_REFERENCE_CLASSES).to_numpy(bool)
        cand_mask = scale_mask & four_candidate
        ref_indices = np.where(ref_mask)[0]
        cand_indices = np.where(cand_mask)[0]
        if len(ref_indices) < 3:
            raise RuntimeError(f'四类参考域 scale={scale:g}s 的参考点不足：{len(ref_indices)}')
        ref_z = z_all[ref_indices]
        k = min(k_neighbors, max(2, len(ref_z) - 1))
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(ref_z)), metric='euclidean').fit(ref_z)
        self_distances = nn.kneighbors(ref_z, return_distance=True)[0]
        loo = self_distances[:, min(k, self_distances.shape[1] - 1)]
        threshold = max(float(np.quantile(loo, NOISE_DISTANCE_QUANTILE)), 1e-09)
        four.loc[ref_indices, 'noise_distance'] = loo
        four.loc[ref_indices, 'noise_distance_threshold'] = threshold
        four.loc[ref_indices, 'noise_distance_ratio'] = loo / threshold
        if len(cand_indices):
            distances = nn.kneighbors(z_all[cand_indices], n_neighbors=min(k, len(ref_z)), return_distance=True)[0]
            cand_distance = distances[:, -1]
            four.loc[cand_indices, 'noise_distance'] = cand_distance
            four.loc[cand_indices, 'noise_distance_threshold'] = threshold
            four.loc[cand_indices, 'noise_distance_ratio'] = cand_distance / threshold
        available_labels: list[str] = []
        class_models: dict[str, NearestNeighbors] = {}
        for label in FOUR_REFERENCE_CLASSES:
            class_indices = np.where(scale_mask & task_class.eq(label).to_numpy(bool))[0]
            if len(class_indices) == 0:
                continue
            class_k = min(max(1, k_neighbors // 2), len(class_indices))
            class_models[label] = NearestNeighbors(n_neighbors=class_k, metric='euclidean').fit(z_all[class_indices])
            available_labels.append(label)
        for indices in [ref_indices, cand_indices]:
            if not len(indices) or not available_labels:
                continue
            class_distance = np.column_stack([class_models[label].kneighbors(z_all[indices], return_distance=True)[0][:, -1] for label in available_labels])
            labels_arr = np.asarray(available_labels, dtype=object)
            four.loc[indices, 'nearest_noise_class'] = labels_arr[np.argmin(class_distance, axis=1)]
            for class_index, label in enumerate(available_labels):
                four.loc[indices, f'distance_to_{label}'] = class_distance[:, class_index]
        counts_by_class = {label: int(np.sum(scale_mask & task_class.eq(label).to_numpy(bool))) for label in FOUR_REFERENCE_CLASSES}
        calibration_rows.append({'scale_s': float(scale), 'reference_domain': '+'.join(FOUR_REFERENCE_CLASSES), 'noise_distance_quantile': NOISE_DISTANCE_QUANTILE, 'noise_distance_threshold': threshold, 'reference_count': int(len(ref_indices)), 'k_neighbors': int(k), 'loo_distance_median': float(np.median(loo)), 'loo_distance_q95': float(np.quantile(loo, 0.95)), 'loo_distance_q99': float(np.quantile(loo, 0.99)), **{f'reference_count_{label}': counts_by_class[label] for label in FOUR_REFERENCE_CLASSES}})
        neighbor_models[float(scale)] = nn
        log(f'[four-ref] scale={scale:g}s；refs={len(ref_indices)}；early30={counts_by_class[EARLY30_REFERENCE_CLASS]}；threshold={threshold:.3f}')
    four['reference_domain'] = '+'.join(FOUR_REFERENCE_CLASSES)
    bundle_four = {'standardizers': baseline_bundle.get('standardizers'), 'standardizer_table': baseline_bundle.get('standardizer_table'), 'neighbors': neighbor_models, 'z': z_all, 'reference_classes': list(FOUR_REFERENCE_CLASSES), 'standardization_source': 'reused_from_baseline_three_reference_branch'}
    return (four, pd.DataFrame(calibration_rows), reference_events, bundle_four)

def apply_four_reference_catalog_tags(catalog: pd.DataFrame) -> pd.DataFrame:
    """Branch B正式标签：保持原筛选逻辑，仅把0--30 min本身按第四类定义固定为非目标。"""
    result = add_formal_screening_tags(catalog)
    result['peak_time'] = pd.to_datetime(result['peak_time'], errors='coerce')
    early = result['peak_time'].ge(EXPERIMENT_START) & result['peak_time'].lt(EARLY30_END)
    result['early30_reference_period'] = early
    result.loc[early, 'automatic_screening_label'] = 'nuisance_like'
    ordinary_early = early.copy()
    for col in ['instrument_anomaly', 'common_mode_or_leadin', 'human_like']:
        if col in result.columns:
            ordinary_early &= ~result[col].fillna(False).astype(bool)
    result.loc[ordinary_early, 'screening_pool'] = 'early30_reference_period'
    result['screening_branch'] = 'four_reference'
    result['reference_domain'] = '+'.join(FOUR_REFERENCE_CLASSES)
    return result

def build_three_vs_four_transition(baseline_catalog: pd.DataFrame, four_catalog: pd.DataFrame, reference_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = baseline_catalog[['physical_event_id', 'peak_time', 'automatic_screening_label', 'max_noise_distance_ratio', 'best_nearest_noise_class']].copy()
    base = base.rename(columns={'automatic_screening_label': 'baseline_label', 'max_noise_distance_ratio': 'baseline_max_noise_distance_ratio', 'best_nearest_noise_class': 'baseline_best_nearest_noise_class'})
    four = four_catalog[['physical_event_id', 'automatic_screening_label', 'max_noise_distance_ratio', 'best_nearest_noise_class']].copy()
    four = four.rename(columns={'automatic_screening_label': 'four_reference_label', 'max_noise_distance_ratio': 'four_reference_max_noise_distance_ratio', 'best_nearest_noise_class': 'four_reference_best_nearest_noise_class'})
    detail = base.merge(four, on='physical_event_id', how='inner', validate='one_to_one')
    detail['peak_time'] = pd.to_datetime(detail['peak_time'], errors='coerce')
    detail['label_changed'] = detail['baseline_label'].astype(str) != detail['four_reference_label'].astype(str)
    detail['post_30min'] = detail['peak_time'].ge(EARLY30_END)
    detail['became_early30_nearest'] = detail['four_reference_best_nearest_noise_class'].eq(EARLY30_REFERENCE_CLASS)
    transition = detail.groupby(['baseline_label', 'four_reference_label'], dropna=False).size().rename('event_count').reset_index().sort_values(['baseline_label', 'four_reference_label'])
    baseline_effective = int(detail['baseline_label'].eq('effective_candidate').sum())
    four_effective = int(detail['four_reference_label'].eq('effective_candidate').sum())
    post30_base_effective = detail[detail['post_30min'] & detail['baseline_label'].eq('effective_candidate')]
    post30_retained = int(post30_base_effective['four_reference_label'].eq('effective_candidate').sum())
    post30_downgraded = int((~post30_base_effective['four_reference_label'].eq('effective_candidate')).sum())
    selected_reference_count = int(reference_events['early30_reference_selected'].fillna(False).astype(bool).sum()) if not reference_events.empty and 'early30_reference_selected' in reference_events.columns else 0
    summary = pd.DataFrame([{'metric': 'baseline_effective_candidate_count', 'value': baseline_effective}, {'metric': 'four_reference_effective_candidate_count', 'value': four_effective}, {'metric': 'selected_early30_reference_event_count', 'value': selected_reference_count}, {'metric': 'post30_baseline_effective_count', 'value': int(len(post30_base_effective))}, {'metric': 'post30_baseline_effective_retained', 'value': post30_retained}, {'metric': 'post30_baseline_effective_downgraded', 'value': post30_downgraded}, {'metric': 'post30_effective_retention_fraction', 'value': float(post30_retained / len(post30_base_effective)) if len(post30_base_effective) else np.nan}, {'metric': 'all_effective_retention_fraction', 'value': float(four_effective / baseline_effective) if baseline_effective else np.nan}, {'metric': 'changed_events_with_early30_as_nearest_reference', 'value': int((detail['label_changed'] & detail['became_early30_nearest']).sum())}])
    return (detail, transition, summary)

def write_four_reference_manual_template(catalog: pd.DataFrame) -> None:
    columns = ['physical_event_id', 'peak_time', 'file_name', 'representative_data_channel', 'burial_depth_cm', 'geometry_type', 'reference_domain', 'automatic_screening_label', 'best_nearest_noise_class', 'novel_scale_count', 'max_consecutive_novel_scales', 'max_noise_distance_ratio', 'manual_label', 'manual_confidence', 'manual_reviewer', 'manual_notes', 'include_in_future_catrms']
    available = [c for c in columns if c in catalog.columns]
    catalog[available].to_csv(OUTPUT_DIR / 'manual_review_label_template_four_reference.csv', index=False, encoding='utf-8-sig')

def figure_three_vs_four_reference_summary(baseline: pd.DataFrame, four_reference: pd.DataFrame) -> None:
    """三类参考域 vs 四类参考域：标签计数与Baseline有效候选去向。"""
    try:
        refine.setup_style()
    except Exception:
        pass
    base_counts = baseline['automatic_screening_label'].value_counts()
    four_counts = four_reference['automatic_screening_label'].value_counts()
    labels = ['effective_candidate', 'uncertain', 'nuisance_like']
    x = np.arange(len(labels))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(getattr(refine, 'WIDE_WIDTH_IN', 6.62), 3.1), constrained_layout=True)
    axes[0].bar(x - width / 2, [int(base_counts.get(v, 0)) for v in labels], width=width, label='3-reference baseline')
    axes[0].bar(x + width / 2, [int(four_counts.get(v, 0)) for v in labels], width=width, label='4-reference')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Effective', 'Uncertain', 'Nuisance'])
    axes[0].set_ylabel('Event count')
    axes[0].set_title('a  Screening-label counts', loc='left', fontweight='bold')
    axes[0].legend(frameon=False, fontsize=8)
    compare = baseline[['physical_event_id', 'automatic_screening_label', 'peak_time']].merge(four_reference[['physical_event_id', 'automatic_screening_label']], on='physical_event_id', suffixes=('_base', '_four'), how='inner')
    compare['peak_time'] = pd.to_datetime(compare['peak_time'], errors='coerce')
    eff = compare[compare['automatic_screening_label_base'].eq('effective_candidate') & compare['peak_time'].ge(EARLY30_END)]
    outcome = eff['automatic_screening_label_four'].value_counts()
    axes[1].bar(['Retained effective', 'Uncertain', 'Nuisance'], [int(outcome.get('effective_candidate', 0)), int(outcome.get('uncertain', 0)), int(outcome.get('nuisance_like', 0))])
    axes[1].set_ylabel('Post-30-min baseline-effective count')
    axes[1].set_title('b  Effect of adding early30 reference', loc='left', fontweight='bold')
    axes[1].tick_params(axis='x', labelrotation=20)
    for tick in axes[1].get_xticklabels():
        tick.set_ha('right')
    stem = FIGURE_DIR / 'Figure07_three_vs_four_reference_sensitivity'
    fig.savefig(str(stem) + '.png', dpi=getattr(refine, 'PNG_DPI', 600))
    fig.savefig(str(stem) + '.pdf')
    plt.close(fig)

def add_formal_screening_tags(catalog: pd.DataFrame) -> pd.DataFrame:
    result = catalog.copy()
    result['human_like'] = result['buffered_operation_overlap'].astype(bool) | result['best_nearest_noise_class'].isin(['personnel', 'pumping']) & (result['max_noise_distance_ratio'] <= 1.0)
    forced_noise = result['human_like'] | result['common_mode_or_leadin'] | result['instrument_anomaly']
    result.loc[forced_noise, 'automatic_screening_label'] = 'nuisance_like'
    result['screening_pool'] = np.select([result.instrument_anomaly, result.common_mode_or_leadin, result.human_like, result.strict_candidate, result.local_priority, result.early_caution], ['instrument_anomaly', 'common_mode_or_leadin', 'human_like', 'strict_candidate', 'local_priority', 'early_caution'], default='expanded_candidate')
    return result

def preserve_manual_labels(catalog: pd.DataFrame) -> pd.DataFrame:
    path = OUTPUT_DIR / 'manual_review_label_template.csv'
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    keep = ['physical_event_id', 'manual_label', 'manual_confidence', 'manual_reviewer', 'manual_notes', 'include_in_future_catrms']
    if not old.empty and set(keep).issubset(old.columns):
        saved = old[keep].copy()
        catalog = catalog.drop(columns=[c for c in keep[1:] if c in catalog], errors='ignore').merge(saved, on='physical_event_id', how='left')
    for column, default in {'manual_label': 'pending', 'manual_confidence': '', 'manual_reviewer': '', 'manual_notes': '', 'include_in_future_catrms': False}.items():
        if column not in catalog:
            catalog[column] = default
        catalog[column] = catalog[column].fillna(default)
    columns = ['physical_event_id', 'peak_time', 'file_name', 'representative_data_channel', 'burial_depth_cm', 'geometry_type', 'screening_pool', 'automatic_screening_label', 'screening_score', 'novel_scale_count', 'max_noise_distance_ratio', 'best_scale_s', 'manual_label', 'manual_confidence', 'manual_reviewer', 'manual_notes', 'include_in_future_catrms']
    catalog[columns].to_csv(path, index=False, encoding='utf-8-sig')
    return catalog

def add_confirmed_effective_types(catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只对人工确认effective聚类；输入不包含时间、深度、坐标或几何类别。"""
    result = catalog.copy()
    result['confirmed_effective_type'] = 'not_assigned'
    confirmed = result['manual_label'].astype(str).str.lower().eq('effective')
    if int(confirmed.sum()) < refine.TYPE_MIN_EVENTS:
        return (result, pd.DataFrame())
    temporary = result.copy()
    temporary['automatic_screening_label'] = np.where(confirmed, 'effective_candidate', 'nuisance_like')
    typed, summary, bic = refine.add_preliminary_types(temporary)
    result.loc[confirmed, 'confirmed_effective_type'] = typed.loc[confirmed, 'preliminary_signal_type']
    summary.to_csv(OUTPUT_DIR / '09_confirmed_effective_type_summary.csv', index=False, encoding='utf-8-sig')
    bic.to_csv(OUTPUT_DIR / '10_confirmed_effective_type_model_selection.csv', index=False, encoding='utf-8-sig')
    return (result, summary)

def event_feature_matrix(scored: pd.DataFrame) -> pd.DataFrame:
    feature_columns = refine.MODEL_FEATURES
    parts = []
    for scale, group in scored.groupby('scale_s'):
        local = group[['task_id', 'task_class', 'physical_event_id', 'time', 'segment_id'] + feature_columns].copy()
        local = local.rename(columns={name: f's{scale:g}_{name}' for name in feature_columns})
        parts.append(local)
    result = parts[0]
    keys = ['task_id', 'task_class', 'physical_event_id', 'time', 'segment_id']
    for part in parts[1:]:
        result = result.merge(part, on=keys, how='inner')
    return result

def train_classifier(scored: pd.DataFrame, catalog: pd.DataFrame) -> None:
    labels = catalog[['physical_event_id', 'manual_label']].copy()
    labels.manual_label = labels.manual_label.astype(str).str.lower().str.strip()
    event_x = event_feature_matrix(scored)
    candidate = event_x[event_x.task_class.eq('candidate')].merge(labels, on='physical_event_id', how='left')
    candidate = candidate[candidate.manual_label.isin(['effective', 'noise'])]
    reference = event_x[event_x.task_class.isin(['stable_rain', 'personnel', 'pumping'])].copy()
    reference['manual_label'] = 'noise'
    data = pd.concat([candidate, reference], ignore_index=True)
    counts = data.manual_label.value_counts()
    if counts.get('effective', 0) < 20 or counts.get('noise', 0) < 40:
        log('[classifier] 人工标签不足：至少需要20个effective和40个noise；已跳过训练。')
        return
    feature_columns = [c for c in data if c.startswith('s') and c not in ['segment_id']]
    x = data[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(data[feature_columns].median()).to_numpy(float)
    y = data.manual_label.eq('effective').astype(int).to_numpy()
    time_block = pd.to_datetime(data.time).dt.floor('30min').astype(str)
    groups = (time_block + '|' + data.segment_id.astype(str)).to_numpy()
    folds = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=folds)
    probability = np.full(len(data), np.nan)
    for train, test in splitter.split(x, y, groups):
        model = RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample', min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
    pr_auc = float(average_precision_score(y, probability))
    precision, recall, thresholds = precision_recall_curve(y, probability)
    eligible = np.where(precision[:-1] >= 0.9)[0]
    threshold = float(thresholds[eligible[np.argmax(recall[:-1][eligible])]]) if len(eligible) else 0.5
    predicted = probability >= threshold
    metrics = {'grouped_pr_auc': pr_auc, 'threshold_at_precision_ge_0.90': threshold, 'recall_at_selected_threshold': float(recall_score(y, predicted)), 'effective_count': int(np.sum(y)), 'noise_count': int(np.sum(1 - y)), 'group_definition': '30-min time block + monitoring segment'}
    OUTPUT_DIR.joinpath('classifier_grouped_cv_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    data.assign(oof_effective_probability=probability).to_csv(OUTPUT_DIR / 'classifier_grouped_cv_predictions.csv', index=False, encoding='utf-8-sig')
    final_model = RandomForestClassifier(n_estimators=800, class_weight='balanced_subsample', min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1).fit(x, y)
    joblib.dump({'model': final_model, 'features': feature_columns, 'threshold': threshold}, OUTPUT_DIR / 'effective_vs_pooled_noise_classifier.joblib')
    log(f'[classifier] grouped PR-AUC={pr_auc:.3f}，阈值={threshold:.3f}')

def create_extended_review_panels(catalog: pd.DataFrame, scored: pd.DataFrame, metadata: pd.DataFrame, records: list[DetectionRecord]) -> None:
    """为高分effective/uncertain输出完整的人工复核证据图。"""
    selection = catalog[catalog.automatic_screening_label.isin(['effective_candidate', 'uncertain'])].copy()
    selection = selection.sort_values('screening_score', ascending=False).head(MAX_REVIEW_PANELS)
    target = target_metadata(metadata)
    controls = control_channels(metadata)
    for row in selection.itertuples(index=False):
        peak = pd.Timestamp(row.peak_time)
        record, local_second = locate_record(peak, records)
        half = 3.0
        start = max(0, int(round((local_second - half) * SAMPLE_RATE_HZ)))
        stop = min(record.samples, int(round((local_second + half) * SAMPLE_RATE_HZ)))
        with h5py.File(record.path, 'r') as h5:
            block = np.asarray(h5[H5_DATASET][start:stop, :], dtype=np.float32)
        relative = np.arange(start, stop) / SAMPLE_RATE_HZ - local_second
        center = int(row.representative_data_channel)
        gauge = np.arange(max(0, center - GAUGE_HALF_CHANNELS), min(block.shape[1] - 1, center + GAUGE_HALF_CHANNELS) + 1)
        local = np.asarray(block[:, gauge], dtype=float)
        trace = np.median(local, axis=1)
        baseline_mask = np.abs(relative) >= 2.0
        baseline_scale = 1.4826 * np.median(np.abs(trace[baseline_mask] - np.median(trace[baseline_mask])))
        trace_z = (trace - np.median(trace[baseline_mask])) / max(baseline_scale, 1e-12)
        local_scale = 1.4826 * np.median(np.abs(local - np.median(local, axis=0)), axis=0)
        heat = (local - np.median(local, axis=0)) / np.maximum(local_scale, 1e-12)
        valid_controls = controls[controls < block.shape[1]]
        control_trace = np.median(block[:, valid_controls], axis=1) if len(valid_controls) else np.zeros(len(block))
        control_scale = 1.4826 * np.median(np.abs(control_trace[baseline_mask] - np.median(control_trace[baseline_mask])))
        control_z = (control_trace - np.median(control_trace[baseline_mask])) / max(control_scale, 1e-12)
        fig, axes = plt.subplots(3, 3, figsize=(refine.WIDE_WIDTH_IN, refine.WIDE_WIDTH_IN * 0.9), constrained_layout=True)
        axes = axes.ravel()
        axes[0].plot(relative, trace_z, color='#333333', linewidth=0.55)
        axes[0].axvspan(-float(row.best_scale_s) / 2, float(row.best_scale_s) / 2, color='#009E73', alpha=0.15)
        axes[0].set(title='Raw gauge-median waveform', xlabel='Relative time (s)', ylabel='Robust amplitude')
        axes[1].imshow(heat.T, aspect='auto', origin='lower', extent=[relative[0], relative[-1], gauge[0], gauge[-1]], cmap='RdBu_r', vmin=-8, vmax=8)
        axes[1].set(title='Gauge channel-time response', xlabel='Relative time (s)', ylabel='DAS channel')
        f, t, power = spectrogram(trace_z, fs=SAMPLE_RATE_HZ, nperseg=512, noverlap=448, scaling='density')
        mask = (f >= 20) & (f <= 2000)
        axes[2].pcolormesh(t + relative[0], f[mask], 10 * np.log10(np.maximum(power[mask], 1e-20)), shading='auto', cmap='magma')
        axes[2].set(title='Time-frequency representation', xlabel='Relative time (s)', ylabel='Frequency (Hz)')
        frame = max(1, int(round(0.05 * SAMPLE_RATE_HZ)))
        usable = len(trace_z) // frame * frame
        envelope = np.sqrt(np.mean(trace_z[:usable].reshape(-1, frame) ** 2, axis=1))
        envelope_time = relative[:usable:frame] + 0.025
        axes[3].plot(envelope_time, envelope, color='#0072B2', linewidth=0.8)
        axes[3].set(title='0.05-s RMS envelope', xlabel='Relative time (s)', ylabel='RMS')
        axes[4].plot(relative, trace_z, color='#009E73', linewidth=0.55, label='slope gauge')
        axes[4].plot(relative, control_z, color='#777777', linewidth=0.55, alpha=0.8, label='lead-in control')
        axes[4].set(title='Slope versus lead-in', xlabel='Relative time (s)', ylabel='Robust amplitude')
        axes[4].legend(frameon=False, fontsize=8)
        event_mask = np.abs(relative) <= float(row.best_scale_s) / 2
        before_mask = (relative >= -2.5) & (relative <= -1.5)
        after_mask = (relative >= 1.5) & (relative <= 2.5)
        for mask_local, label, color in [(before_mask, 'before', '#0072B2'), (event_mask, 'event', '#009E73'), (after_mask, 'after', '#D55E00')]:
            if np.sum(mask_local) >= 64:
                freq, psd = periodogram(trace_z[mask_local], fs=SAMPLE_RATE_HZ, scaling='density')
                keep = (freq >= 20) & (freq <= 2000)
                axes[5].plot(freq[keep], 10 * np.log10(np.maximum(psd[keep], 1e-20)), color=color, linewidth=0.7, label=label)
        axes[5].set_xscale('log')
        axes[5].set(title='Before-event-after spectrum', xlabel='Frequency (Hz)', ylabel='PSD (dB)')
        axes[5].legend(frameon=False, fontsize=8)
        scale_rows = scored[scored.task_class.eq('candidate') & scored.physical_event_id.eq(row.physical_event_id)].sort_values('scale_s')
        axes[6].bar(scale_rows.scale_s.astype(str), scale_rows.noise_distance_ratio, color=np.where(scale_rows.noise_distance_ratio > 1, '#009E73', '#999999'))
        axes[6].axhline(1.0, color='#333333', linestyle='--', linewidth=0.8)
        axes[6].set(title='Multiscale noise-domain distance', xlabel='Window scale (s)', ylabel='Distance ratio')
        axes[7].scatter(target.x_cm, target.y_cm, s=7, color='#BBBBBB', alpha=0.55)
        same_depth = target[np.isclose(target.burial_depth_cm, float(row.burial_depth_cm))]
        local_points = same_depth[np.abs(same_depth.data_channel.astype(int) - center) <= GAUGE_INTERVALS]
        axes[7].scatter(local_points.x_cm, local_points.y_cm, s=20, color='#E69F00', label='gauge overlap')
        axes[7].scatter([row.x_cm], [row.y_cm], s=45, marker='*', color='#D55E00', label='event centre')
        axes[7].set(title=f'Local geometry, depth {row.burial_depth_cm:g} cm', xlabel='x (cm)', ylabel='y (cm)')
        axes[7].legend(frameon=False, fontsize=8)
        axes[8].axis('off')
        evidence = f'Event: {row.physical_event_id}\nTime: {peak}\nAuto label: {row.automatic_screening_label}\nPool: {row.screening_pool}\nNovel scales: {row.novel_scale_count}/6\nBest scale: {row.best_scale_s:g} s\nGauge correlation: {row.max_gauge_abs_correlation:.3f}\nLocality: {row.max_locality_log_ratio:.3f}\nOperation overlap: {row.buffered_operation_overlap}\nCommon/instrument: {row.common_mode_or_leadin}/{row.instrument_anomaly}'
        axes[8].text(0.02, 0.98, evidence, va='top', ha='left', fontsize=8)
        fig.suptitle('Event-level manual review evidence', fontsize=10)
        fig.savefig(REVIEW_DIR / f'{row.physical_event_id}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

def write_method_readme(baseline_catalog: pd.DataFrame, four_reference_catalog: pd.DataFrame | None=None) -> None:
    base_counts = baseline_catalog.automatic_screening_label.value_counts().to_dict()
    four_counts = four_reference_catalog.automatic_screening_label.value_counts().to_dict() if four_reference_catalog is not None else {}
    text = f"# S1 multiscale screening: reference-domain comparison\n\nThe input is the raw DAS HDF5 waveform.\n\n- 标距：{GAUGE_LENGTH_M:g} m；通道间隔：{CHANNEL_SPACING_M:g} m；约{GAUGE_INTERVALS}个采样间隔。\n- 分析尺度：{', '.join((f'{v:g} s' for v in SCALES_S))}。\n- 正式分析截止：{ANALYSIS_END}（首次大规模坡面土体滑落）。\n- 为保证2 s最大特征窗不跨越截止时刻，候选峰值安全截止：{FEATURE_SAFE_END}。\n\n## Branch A：三类非目标参考域（Baseline）\n\n参考类别：stable_rain + personnel + pumping。\n\n- effective_candidate：{base_counts.get('effective_candidate', 0)}\n- uncertain：{base_counts.get('uncertain', 0)}\n- nuisance_like：{base_counts.get('nuisance_like', 0)}\n\n## Branch B：四类非目标参考域\n\n在Branch A完全相同的候选、六尺度特征、稳健标准化、KNN参数、距离分位数阈值、\n多尺度投票规则、局部性/标距相干判据基础上，仅把降雨开始后0--30 min内检测到的\n候选瞬态加入第四类非目标参考信号 `early30_background`。\n\n参考类别：stable_rain + personnel + pumping + early30_background。\n\n- effective_candidate：{four_counts.get('effective_candidate', 0)}\n- uncertain：{four_counts.get('uncertain', 0)}\n- nuisance_like：{four_counts.get('nuisance_like', 0)}\n\n为了避免第四类样本数远大于原三类导致KNN密度偏置，默认最多选择\n{(EARLY30_REFERENCE_MAX_EVENTS if EARLY30_REFERENCE_MAX_EVENTS is not None else '全部')}个早期事件，\n并按埋深×几何分层且覆盖完整0--30 min时间轴。被选事件及选择状态保存在\n`09_early30_fourth_reference_events.csv`。\n\n两套自动结果并行保留；04_formal_candidate_catalog.csv继续指向Baseline，兼容原下游程序。\n自动标签不是物理真值。最终仍需结合原始波形、时频图、空间响应、视频和日志进行人工复核。\n"
    (OUTPUT_DIR / 'README_原始数据重新检测.md').write_text(text, encoding='utf-8')

def raw_scan_signature(target: pd.DataFrame, records: list[DetectionRecord]) -> dict:
    dependency_paths = [BASE_RESULT_DIR / 'monitoring_channel_layout.csv', BASE_RESULT_DIR / 'rainfall_stage_blocks.csv']
    dependencies = []
    for path in dependency_paths:
        if path.exists():
            stat = path.stat()
            dependencies.append({'name': path.name, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    return {'version': 3, 'h5_files': [record.path.name for record in records], 'h5_samples': [record.samples for record in records], 'target_channels': target.data_channel.astype(int).tolist(), 'scales_s': list(SCALES_S), 'gauge_length_m': GAUGE_LENGTH_M, 'channel_spacing_m': CHANNEL_SPACING_M, 'expanded_z': EXPANDED_Z, 'strict_z': STRICT_Z, 'secondary_evidence_z': SECONDARY_EVIDENCE_Z, 'max_global_active_fraction': MAX_GLOBAL_ACTIVE_FRACTION, 'common_mode_control_z': COMMON_MODE_CONTROL_Z, 'chunk_duration_s': CHUNK_DURATION_S, 'chunk_overlap_s': CHUNK_OVERLAP_S, 'calibration_times_per_stage': CALIBRATION_TIMES_PER_STAGE, 'random_seed': RANDOM_SEED, 'analysis_end': ANALYSIS_END.isoformat(), 'feature_safe_end': FEATURE_SAFE_END.isoformat(), 'network_dependencies': dependencies}

def legacy_signature_compatible(old: dict | None, current: dict) -> bool:
    """兼容原版version=2签名；只接受其中已有检测字段全部一致的情况。"""
    if old is None:
        return False
    comparable = ['h5_files', 'h5_samples', 'target_channels', 'scales_s', 'gauge_length_m', 'channel_spacing_m', 'expanded_z', 'strict_z', 'secondary_evidence_z', 'calibration_times_per_stage']
    return all((old.get(key) == current.get(key) for key in comparable))

def run(mode: str) -> None:
    np.random.seed(RANDOM_SEED)
    configure_refiner()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    metadata, stages = load_network_and_stages()
    target = target_metadata(metadata)
    controls = control_channels(metadata)
    all_records = records_from_h5()
    records = [record for record in all_records if record.start < ANALYSIS_END]
    selected = required_dataset_channels(target, controls)
    log(f'监测中心={len(target)}；控制通道={len(controls)}；读取通道={len(selected)}；原始H5={len(all_records)}；22:40前需处理H5={len(records)}')
    log(f'正式分析截止={ANALYSIS_END}；六尺度特征安全截止={FEATURE_SAFE_END}')
    log('Input: raw DAS HDF5 waveform')
    signature = raw_scan_signature(target, records)
    digest = signature_digest(signature)
    signature_path = OUTPUT_DIR / 'raw_scan_signature.json'
    try:
        old_signature = json.loads(signature_path.read_text(encoding='utf-8')) if signature_path.exists() else None
    except Exception:
        old_signature = None
    configuration_changed = old_signature != signature
    if configuration_changed and old_signature is not None:
        if legacy_signature_compatible(old_signature, signature):
            log('[cache] 原版签名的检测字段一致，可验证并复用逐文件缓存')
        else:
            log('[cache] 检测配置发生变化；旧缓存保留，新配置使用独立缓存目录')
    smoke = mode == 'smoke'
    cache_digest = f'smoke_{digest}' if smoke else digest
    active_records = records[:1] if smoke else records
    legacy_cache_dir = OUTPUT_DIR / ('primitive_file_cache_smoke' if smoke else 'primitive_file_cache')
    hashed_cache_dir = OUTPUT_DIR / 'primitive_file_cache_by_signature' / cache_digest
    legacy_complete = complete_legacy_cache_exists(legacy_cache_dir, records, smoke)
    legacy_manifest_path = legacy_cache_dir / 'cache_signature.json'
    try:
        legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding='utf-8')) if legacy_manifest_path.exists() else None
    except Exception:
        legacy_manifest = None
    legacy_registered_for_current = isinstance(legacy_manifest, dict) and legacy_manifest.get('signature_digest') == cache_digest
    unregistered_legacy_may_be_adopted = legacy_manifest is None and (old_signature is None or (isinstance(old_signature, dict) and int(old_signature.get('version', 0)) <= 2 and legacy_signature_compatible(old_signature, signature))) and ADOPT_COMPLETE_LEGACY_FILE_CACHE
    may_adopt_legacy = legacy_complete and (legacy_registered_for_current or unregistered_legacy_may_be_adopted)
    if may_adopt_legacy:
        file_cache_dir = legacy_cache_dir
        if unregistered_legacy_may_be_adopted:
            log('[cache] 找到完整旧式逐文件缓存；将逐个验证结构、H5归属和时间范围后复用')
            legacy_cache_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(legacy_manifest_path, {'signature_digest': cache_digest, 'signature': signature})
        else:
            log(f'[cache] 使用已有逐文件缓存目录：{file_cache_dir}')
    else:
        file_cache_dir = hashed_cache_dir
        log(f'[cache] 使用配置隔离缓存目录：{file_cache_dir}')
    if not smoke:
        atomic_write_json(signature_path, signature)
        log(f'[cache] 当前检测配置签名已保存：{digest}')
    calibration_records = records[:1] if mode == 'smoke' else records
    with timed_stage('阶段1/7 背景校准'):
        samples, models = build_calibration(calibration_records, metadata, stages, target, controls, selected, force=mode == 'smoke' or configuration_changed)
    with timed_stage('阶段2/7 原始H5扫描或逐文件缓存验证'):
        cache_paths, primitive_counts = scan_all_h5(records, metadata, stages, target, controls, selected, models, file_cache_dir=file_cache_dir, smoke=smoke, reuse_existing=True)
    primitive_output = OUTPUT_DIR / ('01_multiscale_primitive_detections_smoke.csv.gz' if smoke else '01_multiscale_primitive_detections.csv.gz')
    if WRITE_COMBINED_PRIMITIVE_CATALOG:
        with timed_stage('阶段3/7 流式汇总唯一primitive目录'):
            combine_primitive_caches(cache_paths, primitive_output, cache_digest)
    else:
        log('[combine] 已按配置跳过总primitive目录；逐文件缓存仍完整保留')
    with timed_stage('阶段4/7 分批归并候选'):
        events, members = merge_primitives_in_batches(cache_paths, active_records, cache_digest)
        if not events.empty:
            events['peak_time'] = pd.to_datetime(events['peak_time'], errors='coerce')
            events['end_time'] = pd.to_datetime(events['end_time'], errors='coerce')
            before_count = len(events)
            keep = events['end_time'].le(ANALYSIS_END) & events['peak_time'].le(FEATURE_SAFE_END)
            events = events.loc[keep].copy().reset_index(drop=True)
            keep_ids = set(events['physical_event_id'].astype(str))
            members = members[members['physical_event_id'].astype(str).isin(keep_ids)].copy()
            log(f'[cutoff] 归并事件 {before_count:,} -> {len(events):,}；仅保留peak<={FEATURE_SAFE_END.time()}且event_end<={ANALYSIS_END.time()}')
        event_output = OUTPUT_DIR / '02_gauge_similarity_merged_events.csv'
        member_output = OUTPUT_DIR / '03_event_primitive_members.csv'
        event_tmp = event_output.with_suffix(event_output.suffix + '.tmp')
        member_tmp = member_output.with_suffix(member_output.suffix + '.tmp')
        events.to_csv(event_tmp, index=False, encoding='utf-8-sig')
        members.to_csv(member_tmp, index=False, encoding='utf-8-sig')
        event_tmp.replace(event_output)
        member_tmp.replace(member_output)
        log(f'[merge] 正式事件目录已保存：{event_output.name}')
    if events.empty:
        log('没有检测到候选；请检查路径、数据通道和门槛。')
        return
    if MAX_EVENTS_FOR_FEATURE_EXTRACTION is not None and len(events) > MAX_EVENTS_FOR_FEATURE_EXTRACTION:
        raise RuntimeError(f'归并后仍有{len(events):,}个事件，超过六尺度特征提取保护上限{MAX_EVENTS_FOR_FEATURE_EXTRACTION:,}。归并结果已安全保存；请先检查背景校准与候选阈值，再决定是否提高上限。')
    with timed_stage('阶段5/7 六尺度特征提取与开放集复筛'):
        tasks = refine.build_feature_tasks(events, metadata, stages, max_events=None)
        if isinstance(tasks, pd.DataFrame) and 'time' in tasks.columns:
            task_time = pd.to_datetime(tasks['time'], errors='coerce')
            before_tasks = len(tasks)
            tasks = tasks.loc[task_time.le(FEATURE_SAFE_END)].copy().reset_index(drop=True)
            log(f'[cutoff] 特征任务 {before_tasks:,} -> {len(tasks):,}；截止={FEATURE_SAFE_END}')
        log(f'[feature] 事件={len(events):,}；特征任务={len(tasks):,}')
        refine_records = [refine.H5Record(r.path, r.start) for r in records]
        log('[feature] 开始从原始H5提取六尺度特征')
        raw_features = refine.extract_all_multiscale_features(tasks, metadata, refine_records)
        if isinstance(raw_features, pd.DataFrame) and 'time' in raw_features.columns:
            feature_time = pd.to_datetime(raw_features['time'], errors='coerce')
            raw_features = raw_features.loc[feature_time.le(FEATURE_SAFE_END)].copy().reset_index(drop=True)
        log(f'[feature] 原始特征提取完成：{len(raw_features):,}行')
        baseline_scored, baseline_calibration, baseline_bundle = refine.add_noise_domain_distances(raw_features)
        log('[feature] Branch A（三类参考域）噪声域距离计算完成')
        baseline_catalog, baseline_votes = refine.aggregate_candidate_screening(baseline_scored, events)
        baseline_catalog = add_formal_screening_tags(baseline_catalog)
        baseline_catalog = preserve_manual_labels(baseline_catalog)
        baseline_catalog, confirmed_type_summary = add_confirmed_effective_types(baseline_catalog)
        baseline_catalog['preliminary_signal_type'] = 'unclassified'
        baseline_catalog['screening_branch'] = 'baseline_three_reference'
        baseline_catalog['reference_domain'] = '+'.join(BASELINE_REFERENCE_CLASSES)
        four_scored, four_calibration, early30_reference_events, four_bundle = build_four_reference_scored(baseline_scored, baseline_bundle, events)
        four_catalog, four_votes = refine.aggregate_candidate_screening(four_scored, events)
        four_catalog = apply_four_reference_catalog_tags(four_catalog)
        manual_cols = ['physical_event_id', 'manual_label', 'manual_confidence', 'manual_reviewer', 'manual_notes', 'include_in_future_catrms']
        available_manual = [c for c in manual_cols if c in baseline_catalog.columns]
        if available_manual:
            four_catalog = four_catalog.drop(columns=[c for c in available_manual[1:] if c in four_catalog.columns], errors='ignore').merge(baseline_catalog[available_manual], on='physical_event_id', how='left', validate='one_to_one')
        write_four_reference_manual_template(four_catalog)
        transition_detail, transition_table, branch_summary = build_three_vs_four_transition(baseline_catalog, four_catalog, early30_reference_events)
        catalog = baseline_catalog
        scored = baseline_scored
        votes = baseline_votes
        calibration = baseline_calibration
        bundle = baseline_bundle
    with timed_stage('阶段6/7 保存正式筛选结果'):
        baseline_catalog.to_csv(OUTPUT_DIR / '04_formal_candidate_catalog.csv', index=False, encoding='utf-8-sig')
        baseline_catalog.to_csv(OUTPUT_DIR / '04A_formal_candidate_catalog_baseline_three_reference.csv', index=False, encoding='utf-8-sig')
        four_catalog.to_csv(OUTPUT_DIR / '04B_formal_candidate_catalog_four_reference.csv', index=False, encoding='utf-8-sig')
        baseline_scored.to_csv(OUTPUT_DIR / '05_multiscale_scored_features.csv.gz', index=False, compression='gzip')
        baseline_scored.to_csv(OUTPUT_DIR / '05A_multiscale_scored_features_baseline_three_reference.csv.gz', index=False, compression='gzip')
        four_scored.to_csv(OUTPUT_DIR / '05B_multiscale_scored_features_four_reference.csv.gz', index=False, compression='gzip')
        baseline_votes.to_csv(OUTPUT_DIR / '06A_event_scale_votes_baseline_three_reference.csv', index=False, encoding='utf-8-sig')
        four_votes.to_csv(OUTPUT_DIR / '06B_event_scale_votes_four_reference.csv', index=False, encoding='utf-8-sig')
        baseline_calibration.to_csv(OUTPUT_DIR / '07A_noise_distance_calibration_baseline_three_reference.csv', index=False, encoding='utf-8-sig')
        four_calibration.to_csv(OUTPUT_DIR / '07B_noise_distance_calibration_four_reference.csv', index=False, encoding='utf-8-sig')
        baseline_bundle['standardizer_table'].to_csv(OUTPUT_DIR / '08_background_standardizers_shared.csv', index=False, encoding='utf-8-sig')
        early30_reference_events.to_csv(OUTPUT_DIR / '09_early30_fourth_reference_events.csv', index=False, encoding='utf-8-sig')
        transition_detail.to_csv(OUTPUT_DIR / '10A_eventwise_transition_three_vs_four_reference.csv', index=False, encoding='utf-8-sig')
        transition_table.to_csv(OUTPUT_DIR / '10B_transition_counts_three_vs_four_reference.csv', index=False, encoding='utf-8-sig')
        branch_summary.to_csv(OUTPUT_DIR / '11_screening_branch_summary.csv', index=False, encoding='utf-8-sig')
        atomic_write_json(OUTPUT_DIR / '12_dual_branch_configuration.json', {'analysis_end': ANALYSIS_END.isoformat(), 'feature_safe_end': FEATURE_SAFE_END.isoformat(), 'early30_reference_start': EXPERIMENT_START.isoformat(), 'early30_reference_end': EARLY30_END.isoformat(), 'baseline_reference_classes': list(BASELINE_REFERENCE_CLASSES), 'four_reference_classes': list(FOUR_REFERENCE_CLASSES), 'early30_reference_max_events': EARLY30_REFERENCE_MAX_EVENTS, 'noise_distance_quantile_shared': NOISE_DISTANCE_QUANTILE, 'k_neighbors_shared': int(getattr(refine, 'K_NEIGHBORS', 10)), 'min_novel_scales_shared': MIN_NOVEL_SCALES, 'min_consecutive_novel_scales_shared': MIN_CONSECUTIVE_NOVEL_SCALES, 'standardization_shared': True, 'standardization_source': 'Baseline three-reference robust standardizers', 'candidate_pool_shared': True, 'feature_matrix_shared': True, 'random_seed_shared': RANDOM_SEED, 'only_intended_branch_difference': 'add early30_background to pooled KNN non-target reference domain'})
        baseline_votes.to_csv(OUTPUT_DIR / '06_event_scale_votes.csv', index=False, encoding='utf-8-sig')
        baseline_calibration.to_csv(OUTPUT_DIR / '07_noise_distance_calibration.csv', index=False, encoding='utf-8-sig')
        log(f'[save] 三类Baseline事件={len(baseline_catalog):,}；四类参考域事件={len(four_catalog):,}；Baseline多尺度评分={len(baseline_scored):,}行；Four-reference评分={len(four_scored):,}行')
        for row in branch_summary.itertuples(index=False):
            log(f'[branch] {row.metric}={row.value}')
    with timed_stage('阶段7/7 生成论文图与人工复核图'):
        figure_steps = [('频谱比较图', lambda: refine.figure_spectral_comparison(scored, catalog)), ('多尺度筛选图', lambda: refine.figure_multiscale_screening(scored, catalog)), ('特征比较图', lambda: refine.figure_feature_comparison(scored, catalog)), ('时间-深度-几何图', lambda: refine.figure_time_depth_geometry(catalog)), ('代表性信号图', lambda: refine.figure_representative_signals(scored, catalog, refine_records)), ('人工复核证据图', lambda: create_extended_review_panels(baseline_catalog, scored, metadata, records)), ('三类-vs-四类参考域敏感性图', lambda: figure_three_vs_four_reference_summary(baseline_catalog, four_catalog))]
        for number, (name, function) in enumerate(figure_steps, start=1):
            log(f'[figure] {number}/{len(figure_steps)} {name}：开始')
            function()
            log(f'[figure] {number}/{len(figure_steps)} {name}：完成')
        write_method_readme(baseline_catalog, four_catalog)
    if mode == 'train':
        train_classifier(scored, catalog)
    log(f'完成：{OUTPUT_DIR}')

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='S1原始H5标距约束多尺度候选筛选')
    parser.add_argument('mode', nargs='?', choices=['all', 'rebuild', 'train', 'smoke'], default=RUN_MODE)
    args = parser.parse_args(argv)
    run(args.mode)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
