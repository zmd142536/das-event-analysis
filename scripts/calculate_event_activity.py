"""Compute gauge-median event squared strain-rate integrals and cumulative activity."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
import h5py
import numpy as np
import pandas as pd
DEFAULT_EXPERIMENT = 'S1'
DATASET_PATH = '/default'
EXPECTED_SAMPLE_RATE_HZ = 5000.0
EXPECTED_BANDPASS_HZ = (20.0, 2000.0)
GAUGE_HALF_CHANNELS = 4

class InputError(RuntimeError):
    """输入数据、元数据或事件目录不满足计算前提。"""

@dataclass(frozen=True)
class ExperimentConfig:
    code: str
    h5_dir: Path
    catalog: Path
    coordinates: Path
    output_dir: Path
    analysis_end: pd.Timestamp
    precursor_end: pd.Timestamp
    macro_observation: pd.Timestamp
EXPERIMENT_CONFIGS = {'S1': ExperimentConfig(code='S1', h5_dir=REPOSITORY_ROOT / 'data/preprocessed/S1', catalog=REPOSITORY_ROOT / 'data/catalogs/S1_activity_catalog_04B.csv.gz', coordinates=REPOSITORY_ROOT / 'data/coordinates/S1_coordinates.xlsx', output_dir=REPOSITORY_ROOT / 'outputs/event_activity/S1', analysis_end=pd.Timestamp('2025-07-07 22:40:00'), precursor_end=pd.Timestamp('2025-07-07 22:38:39'), macro_observation=pd.Timestamp('2025-07-07 22:38:39')), 'S2': ExperimentConfig(code='S2', h5_dir=REPOSITORY_ROOT / 'data/preprocessed/S2', catalog=REPOSITORY_ROOT / 'data/catalogs/S2_activity_catalog_04B.csv.gz', coordinates=REPOSITORY_ROOT / 'data/coordinates/S2_S4_coordinates.xlsx', output_dir=REPOSITORY_ROOT / 'outputs/event_activity/S2', analysis_end=pd.Timestamp('2025-07-19 20:40:00'), precursor_end=pd.Timestamp('2025-07-19 20:37:47'), macro_observation=pd.Timestamp('2025-07-19 20:37:47')), 'S4': ExperimentConfig(code='S4', h5_dir=REPOSITORY_ROOT / 'data/preprocessed/S4', catalog=REPOSITORY_ROOT / 'data/catalogs/S4_activity_catalog_04B.csv.gz', coordinates=REPOSITORY_ROOT / 'data/coordinates/S2_S4_coordinates.xlsx', output_dir=REPOSITORY_ROOT / 'outputs/event_activity/S4', analysis_end=pd.Timestamp('2025-08-11 20:30:00'), precursor_end=pd.Timestamp('2025-08-11 20:28:46'), macro_observation=pd.Timestamp('2025-08-11 20:29:07'))}

@dataclass(frozen=True)
class H5Record:
    path: Path
    start: pd.Timestamp
    end: pd.Timestamp
    sample_rate_hz: float
    sample_count: int
    channel_count: int
    start_channel: int
    bandpass_hz: tuple[float, float]

def log(message: str) -> None:
    print(message, flush=True)

def parse_args(argv: list[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=tuple(EXPERIMENT_CONFIGS), default=DEFAULT_EXPERIMENT, help='实验编号；默认读取脚本顶部 DEFAULT_EXPERIMENT。')
    parser.add_argument('--h5-dir', type=Path, help='覆盖所选实验的默认H5目录。')
    parser.add_argument('--catalog', type=Path, help='覆盖所选实验的默认四参照事件目录。')
    parser.add_argument('--coordinates', type=Path, help='覆盖所选实验的默认坐标文件。')
    parser.add_argument('--output-dir', type=Path, help='覆盖所选实验的默认输出目录。')
    parser.add_argument('--dataset-path', default=DATASET_PATH)
    parser.add_argument('--time-scope', choices=('analysis', 'precursor'), default='analysis', help='analysis=累计至正式分析截止；precursor=仅累计前兆截止之前的事件。')
    parser.add_argument('--cutoff-time', type=pd.Timestamp, help='覆盖配置中的累计截止时刻；筛选规则为 peak_time < cutoff_time。')
    parser.add_argument('--event-policy', choices=('auto_effective', 'auto_effective_plus_uncertain', 'manual_effective', 'include_flag'), default='auto_effective_plus_uncertain', help='默认采用自动effective_candidate＋uncertain事件。')
    parser.add_argument('--input-unit', default='microstrain_per_second', help='H5应变率数值单位标签；数值单位为microstrain_per_second处理。该参数只写入结果元数据，不改变数值。')
    parser.add_argument('--snapshot-minutes', type=int, default=1, help='空间累计输出间隔（分钟）；默认1，即每分钟一个四列TXT。')
    parser.add_argument('--max-events', type=int, help='仅用于快速试算；正式运行不要设置。')
    args = parser.parse_args(argv)
    config = EXPERIMENT_CONFIGS[args.experiment]
    args.h5_dir = args.h5_dir or config.h5_dir
    args.catalog = args.catalog or config.catalog
    args.coordinates = args.coordinates or config.coordinates
    args.output_dir = args.output_dir or config.output_dir / args.time_scope
    if args.cutoff_time is None:
        args.cutoff_time = config.analysis_end if args.time_scope == 'analysis' else config.precursor_end
    args.macro_observation = config.macro_observation
    return args

def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'{label}不存在: {path}')

def file_sha256(path: Path, chunk_size: int=8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while (chunk := handle.read(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()

def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f'不能序列化 {type(value).__name__}')

def inspect_h5_records(h5_dir: Path, dataset_path: str) -> list[H5Record]:
    if not h5_dir.is_dir():
        raise FileNotFoundError(f'H5目录不存在: {h5_dir}')
    paths = sorted([*h5_dir.glob('*.h5'), *h5_dir.glob('*.hdf5')])
    if not paths:
        raise InputError(f'H5目录中没有文件: {h5_dir}')
    records: list[H5Record] = []
    for path in paths:
        with h5py.File(path, 'r') as handle:
            if dataset_path not in handle:
                raise InputError(f'{path.name}: 缺少数据集 {dataset_path}')
            dataset = handle[dataset_path]
            if dataset.ndim != 2:
                raise InputError(f'{path.name}: 数据集应为二维，实际为 {dataset.shape}')
            sample_rate = float(dataset.attrs.get('sampling_rate', np.nan))
            if not np.isfinite(sample_rate):
                raise InputError(f'{path.name}: 缺少sampling_rate')
            epoch = int(dataset.attrs['epoch'])
            ns = int(dataset.attrs.get('ns', 0))
            start = pd.Timestamp(epoch, unit='s', tz='UTC').tz_convert('Asia/Shanghai').tz_localize(None)
            start += pd.to_timedelta(ns, unit='ns')
            metadata = handle.get('/processing_metadata')
            if metadata is None or 'preprocess_parameters_json' not in metadata.attrs:
                raise InputError(f'{path.name}: 缺少预处理来源记录')
            parameters = json.loads(str(metadata.attrs['preprocess_parameters_json']))
            bandpass = tuple((float(v) for v in parameters.get('bandpass_hz', [])))
            if len(bandpass) != 2:
                raise InputError(f'{path.name}: bandpass_hz无效: {bandpass}')
            sample_count, channel_count = map(int, dataset.shape)
            records.append(H5Record(path=path, start=start, end=start + pd.to_timedelta(sample_count / sample_rate, unit='s'), sample_rate_hz=sample_rate, sample_count=sample_count, channel_count=channel_count, start_channel=int(dataset.attrs.get('start_channel', 0)), bandpass_hz=bandpass))
    if {round(r.sample_rate_hz, 9) for r in records} != {EXPECTED_SAMPLE_RATE_HZ}:
        raise InputError('H5采样率不是统一的5000 Hz')
    if len({r.channel_count for r in records}) != 1 or len({r.start_channel for r in records}) != 1:
        raise InputError('H5通道数量或起始通道不一致')
    if {r.bandpass_hz for r in records} != {EXPECTED_BANDPASS_HZ}:
        raise InputError(f'H5并非统一的20--2000 Hz预处理数据: {sorted({r.bandpass_hz for r in records})}')
    return sorted(records, key=lambda r: r.start)

def load_coordinates(path: Path, h5_start_channel: int, h5_channel_count: int) -> pd.DataFrame:
    require_file(path, '坐标文件')
    frame = pd.read_excel(path) if path.suffix.lower() in {'.xlsx', '.xlsm', '.xls'} else pd.read_csv(path)
    required = {'channel', 'segment_id', 'segment_type', 'burial_depth', 'x_cm', 'y_cm', 'z_cm'}
    missing = required - set(frame.columns)
    if missing:
        raise InputError(f'坐标文件缺少字段: {sorted(missing)}')
    result = frame.copy()
    result['layout_channel'] = pd.to_numeric(result['channel'], errors='raise').astype(int)
    if result['layout_channel'].between(0, h5_channel_count - 1).all():
        result['data_channel'] = result['layout_channel']
        channel_offset = 0
    else:
        result['data_channel'] = result['layout_channel'] - h5_start_channel
        channel_offset = h5_start_channel
    if not result['data_channel'].between(0, h5_channel_count - 1).all():
        raise InputError('坐标通道无法映射到H5列')
    if result['data_channel'].duplicated().any():
        raise InputError('坐标文件包含重复通道')
    result['geometry_type'] = result['segment_type'].astype(str).str.lower().str.strip()
    result['burial_depth_cm'] = pd.to_numeric(result['burial_depth'], errors='coerce')
    for column in ('x_cm', 'y_cm', 'z_cm'):
        result[column] = pd.to_numeric(result[column], errors='coerce')
    result['coordinate_valid'] = result[['x_cm', 'y_cm', 'z_cm']].notna().all(axis=1)
    result.attrs['channel_offset'] = channel_offset
    return result.sort_values('data_channel').reset_index(drop=True)

def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({'true', '1', 'yes', 'y'})

def load_events(path: Path, policy: str, cutoff_time: pd.Timestamp, max_events: int | None) -> pd.DataFrame:
    require_file(path, '四参照事件目录')
    required = ['physical_event_id', 'start_time', 'end_time', 'peak_time', 'file_name', 'stage_id', 'representative_data_channel', 'representative_segment_id', 'geometry_type', 'burial_depth_cm', 'x_cm', 'y_cm', 'automatic_screening_label', 'manual_label', 'include_in_future_catrms']
    header = pd.read_csv(path, nrows=0).columns
    missing = set(required) - set(header)
    if missing:
        raise InputError(f'事件目录缺少字段: {sorted(missing)}')
    events = pd.read_csv(path, usecols=required, low_memory=False)
    for column in ('start_time', 'end_time', 'peak_time'):
        events[column] = pd.to_datetime(events[column], errors='coerce')
    if events[['start_time', 'end_time', 'peak_time']].isna().any().any():
        raise InputError('事件目录存在无效时间')
    if events['physical_event_id'].duplicated().any():
        raise InputError('physical_event_id不唯一，不能累计')
    if (events['end_time'] <= events['start_time']).any():
        raise InputError('事件目录存在结束时间不晚于开始时间的记录')
    automatic = events['automatic_screening_label'].astype(str).str.lower().str.strip()
    manual = events['manual_label'].astype(str).str.lower().str.strip()
    if policy == 'auto_effective':
        selected = automatic.eq('effective_candidate')
    elif policy == 'auto_effective_plus_uncertain':
        selected = automatic.isin({'effective_candidate', 'uncertain'})
    elif policy == 'manual_effective':
        selected = manual.eq('effective')
    else:
        selected = truthy(events['include_in_future_catrms'])
    selected &= events['peak_time'] < cutoff_time
    events = events.loc[selected].sort_values(['peak_time', 'physical_event_id']).reset_index(drop=True)
    if max_events is not None:
        events = events.head(max_events).copy()
    if events.empty:
        raise InputError(f'策略{policy!r}在宏观破坏前没有选中事件')
    return events

def attach_coordinates(events: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
    spatial = layout.loc[layout['coordinate_valid']].copy()
    columns = ['data_channel', 'layout_channel', 'segment_id', 'geometry_type', 'burial_depth_cm', 'x_cm', 'y_cm', 'z_cm']
    merged = events.merge(spatial[columns], left_on='representative_data_channel', right_on='data_channel', how='left', suffixes=('_catalog', '_layout'), validate='many_to_one')
    if merged['data_channel'].isna().any():
        examples = merged.loc[merged['data_channel'].isna(), 'representative_data_channel'].head(10).tolist()
        raise InputError(f'代表通道缺少有效坐标: {examples}')
    for left, right, label in (('x_cm_catalog', 'x_cm_layout', 'x'), ('y_cm_catalog', 'y_cm_layout', 'y'), ('burial_depth_cm_catalog', 'burial_depth_cm_layout', '埋深')):
        if (~np.isclose(pd.to_numeric(merged[left]), pd.to_numeric(merged[right]), atol=1e-06, rtol=0)).any():
            raise InputError(f'事件目录与坐标文件的{label}不一致')
    if (merged['geometry_type_catalog'].astype(str).str.lower() != merged['geometry_type_layout'].astype(str).str.lower()).any():
        raise InputError('事件目录与坐标文件的几何类型不一致')
    return merged

def gauge_channels(center: int, listed_channels: set[int]) -> list[int]:
    return sorted((c for c in listed_channels if center - GAUGE_HALF_CHANNELS <= c <= center + GAUGE_HALF_CHANNELS))

def read_event_block(records: list[H5Record], start: pd.Timestamp, end: pd.Timestamp, channels: list[int], dataset_path: str, preferred_file_name: str) -> tuple[np.ndarray, int]:
    """读取事件窗口；跨H5边界时按时间顺序拼接，且不重复使用重叠样本。"""
    matches = [record for record in records if record.path.name == preferred_file_name]
    if len(matches) != 1:
        raise InputError(f'事件目录指定的H5文件不存在或不唯一: {preferred_file_name}')
    preferred = matches[0]
    tolerance = pd.to_timedelta(1.0 / preferred.sample_rate_hz, unit='s')
    expected_samples = int(round((end - start).total_seconds() * preferred.sample_rate_hz))
    if expected_samples <= 0:
        raise InputError(f'事件持续时间无效: {start} -- {end}')
    cursor = start
    current = preferred if preferred.start - tolerance <= cursor < preferred.end + tolerance else None
    used_paths: set[Path] = set()
    blocks: list[np.ndarray] = []
    while cursor < end:
        if current is None:
            candidates = [record for record in records if record.path not in used_paths and record.start - tolerance <= cursor < record.end + tolerance]
            if not candidates:
                raise InputError(f'事件窗口存在未覆盖的H5时间段: {cursor} -- {end}; 目录指定文件={preferred_file_name}')
            current = preferred if preferred in candidates else min(candidates, key=lambda r: r.end)
        segment_end = min(end, current.end)
        first = int(round((cursor - current.start).total_seconds() * current.sample_rate_hz))
        last = int(round((segment_end - current.start).total_seconds() * current.sample_rate_hz))
        if first < 0 or last > current.sample_count or last <= first:
            raise InputError(f'事件样本索引越界: {current.path.name}; first={first}, last={last}, file_samples={current.sample_count}')
        with h5py.File(current.path, 'r') as handle:
            blocks.append(np.asarray(handle[dataset_path][first:last, channels], dtype=np.float64))
        cursor = segment_end
        used_paths.add(current.path)
        current = None
    block = np.concatenate(blocks, axis=0)
    if not np.isfinite(block).all():
        raise InputError(f'事件数据包含NaN或Inf: {start} -- {end}')
    if len(block) != expected_samples:
        raise InputError(f'事件读取长度异常: 期望{expected_samples}点，读取{len(block)}点，时间{start} -- {end}')
    return (block, len(block))

def position_id(experiment: str, geometry: str, depth: float, x: float, y: float, z: float) -> str:
    return f'{experiment}_{geometry}_D{depth:g}_X{x:g}_Y{y:g}_Z{z:g}'

def build_positions(layout: pd.DataFrame, experiment: str) -> pd.DataFrame:
    positions = layout.loc[layout['coordinate_valid'], ['geometry_type', 'burial_depth_cm', 'x_cm', 'y_cm', 'z_cm']].drop_duplicates().copy()
    positions['position_id'] = [position_id(experiment, str(r.geometry_type), float(r.burial_depth_cm), float(r.x_cm), float(r.y_cm), float(r.z_cm)) for r in positions.itertuples(index=False)]
    return positions[['position_id', 'geometry_type', 'burial_depth_cm', 'x_cm', 'y_cm', 'z_cm']].sort_values(['burial_depth_cm', 'geometry_type', 'x_cm', 'y_cm', 'z_cm']).reset_index(drop=True)

def compute_event_metrics(events: pd.DataFrame, layout: pd.DataFrame, records: list[H5Record], dataset_path: str, experiment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    listed = set(layout['data_channel'].astype(int))
    event_rows: list[dict] = []
    channel_rows: list[dict] = []
    fs = records[0].sample_rate_hz
    for number, event in enumerate(events.itertuples(index=False), start=1):
        center = int(event.representative_data_channel)
        gauge = gauge_channels(center, listed)
        if not gauge:
            raise InputError(f'代表通道{center}周围没有坐标通道')
        start = pd.Timestamp(event.start_time)
        end = pd.Timestamp(event.end_time)
        peak = pd.Timestamp(event.peak_time)
        block, sample_count = read_event_block(records, start, end, gauge, dataset_path, str(event.file_name))
        actual_duration_s = sample_count / fs
        channel_rms = np.sqrt(np.mean(np.square(block), axis=0))
        channel_ssri = np.sum(np.square(block), axis=0) / fs
        depth = float(event.burial_depth_cm_layout)
        geometry = str(event.geometry_type_layout)
        x = float(event.x_cm_layout)
        y = float(event.y_cm_layout)
        z = float(event.z_cm)
        pos = position_id(experiment, geometry, depth, x, y, z)
        common = {'physical_event_id': event.physical_event_id, 'start_time': start, 'end_time': end, 'peak_time': peak, 'second_time': peak.floor('s'), 'minute_time': peak.floor('min'), 'stage_id': int(event.stage_id), 'representative_data_channel': center, 'layout_channel': int(event.layout_channel), 'representative_segment_id': event.representative_segment_id, 'geometry_type': geometry, 'burial_depth_cm': depth, 'x_cm': x, 'y_cm': y, 'z_cm': z, 'position_id': pos}
        for index, channel in enumerate(gauge):
            channel_rows.append({**common, 'data_channel': channel, 'is_representative_channel': channel == center, 'sample_count': sample_count, 'duration_s': actual_duration_s, 'strain_rate_rms': float(channel_rms[index]), 'squared_strain_rate_integral': float(channel_ssri[index])})
        event_rows.append({**common, 'file_name': event.file_name, 'gauge_data_channels': ';'.join(map(str, gauge)), 'gauge_channel_count': len(gauge), 'sample_count': sample_count, 'duration_s': actual_duration_s, 'event_rms_gauge_median': float(np.median(channel_rms)), 'event_rms_gauge_min': float(np.min(channel_rms)), 'event_rms_gauge_max': float(np.max(channel_rms)), 'event_squared_strain_rate_integral_gauge_median': float(np.median(channel_ssri)), 'event_squared_strain_rate_integral_gauge_sum': float(np.sum(channel_ssri))})
        if number % 100 == 0 or number == len(events):
            log(f'[事件] 已计算 {number}/{len(events)}')
    event_frame = pd.DataFrame(event_rows).sort_values(['peak_time', 'physical_event_id']).reset_index(drop=True)
    event_frame['cumulative_event_count'] = np.arange(1, len(event_frame) + 1)
    event_frame['cumulative_event_rms_diagnostic'] = event_frame['event_rms_gauge_median'].cumsum()
    event_frame['cumulative_squared_strain_rate_integral'] = event_frame['event_squared_strain_rate_integral_gauge_median'].cumsum()
    return (event_frame, pd.DataFrame(channel_rows))

def aggregate_outputs(events: pd.DataFrame, positions: pd.DataFrame, records: list[H5Record], cutoff_time: pd.Timestamp, snapshot_minutes: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = ['minute_time', 'position_id', 'geometry_type', 'burial_depth_cm', 'x_cm', 'y_cm', 'z_cm']
    sparse = events.groupby(groups, as_index=False).agg(event_count=('physical_event_id', 'size'), rms_sum_diagnostic=('event_rms_gauge_median', 'sum'), squared_strain_rate_integral=('event_squared_strain_rate_integral_gauge_median', 'sum'))
    full_minutes = pd.DataFrame({'minute_time': pd.date_range(records[0].start.floor('min'), cutoff_time.floor('min'), freq='1min')})
    total = sparse.groupby('minute_time', as_index=False).agg(event_count=('event_count', 'sum'), rms_increment_diagnostic=('rms_sum_diagnostic', 'sum'), squared_strain_rate_integral_increment=('squared_strain_rate_integral', 'sum'))
    total = full_minutes.merge(total, on='minute_time', how='left').fillna(0)
    total['event_count'] = total['event_count'].astype(int)
    total['cumulative_event_count'] = total['event_count'].cumsum()
    total['cumulative_event_rms_diagnostic'] = total['rms_increment_diagnostic'].cumsum()
    total['cumulative_squared_strain_rate_integral'] = total['squared_strain_rate_integral_increment'].cumsum()
    occupied = sparse.groupby(['position_id', 'geometry_type', 'burial_depth_cm', 'x_cm', 'y_cm', 'z_cm'], as_index=False).agg(event_count=('event_count', 'sum'), rms_sum_diagnostic=('rms_sum_diagnostic', 'sum'), squared_strain_rate_integral=('squared_strain_rate_integral', 'sum'))
    final = positions.merge(occupied, on=['position_id', 'geometry_type', 'burial_depth_cm', 'x_cm', 'y_cm', 'z_cm'], how='left', validate='one_to_one').fillna({'event_count': 0, 'rms_sum_diagnostic': 0.0, 'squared_strain_rate_integral': 0.0})
    final['event_count'] = final['event_count'].astype(int)
    position_series = sparse.sort_values(['position_id', 'minute_time']).copy()
    position_series['cumulative_event_count'] = position_series.groupby('position_id')['event_count'].cumsum()
    position_series['cumulative_event_rms_diagnostic'] = position_series.groupby('position_id')['rms_sum_diagnostic'].cumsum()
    position_series['cumulative_squared_strain_rate_integral'] = position_series.groupby('position_id')['squared_strain_rate_integral'].cumsum()
    snapshot_times = pd.date_range(records[0].start.floor('min'), cutoff_time.floor('min'), freq=f'{snapshot_minutes}min')
    if len(snapshot_times) == 0 or snapshot_times[-1] < cutoff_time.floor('min'):
        snapshot_times = snapshot_times.append(pd.DatetimeIndex([cutoff_time.floor('min')]))
    by_position = {key: value for key, value in position_series.groupby('position_id')}
    snapshot_rows: list[dict] = []
    for position in positions.itertuples(index=False):
        group = by_position.get(position.position_id)
        for snapshot in snapshot_times:
            latest = None if group is None else group.loc[group['minute_time'] <= snapshot].tail(1)
            row = {'snapshot_time': snapshot, 'position_id': position.position_id, 'geometry_type': position.geometry_type, 'burial_depth_cm': position.burial_depth_cm, 'x_cm': position.x_cm, 'y_cm': position.y_cm, 'z_cm': position.z_cm, 'cumulative_event_count': 0, 'cumulative_event_rms_diagnostic': 0.0, 'cumulative_squared_strain_rate_integral': 0.0}
            if latest is not None and (not latest.empty):
                last = latest.iloc[0]
                row.update({'cumulative_event_count': int(last['cumulative_event_count']), 'cumulative_event_rms_diagnostic': float(last['cumulative_event_rms_diagnostic']), 'cumulative_squared_strain_rate_integral': float(last['cumulative_squared_strain_rate_integral'])})
            snapshot_rows.append(row)
    return (total, final, pd.DataFrame(snapshot_rows))

def write_snapshot_txts(snapshots: pd.DataFrame, output_dir: Path, record_start: pd.Timestamp, cutoff_time: pd.Timestamp, experiment: str, run_slug: str) -> tuple[Path, Path, int]:
    """每个快照写一个无表头四列TXT，并生成文件—时间索引。"""
    txt_dir = output_dir / f'{experiment}_06_minute_txt_{run_slug}'
    txt_dir.mkdir(parents=True, exist_ok=True)
    value_column = 'cumulative_squared_strain_rate_integral'
    columns = [value_column, 'x_cm', 'y_cm', 'z_cm']
    index_rows: list[dict] = []
    for snapshot, frame in snapshots.groupby('snapshot_time', sort=True):
        timestamp = pd.Timestamp(snapshot)
        filename = f'{experiment}_{run_slug}_{timestamp:%Y%m%d_%H%M}.txt'
        path = txt_dir / filename
        frame.sort_values(['z_cm', 'y_cm', 'x_cm'])[columns].to_csv(path, sep='\t', header=False, index=False, encoding='utf-8', float_format='%.12g')
        interval_end = min(timestamp + pd.Timedelta(minutes=1), cutoff_time)
        index_rows.append({'file_name': filename, 'minute_time': timestamp, 'cumulative_definition': 'all selected events with peak_time before the end of this minute', 'actual_interval_start': max(timestamp, record_start), 'interval_end_exclusive_or_cutoff': interval_end, 'is_partial_first_minute': timestamp == record_start.floor('min') and record_start != timestamp, 'is_partial_last_minute': timestamp == cutoff_time.floor('min') and cutoff_time != timestamp + pd.Timedelta(minutes=1), 'row_count': len(frame), 'column_1': value_column, 'column_2': 'x_cm', 'column_3': 'y_cm', 'column_4': 'z_cm'})
    index_path = output_dir / f'{experiment}_06_minute_txt_index_{run_slug}.csv'
    pd.DataFrame(index_rows).to_csv(index_path, index=False, encoding='utf-8-sig')
    columns_path = txt_dir / 'README_columns.txt'
    columns_path.write_text('Each data file has no header and contains four tab-separated columns:\n1 cumulative_squared_strain_rate_integral\n2 x_cm\n3 y_cm\n4 z_cm\nThe timestamp in the file name is the calendar minute label; the value includes all selected events whose peak_time falls in that minute or earlier.\n', encoding='utf-8')
    return (txt_dir, index_path, len(index_rows))

def validate_results(events: pd.DataFrame, event_channels: pd.DataFrame, total: pd.DataFrame, final: pd.DataFrame, snapshots: pd.DataFrame, positions: pd.DataFrame, cutoff_time: pd.Timestamp, snapshot_minutes: int) -> dict[str, bool]:
    expected_snapshots = pd.date_range(total['minute_time'].min(), total['minute_time'].max(), freq=f'{snapshot_minutes}min')
    if len(expected_snapshots) == 0 or expected_snapshots[-1] < total['minute_time'].max():
        expected_snapshots = expected_snapshots.append(pd.DatetimeIndex([total['minute_time'].max()]))
    checks = {'unique_physical_event_id': not events['physical_event_id'].duplicated().any(), 'all_events_before_cutoff': bool((events['peak_time'] < cutoff_time).all()), 'metrics_finite': bool(np.isfinite(events[['event_rms_gauge_median', 'event_squared_strain_rate_integral_gauge_median']].to_numpy()).all()), 'metrics_nonnegative': bool((events[['event_rms_gauge_median', 'event_squared_strain_rate_integral_gauge_median']] >= 0).all().all()), 'channel_identity_J_equals_RMS2T': bool(np.allclose(event_channels['squared_strain_rate_integral'], np.square(event_channels['strain_rate_rms']) * event_channels['duration_s'], rtol=1e-10, atol=1e-12)), 'cumulative_count_nondecreasing': bool(total['cumulative_event_count'].is_monotonic_increasing), 'cumulative_integral_nondecreasing': bool(total['cumulative_squared_strain_rate_integral'].is_monotonic_increasing), 'total_event_count_reconciles': int(total['event_count'].sum()) == len(events), 'total_integral_reconciles': bool(np.isclose(total['squared_strain_rate_integral_increment'].sum(), events['event_squared_strain_rate_integral_gauge_median'].sum(), rtol=1e-10, atol=1e-12)), 'spatial_integral_reconciles': bool(np.isclose(final['squared_strain_rate_integral'].sum(), events['event_squared_strain_rate_integral_gauge_median'].sum(), rtol=1e-10, atol=1e-12)), 'all_layout_positions_present': len(final) == len(positions), 'one_row_per_position_per_snapshot': len(snapshots) == len(positions) * len(expected_snapshots)}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise InputError(f'结果自检失败: {failed}')
    return checks

def main(argv: list[str] | None=None) -> int:
    args = parse_args(argv)
    experiment = args.experiment
    run_slug = f'{args.event_policy}_{args.time_scope}'
    if args.snapshot_minutes <= 0:
        raise InputError('--snapshot-minutes必须大于0')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log('[1/6] 核查H5记录与预处理来源')
    records = inspect_h5_records(args.h5_dir, args.dataset_path)
    record_end = max((record.end for record in records))
    if not records[0].start < args.cutoff_time <= record_end:
        raise InputError(f'累计截止时刻{args.cutoff_time}不在H5覆盖范围{records[0].start} -- {record_end}内')
    log(f'  实验={experiment}; time_scope={args.time_scope}; cutoff(exclusive)={args.cutoff_time}')
    log(f'  文件={len(records)}; fs={records[0].sample_rate_hz:g} Hz; bandpass={records[0].bandpass_hz}')
    log('[2/6] 读取坐标和有效事件')
    layout = load_coordinates(args.coordinates, records[0].start_channel, records[0].channel_count)
    events = load_events(args.catalog, args.event_policy, args.cutoff_time, args.max_events)
    events = attach_coordinates(events, layout)
    positions = build_positions(layout, experiment)
    log(f'  事件={len(events)}; 空间位置={len(positions)}; policy={args.event_policy}')
    log('[3/6] 直接计算事件RMS与平方应变率积分（无背景阈值）')
    event_metrics, event_channel_metrics = compute_event_metrics(events, layout, records, args.dataset_path, experiment)
    log(f'[4/6] 生成每分钟累计量和每{args.snapshot_minutes}分钟空间快照')
    total, final, snapshots = aggregate_outputs(event_metrics, positions, records, args.cutoff_time, args.snapshot_minutes)
    log('[5/6] 执行一致性自检')
    checks = validate_results(event_metrics, event_channel_metrics, total, final, snapshots, positions, args.cutoff_time, args.snapshot_minutes)
    outputs = {'event_metrics': args.output_dir / f'{experiment}_01_event_metrics_no_threshold_{run_slug}.csv', 'event_channel_metrics': args.output_dir / f'{experiment}_02_event_channel_metrics_no_threshold_{run_slug}.csv', 'total_1min': args.output_dir / f'{experiment}_03_total_activity_1min_no_threshold_{run_slug}.csv', 'position_final': args.output_dir / f'{experiment}_04_position_final_no_threshold_{run_slug}.csv', 'position_snapshots': args.output_dir / f'{experiment}_05_position_snapshots_{args.snapshot_minutes}min_no_threshold_{run_slug}.csv'}
    for frame, path in zip((event_metrics, event_channel_metrics, total, final, snapshots), outputs.values()):
        frame.to_csv(path, index=False, encoding='utf-8-sig')
    txt_dir, txt_index_path, txt_file_count = write_snapshot_txts(snapshots, args.output_dir, records[0].start, args.cutoff_time, experiment, run_slug)
    status = 'provisional_auto_labels' if args.event_policy.startswith('auto_') else 'selected_review_policy'
    manifest = {'analysis': f'{experiment} no-threshold strain-rate activity: {args.event_policy}', 'experiment': experiment, 'status': status, 'created_by': Path(__file__).name, 'definitions': {'event_rms': 'median across representative channel +/-4 of sqrt(mean(strain_rate^2))', 'primary_event_activity': 'median across representative channel +/-4 of sum(strain_rate^2)/sampling_rate', 'primary_cumulative_activity': 'cumulative sum of primary_event_activity ordered by event peak time', 'minute_definition': 'calendar-minute label; cumulative value includes all selected events whose peak_time falls in that minute or earlier', 'threshold': None, 'background_subtraction': None, 'input_unit_label': args.input_unit, 'input_unit_basis': 'user-confirmed; not encoded in the H5 attributes', 'rms_unit': args.input_unit, 'integral_unit': f'({args.input_unit})^2*s'}, 'inputs': {'h5_dir': args.h5_dir, 'catalog': args.catalog, 'catalog_sha256': file_sha256(args.catalog), 'coordinates': args.coordinates, 'coordinates_sha256': file_sha256(args.coordinates), 'dataset_path': args.dataset_path, 'sample_rate_hz': records[0].sample_rate_hz, 'bandpass_hz': records[0].bandpass_hz, 'time_scope': args.time_scope, 'cutoff_time_exclusive': args.cutoff_time, 'macro_observation': args.macro_observation, 'event_policy': args.event_policy, 'max_events': args.max_events}, 'counts': {'h5_files': len(records), 'selected_events': len(event_metrics), 'event_channel_rows': len(event_channel_metrics), 'all_spatial_positions': len(positions), 'occupied_positions': int((final['event_count'] > 0).sum()), 'spatial_snapshot_count': int(snapshots['snapshot_time'].nunique()), 'four_column_txt_files': txt_file_count}, 'totals': {'event_count': int(len(event_metrics)), 'event_rms_sum_diagnostic': float(event_metrics['event_rms_gauge_median'].sum()), 'squared_strain_rate_integral': float(event_metrics['event_squared_strain_rate_integral_gauge_median'].sum())}, 'validation': checks, 'outputs': outputs, 'minute_txt_directory': txt_dir, 'minute_txt_index': txt_index_path, 'interpretation_limits': ['The primary metric is a band-limited DAS squared strain-rate integral, not energy in joules.', 'No independent cable-soil coupling calibration is applied.', 'Absolute spatial amplitude differences may include channel/coupling effects.', 'Directly accumulated event RMS is diagnostic because it depends on event segmentation.', 'Cross-depth synchronized duplicate-event review is not performed by this program.']}
    manifest_path = args.output_dir / f'{experiment}_00_run_manifest_no_threshold_{run_slug}.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_ready), encoding='utf-8')
    log('[6/6] 完成')
    log(f'  输出目录: {args.output_dir}')
    log(f'  事件数: {len(event_metrics)}')
    log(f"  累计平方应变率积分: {manifest['totals']['squared_strain_rate_integral']:.8g}")
    log(f'  状态: {status}')
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (InputError, FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
