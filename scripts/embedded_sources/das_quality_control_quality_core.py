"""DAS processing functions for quality core."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import h5py
import numpy as np
H5_SUFFIXES = {'.h5', '.hdf5', '.hdf'}
FILE_INFO_COLUMNS = ['file_name', 'experiment', 'data_type', 'file_start_time', 'rainfall_elapsed_time', 'rainfall_intensity', 'event_start_second', 'event_end_second', 'event_location', 'operation_type', 'previous_file', 'next_file', 'notes']
CHANNEL_MAP_COLUMNS = ['channel', 'fiber_distance_m', 'x_m', 'y_m', 'z_m', 'burial_depth', 'ring_id', 'fiber_segment', 'slope_or_leadin', 'valid_or_bad']

class QCError(RuntimeError):
    pass

def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8-sig') as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise QCError('配置文件顶层必须是 JSON 对象。')
    return value

def deep_get(mapping: dict[str, Any], keys: Iterable[str], default: Any) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value

def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return [{k: (v or '').strip() for k, v in row.items()} for row in csv.DictReader(handle)]

def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def parse_datetime(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    normalized = value.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise QCError(f'无法解析时间 {value!r}，请使用 ISO 8601，例如 2026-08-12 14:30:00。') from exc

def parse_float(value: str) -> float | None:
    if not value.strip():
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise QCError(f'无法解析数值 {value!r}') from exc

def numeric_2d_datasets(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with h5py.File(path, 'r') as handle:

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and np.issubdtype(obj.dtype, np.number):
                result.append({'path': '/' + name.strip('/'), 'shape': list(obj.shape), 'dtype': str(obj.dtype), 'chunks': obj.chunks, 'compression': obj.compression})
        handle.visititems(visitor)
    return result

def find_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        files = [input_path] if input_path.suffix.lower() in H5_SUFFIXES else []
    else:
        iterator = input_path.rglob('*') if recursive else input_path.glob('*')
        files = [p for p in iterator if p.is_file() and p.suffix.lower() in H5_SUFFIXES]
    files.sort(key=lambda p: str(p).lower())
    if not files:
        raise QCError(f'未找到 H5 文件: {input_path}')
    return files

def choose_dataset(path: Path, configured: str | None) -> str:
    if configured:
        dataset_path = '/' + configured.strip('/')
        with h5py.File(path, 'r') as handle:
            if dataset_path not in handle:
                raise QCError(f'{path.name} 中不存在数据集 {dataset_path}')
            ds = handle[dataset_path]
            if not isinstance(ds, h5py.Dataset) or ds.ndim != 2 or (not np.issubdtype(ds.dtype, np.number)):
                raise QCError(f'{dataset_path} 不是数值型二维数据集。')
        return dataset_path
    candidates = numeric_2d_datasets(path)
    if len(candidates) != 1:
        details = ', '.join((f"{v['path']} shape={v['shape']}" for v in candidates)) or '无'
        raise QCError(f'无法唯一识别波形数据集（候选: {details}），请配置 hdf5.data_dataset。')
    return str(candidates[0]['path'])

def choose_channel_axis(shape: tuple[int, int], configured: Any, expected: int) -> int:
    if configured in (0, 1):
        return int(configured)
    exact = [axis for axis in (0, 1) if shape[axis] == expected]
    if len(exact) == 1:
        return exact[0]
    smaller = 0 if shape[0] < shape[1] else 1
    if shape[smaller] <= max(10000, expected * 4):
        return smaller
    raise QCError(f'无法从 shape={shape} 判断通道轴，请设置 hdf5.channel_axis 为 0 或 1。')

@dataclass
class Moments:
    channels: int
    n: np.ndarray = field(init=False)
    mean: np.ndarray = field(init=False)
    m2: np.ndarray = field(init=False)
    m3: np.ndarray = field(init=False)
    m4: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.n = np.zeros(self.channels, dtype=np.int64)
        self.mean = np.zeros(self.channels, dtype=np.float64)
        self.m2 = np.zeros(self.channels, dtype=np.float64)
        self.m3 = np.zeros(self.channels, dtype=np.float64)
        self.m4 = np.zeros(self.channels, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        nb = finite.sum(axis=0).astype(np.int64)
        if not np.any(nb):
            return
        safe = np.where(finite, values, 0.0)
        mean_b = np.divide(safe.sum(axis=0), nb, out=np.zeros(self.channels), where=nb > 0)
        dev = np.where(finite, values - mean_b, 0.0)
        dev2 = dev * dev
        m2b = dev2.sum(axis=0)
        m3b = (dev2 * dev).sum(axis=0)
        m4b = (dev2 * dev2).sum(axis=0)
        na = self.n.astype(np.float64)
        nbf = nb.astype(np.float64)
        n = na + nbf
        delta = mean_b - self.mean
        active = nb > 0
        n_safe = np.where(n > 0, n, 1.0)
        old_m2, old_m3, old_m4 = (self.m2.copy(), self.m3.copy(), self.m4.copy())
        merged_mean = self.mean + delta * nbf / n_safe
        merged_m2 = old_m2 + m2b + delta ** 2 * na * nbf / n_safe
        merged_m3 = old_m3 + m3b + delta ** 3 * na * nbf * (na - nbf) / n_safe ** 2 + 3.0 * delta * (na * m2b - nbf * old_m2) / n_safe
        merged_m4 = old_m4 + m4b + delta ** 4 * na * nbf * (na ** 2 - na * nbf + nbf ** 2) / n_safe ** 3 + 6.0 * delta ** 2 * (na ** 2 * m2b + nbf ** 2 * old_m2) / n_safe ** 2 + 4.0 * delta * (na * m3b - nbf * old_m3) / n_safe
        self.mean[active] = merged_mean[active]
        self.m2[active] = merged_m2[active]
        self.m3[active] = merged_m3[active]
        self.m4[active] = merged_m4[active]
        self.n += nb

@dataclass
class FileAnalysis:
    file_row: dict[str, Any]
    channel_rows: list[dict[str, Any]]
    first_boundary: np.ndarray
    last_boundary: np.ndarray
    heatmap: np.ndarray
    raw_time: np.ndarray
    raw_traces: np.ndarray

def robust_z(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    good = np.isfinite(values)
    if good.sum() < 3:
        return result
    med = np.median(values[good])
    mad = np.median(np.abs(values[good] - med))
    if mad > 0:
        result[good] = 0.67448975 * (values[good] - med) / mad
    return result

def read_block(ds: h5py.Dataset, channel_axis: int, start: int, end: int) -> np.ndarray:
    raw = ds[:, start:end].T if channel_axis == 0 else ds[start:end, :]
    return np.asarray(raw, dtype=np.float64)

def channel_display_number(index: int, one_based: bool) -> int:
    return index + 1 if one_based else index

def analyze_file(path: Path, dataset_path: str, channel_axis: int, config: dict[str, Any], info: dict[str, str], selected_channels: list[int], make_plot_data: bool) -> FileAnalysis:
    fs = float(deep_get(config, ('acquisition', 'sample_rate_hz'), 5000.0))
    expected_channels = int(deep_get(config, ('acquisition', 'expected_channels'), 500))
    expected_duration = float(deep_get(config, ('acquisition', 'expected_duration_seconds'), 600.0))
    block_seconds = float(deep_get(config, ('processing', 'block_seconds'), 1.0))
    heat_seconds = float(deep_get(config, ('plots', 'heatmap_bin_seconds'), 1.0))
    boundary_seconds = float(deep_get(config, ('processing', 'boundary_seconds'), 1.0))
    one_based = bool(deep_get(config, ('hdf5', 'channel_numbers_are_one_based'), False))
    saturation_values = deep_get(config, ('thresholds', 'saturation_values'), None)
    with h5py.File(path, 'r') as handle:
        if dataset_path not in handle:
            raise QCError(f'{path.name} 缺少数据集 {dataset_path}')
        ds = handle[dataset_path]
        shape = tuple((int(v) for v in ds.shape))
        channels, samples = (shape[channel_axis], shape[1 - channel_axis])
        dtype = ds.dtype
        block = max(1, int(round(block_seconds * fs)))
        heat_bin = max(1, int(round(heat_seconds * fs)))
        boundary_n = min(samples, max(1, int(round(boundary_seconds * fs))))
        moments = Moments(channels)
        zeros = np.zeros(channels, dtype=np.int64)
        saturation = np.zeros(channels, dtype=np.int64)
        flat = np.zeros(channels, dtype=np.int64)
        diff_count = np.zeros(channels, dtype=np.int64)
        diff_sq_sum = np.zeros(channels, dtype=np.float64)
        max_jump = np.zeros(channels, dtype=np.float64)
        previous_last: np.ndarray | None = None
        bins = int(math.ceil(samples / heat_bin))
        heat_sum = np.zeros((channels, bins), dtype=np.float64) if make_plot_data else np.empty((0, 0))
        heat_sq = np.zeros((channels, bins), dtype=np.float64) if make_plot_data else np.empty((0, 0))
        heat_n = np.zeros((channels, bins), dtype=np.int64) if make_plot_data else np.empty((0, 0), dtype=np.int64)
        raw_offset = max(0, int(round(float(deep_get(config, ('plots', 'raw_start_second'), 0.0)) * fs)))
        raw_length = max(1, int(round(float(deep_get(config, ('plots', 'raw_duration_seconds'), 5.0)) * fs)))
        raw_end = min(samples, raw_offset + raw_length)
        raw_parts: list[np.ndarray] = []
        if np.issubdtype(dtype, np.integer):
            dtype_info = np.iinfo(dtype)
            sat_low, sat_high = (dtype_info.min, dtype_info.max)
        elif saturation_values and len(saturation_values) == 2:
            sat_low, sat_high = (float(saturation_values[0]), float(saturation_values[1]))
        else:
            sat_low = sat_high = None
        for start in range(0, samples, block):
            end = min(samples, start + block)
            values = read_block(ds, channel_axis, start, end)
            finite = np.isfinite(values)
            moments.update(values)
            zeros += ((values == 0) & finite).sum(axis=0)
            if sat_low is not None:
                saturation += (((values <= sat_low) | (values >= sat_high)) & finite).sum(axis=0)
            if values.shape[0]:
                if previous_last is not None:
                    d0 = values[0] - previous_last
                    good0 = np.isfinite(d0)
                    flat += (good0 & (d0 == 0)).astype(np.int64)
                    diff_count += good0.astype(np.int64)
                    diff_sq_sum += np.where(good0, d0 * d0, 0.0)
                    max_jump = np.maximum(max_jump, np.where(good0, np.abs(d0), 0.0))
                if values.shape[0] > 1:
                    diffs = np.diff(values, axis=0)
                    good = np.isfinite(diffs)
                    flat += ((diffs == 0) & good).sum(axis=0)
                    diff_count += good.sum(axis=0)
                    diff_sq_sum += np.where(good, diffs * diffs, 0.0).sum(axis=0)
                    max_jump = np.maximum(max_jump, np.max(np.where(good, np.abs(diffs), 0.0), axis=0))
                previous_last = values[-1].copy()
            if make_plot_data:
                cursor = start
                while cursor < end:
                    bin_index = cursor // heat_bin
                    sub_end = min(end, (bin_index + 1) * heat_bin)
                    sub = values[cursor - start:sub_end - start]
                    sub_good = np.isfinite(sub)
                    safe = np.where(sub_good, sub, 0.0)
                    heat_sum[:, bin_index] += safe.sum(axis=0)
                    heat_sq[:, bin_index] += (safe * safe).sum(axis=0)
                    heat_n[:, bin_index] += sub_good.sum(axis=0)
                    cursor = sub_end
                overlap_a, overlap_b = (max(start, raw_offset), min(end, raw_end))
                if overlap_b > overlap_a:
                    raw_parts.append(values[overlap_a - start:overlap_b - start, selected_channels].copy())
        first_boundary = read_block(ds, channel_axis, 0, boundary_n)
        last_boundary = read_block(ds, channel_axis, samples - boundary_n, samples)
        fingerprint = hashlib.sha256()
        fingerprint.update(np.ascontiguousarray(first_boundary).tobytes())
        middle_a = max(0, samples // 2 - boundary_n // 2)
        fingerprint.update(np.ascontiguousarray(read_block(ds, channel_axis, middle_a, min(samples, middle_a + boundary_n))).tobytes())
        fingerprint.update(np.ascontiguousarray(last_boundary).tobytes())
        n_float = moments.n.astype(np.float64)
        variance = np.divide(moments.m2, n_float, out=np.full(channels, np.nan), where=n_float > 0)
        std = np.sqrt(np.maximum(variance, 0.0))
        rms = np.sqrt(np.maximum(variance + moments.mean ** 2, 0.0))
        kurtosis = np.divide(moments.m4 * n_float, moments.m2 ** 2, out=np.full(channels, np.nan), where=moments.m2 > 0)
        nonfinite_fraction = 1.0 - n_float / max(samples, 1)
        zero_fraction = zeros / max(samples, 1)
        saturation_fraction = saturation / max(samples, 1)
        flat_fraction = np.divide(flat, diff_count, out=np.ones(channels), where=diff_count > 0)
        diff_rms = np.sqrt(np.divide(diff_sq_sum, diff_count, out=np.full(channels, np.nan), where=diff_count > 0))
        jump_std_ratio = np.divide(max_jump, std, out=np.full(channels, np.nan), where=std > 0)
        log_std = np.log10(np.maximum(std, np.finfo(float).tiny))
        std_z = robust_z(log_std)
        thresholds = config.get('thresholds', {})
        rows: list[dict[str, Any]] = []
        for ch in range(channels):
            reasons: list[str] = []
            if nonfinite_fraction[ch] > float(thresholds.get('max_nonfinite_fraction', 0.01)):
                reasons.append('NONFINITE')
            if zero_fraction[ch] > float(thresholds.get('max_zero_fraction', 0.99)):
                reasons.append('ZERO_OR_DEAD')
            if saturation_fraction[ch] > float(thresholds.get('max_saturation_fraction', 0.001)):
                reasons.append('SATURATED')
            if flat_fraction[ch] > float(thresholds.get('max_flat_fraction', 0.99)):
                reasons.append('FLATLINE')
            if std[ch] == 0 or not np.isfinite(std[ch]):
                reasons.append('NO_VARIANCE')
            elif std_z[ch] < -float(thresholds.get('std_robust_z', 6.0)):
                reasons.append('LOW_NOISE_OUTLIER')
            elif std_z[ch] > float(thresholds.get('std_robust_z', 6.0)):
                reasons.append('HIGH_NOISE_OUTLIER')
            if np.isfinite(kurtosis[ch]) and kurtosis[ch] > float(thresholds.get('max_kurtosis', 100.0)):
                reasons.append('IMPULSIVE_KURTOSIS')
            if np.isfinite(jump_std_ratio[ch]) and jump_std_ratio[ch] > float(thresholds.get('max_jump_std_ratio', 50.0)):
                reasons.append('ABRUPT_JUMP')
            reasons = list(dict.fromkeys(reasons))
            rows.append({'file_name': path.name, 'channel_index': ch, 'channel': channel_display_number(ch, one_based), 'sample_count': samples, 'finite_count': int(moments.n[ch]), 'valid_fraction': float(1.0 - nonfinite_fraction[ch] - saturation_fraction[ch]), 'nonfinite_fraction': float(nonfinite_fraction[ch]), 'zero_fraction': float(zero_fraction[ch]), 'saturation_fraction': float(saturation_fraction[ch]), 'flat_difference_fraction': float(flat_fraction[ch]), 'mean': float(moments.mean[ch]), 'std': float(std[ch]), 'rms': float(rms[ch]), 'kurtosis_pearson': float(kurtosis[ch]), 'difference_rms': float(diff_rms[ch]), 'max_abs_jump': float(max_jump[ch]), 'max_jump_divided_by_std': float(jump_std_ratio[ch]), 'log_std_robust_z': float(std_z[ch]), 'bad_channel': bool(reasons), 'bad_reasons': ';'.join(reasons)})
        duration = samples / fs
        structural: list[str] = []
        if channels != expected_channels:
            structural.append(f'CHANNEL_COUNT:{channels}!={expected_channels}')
        tolerance = max(1.0 / fs, float(deep_get(config, ('acquisition', 'duration_tolerance_seconds'), 0.01)))
        if abs(duration - expected_duration) > tolerance:
            structural.append(f'DURATION:{duration:.6f}!={expected_duration:.6f}')
        bad_count = sum((bool(row['bad_channel']) for row in rows))
        file_row = {'file_name': path.name, 'file_path': str(path.resolve()), 'dataset_path': dataset_path, 'shape': str(shape), 'channel_axis': channel_axis, 'channel_count': channels, 'sample_count': samples, 'sample_rate_hz': fs, 'duration_seconds': duration, 'dtype': str(dtype), 'chunks': str(ds.chunks), 'compression': str(ds.compression or ''), 'file_size_bytes': path.stat().st_size, 'fingerprint_first_middle_last_sha256': fingerprint.hexdigest(), 'bad_channel_count': bad_count, 'bad_channel_fraction': bad_count / max(channels, 1), 'structural_issues': ';'.join(structural), 'quality_status': 'FAIL' if structural else 'WARN' if bad_count else 'PASS', 'experiment': info.get('experiment', ''), 'data_type': info.get('data_type', ''), 'file_start_time': info.get('file_start_time', '')}
        if make_plot_data:
            heat_mean = np.divide(heat_sum, heat_n, out=np.zeros_like(heat_sum), where=heat_n > 0)
            heat_var = np.divide(heat_sq, heat_n, out=np.zeros_like(heat_sq), where=heat_n > 0) - heat_mean ** 2
            heatmap = np.sqrt(np.maximum(heat_var, 0.0))
            raw_traces = np.concatenate(raw_parts, axis=0) if raw_parts else np.empty((0, len(selected_channels)))
            raw_time = (np.arange(raw_traces.shape[0]) + raw_offset) / fs
        else:
            heatmap, raw_traces, raw_time = (np.empty((0, 0)), np.empty((0, 0)), np.empty(0))
        return FileAnalysis(file_row, rows, first_boundary, last_boundary, heatmap, raw_time, raw_traces)

def metadata_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row.get('file_name', '').strip()
        if not name:
            raise QCError('文件信息表存在空 file_name。')
        for key in {name.lower(), Path(name).name.lower()}:
            if key in result and result[key] is not row:
                raise QCError(f'文件信息表 file_name 不唯一: {name}')
            result[key] = row
    return result

def get_info(path: Path, input_path: Path, index: dict[str, dict[str, str]]) -> dict[str, str]:
    candidates = [path.name.lower()]
    if input_path.is_dir():
        try:
            candidates.insert(0, path.relative_to(input_path).as_posix().lower())
        except ValueError:
            pass
    for key in candidates:
        if key in index:
            return index[key]
    return {}

def validate_metadata(files: list[Path], input_path: Path, info_rows: list[dict[str, str]], analyses: list[FileAnalysis]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = metadata_index(info_rows) if info_rows else {}
    validation: list[dict[str, Any]] = []
    timeline_items: list[tuple[datetime | None, Path, dict[str, str], FileAnalysis]] = []
    allowed_types = {'stable_rain', 'human_noise', 'visible_slide', 'suspected'}
    for path, analysis in zip(files, analyses):
        info = get_info(path, input_path, index)
        issues: list[str] = []
        if info_rows and (not info):
            issues.append('MISSING_FILE_INFO_ROW')
        data_type = info.get('data_type', '')
        if data_type and data_type not in allowed_types:
            issues.append('UNKNOWN_DATA_TYPE')
        start = parse_datetime(info.get('file_start_time', '')) if info else None
        event_a = parse_float(info.get('event_start_second', '')) if info else None
        event_b = parse_float(info.get('event_end_second', '')) if info else None
        duration = float(analysis.file_row['duration_seconds'])
        if (event_a is None) != (event_b is None):
            issues.append('INCOMPLETE_EVENT_INTERVAL')
        if event_a is not None and event_b is not None and (not 0 <= event_a <= event_b <= duration):
            issues.append('EVENT_OUTSIDE_FILE')
        if data_type in {'human_noise', 'visible_slide'} and event_a is None:
            issues.append('MISSING_REFERENCE_EVENT_TIME')
        rain_elapsed = parse_float(info.get('rainfall_elapsed_time', '')) if info else None
        if rain_elapsed is not None and rain_elapsed < 0:
            issues.append('NEGATIVE_RAINFALL_ELAPSED')
        validation.append({'file_name': path.name, 'metadata_status': 'WARN' if issues else 'PASS', 'metadata_issues': ';'.join(issues)})
        timeline_items.append((start, path, info, analysis))
    timeline_items.sort(key=lambda x: (x[0] is None, x[0] or datetime.max, x[1].name.lower()))
    timeline: list[dict[str, Any]] = []
    fingerprint_owner: dict[str, str] = {}
    for i, (start, path, info, analysis) in enumerate(timeline_items):
        prev = timeline_items[i - 1] if i else None
        gap: float | None = None
        issues: list[str] = []
        if start is None:
            issues.append('START_TIME_UNAVAILABLE')
        if prev and start is not None and (prev[0] is not None):
            try:
                gap = (start - prev[0]).total_seconds() - float(prev[3].file_row['duration_seconds'])
                if abs(gap) > 1e-06:
                    issues.append('TIME_GAP' if gap > 0 else 'TIME_OVERLAP')
            except TypeError:
                issues.append('TIMEZONE_MISMATCH')
        expected_prev = info.get('previous_file', '')
        expected_next = info.get('next_file', '')
        actual_prev = prev[1].name if prev else ''
        actual_next = timeline_items[i + 1][1].name if i + 1 < len(timeline_items) else ''
        if expected_prev and Path(expected_prev).name.lower() != actual_prev.lower():
            issues.append('PREVIOUS_FILE_MISMATCH')
        if expected_next and Path(expected_next).name.lower() != actual_next.lower():
            issues.append('NEXT_FILE_MISMATCH')
        fingerprint = str(analysis.file_row['fingerprint_first_middle_last_sha256'])
        duplicate_data_of = fingerprint_owner.get(fingerprint, '')
        if duplicate_data_of:
            issues.append('POSSIBLE_DUPLICATE_FILE_DATA')
        else:
            fingerprint_owner[fingerprint] = path.name
        boundary_duplicate = False
        if prev and prev[3].last_boundary.shape == analysis.first_boundary.shape:
            boundary_duplicate = bool(np.array_equal(prev[3].last_boundary, analysis.first_boundary, equal_nan=True))
            if boundary_duplicate:
                issues.append('EXACT_DUPLICATE_BOUNDARY_BLOCK')
        timeline.append({'order': i + 1, 'file_name': path.name, 'start_time': info.get('file_start_time', ''), 'duration_seconds': analysis.file_row['duration_seconds'], 'previous_actual': actual_prev, 'next_actual': actual_next, 'gap_after_previous_seconds': '' if gap is None else gap, 'possible_duplicate_data_of': duplicate_data_of, 'exact_duplicate_boundary_block': boundary_duplicate, 'timeline_status': 'WARN' if issues else 'PASS', 'timeline_issues': ';'.join(issues)})
    return (validation, timeline)

def summarize_channels(channel_rows: list[dict[str, Any]], config: dict[str, Any], channel_map: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in channel_rows:
        grouped.setdefault(int(row['channel_index']), []).append(row)
    map_by_channel = {row.get('channel', ''): row for row in channel_map}
    threshold = float(deep_get(config, ('thresholds', 'fixed_bad_min_file_fraction'), 0.5))
    result: list[dict[str, Any]] = []
    for index, rows in sorted(grouped.items()):
        bad_rows = [r for r in rows if r['bad_channel']]
        reasons = sorted({reason for r in bad_rows for reason in str(r['bad_reasons']).split(';') if reason})
        display = rows[0]['channel']
        mapped = map_by_channel.get(str(display), {})
        bad_fraction = len(bad_rows) / len(rows)
        std_values = np.asarray([float(r['std']) for r in rows], dtype=float)
        kurtosis_values = np.asarray([float(r['kurtosis_pearson']) for r in rows], dtype=float)
        finite_std = std_values[np.isfinite(std_values)]
        finite_kurtosis = kurtosis_values[np.isfinite(kurtosis_values)]
        result.append({'channel_index': index, 'channel': display, 'files_analyzed': len(rows), 'bad_file_count': len(bad_rows), 'bad_file_fraction': bad_fraction, 'fixed_bad_candidate': bad_fraction >= threshold, 'bad_reasons': ';'.join(reasons), 'median_std': float(np.median(finite_std)) if finite_std.size else float('nan'), 'median_kurtosis_pearson': float(np.median(finite_kurtosis)) if finite_kurtosis.size else float('nan'), 'mapped_valid_or_bad': mapped.get('valid_or_bad', ''), 'fiber_distance_m': mapped.get('fiber_distance_m', ''), 'fiber_segment': mapped.get('fiber_segment', ''), 'slope_or_leadin': mapped.get('slope_or_leadin', '')})
    return result

def make_plots(output: Path, analyses: list[FileAnalysis], selected_channels: list[int], config: dict[str, Any]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise QCError('绘图需要 matplotlib；请在 DASpy 环境中安装后重试，或使用 --no-plots。') from exc
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    heat_seconds = float(deep_get(config, ('plots', 'heatmap_bin_seconds'), 1.0))
    spacing = float(deep_get(config, ('acquisition', 'spatial_interval_m'), 0.2))
    for result in analyses:
        stem = Path(str(result.file_row['file_name'])).stem
        if result.raw_traces.size:
            fig, axes = plt.subplots(len(selected_channels), 1, figsize=(12, max(3, 2 * len(selected_channels))), sharex=True)
            axes = np.atleast_1d(axes)
            for i, (ax, ch) in enumerate(zip(axes, selected_channels)):
                ax.plot(result.raw_time, result.raw_traces[:, i], linewidth=0.5)
                ax.set_ylabel(f'Ch {ch}')
                ax.grid(alpha=0.2)
            axes[-1].set_xlabel('Time in file (s)')
            fig.suptitle(f"Representative raw waveforms: {result.file_row['file_name']}")
            fig.tight_layout()
            path = figures / f'{stem}_raw_waveforms.png'
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(str(path.resolve()))
        if result.heatmap.size:
            display = np.log10(np.maximum(result.heatmap, np.finfo(float).tiny))
            fig, ax = plt.subplots(figsize=(13, 6))
            extent = [0, display.shape[1] * heat_seconds, 0, display.shape[0] * spacing]
            image = ax.imshow(display, origin='lower', aspect='auto', extent=extent, cmap='viridis')
            ax.set_xlabel('Time in file (s)')
            ax.set_ylabel('Fiber distance from channel 0 (m)')
            ax.set_title(f"Channel-time QC map (log10 block STD): {result.file_row['file_name']}")
            fig.colorbar(image, ax=ax, label='log10(STD)')
            fig.tight_layout()
            path = figures / f'{stem}_channel_time_std.png'
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(str(path.resolve()))
    return paths

def run_qc(args: argparse.Namespace) -> int:
    input_path, output, config_path = (Path(args.input), Path(args.output), Path(args.config))
    config = load_json(config_path)
    output.mkdir(parents=True, exist_ok=True)
    files = find_files(input_path, args.recursive)
    info_rows = read_csv(Path(args.file_info)) if args.file_info else []
    channel_map = read_csv(Path(args.channel_map)) if args.channel_map else []
    index = metadata_index(info_rows) if info_rows else {}
    dataset_path = choose_dataset(files[0], deep_get(config, ('hdf5', 'data_dataset'), None))
    with h5py.File(files[0], 'r') as handle:
        shape = tuple((int(v) for v in handle[dataset_path].shape))
    expected_channels = int(deep_get(config, ('acquisition', 'expected_channels'), 500))
    channel_axis = choose_channel_axis(shape, deep_get(config, ('hdf5', 'channel_axis'), 'auto'), expected_channels)
    selected = [int(v) for v in deep_get(config, ('plots', 'representative_channels'), [0, expected_channels // 2, expected_channels - 1])]
    if bool(deep_get(config, ('hdf5', 'channel_numbers_are_one_based'), False)):
        selected = [v - 1 for v in selected]
    if min(selected) < 0 or max(selected) >= shape[channel_axis]:
        raise QCError(f'代表通道 {selected} 超出 0..{shape[channel_axis] - 1}')
    analyses: list[FileAnalysis] = []
    print(f'数据集: {dataset_path}; 通道轴: {channel_axis}; 文件数: {len(files)}')
    for number, path in enumerate(files, 1):
        print(f'[{number}/{len(files)}] {path.name}', flush=True)
        info = get_info(path, input_path, index)
        analyses.append(analyze_file(path, dataset_path, channel_axis, config, info, selected, not args.no_plots))
    file_rows = [a.file_row for a in analyses]
    channel_rows = [row for a in analyses for row in a.channel_rows]
    validation, timeline = validate_metadata(files, input_path, info_rows, analyses)
    validation_by_name = {r['file_name']: r for r in validation}
    timeline_by_name = {r['file_name']: r for r in timeline}
    for row in file_rows:
        row.update(validation_by_name.get(row['file_name'], {}))
        row.update({k: v for k, v in timeline_by_name.get(row['file_name'], {}).items() if k in {'timeline_status', 'timeline_issues'}})
    summaries = summarize_channels(channel_rows, config, channel_map)
    figures = [] if args.no_plots else make_plots(output, analyses, selected, config)
    write_csv(output / 'file_quality.csv', file_rows, list(file_rows[0].keys()))
    write_csv(output / 'channel_statistics.csv', channel_rows, list(channel_rows[0].keys()))
    write_csv(output / 'bad_channel_summary.csv', summaries, list(summaries[0].keys()))
    write_csv(output / 'continuous_timeline.csv', timeline, list(timeline[0].keys()))
    manifest = {'program': 'das_qc.py', 'step': 'data_integrity_and_basic_quality_control', 'input': str(input_path.resolve()), 'dataset_path': dataset_path, 'channel_axis': channel_axis, 'config': config, 'files': [p.name for p in files], 'metadata_table_supplied': bool(info_rows), 'channel_map_supplied': bool(channel_map), 'figures': figures, 'important_note': 'Bad-channel flags are based only on raw-data QC thresholds, never on CAT/RMS event maps.'}
    with (output / 'qc_manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    fixed = sum((bool(row['fixed_bad_candidate']) for row in summaries))
    print(f'完成：{len(files)} 个文件，固定坏通道候选 {fixed} 个。输出目录: {output.resolve()}')
    return 0

def write_templates(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / 'file_info_template.csv', [], FILE_INFO_COLUMNS)
    write_csv(directory / 'channel_map_template.csv', [], CHANNEL_MAP_COLUMNS)
    print(f'已生成模板: {directory.resolve()}')
    return 0

def inspect_file(path: Path) -> int:
    print(json.dumps({'file': str(path.resolve()), 'numeric_2d_datasets': numeric_2d_datasets(path)}, ensure_ascii=False, indent=2))
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='DAS H5 第1步：数据完整性和基础质量检查')
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='查看 H5 中的二维数值数据集')
    inspect.add_argument('h5_file')
    templates = sub.add_parser('templates', help='生成两张 CSV 说明表模板')
    templates.add_argument('output_dir')
    run = sub.add_parser('run', help='执行流式质量检查')
    run.add_argument('--input', required=True, help='H5 文件或目录')
    run.add_argument('--output', required=True, help='质控报告输出目录')
    run.add_argument('--config', required=True, help='JSON 配置')
    run.add_argument('--file-info', help='文件信息 CSV')
    run.add_argument('--channel-map', help='通道—物理位置映射 CSV')
    run.add_argument('--recursive', action='store_true', help='递归查找 H5')
    run.add_argument('--no-plots', action='store_true', help='不生成 PNG（统计表仍完整生成）')
    return parser

def main(argv: list[str] | None=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'inspect':
            return inspect_file(Path(args.h5_file))
        if args.command == 'templates':
            return write_templates(Path(args.output_dir))
        return run_qc(args)
    except (QCError, OSError, KeyError, ValueError) as exc:
        print(f'错误: {exc}', file=sys.stderr)
        return 2
if __name__ == '__main__':
    raise SystemExit(main())
