"""Center, detrend and band-pass DAS HDF5 data in bounded memory."""
from __future__ import annotations
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
from typing import Any
import h5py
import numpy as np
from scipy import signal
INPUT_PATH = REPOSITORY_ROOT / 'data/raw/S4'
OUTPUT_DIR = REPOSITORY_ROOT / 'data/preprocessed/S4'
DATASET_PATH = '/default'
TIME_AXIS = 0
SAMPLE_RATE_HZ: float | None = None
CENTER_METHOD = 'median'
BANDPASS_HZ: tuple[float, float] | None = (20.0, 2000.0)
FILTER_ORDER = 4
BLOCK_SECONDS = 5.0
PAD_SECONDS = 3.0
CHANNELS_PER_BATCH = 64
OUTPUT_DTYPE = np.float32
OVERWRITE = False

class PreprocessError(RuntimeError):
    pass

def list_input_files(input_path: Path, output_dir: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in {'.h5', '.hdf5'}:
            raise PreprocessError(f'输入文件不是 H5：{input_path}')
        return [input_path]
    if not input_path.is_dir():
        raise PreprocessError(f'输入路径不存在：{input_path}')
    output_resolved = output_dir.resolve()
    files: list[Path] = []
    for path in sorted(input_path.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.h5', '.hdf5'}:
            continue
        if output_resolved in path.resolve().parents:
            continue
        files.append(path)
    if not files:
        raise PreprocessError(f'未在 {input_path} 找到 .h5/.hdf5 文件。')
    return files

def sample_rate(dataset: h5py.Dataset) -> float:
    if SAMPLE_RATE_HZ is not None:
        rate = float(SAMPLE_RATE_HZ)
    elif 'sampling_rate' in dataset.attrs:
        rate = float(dataset.attrs['sampling_rate'])
    elif 'sampling_rate_hz' in dataset.attrs:
        rate = float(dataset.attrs['sampling_rate_hz'])
    else:
        raise PreprocessError('未找到采样率属性；请在用户配置区填写 SAMPLE_RATE_HZ。')
    if not np.isfinite(rate) or rate <= 0:
        raise PreprocessError(f'无效采样率：{rate}')
    return rate

def check_configuration(first_file: Path) -> tuple[float, np.ndarray | None]:
    with h5py.File(first_file, 'r') as handle:
        if DATASET_PATH not in handle:
            raise PreprocessError(f'{first_file.name} 中不存在数据集：{DATASET_PATH}')
        dataset = handle[DATASET_PATH]
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 2:
            raise PreprocessError(f'{DATASET_PATH} 必须是二维波形数据集。')
        rate = sample_rate(dataset)
    if TIME_AXIS not in (0, 1):
        raise PreprocessError('TIME_AXIS 必须为 0 或 1。')
    if CENTER_METHOD not in {'median', 'mean', 'none'}:
        raise PreprocessError('CENTER_METHOD 只能是 median、mean 或 none。')
    if BLOCK_SECONDS <= 0 or PAD_SECONDS < 0 or CHANNELS_PER_BATCH < 1:
        raise PreprocessError('BLOCK_SECONDS 必须大于 0，PAD_SECONDS 不能小于 0，CHANNELS_PER_BATCH 必须为正整数。')
    sos = None
    if BANDPASS_HZ is not None:
        low, high = map(float, BANDPASS_HZ)
        if not 0 < low < high < rate / 2:
            raise PreprocessError(f'带通必须满足 0 < low < high < {rate / 2:g} Hz。')
        if FILTER_ORDER < 1:
            raise PreprocessError('FILTER_ORDER 必须为正整数。')
        sos = signal.butter(FILTER_ORDER, [low, high], btype='bandpass', fs=rate, output='sos')
    return (rate, sos)

def copy_h5_except_waveform(input_file: Path, partial_file: Path) -> None:
    """复制非波形内容，避免先复制大型原始波形再删除而浪费磁盘。"""
    with h5py.File(input_file, 'r') as source, h5py.File(partial_file, 'w') as destination:
        for key, value in source.attrs.items():
            destination.attrs[key] = value
        copy_children_except_waveform(source, destination, '/')

def copy_children_except_waveform(source: h5py.Group, destination: h5py.Group, parent_path: str) -> None:
    """递归复制 H5 层级，但跳过唯一要替换的波形数据集。"""
    for name in source:
        path = parent_path.rstrip('/') + '/' + name
        if path == DATASET_PATH:
            continue
        item = source[name]
        if isinstance(item, h5py.Group):
            group = destination.create_group(name)
            for key, value in item.attrs.items():
                group.attrs[key] = value
            copy_children_except_waveform(item, group, path)
        else:
            source.copy(name, destination)

def recreate_waveform_dataset(destination: h5py.File, source: h5py.Dataset) -> h5py.Dataset:
    """以浮点型重建波形数据集，并尽可能保持原始存储设置和属性。"""
    attrs = dict(source.attrs.items())
    kwargs: dict[str, Any] = {'shape': source.shape, 'dtype': OUTPUT_DTYPE, 'chunks': source.chunks}
    if source.compression is not None:
        kwargs.update({'compression': source.compression, 'compression_opts': source.compression_opts, 'shuffle': source.shuffle, 'fletcher32': source.fletcher32})
    output = destination.create_dataset(DATASET_PATH, **kwargs)
    for key, value in attrs.items():
        output.attrs[key] = value
    return output

def read_time_block(dataset: h5py.Dataset, start: int, end: int, channel_start: int, channel_end: int) -> np.ndarray:
    if TIME_AXIS == 0:
        raw = dataset[start:end, channel_start:channel_end]
    else:
        raw = dataset[channel_start:channel_end, start:end].T
    return np.asarray(raw, dtype=np.float64)

def write_time_block(dataset: h5py.Dataset, start: int, end: int, channel_start: int, channel_end: int, values: np.ndarray) -> None:
    if TIME_AXIS == 0:
        dataset[start:end, channel_start:channel_end] = values
    else:
        dataset[channel_start:channel_end, start:end] = values.T

def process_block(values: np.ndarray, sos: np.ndarray | None) -> np.ndarray:
    if not np.isfinite(values).all():
        count = int(values.size - np.isfinite(values).sum())
        raise PreprocessError(f'发现 {count} 个 NaN/Inf；为防止滤波扩散，停止处理。')
    if CENTER_METHOD == 'median':
        values = values - np.median(values, axis=0, keepdims=True)
    elif CENTER_METHOD == 'mean':
        values = values - np.mean(values, axis=0, keepdims=True)
    values = signal.detrend(values, axis=0, type='linear')
    if sos is not None:
        values = signal.sosfiltfilt(sos, values, axis=0, padtype='odd')
    return values

def process_one(input_file: Path, output_file: Path, rate: float, sos: np.ndarray | None) -> None:
    if output_file.exists() and (not OVERWRITE):
        raise PreprocessError(f'输出已存在，已停止以避免覆盖：{output_file}')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    partial_file = output_file.with_suffix(output_file.suffix + '.partial')
    if partial_file.exists():
        partial_file.unlink()
    block = max(1, int(round(BLOCK_SECONDS * rate)))
    pad = int(round(PAD_SECONDS * rate))
    try:
        copy_h5_except_waveform(input_file, partial_file)
        with h5py.File(input_file, 'r') as source, h5py.File(partial_file, 'r+') as destination:
            input_data = source[DATASET_PATH]
            if input_data.ndim != 2:
                raise PreprocessError(f'{input_file.name}: 波形数据集不是二维。')
            if not np.isclose(sample_rate(input_data), rate):
                raise PreprocessError(f'{input_file.name}: 采样率与首文件不一致。')
            output_data = recreate_waveform_dataset(destination, input_data)
            count = int(input_data.shape[TIME_AXIS])
            channel_count = int(input_data.shape[1 - TIME_AXIS])
            for core_start in range(0, count, block):
                core_end = min(count, core_start + block)
                extended_start = max(0, core_start - pad)
                extended_end = min(count, core_end + pad)
                for channel_start in range(0, channel_count, CHANNELS_PER_BATCH):
                    channel_end = min(channel_count, channel_start + CHANNELS_PER_BATCH)
                    filtered = process_block(read_time_block(input_data, extended_start, extended_end, channel_start, channel_end), sos)
                    begin = core_start - extended_start
                    write_time_block(output_data, core_start, core_end, channel_start, channel_end, filtered[begin:begin + core_end - core_start])
                print(f'  {input_file.name}: {core_end}/{count} samples', flush=True)
            metadata = destination.require_group('processing_metadata')
            metadata.attrs['preprocess_parameters_json'] = json.dumps({'program': Path(__file__).name, 'created_utc': datetime.now(timezone.utc).isoformat(), 'source_file': str(input_file.resolve()), 'dataset_path': DATASET_PATH, 'time_axis': TIME_AXIS, 'sample_rate_hz': rate, 'center_method': CENTER_METHOD, 'linear_detrend': True, 'bandpass_hz': BANDPASS_HZ, 'filter_order': FILTER_ORDER if BANDPASS_HZ is not None else None, 'block_seconds': BLOCK_SECONDS, 'pad_seconds': PAD_SECONDS, 'channels_per_batch': CHANNELS_PER_BATCH, 'parameter_status': 'provisional; validate scientifically before treating as final'}, ensure_ascii=False)
        if output_file.exists():
            output_file.unlink()
        partial_file.replace(output_file)
    except Exception:
        if partial_file.exists():
            partial_file.unlink()
        raise

def preflight_storage(files: list[Path]) -> None:
    """写入前预估输出空间，避免大数据处理中途磁盘耗尽。"""
    waveform_bytes = 0
    input_bytes = 0
    non_wave_bytes = 0
    for path in files:
        input_bytes += path.stat().st_size
        with h5py.File(path, 'r') as handle:
            dataset = handle[DATASET_PATH]
            waveform_bytes += int(np.prod(dataset.shape)) * np.dtype(OUTPUT_DTYPE).itemsize
            non_wave_bytes += max(0, path.stat().st_size - int(dataset.id.get_storage_size()))
    expected_output = non_wave_bytes + waveform_bytes
    required = int(expected_output * 1.15) + 2 * 1024 ** 3
    free = shutil.disk_usage(OUTPUT_DIR).free
    print(f'存储预检：输入文件 {input_bytes / 1024 ** 3:.1f} GiB；预计至少需要 {required / 1024 ** 3:.1f} GiB 可用空间；目标盘当前可用 {free / 1024 ** 3:.1f} GiB。')
    if free < required:
        raise PreprocessError('输出盘可用空间不足；为避免处理中断和残留大文件，未开始写入。')

def main() -> int:
    input_path = INPUT_PATH.resolve()
    output_dir = OUTPUT_DIR.resolve()
    if input_path == output_dir:
        raise PreprocessError('INPUT_PATH 与 OUTPUT_DIR 不能相同。')
    files = list_input_files(input_path, output_dir)
    rate, sos = check_configuration(files[0])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preflight_storage(files)
    print(f'文件数：{len(files)}；数据集：{DATASET_PATH}；采样率：{rate:g} Hz')
    if BANDPASS_HZ is None:
        print('带通：关闭')
    else:
        print(f'带通：{BANDPASS_HZ[0]:g}--{BANDPASS_HZ[1]:g} Hz（暂定参数）')
    for index, input_file in enumerate(files, 1):
        relative = input_file.name if input_path.is_file() else input_file.relative_to(input_path)
        print(f'[{index}/{len(files)}] {input_file.name}')
        process_one(input_file, output_dir / relative, rate, sos)
    print('完成。未生成额外 TXT/JSON 文件；参数已写入每个输出 H5 内部。')
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (PreprocessError, OSError, ValueError) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        raise SystemExit(2)
