"""DAS processing functions for quality reporting."""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
from openpyxl import load_workbook
from das_qc import QCError, analyze_file, choose_channel_axis, choose_dataset, deep_get, find_files, get_info, load_json, metadata_index, read_csv, robust_z, validate_metadata, write_csv

def read_layout_xlsx(path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the supplied layout workbook to every channel in the H5 dataset."""
    sheet_name = str(deep_get(config, ('layout', 'sheet_name'), 'Sheet1'))
    source_offset = int(deep_get(config, ('layout', 'source_channel_offset'), 3000))
    expected = int(deep_get(config, ('acquisition', 'expected_channels'), 500))
    one_based = bool(deep_get(config, ('hdf5', 'channel_numbers_are_one_based'), False))
    channel_min = 1 if one_based else 0
    channel_max = expected if one_based else expected - 1
    workbook = load_workbook(path, read_only=True, data_only=True)
    if sheet_name not in workbook.sheetnames:
        raise QCError(f'布设表不存在工作表 {sheet_name!r}；实际为 {workbook.sheetnames}')
    sheet = workbook[sheet_name]
    values = sheet.iter_rows(values_only=True)
    headers = [str(v).strip() if v is not None else '' for v in next(values)]
    required = {'channel', 'segment_id', 'segment_type', 'burial_depth', 'x_cm', 'y_cm', 'z_cm', 'slope_or_leadin'}
    missing = sorted(required - set(headers))
    if missing:
        raise QCError(f'布设表缺少列: {missing}')
    mapped: dict[int, dict[str, Any]] = {}
    for line_number, values_row in enumerate(values, 2):
        raw = dict(zip(headers, values_row))
        if raw.get('channel') in (None, ''):
            continue
        source_channel = int(raw['channel'])
        channel = source_channel - source_offset
        if not channel_min <= channel <= channel_max:
            raise QCError(f'布设表第{line_number}行映射后通道{channel}超出{channel_min}..{channel_max}')
        if channel in mapped:
            raise QCError(f'布设表通道重复: {channel}（原编号{source_channel}）')
        role = str(raw.get('slope_or_leadin') or 'slope').strip().lower()
        segment_type = str(raw.get('segment_type') or 'unknown').strip().lower()
        burial = _finite_or_blank(raw.get('burial_depth'))
        mapped[channel] = {'channel': channel, 'source_channel': source_channel, 'segment_id': str(raw.get('segment_id') or '').strip(), 'segment_type': segment_type, 'installation_role': role, 'depth_group_original': str(raw.get('depth_group') or '').strip(), 'burial_depth_cm': burial, 'x_cm': _finite_or_blank(raw.get('x_cm')), 'y_cm': _finite_or_blank(raw.get('y_cm')), 'z_cm': _finite_or_blank(raw.get('z_cm')), 'listed_in_layout': True}
    result: list[dict[str, Any]] = []
    for channel in range(channel_min, channel_max + 1):
        row = mapped.get(channel, {'channel': channel, 'source_channel': channel + source_offset, 'segment_id': 'leadin_unlisted', 'segment_type': 'leadin', 'installation_role': 'leadin', 'depth_group_original': '', 'burial_depth_cm': '', 'x_cm': '', 'y_cm': '', 'z_cm': '', 'listed_in_layout': False})
        depth = row['burial_depth_cm'] if row['burial_depth_cm'] != '' else 'unknown'
        row['qc_group'] = f"{row['installation_role']}|{row['segment_type']}|depth_{depth}cm"
        result.append(row)
    return result

def _finite_or_blank(value: Any) -> float | str:
    if value in (None, ''):
        return ''
    number = float(value)
    return number if math.isfinite(number) else ''

def choose_representative_names(files: list[Path], info_rows: list[dict[str, str]], config: dict[str, Any]) -> set[str]:
    explicit = [str(v) for v in deep_get(config, ('paper_figures', 'representative_files'), [])]
    if explicit:
        return {Path(v).name.lower() for v in explicit}
    index = metadata_index(info_rows) if info_rows else {}
    stable = [p for p in files if get_info(p, files[0].parent, index).get('data_type') == 'stable_rain']
    pool = stable or files
    count = max(1, min(int(deep_get(config, ('paper_figures', 'auto_representative_count'), 3)), len(pool)))
    positions = np.linspace(0, len(pool) - 1, count).round().astype(int)
    return {pool[int(i)].name.lower() for i in positions}

def sort_continuous_files(files: list[Path], input_path: Path, info_rows: list[dict[str, str]]) -> list[Path]:
    """Use true start time when supplied; otherwise preserve deterministic name order."""
    if not info_rows:
        return files
    index = metadata_index(info_rows)

    def key(path: Path) -> tuple[int, datetime, str]:
        info = get_info(path, input_path, index)
        raw = str(info.get('file_start_time', '')).strip()
        if raw:
            try:
                return (0, datetime.fromisoformat(raw.replace('Z', '+00:00')), path.name.lower())
            except ValueError:
                pass
        return (1, datetime.max, path.name.lower())
    try:
        return sorted(files, key=key)
    except TypeError as exc:
        raise QCError('file_start_time不能混用带时区和不带时区的时间。') from exc

def enrich_and_reclassify(channel_rows: list[dict[str, Any]], layout: list[dict[str, Any]], config: dict[str, Any]) -> None:
    layout_by_channel = {int(r['channel']): r for r in layout}
    thresholds = config.get('thresholds', {})
    min_group = int(thresholds.get('min_soft_qc_group_size', 8))
    std_limit = float(thresholds.get('std_group_robust_z', 6.0))
    by_file_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in channel_rows:
        channel = int(row['channel'])
        if channel not in layout_by_channel:
            available = f'{min(layout_by_channel)}..{max(layout_by_channel)}'
            raise QCError(f'H5通道{channel}没有布设映射（当前映射范围{available}）。请核对EXPECTED_CHANNELS和DATA_CHANNEL_NUMBERS_ARE_ONE_BASED。')
        mapped = layout_by_channel[channel]
        row.update({key: value for key, value in mapped.items() if key != 'channel'})
        by_file_group[str(row['file_name']), str(mapped['qc_group'])].append(row)
    for rows in by_file_group.values():
        values = np.asarray([math.log10(max(float(r['std']), np.finfo(float).tiny)) for r in rows])
        z = robust_z(values) if len(rows) >= min_group else np.zeros(len(rows))
        for row, value in zip(rows, z):
            row['group_log_std_robust_z'] = float(value)
    for row in channel_rows:
        hard: list[str] = []
        if float(row['nonfinite_fraction']) > float(thresholds.get('max_nonfinite_fraction', 0.01)):
            hard.append('NONFINITE')
        if float(row['zero_fraction']) > float(thresholds.get('max_zero_fraction', 0.99)):
            hard.append('ZERO_OR_DEAD')
        if float(row['saturation_fraction']) > float(thresholds.get('max_saturation_fraction', 0.001)):
            hard.append('SATURATED')
        if float(row['flat_difference_fraction']) > float(thresholds.get('max_flat_fraction', 0.99)):
            hard.append('FLATLINE')
        if not np.isfinite(float(row['std'])) or float(row['std']) == 0:
            hard.append('NO_VARIANCE')
        soft: list[str] = []
        z = float(row['group_log_std_robust_z'])
        if z < -std_limit:
            soft.append('LOW_STD_WITHIN_LAYOUT_GROUP')
        elif z > std_limit:
            soft.append('HIGH_STD_WITHIN_LAYOUT_GROUP')
        kurtosis = float(row['kurtosis_pearson'])
        if np.isfinite(kurtosis) and kurtosis > float(thresholds.get('max_kurtosis', 100.0)):
            soft.append('HIGH_KURTOSIS_REVIEW')
        jump_ratio = float(row['max_jump_divided_by_std'])
        if np.isfinite(jump_ratio) and jump_ratio > float(thresholds.get('max_jump_std_ratio', 50.0)):
            soft.append('ABRUPT_JUMP_REVIEW')
        row['hard_bad'] = bool(hard)
        row['hard_bad_reasons'] = ';'.join(hard)
        row['review_candidate'] = bool(soft)
        row['review_reasons'] = ';'.join(soft)
        row['qc_class'] = 'HARD_BAD' if hard else 'REVIEW' if soft else 'PASS'

def summarize_channels_paper(rows: list[dict[str, Any]], info_rows: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    info_by_name = {Path(r.get('file_name', '')).name.lower(): r for r in info_rows}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row['channel'])].append(row)
    threshold = float(deep_get(config, ('thresholds', 'fixed_bad_min_file_fraction'), 0.5))
    result: list[dict[str, Any]] = []
    for channel, values in sorted(grouped.items()):
        hard = [r for r in values if r['hard_bad']]
        stable = [r for r in values if info_by_name.get(Path(str(r['file_name'])).name.lower(), {}).get('data_type') == 'stable_rain']
        baseline = stable or values
        review = [r for r in baseline if r['review_candidate']]
        first = values[0]

        def med(key: str) -> float:
            numbers = np.asarray([float(v[key]) for v in values], dtype=float)
            numbers = numbers[np.isfinite(numbers)]
            return float(np.median(numbers)) if numbers.size else float('nan')
        result.append({'channel': channel, 'source_channel': first['source_channel'], 'segment_id': first['segment_id'], 'segment_type': first['segment_type'], 'installation_role': first['installation_role'], 'burial_depth_cm': first['burial_depth_cm'], 'x_cm': first['x_cm'], 'y_cm': first['y_cm'], 'z_cm': first['z_cm'], 'listed_in_layout': first['listed_in_layout'], 'qc_group': first['qc_group'], 'files_analyzed': len(values), 'hard_bad_file_count': len(hard), 'hard_bad_file_fraction': len(hard) / len(values), 'fixed_bad_candidate': len(hard) / len(values) >= threshold, 'hard_bad_reasons': ';'.join(sorted({x for r in hard for x in str(r['hard_bad_reasons']).split(';') if x})), 'stable_baseline_files': len(stable), 'review_file_count_in_baseline': len(review), 'review_file_fraction_in_baseline': len(review) / len(baseline), 'review_reasons': ';'.join(sorted({x for r in review for x in str(r['review_reasons']).split(';') if x})), 'median_valid_fraction': med('valid_fraction'), 'median_rms': med('rms'), 'median_std': med('std'), 'median_kurtosis_pearson': med('kurtosis_pearson')})
    return result

def build_file_summary(file_rows: list[dict[str, Any]], channel_rows: list[dict[str, Any]]) -> None:
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in channel_rows:
        by_file[str(row['file_name'])].append(row)
    for row in file_rows:
        values = by_file[str(row['file_name'])]
        row['hard_bad_channel_count'] = sum((bool(v['hard_bad']) for v in values))
        row['review_channel_count'] = sum((bool(v['review_candidate']) for v in values))
        row['mean_valid_fraction'] = float(np.nanmean([float(v['valid_fraction']) for v in values]))
        row['mean_nonfinite_fraction'] = float(np.nanmean([float(v['nonfinite_fraction']) for v in values]))
        row['mean_saturation_fraction'] = float(np.nanmean([float(v['saturation_fraction']) for v in values]))
        row['quality_status'] = 'FAIL' if row['structural_issues'] else 'WARN' if row['hard_bad_channel_count'] else 'PASS'

def group_summary(channel_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in channel_summary:
        groups[str(row['qc_group'])].append(row)
    result: list[dict[str, Any]] = []
    for name, rows in sorted(groups.items()):
        result.append({'qc_group': name, 'channel_count': len(rows), 'fixed_bad_channel_count': sum((bool(r['fixed_bad_candidate']) for r in rows)), 'valid_channel_rate': 1.0 - sum((bool(r['fixed_bad_candidate']) for r in rows)) / len(rows), 'median_rms': float(np.nanmedian([float(r['median_rms']) for r in rows])), 'median_valid_fraction': float(np.nanmedian([float(r['median_valid_fraction']) for r in rows]))})
    return result

def write_manuscript_summary(output: Path, files: list[Path], file_rows: list[dict[str, Any]], channel_summary: list[dict[str, Any]], timeline: list[dict[str, Any]], config: dict[str, Any]) -> None:
    duration = sum((float(r['duration_seconds']) for r in file_rows))
    total_bytes = sum((int(r['file_size_bytes']) for r in file_rows))
    fixed_bad = sum((bool(r['fixed_bad_candidate']) for r in channel_summary))
    gaps = sum((max(0.0, float(r['gap_after_previous_seconds'])) for r in timeline if r['gap_after_previous_seconds'] not in ('', None)))
    overlaps = -sum((min(0.0, float(r['gap_after_previous_seconds'])) for r in timeline if r['gap_after_previous_seconds'] not in ('', None)))
    summary = [{'experiment': next((str(r.get('experiment')) for r in file_rows if r.get('experiment')), 'S1'), 'file_count': len(files), 'data_size_gb': total_bytes / 1000000000.0, 'record_duration_hours': duration / 3600.0, 'time_gap_seconds': gaps, 'time_overlap_seconds': overlaps, 'total_channels': len(channel_summary), 'fixed_bad_channels': fixed_bad, 'valid_channel_rate': 1.0 - fixed_bad / max(len(channel_summary), 1), 'files_with_structural_issues': sum((bool(r['structural_issues']) for r in file_rows)), 'files_with_hard_bad_channels': sum((int(r['hard_bad_channel_count']) > 0 for r in file_rows))}]
    write_csv(output / 'paper_tables' / 'Table_QC_summary.csv', summary, list(summary[0].keys()))
    method = f"DAS DATA QUALITY CONTROL (processing summary)\n\nAll {len(files)} raw HDF5 files were subjected to automated integrity and channel-level quality control before event analysis. Records were read sequentially in {deep_get(config, ('processing', 'block_seconds'), 1.0)} s blocks without modifying the raw files. File readability, dimensions, channel count, duration, temporal continuity, duplicate content, non-finite values, zero-valued samples, hard saturation, repeated values, channel mean, standard deviation, root-mean-square amplitude, Pearson kurtosis, and maximum sample-to-sample jump were evaluated.\n\nDefinite acquisition/storage defects (non-finite data, persistent zeros, hard saturation, flatlining, or zero variance) were classified as hard quality-control failures. Layout-dependent amplitude, kurtosis, and jump anomalies were evaluated within groups defined by installation role, segment type, and burial depth and were retained as review candidates rather than automatically removed. A fixed bad-channel candidate was defined as a channel with hard failures in at least {100 * deep_get(config, ('thresholds', 'fixed_bad_min_file_fraction'), 0.5):.0f}% of files. Suspected slope activity, visible sliding, human-noise labels, and subsequent CAT-RMS results were not used to define bad channels.\n"
    (output / 'paper_tables' / 'quality_control_methods.txt').write_text(method, encoding='utf-8')

def setup_plotting() -> Any:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise QCError('论文图需要 matplotlib，请在DASpy环境安装 requirements.txt。') from exc
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight'})
    return plt

def save_figure(fig: Any, base: Path, dpi: int) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = (base.with_suffix('.png'), base.with_suffix('.pdf'))
    fig.savefig(png, dpi=dpi, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    return [str(png.resolve()), str(pdf.resolve())]

def make_paper_figures(output: Path, file_rows: list[dict[str, Any]], channel_summary: list[dict[str, Any]], analyses: list[Any], config: dict[str, Any]) -> list[str]:
    plt = setup_plotting()
    dpi = int(deep_get(config, ('paper_figures', 'dpi'), 300))
    figures = output / 'paper_figures'
    paths: list[str] = []
    channels = np.asarray([int(r['channel']) for r in channel_summary])
    valid = np.asarray([float(r['median_valid_fraction']) for r in channel_summary])
    rms = np.asarray([float(r['median_rms']) for r in channel_summary])
    hard = np.asarray([float(r['hard_bad_file_fraction']) for r in channel_summary])
    fixed = np.asarray([bool(r['fixed_bad_candidate']) for r in channel_summary])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
    x = np.arange(1, len(file_rows) + 1)
    axes[0, 0].plot(x, [float(r['duration_seconds']) for r in file_rows], color='#276FBF', lw=1)
    expected = float(deep_get(config, ('acquisition', 'expected_duration_seconds'), 600))
    axes[0, 0].axhline(expected, color='black', ls='--', lw=0.8)
    axes[0, 0].set(xlabel='File order', ylabel='Duration (s)', title='a  Record duration')
    axes[0, 1].bar(x, [int(r['hard_bad_channel_count']) for r in file_rows], color='#D1495B', width=0.8)
    axes[0, 1].set(xlabel='File order', ylabel='Channels', title='b  Hard-QC channels per file')
    axes[1, 0].plot(channels, valid, color='#2A9D8F', lw=0.8)
    axes[1, 0].scatter(channels[fixed], valid[fixed], s=12, color='#D1495B', zorder=3, label='Fixed bad candidate')
    axes[1, 0].set(xlabel='DAS channel', ylabel='Median valid fraction', title='c  Channel data availability', ylim=(-0.02, 1.02))
    if fixed.any():
        axes[1, 0].legend(frameon=False, fontsize=7)
    axes[1, 1].plot(channels, hard, color='#6A4C93', lw=0.8)
    axes[1, 1].set(xlabel='DAS channel', ylabel='Fraction of files', title='d  Persistent hard-QC occurrence', ylim=(-0.02, 1.02))
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.2, lw=0.5)
    paths += save_figure(fig, figures / 'Figure_QC_overview', dpi)
    plt.close(fig)
    roles = [str(r['installation_role']) for r in channel_summary]
    role_colors = {'slope': '#2A9D8F', 'connection': '#E9C46A', 'leadin': '#8D99AE'}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, constrained_layout=True)
    for role in sorted(set(roles)):
        mask = np.asarray([v == role for v in roles])
        axes[0].scatter(channels[mask], rms[mask], s=9, color=role_colors.get(role, '#777777'), label=role, alpha=0.85)
        review = np.asarray([float(r['review_file_fraction_in_baseline']) for r in channel_summary])
        axes[1].scatter(channels[mask], review[mask], s=9, color=role_colors.get(role, '#777777'), label=role, alpha=0.85)
    if np.all(rms[np.isfinite(rms)] > 0):
        axes[0].set_yscale('log')
    axes[0].set(ylabel='Median RMS', title='a  Background amplitude by channel')
    axes[1].set(xlabel='DAS channel', ylabel='Baseline review fraction', title='b  Layout-group statistical review flags', ylim=(-0.02, 1.02))
    axes[0].legend(frameon=False, ncol=3, fontsize=7)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.2, lw=0.5)
    paths += save_figure(fig, figures / 'Figure_channel_QC_metrics', dpi)
    plt.close(fig)
    layout_points = [r for r in channel_summary if r['x_cm'] != '' and r['y_cm'] != '']
    depths = sorted({float(r['burial_depth_cm']) for r in layout_points if r['burial_depth_cm'] != ''}, reverse=True)
    if layout_points and depths:
        fig, axes = plt.subplots(1, len(depths), figsize=(max(7.2, 2.4 * len(depths)), 2.8), squeeze=False, constrained_layout=True)
        for ax, depth in zip(axes[0], depths):
            points = [r for r in layout_points if float(r['burial_depth_cm']) == depth]
            colors = ['#D1495B' if r['fixed_bad_candidate'] else role_colors.get(str(r['installation_role']), '#777777') for r in points]
            ax.scatter([float(r['x_cm']) for r in points], [float(r['y_cm']) for r in points], c=colors, s=14, alpha=0.85)
            ax.set(title=f'Depth {depth:g} cm', xlabel='x (cm)', ylabel='y (cm)')
            ax.set_aspect('equal', adjustable='box')
            ax.spines[['top', 'right']].set_visible(False)
        paths += save_figure(fig, figures / 'Figure_spatial_QC', dpi)
        plt.close(fig)
    heat_seconds = float(deep_get(config, ('plots', 'heatmap_bin_seconds'), 1.0))
    for result in analyses:
        if not result.heatmap.size:
            continue
        if result.raw_traces.size:
            fig, axes = plt.subplots(result.raw_traces.shape[1], 1, figsize=(7.2, 1.5 * result.raw_traces.shape[1]), sharex=True, squeeze=False, constrained_layout=True)
            selected_display = [int(v) for v in deep_get(config, ('plots', 'representative_channels'), [])]
            for index, ax in enumerate(axes[:, 0]):
                label = selected_display[index] if index < len(selected_display) else index + 1
                ax.plot(result.raw_time, result.raw_traces[:, index], color='#276FBF', lw=0.45)
                ax.set_ylabel(f'Ch {label}')
                ax.spines[['top', 'right']].set_visible(False)
                ax.grid(axis='y', alpha=0.2, lw=0.5)
            axes[-1, 0].set_xlabel('Time in file (s)')
            axes[0, 0].set_title(f"Representative raw waveforms: {result.file_row['file_name']}")
            paths += save_figure(fig, figures / f"Figure_raw_{Path(str(result.file_row['file_name'])).stem}", dpi)
            plt.close(fig)
        display = np.log10(np.maximum(result.heatmap, np.finfo(float).tiny))
        lo, hi = np.nanpercentile(display[np.isfinite(display)], [2, 98])
        fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
        image = ax.imshow(display, origin='lower', aspect='auto', extent=[0, display.shape[1] * heat_seconds, 0.5, display.shape[0] + 0.5], cmap='viridis', vmin=lo, vmax=hi)
        ax.set(xlabel='Time in file (s)', ylabel='DAS channel', title=f"Channel–time background variability: {result.file_row['file_name']}")
        fig.colorbar(image, ax=ax, label='log10(block standard deviation)')
        paths += save_figure(fig, figures / f"Figure_heatmap_{Path(str(result.file_row['file_name'])).stem}", dpi)
        plt.close(fig)
    return paths

def run(args: argparse.Namespace) -> int:
    input_path, output, config_path = (Path(args.input), Path(args.output), Path(args.config))
    config = load_json(config_path)
    setup_plotting()
    output.mkdir(parents=True, exist_ok=True)
    files = find_files(input_path, args.recursive)
    layout = read_layout_xlsx(Path(args.layout), config)
    info_rows = read_csv(Path(args.file_info)) if args.file_info else []
    files = sort_continuous_files(files, input_path, info_rows)
    info_index = metadata_index(info_rows) if info_rows else {}
    dataset_path = choose_dataset(files[0], deep_get(config, ('hdf5', 'data_dataset'), None))
    import h5py
    with h5py.File(files[0], 'r') as handle:
        shape = tuple((int(v) for v in handle[dataset_path].shape))
    expected = int(deep_get(config, ('acquisition', 'expected_channels'), 500))
    axis = choose_channel_axis(shape, deep_get(config, ('hdf5', 'channel_axis'), 'auto'), expected)
    actual_channels = int(shape[axis])
    if actual_channels != expected:
        raise QCError(f'H5实际通道数为{actual_channels}，但EXPECTED_CHANNELS={expected}；数据集{dataset_path}的shape={shape}、通道轴={axis}。请修改入口文件参数。')
    selected_display = [int(v) for v in deep_get(config, ('plots', 'representative_channels'), [80, 250, 495])]
    one_based = bool(deep_get(config, ('hdf5', 'channel_numbers_are_one_based'), False))
    selected_indices = [v - 1 if one_based else v for v in selected_display]
    if min(selected_indices) < 0 or max(selected_indices) >= shape[axis]:
        raise QCError(f'代表通道{selected_display}超出H5通道范围；数据集shape={shape}，通道轴={axis}。')
    representatives = choose_representative_names(files, info_rows, config)
    analyses = []
    display_min, display_max = (1, expected) if one_based else (0, expected - 1)
    print(f'Found {len(files)} H5 files; dataset={dataset_path}; shape={shape}; channel_axis={axis}')
    print(f'DAS channel numbering: {display_min}..{display_max} ({expected} channels)')
    print(f'Representative heatmaps: {sorted(representatives)}')
    for i, path in enumerate(files, 1):
        print(f'[{i}/{len(files)}] {path.name}', flush=True)
        info = get_info(path, input_path, info_index)
        analyses.append(analyze_file(path, dataset_path, axis, config, info, selected_indices, path.name.lower() in representatives))
    file_rows = [a.file_row for a in analyses]
    channel_rows = [row for a in analyses for row in a.channel_rows]
    enrich_and_reclassify(channel_rows, layout, config)
    validation, timeline = validate_metadata(files, input_path, info_rows, analyses)
    validation_by_name = {r['file_name']: r for r in validation}
    timeline_by_name = {r['file_name']: r for r in timeline}
    for row in file_rows:
        row.update(validation_by_name.get(row['file_name'], {}))
        row.update({k: v for k, v in timeline_by_name.get(row['file_name'], {}).items() if k in {'timeline_status', 'timeline_issues'}})
    build_file_summary(file_rows, channel_rows)
    channel_summary = summarize_channels_paper(channel_rows, info_rows, config)
    grouped = group_summary(channel_summary)
    write_csv(output / 'tables' / 'file_quality.csv', file_rows, list(file_rows[0].keys()))
    write_csv(output / 'tables' / 'channel_statistics_all_files.csv', channel_rows, list(channel_rows[0].keys()))
    write_csv(output / 'tables' / 'channel_QC_summary.csv', channel_summary, list(channel_summary[0].keys()))
    write_csv(output / 'tables' / 'layout_group_QC_summary.csv', grouped, list(grouped[0].keys()))
    write_csv(output / 'tables' / 'continuous_timeline.csv', timeline, list(timeline[0].keys()))
    write_csv(output / 'tables' / 'expanded_500_channel_layout.csv', layout, list(layout[0].keys()))
    write_manuscript_summary(output, files, file_rows, channel_summary, timeline, config)
    figures = make_paper_figures(output, file_rows, channel_summary, analyses, config)
    manifest = {'program': 'das_qc_paper.py', 'input': str(input_path.resolve()), 'layout': str(Path(args.layout).resolve()), 'dataset_path': dataset_path, 'channel_axis': axis, 'files': [str(p.resolve()) for p in files], 'representative_files': sorted(representatives), 'config': config, 'figures': figures, 'classification': {'hard_bad': 'definite acquisition/storage defects', 'review_candidate': 'layout-group statistical anomaly; not automatically removed'}}
    (output / 'qc_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'QC complete. Results: {output.resolve()}')
    return 0

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description='S1 DAS paper-ready streaming quality control')
    sub = value.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='Inspect numeric 2-D H5 datasets')
    inspect.add_argument('h5_file')
    run_parser = sub.add_parser('run', help='Run all-file QC and make paper figures')
    run_parser.add_argument('--input', required=True, help='H5 directory')
    run_parser.add_argument('--output', required=True, help='Result directory')
    run_parser.add_argument('--config', required=True)
    run_parser.add_argument('--layout', required=True, help='S1 layout xlsx')
    run_parser.add_argument('--file-info', help='File metadata CSV')
    run_parser.add_argument('--recursive', action='store_true')
    return value

def main(argv: list[str] | None=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == 'inspect':
            from das_qc import inspect_file
            return inspect_file(Path(args.h5_file))
        return run(args)
    except (QCError, OSError, ValueError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
if __name__ == '__main__':
    raise SystemExit(main())
