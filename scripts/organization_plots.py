"""Plot saved spatial-organization scan results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.text import Text
from pypdf import PdfReader
MM = 1 / 25.4
SINGLE_WIDTH_MM = 84.0
WIDE_WIDTH_MM = 168.0
QUALITY = []
RED = '#B32624'
BLUE = '#246A93'
GREY = '#6C737A'

def configure_font():
    path = font_manager.findfont('Times New Roman', fallback_to_default=False)
    if not Path(path).is_file():
        raise FileNotFoundError('Times New Roman is required')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 8, 'figure.titlesize': 10, 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.65, 'lines.linewidth': 1.1, 'xtick.major.width': 0.65, 'ytick.major.width': 0.65, 'xtick.major.size': 3, 'ytick.major.size': 3, 'savefig.bbox': None})
    return path

def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.15, lw=0.5)
    ax.tick_params(pad=2)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

def time_axis(ax, result):
    duration = (pd.Timestamp(result['spec'].end) - pd.Timestamp(result['spec'].start)).total_seconds() / 3600
    ax.set_xlim(0, duration)
    ax.set_xticks(np.linspace(0, duration, 4))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.set_xlabel('Time since start (h)', labelpad=3)

def save(fig, path, width_mm, height_mm, dpi):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outer = fig.bbox
    clipping = []
    font_errors = []
    hidden_tick_text = set()
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            low, high = sorted(limits)
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                if not low - 1e-10 <= tick.get_loc() <= high + 1e-10:
                    hidden_tick_text.update([id(tick.label1), id(tick.label2)])
    for text in fig.findobj(match=Text):
        if id(text) in hidden_tick_text or not text.get_visible() or (not text.get_text().strip()):
            continue
        box = text.get_window_extent(renderer)
        if text.axes is not None and text.get_clip_on():
            continue
        if box.x0 < outer.x0 - 2 or box.y0 < outer.y0 - 2 or box.x1 > outer.x1 + 2 or (box.y1 > outer.y1 + 2):
            clipping.append(text.get_text())
        if text.get_fontfamily() != ['Times New Roman'] or round(text.get_fontsize(), 3) not in (8, 10):
            font_errors.append(text.get_text())
    if font_errors:
        raise ValueError(f'Unexpected plot font/size: {path.name}: {font_errors}')
    if clipping:
        raise ValueError(f'Plot text outside fixed canvas: {path.name}: {clipping}')
    fig.savefig(path.with_suffix('.png'), dpi=dpi)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)
    reader = PdfReader(str(path.with_suffix('.pdf')))
    page = reader.pages[0]
    width = float(page.mediabox.width) / 72 * 25.4
    height = float(page.mediabox.height) / 72 * 25.4
    fonts = []
    resources = page['/Resources'].get_object()
    if '/Font' in resources:
        for item in resources['/Font'].get_object().values():
            fonts.append(str(item.get_object()['/BaseFont']))
    if not np.isclose(width, width_mm, atol=0.02) or not np.isclose(height, height_mm, atol=0.02):
        raise ValueError(f'Unexpected exported size: {path.name}, {width}, {height}')
    if not fonts or any(('TimesNewRoman' not in f for f in fonts)):
        raise ValueError(f'PDF must embed Times New Roman: {fonts}')
    QUALITY.append(dict(file=str(path.with_suffix('.pdf')), width_mm=width, height_mm=height, requested_width_mm=width_mm, body_font_pt=10, annotation_font_pt=8, pdf_fonts=';'.join(fonts), text_within_canvas=True))

def draw_moran(ax, result, title):
    frame = result['windows']
    view = frame[frame.duration_minutes.eq(10)].sort_values('start_time')
    x = (view.start_hours + view.end_hours) / 2
    ax.fill_between(x, view.null_q025, view.null_q975, color='#E1E5E8', label='Pointwise null 95%')
    ax.plot(x, view.null_mean, color=GREY, lw=0.8, label='Null mean')
    ax.plot(x, view.simultaneous_upper_study, color=BLUE, ls='--', lw=0.9, label='Scan-wide upper bound')
    ax.plot(x, view.observed_moran_i, color=RED, lw=1.0, label='Observed')
    ax.set_title(title, loc='left', pad=7)
    ax.set_ylabel("Event-frequency Moran's I", labelpad=3)
    time_axis(ax, result)
    style(ax)

def draw_maxscan(ax, result, title):
    w = result['windows']
    time = w.groupby('start_time', sort=True).agg(start_hours=('start_hours', 'first'), scan_z=('scan_z', 'max'))
    ax.plot(time.start_hours, time.scan_z, color=RED, lw=0.9)
    if result['critical'] is not None:
        ax.axhline(result['critical'], color=BLUE, ls='--', lw=0.9)
    start = pd.Timestamp(result['spec'].start)
    for r in result['episodes'].itertuples():
        a = (r.start_time - start).total_seconds() / 3600
        b = (r.end_time - start).total_seconds() / 3600
        ax.axvspan(a, b, color='#DDEADF', zorder=0)
    ax.set_title(title, loc='left', pad=7)
    ax.set_ylabel('Maximum scan score', labelpad=3)
    time_axis(ax, result)
    style(ax)

def plot_experiment(result, settings, dpi):
    name = result['spec'].name
    out = result['output']
    fig, axes = plt.subplots(1, 2, figsize=(168 * MM, 82 * MM))
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.205, top=0.86, wspace=0.4)
    draw_moran(axes[0], result, f'a  {name}: 10-min display')
    draw_maxscan(axes[1], result, f'b  {name}: all scanned durations')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=2, frameon=False, columnspacing=1.1, handlelength=1.5, labelspacing=0.15)
    save(fig, out / 'Figure01_time_and_scan', 168, 82, dpi)
    fig, ax = plt.subplots(figsize=(84 * MM, 74 * MM))
    fig.subplots_adjust(left=0.205, right=0.965, bottom=0.205, top=0.86)
    maxima = result['maxima']
    if len(maxima):
        ax.hist(maxima[1:], bins=25, density=True, color='#B8CAD6', edgecolor='white', linewidth=0.25)
        ax.axvline(maxima[0], color=RED, lw=1.2)
        if result['critical'] is not None:
            ax.axvline(result['critical'], color=BLUE, ls='--', lw=1)
        p = result['summary']['global_p_study']
        ax.text(0.03, 0.96, f'Study-adjusted p = {p:.4g}', transform=ax.transAxes, ha='left', va='top', fontsize=8, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1.5))
    else:
        ax.text(0.5, 0.5, 'Insufficient test information', transform=ax.transAxes, ha='center', fontsize=8)
    ax.set(title=f'{name}: global scan test', xlabel='Maximum scan score', ylabel='Density')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    style(ax)
    save(fig, out / 'Figure02_global_null', 84, 74, dpi)
    fig, ax = plt.subplots(figsize=(168 * MM, 78 * MM))
    fig.subplots_adjust(left=0.115, right=0.86, bottom=0.23, top=0.86)
    w = result['windows']
    durations = list(settings.window_minutes)
    y = w.duration_minutes.map({v: i for i, v in enumerate(durations)})
    valid = w.scan_z.notna()
    if valid.any():
        artist = ax.scatter(w.loc[valid, 'start_hours'], y[valid], c=w.loc[valid, 'scan_z'], s=10, marker='s', cmap='viridis', rasterized=True)
        cax = fig.add_axes([0.885, 0.25, 0.018, 0.57])
        bar = fig.colorbar(artist, cax=cax)
        bar.set_label('Scan score', fontsize=10, labelpad=4)
        bar.ax.tick_params(labelsize=10)
        sig = w.significant_study
        ax.scatter(w.loc[sig, 'start_hours'], y[sig], s=14, marker='o', facecolors='none', edgecolors=RED, lw=0.45)
    ax.set_yticks(range(len(durations)), [f'{v:g}' for v in durations])
    ax.set_ylim(-0.6, len(durations) - 0.4)
    ax.set_ylabel('Window duration (min)')
    ax.set_title(f'{name}: event-frequency scan (outlined points pass study correction)', loc='left', pad=8)
    time_axis(ax, result)
    ax.set_xlabel('Window start since analysis start (h)')
    save(fig, out / 'Figure03_multiscale_scan', 168, 78, dpi)
    fig, axes = plt.subplots(1, 2, figsize=(168 * MM, 86 * MM))
    fig.subplots_adjust(left=0.08, right=0.85, bottom=0.24, top=0.84, wspace=0.4)
    nodes = result['nodes'].copy()
    nodes['event_records'] = 0 if result['best'] is None else result['counts'][result['best']]
    plan = nodes.groupby(['x_cm', 'y_cm'], as_index=False).event_records.sum()
    side = nodes.groupby(['x_cm', 'z_cm'], as_index=False).event_records.sum()
    vmax = max(float(plan.event_records.max()), float(side.event_records.max()), 1)
    for ax, frame, second, title in zip(axes, [plan, side], ['y_cm', 'z_cm'], ['Plan projection', 'Side projection']):
        artist = ax.scatter(frame.x_cm, frame[second], c=frame.event_records, cmap='YlOrRd', vmin=0, vmax=vmax, s=18, edgecolors='#555555', linewidths=0.3)
        ax.set(xlabel='x (cm)', ylabel=f'{second[0]} (cm)', title=f'{name}: {title}')
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 60, 120, 180])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        style(ax)
    cax = fig.add_axes([0.875, 0.25, 0.02, 0.53])
    bar = fig.colorbar(artist, cax=cax)
    bar.set_label('Projected event count', fontsize=10)
    bar.ax.tick_params(labelsize=10)
    b = result['best']
    label = 'No eligible candidate' if b is None else f"{result['windows'].loc[b, 'start_time']:%H:%M:%S} - {result['windows'].loc[b, 'end_time']:%H:%M:%S}; " + ('significant' if result['windows'].loc[b, 'significant_study'] else 'not significant')
    fig.text(0.46, 0.055, f'Strongest candidate: {label}', ha='center', fontsize=8)
    save(fig, out / 'Figure04_strongest_window_space', 168, 86, dpi)

def plot_study(results, settings, run_dir, dpi):
    rows = len(results)
    height = 55 * rows + 12
    fig, axes = plt.subplots(rows, 2, figsize=(168 * MM, height * MM), squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.965, bottom=0.125, top=0.965, wspace=0.31, hspace=0.43)
    for k, result in enumerate(results):
        draw_moran(axes[k, 0], result, f"{chr(97 + 2 * k)}  {result['spec'].name}: 10-min display")
        draw_maxscan(axes[k, 1], result, f"{chr(98 + 2 * k)}  {result['spec'].name}: multiscale scan")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.53, 0.012), ncol=4, frameon=False, columnspacing=0.9, handlelength=1.5, labelspacing=0.15)
    save(fig, run_dir / 'Figure05_three_experiment_comparison', 168, height, dpi)
    fig, axes = plt.subplots(1, 2, figsize=(168 * MM, 82 * MM))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.205, top=0.76, wspace=0.3)
    nodes = results[0]['nodes']
    palette = {'surface_2cm_top': '#0072B2', 'surface_2cm_slope': '#009E73', 'platform_20cm': '#E69F00', 'near_face_50cm': '#D55E00', 'near_face_80cm': '#8B4EAF'}
    for ax, coordinate, title in zip(axes, ['z_cm', 'y_cm'], ['a  Side projection', 'b  Plan projection']):
        for region, g in nodes.groupby('region'):
            q = g.drop_duplicates(['x_cm', coordinate])
            ax.scatter(q.x_cm, q[coordinate], color=palette[region], s=17)
        ax.set(xlabel='x (cm)', ylabel=f'{coordinate[0]} (cm)', title=title, xlim=(0, 180))
        ax.set_xticks([0, 60, 120, 180])
        style(ax)
    fig.text(0.53, 0.965, 'Confirmed near-surface domain; full slope width', ha='center', va='top', fontsize=10)
    fig.text(0.53, 0.895, 'Blue/green: 2 cm; yellow: 20 cm; orange/purple: near-face 50/80 cm rows', ha='center', va='top', fontsize=8)
    save(fig, run_dir / 'Figure06_analysis_domain', 168, 82, dpi)

def write_figure_quality(run_dir):
    pd.DataFrame(QUALITY).to_csv(Path(run_dir) / 'figure_quality.csv', index=False, encoding='utf-8-sig')

def render_saved_results(run_dir):
    """Explicit redraw command: use saved numbers, do not perform another statistical analysis."""
    import json
    from types import SimpleNamespace
    from scan_spatial_organization import Settings, PNG_DPI
    run_dir = Path(run_dir)
    configure_font()
    QUALITY.clear()
    results = []
    for name in ('S1', 'S2', 'S4'):
        output = run_dir / name
        if not output.is_dir():
            continue
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        settings = Settings(**manifest['settings'])
        windows = pd.read_csv(output / '08_all_scan_windows.csv', parse_dates=['start_time', 'end_time'])
        episodes = pd.read_csv(output / '09_detected_episodes.csv', parse_dates=['start_time', 'end_time', 'support_union_start', 'support_union_end'])
        with np.load(output / '11_reproducibility_arrays.npz') as arrays:
            counts = arrays['window_merged_event_counts']
            maxima = arrays['maximum_scan_z']
        indices = windows.index[windows.information_status.eq('eligible')]
        best = None if not len(indices) else int(windows.loc[indices, 'scan_z'].idxmax())
        result = dict(spec=SimpleNamespace(**manifest['experiment']), nodes=pd.read_csv(output / '02_selected_nodes.csv'), edges=pd.read_csv(output / '03_geometric_edges.csv'), windows=windows, episodes=episodes, maxima=maxima, critical=manifest['summary']['critical_max_z_study'], best=best, counts=counts, summary=manifest['summary'], output=output)
        plot_experiment(result, settings, PNG_DPI)
        results.append(result)
    if not results:
        raise ValueError('No saved experiment results found')
    plot_study(results, settings, run_dir, PNG_DPI)
    write_figure_quality(run_dir)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Redraw saved scan results without rerunning permutations')
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    render_saved_results(args.run_directory)
