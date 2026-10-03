"""Recompute the original comparison-figure statistics from archived events and permutations."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scan_spatial_organization import Settings, candidate_windows, window_event_counts, make_graph, merge_overlapping, permutation_moran, max_scan_inference, holm_adjusted

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/spatial_scan_original_figure'

def reproduce_one(experiment):
    folder = SOURCE / experiment
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    settings = Settings(**manifest['settings'])
    nodes = pd.read_csv(folder / '02_selected_nodes.csv')
    edges = pd.read_csv(folder / '03_geometric_edges.csv')
    rebuilt_edges, _ = make_graph(nodes, settings.radius_cm)
    np.testing.assert_array_equal(rebuilt_edges[['node_i','node_j']], edges[['node_i','node_j']])
    source_records = pd.read_csv(folder / '05_selected_source_records.csv')
    for column in ('start_time','end_time','peak_time'):
        source_records[column] = pd.to_datetime(source_records[column],format='mixed')
    events, membership = merge_overlapping(source_records)
    saved_events = pd.read_csv(folder / '06_merged_local_records.csv')
    np.testing.assert_array_equal(events.node_index, saved_events.node_index)
    np.testing.assert_array_equal(events.peak_time.to_numpy(), pd.to_datetime(saved_events.peak_time,format='mixed').to_numpy())
    saved_membership = pd.read_csv(folder / '07_merge_membership.csv')
    columns = ['physical_event_id','cluster_id','node_index']
    np.testing.assert_array_equal(membership.sort_values('physical_event_id')[columns],
                                 saved_membership.sort_values('physical_event_id')[columns])
    spec = manifest['experiment']
    windows = candidate_windows(spec['start'], spec['end'], settings)
    counts = window_event_counts(events, windows, len(nodes), spec['end'])
    with np.load(folder / '11_reproducibility_arrays.npz', allow_pickle=False) as arrays:
        permutations = arrays['permutations'].copy()
        archived_maxima = arrays['maximum_scan_z'].copy()
        np.testing.assert_array_equal(counts, arrays['window_merged_event_counts'])
    np.testing.assert_array_equal(permutations[0], np.arange(len(nodes)))
    groups = nodes.permutation_group.to_numpy()
    np.testing.assert_array_equal(groups[permutations], np.broadcast_to(groups, permutations.shape))
    eligible = (counts.sum(axis=1)>=settings.min_merged_event_records) & ((counts>0).sum(axis=1)>=settings.min_active_nodes) & (counts.var(axis=1)>0)
    scores = permutation_moran(counts[eligible], edges.node_i.to_numpy(), edges.node_j.to_numpy(), permutations, settings.batch_size)
    inference = max_scan_inference(scores, settings.alpha)
    np.testing.assert_allclose(inference['maxima'], archived_maxima, rtol=1e-10, atol=2e-11)
    saved = pd.read_csv(folder / '08_all_scan_windows.csv')
    for actual, column in ((inference['z_observed'],'scan_z'), (inference['p_within'],'p_scan_within_experiment'), (scores[0],'observed_moran_i')):
        np.testing.assert_allclose(actual, saved.loc[eligible,column], rtol=1e-10, atol=2e-11)
    np.testing.assert_allclose(inference['global_p'], manifest['summary']['global_p_scan'], rtol=0, atol=1e-14)
    return {'experiment':experiment, 'global_p_scan':float(inference['global_p']), 'verified_source_record_merging':True, 'verified_window_counts':True,
            'verified_graph':True, 'verified_permutations':True, 'verified_scan_maxima':True,
            'eligible_windows':int(eligible.sum()), 'permutations':settings.simulations}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/original_figure_verification')
    args = parser.parse_args()
    results = [reproduce_one(experiment) for experiment in ('S1','S2','S4')]
    _, adjusted = holm_adjusted([result['global_p_scan'] for result in results])
    saved = pd.read_csv(SOURCE / 'study_summary.csv').set_index('experiment')
    for result, p_value in zip(results, adjusted):
        result['global_p_study'] = float(p_value)
        np.testing.assert_allclose(p_value, saved.loc[result['experiment'],'global_p_study'], rtol=0, atol=1e-14)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    pd.DataFrame(results).to_csv(args.output_dir / 'study_summary.csv', index=False)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
