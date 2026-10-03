"""Scientific checks for the independently named event-frequency analysis."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import unittest
from itertools import permutations
import numpy as np
import pandas as pd
from scan_spatial_organization import Settings, candidate_windows, window_event_counts, merge_overlapping, draw_permutations, moran_direct, permutation_moran, max_scan_inference, holm_adjusted, region_name

class EventFrequencyChecks(unittest.TestCase):

    def test_confirmed_domain(self):
        self.assertEqual(region_name(20, 30, 80), 'platform_20cm')
        self.assertEqual(region_name(50, 120, 50), 'near_face_50cm')
        self.assertEqual(region_name(80, 150, 20), 'near_face_80cm')
        self.assertEqual(region_name(80, 120, 20), 'excluded_interior')

    def test_merge_overlapping_only_within_node_and_count_once(self):
        base = pd.Timestamp('2025-01-01')
        rows = []
        source = [(0, 0, 2, 1, 1), (0, 1, 3, 2, 50), (0, 2.5, 4, 3, 2), (0, 6, 7, 6.5, 1), (1, 0, 2, 1, 1)]
        for k, (node, a, b, peak, activity) in enumerate(source):
            rows.append(dict(physical_event_id=str(k), node_index=node, start_time=base + pd.Timedelta(seconds=a), end_time=base + pd.Timedelta(seconds=b), peak_time=base + pd.Timedelta(seconds=peak), activity=activity, gauge_crosses_domain=False, gauge_has_unlocated_channel=False))
        merged, membership = merge_overlapping(pd.DataFrame(rows))
        self.assertEqual(len(merged), 3)
        self.assertEqual(len(membership), 5)
        group = merged[merged.source_records.eq(3)].iloc[0]
        self.assertEqual(group.peak_time, base + pd.Timedelta(seconds=2))
        self.assertEqual(group.end_time, base + pd.Timedelta(seconds=4))
        self.assertEqual(group.representative_activity, 50)

    def test_window_boundaries_and_fixed_end_inclusion(self):
        settings = Settings(window_minutes=(1,), scan_step_seconds=60)
        windows = candidate_windows('2025-01-01', '2025-01-01 00:02:07', settings)
        events = pd.DataFrame({'node_index': [0] * 4, 'peak_time': pd.to_datetime(['2025-01-01 00:00:00', '2025-01-01 00:01:00', '2025-01-01 00:02:00', '2025-01-01 00:02:07'])})
        counts = window_event_counts(events, windows, 1, '2025-01-01 00:02:07')
        self.assertEqual(counts[:, 0].tolist(), [1, 1, 2])
        self.assertTrue(windows.end_time.le(pd.Timestamp('2025-01-01 00:02:07')).all())

    def test_whole_history_permutation_preserves_totals_and_groups(self):
        groups = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5])]
        p = draw_permutations(groups, 6, 30, np.random.default_rng(42))
        history = np.arange(24).reshape(4, 6)
        for row in p:
            self.assertEqual(row[5], 5)
            self.assertEqual(set(row[:3]), {0, 1, 2})
            np.testing.assert_array_equal(history[:, row].sum(axis=1), history.sum(axis=1))

    def test_fast_moran_matches_direct_for_event_counts(self):
        counts = np.array([[1, 2, 5, 0], [8, 0, 0, 3], [0, 3, 0, 1]], float)
        left = np.array([0, 1, 2])
        right = np.array([1, 2, 3])
        p = np.array(list(permutations(range(4))))
        actual = permutation_moran(counts, left, right, p, batch_size=7)
        expected = np.array([moran_direct(counts[:, v], left, right) for v in p])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_global_scan_and_study_correction(self):
        scores = np.array([[10, 0, 1], [0, 4, 1], [1, 3, 1], [2, 2, 1], [3, 1, 1]], float)
        result = max_scan_inference(scores, 0.05)
        self.assertAlmostEqual(result['global_p'], 0.2)
        self.assertAlmostEqual(result['global_p_study'], 0.2)
        self.assertEqual(result['p_within'][2], 1)
        self.assertFalse(result['informative'][2])

    def test_holm_three_experiment_adjustment(self):
        p = np.array([0.028994201159768047, 0.002599480103979204, 0.00639872025594881])
        order, adjusted = holm_adjusted(p)
        np.testing.assert_array_equal(order, [1, 2, 0])
        np.testing.assert_allclose(adjusted, [p[0], 3 * p[1], 2 * p[2]])
        self.assertTrue(np.all(adjusted < 0.05))

    def test_constant_surrogates_cannot_be_significant(self):
        result = max_scan_inference(np.ones((101, 4)), 0.05)
        self.assertEqual(result['global_p_study'], 1)
        self.assertFalse(result['informative'].any())

    def test_symmetric_standardization_relabels_rows(self):
        scores = np.random.default_rng(3).normal(size=(41, 9))
        a = max_scan_inference(scores, 0.05)
        order = np.r_[7, np.arange(7), np.arange(8, 41)]
        b = max_scan_inference(scores[order], 0.05)
        np.testing.assert_allclose(b['maxima'], a['maxima'][order], atol=1e-12)

    def test_familywise_null_size_by_exact_orbit(self):
        counts = np.array([[7, 3, 0, 1], [1, 6, 2, 0], [4, 0, 3, 2]], float)
        p = np.array(list(permutations(range(4))))
        scores = permutation_moran(counts, np.array([0, 1, 2]), np.array([1, 2, 3]), p)
        rejected = 0
        for i in range(len(scores)):
            order = np.r_[i, np.delete(np.arange(len(scores)), i)]
            rejected += max_scan_inference(scores[order], 0.05)['global_p'] <= 0.05
        self.assertLessEqual(rejected / len(scores), 0.05)

    def test_detect_transient_frequency_patch(self):
        n = 30
        counts = np.zeros((3, n))
        counts[0, :] = np.arange(n) % 3
        counts[1, 10:16] = 12
        counts[2, :] = np.arange(n) % 3
        left = np.arange(n - 1)
        right = left + 1
        p = np.vstack([np.arange(n), draw_permutations([np.arange(n)], n, 499, np.random.default_rng(8))])
        result = max_scan_inference(permutation_moran(counts, left, right, p), 0.05)
        self.assertLessEqual(result['global_p'], 0.05)
        self.assertEqual(int(np.argmax(result['z_observed'])), 1)
if __name__ == '__main__':
    unittest.main(verbosity=2)
