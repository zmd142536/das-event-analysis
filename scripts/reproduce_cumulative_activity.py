"""Reconstruct and verify the deposited one-minute cumulative activity tables."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VALUE = 'event_squared_strain_rate_integral_gauge_median'
PERIODS = {
    'S1': ('2025-07-07 14:06:00', '2025-07-07 22:38:39'),
    'S2': ('2025-07-19 09:00:00', '2025-07-19 20:40:00'),
    'S4': ('2025-08-11 09:00:00', '2025-08-11 20:30:00'),
}

def reconstruct(experiment):
    start, cutoff = map(pd.Timestamp, PERIODS[experiment])
    events = pd.read_csv(ROOT / 'data/events' / f'{experiment}_event_activity_04B_effective.csv',
                         parse_dates=['peak_time', 'minute_time'])
    events = events.loc[events.peak_time.lt(cutoff)]
    minutes = pd.date_range(start, cutoff.floor('min'), freq='1min', name='minute_time')
    frame = events.groupby('minute_time').agg(
        event_count=('physical_event_id', 'size'), activity_increment=(VALUE, 'sum')
    ).reindex(minutes, fill_value=0).reset_index()
    excluded = frame.minute_time.lt(pd.Timestamp('2025-08-11 09:30:00')) if experiment == 'S4' else pd.Series(False, index=frame.index)
    frame.loc[excluded, ['event_count', 'activity_increment']] = 0
    frame['excluded_from_analysis'] = excluded
    frame['cumulative_activity'] = frame.activity_increment.cumsum()
    frame['count_trailing_10min_mean'] = frame.event_count.rolling(10, min_periods=1).mean()
    return frame

def verify(experiment, actual):
    saved = pd.read_csv(ROOT / 'data/cumulative' / f'{experiment}_activity_04B_effective_1min.csv',
                        parse_dates=['minute_time'])
    np.testing.assert_array_equal(actual.minute_time.to_numpy(), saved.minute_time.to_numpy())
    for column in ('event_count', 'excluded_from_analysis'):
        np.testing.assert_array_equal(actual[column], saved[column])
    for column in ('activity_increment', 'cumulative_activity', 'count_trailing_10min_mean'):
        np.testing.assert_allclose(actual[column], saved[column], rtol=1e-10, atol=1e-12)
    return {'experiment':experiment, 'rows':len(actual), 'verified':True,
            'retained_events':int(actual.event_count.sum()), 'final_cumulative_activity':float(actual.cumulative_activity.iloc[-1])}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/cumulative')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for experiment in PERIODS:
        frame = reconstruct(experiment)
        summaries.append(verify(experiment, frame))
        frame.to_csv(args.output_dir / f'{experiment}_activity_04B_effective_1min.csv', index=False)
    (args.output_dir / 'verification.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print(json.dumps(summaries, indent=2))

if __name__ == '__main__':
    main()
