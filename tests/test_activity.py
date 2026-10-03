"""Check event integration and overlapping HDF5 file boundaries."""
from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from calculate_event_activity import H5Record, read_event_block, compute_event_metrics
from preprocess_das import process_block, PreprocessError

class ActivityChecks(unittest.TestCase):
    def record(self, folder, name, start, values):
        path = folder / name
        with h5py.File(path, 'w') as handle:
            handle['default'] = values
        return H5Record(path, start, start + pd.Timedelta(seconds=len(values)/10),
                        10.0, len(values), values.shape[1], 3000, (20.0,2000.0))

    def test_overlap_does_not_duplicate_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            start = pd.Timestamp('2025-01-01')
            first = self.record(folder, 'first.h5', start, np.arange(10).reshape(-1,1))
            second = self.record(folder, 'second.h5', start+pd.Timedelta(seconds=.8), np.arange(8,18).reshape(-1,1))
            block, count = read_event_block([first,second], start+pd.Timedelta(seconds=.6),
                                           start+pd.Timedelta(seconds=1.3), [0], '/default', 'second.h5')
            self.assertEqual(count,7)
            np.testing.assert_array_equal(block[:,0],np.arange(6,13))

    def test_gauge_median_integral_has_expected_units(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); start=pd.Timestamp('2025-01-01')
            record=self.record(folder,'constant.h5',start,np.tile([2.,4.,6.],(10,1)))
            event=pd.DataFrame([dict(physical_event_id='event_1',start_time=start,end_time=start+pd.Timedelta(seconds=1),
                peak_time=start+pd.Timedelta(seconds=.5),representative_data_channel=1,file_name='constant.h5',
                burial_depth_cm_layout=2,geometry_type_layout='ring',x_cm_layout=1,y_cm_layout=2,z_cm=3,
                stage_id=0,layout_channel=3001,representative_segment_id='ring_1')])
            metrics,channels=compute_event_metrics(event,pd.DataFrame({'data_channel':[0,1,2]}),[record],'/default','S1')
            self.assertEqual(metrics.event_rms_gauge_median.iloc[0],4)
            self.assertEqual(metrics.event_squared_strain_rate_integral_gauge_median.iloc[0],16)
            np.testing.assert_allclose(channels.squared_strain_rate_integral,
                                       channels.strain_rate_rms**2*channels.duration_s)

    def test_preprocessing_removes_linear_trend(self):
        x=np.arange(1000,dtype=float)
        output=process_block(np.column_stack([2*x+10,-3*x-20]),None)
        np.testing.assert_allclose(output,0,atol=2e-12)

    def test_nonfinite_input_is_rejected(self):
        with self.assertRaises(PreprocessError):
            process_block(np.array([[0.],[np.nan],[1.]]),None)

if __name__=='__main__':
    unittest.main()
