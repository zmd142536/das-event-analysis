"""Run the original-figure scan from the full S1 table and archived S2/S4 source records."""
from dataclasses import replace
import scan_spatial_organization as scan

def main(argv=None):
    scan.CONFIG = dict(scan.CONFIG)
    scan.CONFIG['S1'] = replace(
        scan.CONFIG['S1'],
        coordinate_file=scan.COORDINATE_DIRECTORY / 'supplied_layout/S1_coordinates.xlsx',
        event_file=scan.EVENT_DIRECTORY / 'S1_event_activity_original_figure_effective_plus_uncertain.csv',
    )
    for experiment in ('S2', 'S4'):
        scan.CONFIG[experiment] = replace(
            scan.CONFIG[experiment],
            coordinate_file=scan.COORDINATE_DIRECTORY / 'supplied_layout/S2_S4_coordinates.xlsx',
            event_file=scan.REPOSITORY_ROOT / 'data/spatial_scan_original_figure' / experiment / '05_selected_source_records.csv',
        )
    scan.OUTPUT_DIRECTORY = scan.REPOSITORY_ROOT / 'outputs/spatial_scan_original_figure'
    return scan.main(argv)

if __name__ == '__main__':
    main()
