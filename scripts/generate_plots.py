#!/usr/bin/env python3
"""
Generate traffic volume plots for each intersection.

This script processes all intersection folders, reads the detector dictionaries
and traffic data files, and generates plots showing volume patterns over time.
Plots are saved as PNG files in the specified output directory.

Usage:
    python scripts/generate_plots.py --data-dir "data/Oct Volume" --output-dir "data/Oct Volume/organized_subfolders2"
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def excel_frac_to_minutes_of_day(excel_val):
    """Convert Excel fractional date or datetime to minutes of day."""
    if isinstance(excel_val, (int, float)):
        frac = excel_val - np.floor(excel_val)
        return int(np.round(frac * 24 * 60))
    elif hasattr(excel_val, 'hour'):  # datetime object
        return excel_val.hour * 60 + excel_val.minute
    else:
        return 0


def minute_to_hhmm(minute_of_day):
    """Convert minutes of day to HH:MM format."""
    h = minute_of_day // 60
    m = minute_of_day % 60
    return f"{h:02d}:{m:02d}"


def detect_granularity(data):
    """Detect data granularity and aggregation needs."""
    if len(data) < 50:
        return {'resolution_min': None, 'is_already_15min': False}

    sample = data.head(6000)
    times = sample['dateTime'].dropna()

    if len(times) < 10:
        return {'resolution_min': None, 'is_already_15min': False}

    minutes = times.apply(excel_frac_to_minutes_of_day)

    # Calculate time differences
    deltas = []
    for det in sample['detector'].unique():
        det_times = minutes[sample['detector'] == det].sort_values()
        if len(det_times) > 1:
            det_deltas = det_times.diff().dropna()
            deltas.extend(det_deltas[(det_deltas > 0) & (det_deltas <= 1000)].tolist())

    med_delta = np.median(deltas) if deltas else None

    # Check alignment with 15-min intervals
    aligned_15 = sum(1 for m in minutes if m % 15 == 0)
    aligned_15_ratio = aligned_15 / len(minutes) if len(minutes) > 0 else 0

    is_already_15min = (med_delta and 13 <= med_delta <= 17) or aligned_15_ratio >= 0.85

    return {
        'resolution_min': med_delta,
        'is_already_15min': is_already_15min,
        'aligned_15_ratio': aligned_15_ratio
    }


def extract_date(dt):
    """Extract date from dateTime, handling Excel dates and datetime objects."""
    if isinstance(dt, (int, float)):
        return pd.to_datetime(int(dt), origin='1899-12-30', unit='D').date()
    elif hasattr(dt, 'date'):
        return dt.date()
    else:
        return pd.to_datetime(dt, errors='coerce').date()


def process_traffic_data(excel_files, detector_dict=None):
    """Process traffic data files similar to the HTML app."""
    all_data = []

    for file_path in excel_files:
        try:
            # Read Excel file
            df = pd.read_excel(file_path)

            # Detect format (old vs new)
            columns = df.columns.tolist()
            has_old_format = any('System Detector History Report' in col for col in columns)

            if has_old_format:
                # Old format - handle pandas dot conversion
                clean_df = pd.DataFrame({
                    'dateTime': df['Print Date'],
                    'zone': df['System Detector History Report'],
                    'assetId': df.get('System Detector History Report.1', df.get('System Detector History Report_1', 'Unknown')),
                    'location': df.get('System Detector History Report.2', df.get('System Detector History Report_2', 'Unknown')),
                    'detector': df.get('System Detector History Report.3', df.get('System Detector History Report_3', 'Unknown')),
                    'volume': pd.to_numeric(df.get('System Detector History Report.4', df.get('System Detector History Report_4', 0)), errors='coerce'),
                    'occupancy': pd.to_numeric(df.get('System Detector History Report.5', df.get('System Detector History Report_5', 0)), errors='coerce'),
                    'speed': pd.to_numeric(df.get('System Detector History Report.6', df.get('System Detector History Report_6', 0)), errors='coerce'),
                    'status': df['Print Time']
                })
            else:
                # New format - detect columns
                date_col = next((col for col in columns if 'date' in col.lower() or 'time' in col.lower()), columns[0])
                detector_col = next((col for col in columns if 'channel' in col.lower() or 'detector' in col.lower() or 'lane' in col.lower()), None)
                volume_col = next((col for col in columns if 'volume' in col.lower() or 'count' in col.lower()), None)

                if not detector_col or not volume_col:
                    print(f"Warning: Could not identify required columns in {file_path}")
                    continue

                clean_df = pd.DataFrame({
                    'dateTime': df[date_col],
                    'zone': 'Zone 1',
                    'assetId': os.path.basename(file_path).split('_')[0],
                    'location': 'Route 50',
                    'detector': df[detector_col],
                    'volume': pd.to_numeric(df[volume_col], errors='coerce'),
                    'occupancy': pd.to_numeric(df.get('occupancy', 0), errors='coerce'),
                    'speed': pd.to_numeric(df.get('speed', 0), errors='coerce'),
                    'status': df.get('status', 'Valid')
                })

            # Filter valid data
            clean_df = clean_df[
                (clean_df['status'] == 'Valid') &
                clean_df['volume'].notna() &
                clean_df['detector'].notna()
            ].copy()

            all_data.append(clean_df)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    # Extract date from dateTime
    combined_df['date'] = combined_df['dateTime'].apply(extract_date)

    # Detect granularity
    gran = detect_granularity(combined_df)

    # Aggregate to 15-minute intervals
    grouped_data = []

    if gran['is_already_15min']:
        # Data is already in 15-min intervals
        for (detector, date_key, time_key), group in combined_df.groupby(['detector', 'date', 'dateTime']):
            grouped_data.append({
                'detector': detector,
                'time': minute_to_hhmm(excel_frac_to_minutes_of_day(time_key)),
                'volume': group['volume'].sum(),
                'date': str(date_key)
            })
    else:
        # Aggregate to 15-minute intervals
        combined_df['minutes'] = combined_df['dateTime'].apply(excel_frac_to_minutes_of_day)
        combined_df['interval'] = (combined_df['minutes'] // 15) * 15

        for (detector, date_key, interval), group in combined_df.groupby(['detector', 'date', 'interval']):
            grouped_data.append({
                'detector': detector,
                'time': minute_to_hhmm(int(interval)),
                'volume': group['volume'].sum(),
                'date': str(date_key)
            })

    return pd.DataFrame(grouped_data)


def load_detector_dict(dict_path):
    """Load detector dictionary from CSV."""
    try:
        df = pd.read_csv(dict_path)
        # Map detector names to approaches
        detector_to_approach = {}
        approach_to_phase = {}

        for _, row in df.iterrows():
            det_str = str(row.get('Det', ''))
            if det_str and det_str != 'nan':
                # Handle multiple detectors like "1,21,29"
                detectors = [d.strip() for d in det_str.split(',') if d.strip()]
                approach = str(row.get('movement_name', '')).strip()

                if approach:
                    for det in detectors:
                        detector_to_approach[det] = approach

                    # Also map by KITS_det_name if available
                    kits_det = str(row.get('KITS_det_name', '')).strip()
                    if kits_det:
                        detector_to_approach[kits_det] = approach

                    # Phase mapping
                    phase = row.get('Phase')
                    if pd.notna(phase):
                        approach_to_phase[approach] = int(phase)

        return detector_to_approach, approach_to_phase

    except Exception as e:
        print(f"Error loading detector dictionary {dict_path}: {e}")
        return {}, {}


def create_detector_plot(data_df, intersection_id, output_path, date_filter=None):
    """Create plot showing volume by detector."""
    if data_df.empty:
        print(f"No data for {intersection_id}")
        return

    # Filter by date if specified
    if date_filter:
        data_df = data_df[data_df['date'] == date_filter]
        if data_df.empty:
            print(f"No data for {intersection_id} on {date_filter}")
            return

    # Get all unique times and detectors
    times = sorted(data_df['time'].unique())
    detectors = sorted(data_df['detector'].unique())

    plt.figure(figsize=(12, 8))

    colors = ['#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea', '#0891b2', '#db2777', '#65a30d',
              '#f59e0b', '#ef4444', '#10b981', '#3b82f6', '#8b5cf6', '#f97316', '#06b6d4']

    for i, detector in enumerate(detectors):
        detector_data = data_df[data_df['detector'] == detector]
        volumes = []
        for t in times:
            time_data = detector_data[detector_data['time'] == t]
            volume = time_data['volume'].sum() if not time_data.empty else 0
            volumes.append(volume)

        plt.plot(times, volumes, marker='o', label=f'Detector {detector}',
                color=colors[i % len(colors)], linewidth=2, markersize=4)

    date_str = f" - {date_filter}" if date_filter else ""
    plt.title(f"Traffic Volume - Intersection {intersection_id}{date_str}", fontsize=16, fontweight='bold')
    plt.xlabel("Time (15-minute intervals)", fontsize=12)
    plt.ylabel("Volume (vehicles)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved detector plot to {output_path}")


def create_approach_plot(data_df, detector_dict, approach_phase_dict, intersection_id, output_path, date_filter=None):
    """Create plot showing volume by approach."""
    if data_df.empty or not detector_dict:
        print(f"No approach data for {intersection_id}")
        return

    # Filter by date if specified
    if date_filter:
        data_df = data_df[data_df['date'] == date_filter]
        if data_df.empty:
            print(f"No approach data for {intersection_id} on {date_filter}")
            return

    # Group data by approach
    approach_data = []
    times = sorted(data_df['time'].unique())

    for time in times:
        time_data = data_df[data_df['time'] == time]

        # Group by approach
        approach_volumes = {}
        for _, row in time_data.iterrows():
            detector = str(row['detector'])
            approach = detector_dict.get(detector)

            if approach:
                if approach not in approach_volumes:
                    approach_volumes[approach] = 0
                approach_volumes[approach] += row['volume']

        for approach, volume in approach_volumes.items():
            approach_data.append({
                'approach': approach,
                'time': time,
                'volume': volume,
                'phase': approach_phase_dict.get(approach, 'Unknown')
            })

    if not approach_data:
        print(f"No approach data to plot for {intersection_id}")
        return

    approach_df = pd.DataFrame(approach_data)

    # Create figure
    plt.figure(figsize=(12, 8))

    approaches = sorted(approach_df['approach'].unique())
    colors = ['#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea', '#0891b2', '#db2777', '#65a30d',
              '#f59e0b', '#ef4444', '#10b981', '#3b82f6', '#8b5cf6', '#f97316', '#06b6d4']

    for i, approach in enumerate(approaches):
        approach_data_filtered = approach_df[approach_df['approach'] == approach]
        phase = approach_data_filtered['phase'].iloc[0] if not approach_data_filtered.empty else 'Unknown'

        plt.plot(approach_data_filtered['time'], approach_data_filtered['volume'],
                marker='o', label=f'{approach} (Phase {phase})',
                color=colors[i % len(colors)], linewidth=2, markersize=4)

    date_str = f" - {date_filter}" if date_filter else ""
    plt.title(f"Traffic Volume by Approach - Intersection {intersection_id}{date_str}", fontsize=16, fontweight='bold')
    plt.xlabel("Time (15-minute intervals)", fontsize=12)
    plt.ylabel("Volume (vehicles)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved approach plot to {output_path}")


def process_intersection(intersection_dir, output_dir):
    """Process a single intersection folder."""
    intersection_path = Path(intersection_dir)
    intersection_name = intersection_path.name

    # Extract intersection ID (handle different naming patterns)
    if '_' in intersection_name:
        parts = intersection_name.split('_')
        intersection_id = parts[1] if len(parts) > 1 else intersection_name
    elif '-' in intersection_name:
        intersection_id = intersection_name.split('-')[0]
    else:
        intersection_id = intersection_name

    # Clean intersection ID (remove suffixes like _Sys15, Sys23)
    intersection_id = intersection_id.split('_')[0]

    print(f"\nProcessing intersection {intersection_id}...")

    # Create output subdirectory for this intersection
    intersection_output_dir = output_dir / intersection_name
    intersection_output_dir.mkdir(parents=True, exist_ok=True)

    # Remove old plot files
    for old_file in intersection_output_dir.glob("*_plot*.png"):
        old_file.unlink()

    # Find detector dictionary
    dict_path = None
    dict_candidates = [
        intersection_path / f"{intersection_id}_dict.csv",
        intersection_path / f"{intersection_id}_dict.xlsx",
        Path(f"data/dicts/{intersection_id}.csv"),
        Path(f"data/dicts/{intersection_id}.xlsx")
    ]

    for candidate in dict_candidates:
        if candidate.exists():
            dict_path = candidate
            break

    detector_dict = {}
    approach_phase_dict = {}

    if dict_path:
        print(f"Loading detector dictionary: {dict_path}")
        detector_dict, approach_phase_dict = load_detector_dict(dict_path)
        print(f"Found {len(detector_dict)} detector mappings")
    else:
        print(f"No detector dictionary found for {intersection_id}")

    # Find all Excel files recursively
    excel_files = []
    for pattern in ['**/*.xlsx', '**/*.xls']:
        excel_files.extend(intersection_path.glob(pattern))

    if not excel_files:
        print(f"No Excel files found in {intersection_path}")
        return

    print(f"Found {len(excel_files)} Excel files")

    # Process traffic data
    data_df = process_traffic_data(excel_files, detector_dict)

    if data_df.empty:
        print(f"No valid data processed for {intersection_id}")
        return

    print(f"Processed {len(data_df)} data points")

    # Get unique dates
    unique_dates = sorted(data_df['date'].unique())

    # Create plots for each date
    for date in unique_dates:
        date_str = str(date).replace('-', '_')  # For filename

        # Create detector plot
        detector_plot_path = intersection_output_dir / f"{intersection_id}_detector_plot_{date_str}.png"
        create_detector_plot(data_df, intersection_id, detector_plot_path, date_filter=date)

        # Create approach plot if dictionary available
        if detector_dict:
            approach_plot_path = intersection_output_dir / f"{intersection_id}_approach_plot_{date_str}.png"
            create_approach_plot(data_df, detector_dict, approach_phase_dict, intersection_id, approach_plot_path, date_filter=date)


def main():
    parser = argparse.ArgumentParser(description='Generate traffic volume plots for intersections')
    parser.add_argument('--data-dir', required=True, help='Directory containing intersection folders')
    parser.add_argument('--output-dir', help='Directory to save plots (default: organized_subfolders2 under data-dir)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing plot files')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist")
        return

    # Default output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "organized_subfolders2"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all intersection folders
    intersection_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    print(f"Found {len(intersection_dirs)} intersection folders")
    print(f"Saving plots to: {output_dir}")

    for intersection_dir in intersection_dirs:
        try:
            process_intersection(intersection_dir, output_dir)
        except Exception as e:
            print(f"Error processing {intersection_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nPlot generation complete!")


if __name__ == '__main__':
    main()