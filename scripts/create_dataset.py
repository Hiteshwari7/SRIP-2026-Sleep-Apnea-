import argparse
import os
import pandas as pd
import numpy as np
import pickle
from scipy.signal import butter, filtfilt, resample

def load_signal(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    start_idx = next(i for i, line in enumerate(lines) if 'Data:' in line) + 1
    df = pd.read_csv(filepath, skiprows=start_idx, sep=';', header=None, names=['timestamp', 'value'])
    df['timestamp'] = df['timestamp'].str.strip().str.replace(',', '.')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S.%f')
    df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
    return df

def load_events(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    start_idx = next(i for i, line in enumerate(lines) if line.strip() == '' and i > 2) + 1
    data = []
    for line in lines[start_idx:]:
        if line.strip():
            parts = line.strip().split(';')
            date_str, times = parts[0].strip().split(' ')
            start_t, end_t = times.split('-')
            start_dt = pd.to_datetime(f"{date_str} {start_t.replace(',', '.')}", format='%d.%m.%Y %H:%M:%S.%f')
            end_dt   = pd.to_datetime(f"{date_str} {end_t.replace(',', '.')}", format='%d.%m.%Y %H:%M:%S.%f')
            if end_dt < start_dt:
                end_dt += pd.Timedelta(days=1)
            data.append({'start_time': start_dt, 'end_time': end_dt, 'event_type': parts[2].strip()})
    return pd.DataFrame(data)

def butter_bandpass_filter(data, lowcut=0.17, highcut=0.4, fs=32.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def get_file(folder, keyword, exclude=None):
    for f in os.listdir(folder):
        name = f.lower()
        if keyword.lower() in name:
            if exclude and exclude.lower() in name:
                continue
            return os.path.join(folder, f)
    raise FileNotFoundError(f"Missing file with '{keyword}' in {folder}")

def get_label(w_start, w_end, events, window_sec=30):
    label = 'Normal'
    for _, event in events.iterrows():
        if 'Apnea' in event['event_type'] or 'Hypopnea' in event['event_type']:
            overlap = (min(w_end, event['end_time']) - max(w_start, event['start_time'])).total_seconds()
            if overlap > (window_sec / 2):
                label = event['event_type']
                break
    return label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir',  type=str, required=True)
    parser.add_argument('-out_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    fs_resp      = 32
    fs_spo2      = 4
    window_sec   = 30
    overlap_sec  = 15
    window_samples_resp = window_sec * fs_resp        # 960
    step_samples        = overlap_sec * fs_resp       # 480
    window_samples_spo2 = window_sec * fs_spo2        # 120

    all_windows, all_labels, all_groups = [], [], []
    participant_folders = sorted([f.path for f in os.scandir(args.in_dir) if f.is_dir()])

    for group_id, folder in enumerate(participant_folders):
        p_id = os.path.basename(folder)
        print(f"Processing {p_id}...")

        flow_file   = get_file(folder, 'flow', exclude='event')
        thorac_file = get_file(folder, 'thorac')
        spo2_file   = get_file(folder, 'spo2')
        events_file = get_file(folder, 'event')

        airflow  = load_signal(flow_file)
        thoracic = load_signal(thorac_file)
        spo2     = load_signal(spo2_file)
        spo2['value'] = spo2['value'].clip(70, 100)   # Remove sensor artifacts
        events   = load_events(events_file)

        # Filter respiration signals
        airflow['filtered']  = butter_bandpass_filter(airflow['value'].values,  fs=fs_resp)
        thoracic['filtered'] = butter_bandpass_filter(thoracic['value'].values, fs=fs_resp)

        # Filter SpO2 with lower freq range (4 Hz signal)
        spo2['filtered'] = butter_bandpass_filter(
            spo2['value'].values, lowcut=0.017, highcut=0.4, fs=fs_spo2, order=2
        )

        # Determine total windows from shortest signal
        n_windows_resp = (len(airflow) - window_samples_resp) // step_samples + 1
        n_windows_spo2 = (len(spo2)    - window_samples_spo2) // (overlap_sec * fs_spo2) + 1
        n_windows = min(n_windows_resp, n_windows_spo2)

        for i in range(n_windows):
            # Respiration indices
            r_start = i * step_samples
            r_end   = r_start + window_samples_resp

            # SpO2 indices (proportional, since 4 Hz)
            s_start = i * (overlap_sec * fs_spo2)
            s_end   = s_start + window_samples_spo2

            if r_end > len(airflow) or s_end > len(spo2):
                break

            w_start = airflow['timestamp'].iloc[r_start]
            w_end   = airflow['timestamp'].iloc[r_end - 1]

            # Extract features from each signal
            flow_window    = airflow['filtered'].iloc[r_start:r_end].values       # shape (960,)
            thorac_window  = thoracic['filtered'].iloc[r_start:r_end].values      # shape (960,)
            spo2_window_raw = spo2['filtered'].iloc[s_start:s_end].values         # shape (120,)

            # Upsample SpO2 to 32 Hz so all signals have same length (960,)
            spo2_upsampled = resample(spo2_window_raw, window_samples_resp)

            # Stack into (960, 3) — 3 channels
            combined = np.stack([flow_window, thorac_window, spo2_upsampled], axis=1)

            label = get_label(w_start, w_end, events, window_sec)

            all_windows.append(combined)
            all_labels.append(label)
            all_groups.append(group_id)

    X      = np.array(all_windows)   # shape: (N, 960, 3)
    y      = np.array(all_labels)
    groups = np.array(all_groups)

    # Save as pickle
    dataset = {'X': X, 'y': y, 'groups': groups}
    with open(os.path.join(args.out_dir, 'breathing_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

    # Also save a CSV summary (label counts per participant)
    summary = pd.DataFrame({'label': y, 'participant': groups})
    summary.to_csv(os.path.join(args.out_dir, 'label_summary.csv'), index=False)

    print(f"\nDataset saved! Shape: X={X.shape}, y={y.shape}")
    print(f"Label distribution:\n{pd.Series(y).value_counts()}")

if __name__ == "__main__":
    main()
