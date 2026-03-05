import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def load_signal(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    start_idx = next(i for i, line in enumerate(lines) if 'Data:' in line) + 1
    df = pd.read_csv(filepath, skiprows=start_idx, sep=';', header=None, names=['timestamp', 'value'])
    df['timestamp'] = df['timestamp'].str.strip().str.replace(',', '.')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S.%f')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    # Clip SpO2 anomalies if this is an SpO2 file
    return df

def load_spo2(filepath):
    df = load_signal(filepath)
    df['value'] = df['value'].clip(70, 100)  # Remove sensor artifacts
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
            end_dt = pd.to_datetime(f"{date_str} {end_t.replace(',', '.')}", format='%d.%m.%Y %H:%M:%S.%f')
            if end_dt < start_dt:
                end_dt += pd.Timedelta(days=1)
            data.append({'start_time': start_dt, 'end_time': end_dt, 'event_type': parts[2].strip()})
    return pd.DataFrame(data)

def get_file(folder, keyword, exclude=None):
    for f in os.listdir(folder):
        name = f.lower()
        if keyword.lower() in name:
            if exclude and exclude.lower() in name:
                continue
            return os.path.join(folder, f)
    raise FileNotFoundError(f"Missing file with '{keyword}' in {folder}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, required=True)
    args = parser.parse_args()

    participant_folder = args.name
    p_id = os.path.basename(os.path.normpath(participant_folder))
    out_dir = os.path.join(os.path.dirname(os.path.dirname(participant_folder)), 'Visualizations')
    os.makedirs(out_dir, exist_ok=True)

    flow_file   = get_file(participant_folder, 'flow', exclude='event')
    thorac_file = get_file(participant_folder, 'thorac')
    spo2_file   = get_file(participant_folder, 'spo2')
    events_file = get_file(participant_folder, 'event')

    airflow  = load_signal(flow_file)
    thoracic = load_signal(thorac_file)
    spo2     = load_spo2(spo2_file)
    events   = load_events(events_file)

    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
    fig.suptitle(f'Sleep Study - Participant {p_id}', fontsize=14, fontweight='bold')

    # Plot signals
    axes[0].plot(airflow['timestamp'],  airflow['value'],  color='steelblue',  alpha=0.8, linewidth=0.5)
    axes[0].set_title('Nasal Airflow (32 Hz)')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(thoracic['timestamp'], thoracic['value'], color='seagreen',   alpha=0.8, linewidth=0.5)
    axes[1].set_title('Thoracic Movement (32 Hz)')
    axes[1].set_ylabel('Amplitude')

    axes[2].plot(spo2['timestamp'],     spo2['value'],     color='crimson',    alpha=0.8, linewidth=0.7)
    axes[2].set_title('SpO₂ - Oxygen Saturation (4 Hz)')
    axes[2].set_ylabel('SpO₂ (%)')
    axes[2].set_ylim(70, 105)

    # Overlay breathing events
    event_colors = {
        'Obstructive Apnea': 'red',
        'Hypopnea': 'orange',
        'Central Apnea': 'purple',
        'Mixed Apnea': 'brown'
    }
    legend_added = set()
    for _, row in events.iterrows():
        etype = row['event_type']
        if 'Apnea' in etype or 'Hypopnea' in etype:
            color = event_colors.get(etype, 'orange')
            for ax in axes:
                label = etype if etype not in legend_added else None
                ax.axvspan(row['start_time'], row['end_time'], color=color, alpha=0.35, label=label)
            legend_added.add(etype)

    axes[0].legend(loc='upper right', fontsize=8)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[2].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xlabel('Time (HH:MM)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{p_id}_visualization.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Visualization saved: {out_path}")

if __name__ == "__main__":
    main()
