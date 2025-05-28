import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

try:
    df = pd.read_csv("psp_data.csv")
except FileNotFoundError:
    print("No psp_data.csv found. Run the demo script first.")
    exit()

if 't' not in df or df.empty:
    print("Invalid or empty CSV.")
    exit()

df['t'] = pd.to_datetime(df['t'], unit='s', errors='coerce')
df = df.dropna(subset=['t'])

duration_min = (df['t'].iloc[-1] - df['t'].iloc[0]).total_seconds() / 60
blink_rate = df['blink'].sum() / duration_min
saccades = df[df['sac_start'].notna()]
saccade_rate = len(saccades) / duration_min
avg_sac_vel = saccades['sac_vel'].astype(float).mean()
avg_sac_amp = saccades['sac_amp'].astype(float).mean()
jitter_count = df['jitter_t'].notna().sum()
v_range = df['v_ratio'].max() - df['v_ratio'].min()

def classify(val, thresholds, labels):
    for t, l in zip(thresholds, labels):
        if val <= t:
            return l
    return labels[-1]

blink_class = classify(blink_rate, [5, 10], ["Very Low", "Reduced", "Normal"])
sac_vel_class = classify(avg_sac_vel, [0.6, 1.0], ["Low", "Medium", "High"])
sac_amp_class = classify(avg_sac_amp, [0.08, 0.15], ["Severely Reduced", "Reduced", "Normal"])
jitter_class = classify(jitter_count, [10, 30], ["Stable", "Mild Instability", "Unstable"])
v_range_class = classify(v_range, [0.2, 0.35], ["Severe Loss", "Mild Loss", "Normal"])

summary = {
    "Duration (min)": duration_min,
    "Blink Rate (/min)": blink_rate,
    "Blink Classification": blink_class,
    "Saccade Rate (/min)": saccade_rate,
    "Avg Saccade Velocity": avg_sac_vel,
    "Saccade Velocity Class": sac_vel_class,
    "Avg Saccade Amplitude": avg_sac_amp,
    "Saccade Amplitude Class": sac_amp_class,
    "Fixation Jitter Count": jitter_count,
    "Jitter Classification": jitter_class,
    "Vertical Gaze Range": v_range,
    "Vertical Range Class": v_range_class
}

print("\n=== PSP Session Summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")

sum_df = pd.DataFrame([summary])
sum_df.to_csv("psp_summary.csv", index=False)

plt.figure(figsize=(10, 4))
plt.plot(df['t'], df['v_ratio'], label='Vertical Gaze', color='blue')
if not saccades.empty:
    plt.scatter(saccades['t'], saccades['v_ratio'], color='red', label='Saccades')
plt.xlabel("Time")
plt.ylabel("Vertical Ratio")
plt.title("Vertical Gaze Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("psp_plot.png")
print("\nSaved plot to psp_plot.png and summary to psp_summary.csv")
