#tabular method for analysis 

import pandas as pd
import matplotlib.pyplot as plt

#NOTES: FIX THRESHOLDS BASED ON RESEARCH
def classify_blink_rate(r):
    if r > 10:   return "High"
    elif r > 5:  return "Moderate"
    else:        return "Low"

def classify_saccade_velocity(v):
    if v > 1.0:  return "High"
    elif v > 0.6: return "Moderate"
    else:         return "Low"

def classify_saccade_amplitude(a):
    if a > 0.15:  return "High"
    elif a > 0.08: return "Moderate"
    else:          return "Low"

def classify_jitter(c):
    if c < 10:    return "Low"
    elif c < 30:  return "Moderate"
    else:         return "High"

def classify_vrange(r):
    if r > 0.35:  return "High"
    elif r > 0.2:  return "Moderate"
    else:          return "Low"


def main():
    df = pd.read_csv("psp_data.csv")
    df['t'] = pd.to_datetime(df['t'], unit='s', errors='coerce')
    df = df.dropna(subset=['t'])

    duration_min   = (df['t'].iloc[-1] - df['t'].iloc[0]).total_seconds() / 60
    blink_rate     = df['blink'].sum() / duration_min
    saccades       = df[df['sac_vel'].notna()]
    avg_sac_vel    = saccades['sac_vel'].astype(float).mean() if not saccades.empty else 0
    avg_sac_amp    = saccades['sac_amp'].astype(float).mean() if not saccades.empty else 0
    jitter_count   = df['jitter_t'].notna().sum()
    v_range        = df['v_ratio'].max() - df['v_ratio'].min()

    #print table
    summary = pd.DataFrame({
        "Feature": [
            "Blink Rate (/min)",
            "Avg Saccade Velocity",
            "Avg Saccade Amplitude",
            "Fixation Jitter Count",
            "Vertical Gaze Range"
        ],
        "Value": [
            round(blink_rate, 2),
            round(avg_sac_vel, 2),
            round(avg_sac_amp, 2),
            jitter_count,
            round(v_range, 2)
        ],
        "Category": [
            classify_blink_rate(blink_rate),
            classify_saccade_velocity(avg_sac_vel),
            classify_saccade_amplitude(avg_sac_amp),
            classify_jitter(jitter_count),
            classify_vrange(v_range)
        ]
    })
    print("\nPSP Session Feature Summary:\n")
    print(summary.to_string(index=False))

    #plot vertical gaze with saccades
    plt.figure(figsize=(10,4))
    plt.plot(df['t'], df['v_ratio'], label='Vertical Gaze')
    if not saccades.empty:
        plt.scatter(saccades['t'], saccades['v_ratio'],
                    color='red', label='Saccades', zorder=5)
    plt.xlabel("Time")
    plt.ylabel("Vertical Ratio")
    plt.title("Vertical Gaze Over Time with Saccades Highlighted")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
