from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'oran_kpis.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
n = 1600  # 16.7 days at 15-minute intervals
idx = pd.date_range('2026-09-01', periods=n, freq='15min')
t = np.arange(n)
daily = np.sin(2 * np.pi * (t % 96) / 96 - 1.2)
weekly = np.sin(2 * np.pi * t / (96 * 7))
rush = np.maximum(0, np.sin(2 * np.pi * (t % 96) / 96 - 0.6))
traffic = 55 + 18*daily + 9*rush + 4*weekly + rng.normal(0, 3, n)
users = 600 + 170*daily + 80*rush + 30*weekly + rng.normal(0, 20, n)
# Add a high-load event toward the end so the warning logic is easy to demonstrate.
stress = np.clip((t - int(n*0.82)) / (n*0.12), 0, 1)
traffic += 35*stress
users += 220*stress
prb = np.clip(20 + 0.52*traffic + 0.018*users + rng.normal(0, 2.5, n), 5, 99)
throughput = np.clip(110 - 0.45*prb + 0.012*users + rng.normal(0, 3, n), 5, 120)
latency = np.clip(10 + 0.34*prb + 0.22*np.maximum(prb-75, 0) + rng.normal(0, 1.8, n), 5, 100)
packet_loss = np.clip(0.15 + 0.045*np.maximum(prb-70, 0) + rng.normal(0, 0.12, n), 0, 10)
df = pd.DataFrame({
    'timestamp': idx,
    'throughput_mbps': throughput,
    'latency_ms': latency,
    'packet_loss_pct': packet_loss,
    'prb_utilization': prb,
    'active_ues': np.clip(users, 50, None),
    'traffic_mbps': np.clip(traffic, 1, None),
})
df.to_csv(OUT, index=False)
print(f'Created {OUT} ({len(df)} rows).')
