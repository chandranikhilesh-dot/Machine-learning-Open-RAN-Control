from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from model import CompactTFT

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'oran_kpis.csv'
MODEL_DIR = ROOT / 'models'; MODEL_DIR.mkdir(exist_ok=True)
FEATURES = ['throughput_mbps', 'latency_ms', 'packet_loss_pct', 'prb_utilization', 'active_ues']
LOOKBACK, HORIZON = 48, 12

class WindowDataset(Dataset):
    def __init__(self, values, starts):
        self.values, self.starts = values, starts
    def __len__(self): return len(self.starts)
    def __getitem__(self, i):
        s = self.starts[i]
        return (torch.tensor(self.values[s:s+LOOKBACK], dtype=torch.float32),
                torch.tensor(self.values[s+LOOKBACK:s+LOOKBACK+HORIZON], dtype=torch.float32))

def main():
    if not DATA.exists():
        raise FileNotFoundError('Dataset missing. Run: python src/generate_data.py')
    torch.manual_seed(7); np.random.seed(7)
    df = pd.read_csv(DATA)
    raw = df[FEATURES].to_numpy(dtype=np.float32)
    split = int(len(raw)*0.8)
    scaler = StandardScaler().fit(raw[:split])
    values = scaler.transform(raw).astype(np.float32)
    starts = list(range(0, len(values)-LOOKBACK-HORIZON))
    train_starts = [s for s in starts if s+LOOKBACK+HORIZON <= split]
    val_starts = [s for s in starts if s >= split-LOOKBACK and s+LOOKBACK+HORIZON <= len(values)]
    train_ds, val_ds = WindowDataset(values, train_starts), WindowDataset(values, val_starts)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    model = CompactTFT(n_features=len(FEATURES), hidden=48, horizon=HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    quantiles = torch.tensor([0.1, 0.5, 0.9]).view(1,1,1,3)
    best = float('inf')
    epochs = 8
    for epoch in range(epochs):
        model.train(); train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            err = y.unsqueeze(-1) - pred
            loss = torch.maximum(quantiles*err, (quantiles-1)*err).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss += loss.item()*len(x)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                pred=model(x); err=y.unsqueeze(-1)-pred
                loss=torch.maximum(quantiles*err,(quantiles-1)*err).mean()
                val_loss += loss.item()*len(x)
        train_loss /= max(len(train_ds),1); val_loss /= max(len(val_ds),1)
        print(f'Epoch {epoch+1}/{epochs} | train pinball loss={train_loss:.4f} | validation={val_loss:.4f}')
        if val_loss < best:
            best = val_loss
            torch.save({'state_dict':model.state_dict(), 'features':FEATURES, 'lookback':LOOKBACK, 'horizon':HORIZON, 'hidden':48}, MODEL_DIR/'compact_tft.pt')
    np.savez(MODEL_DIR/'scaler.npz', mean=scaler.mean_, scale=scaler.scale_)
    (MODEL_DIR/'metadata.json').write_text(json.dumps({'features':FEATURES,'lookback':LOOKBACK,'horizon':HORIZON,'best_validation_pinball_loss':best}, indent=2))
    print(f'Training complete. Model saved to {MODEL_DIR / "compact_tft.pt"}')

if __name__ == '__main__': main()
