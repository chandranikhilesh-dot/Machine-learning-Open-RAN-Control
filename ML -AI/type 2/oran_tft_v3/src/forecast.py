from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import CompactTFT

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['throughput_mbps', 'latency_ms', 'packet_loss_pct', 'prb_utilization', 'active_ues']
LABELS = {'throughput_mbps':'Throughput (Mbps)','latency_ms':'Latency (ms)','packet_loss_pct':'Packet loss (%)','prb_utilization':'PRB utilization (%)','active_ues':'Active UEs'}

def forecast():
    model_path = ROOT/'models'/'compact_tft.pt'; scaler_path = ROOT/'models'/'scaler.npz'
    data_path = ROOT/'data'/'oran_kpis.csv'
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError('Model not trained. Run: python src/train.py')
    df = pd.read_csv(data_path)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    scaler = np.load(scaler_path)
    mean, scale = scaler['mean'], scaler['scale']
    values = df[FEATURES].to_numpy(dtype=np.float32)
    x = (values[-ckpt['lookback']:] - mean) / scale
    model = CompactTFT(n_features=len(FEATURES), hidden=ckpt['hidden'], horizon=ckpt['horizon'])
    model.load_state_dict(ckpt['state_dict']); model.eval()
    with torch.no_grad(): pred = model(torch.tensor(x[None], dtype=torch.float32)).numpy()[0]
    pred = pred * scale[None,:,None] + mean[None,:,None]
    rows=[]
    last_time=pd.to_datetime(df['timestamp']).iloc[-1]
    for step in range(ckpt['horizon']):
        row={'timestamp':last_time+pd.Timedelta(minutes=15*(step+1))}
        for j, feature in enumerate(FEATURES):
            row[feature+'_p10']=float(pred[step,j,0])
            row[feature+'_p50']=float(pred[step,j,1])
            row[feature+'_p90']=float(pred[step,j,2])
        rows.append(row)
    return pd.DataFrame(rows)
