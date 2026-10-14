from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from forecast import forecast, FEATURES, LABELS
from control import control_recommendation

st.set_page_config(page_title='O-RAN TFT Forecasting', page_icon='📡', layout='wide')
st.title('📡 O-RAN Multi-KPI Forecasting')
st.caption('Student software prototype • simulated KPI data • 12-step (3-hour) forecast')
data_path=ROOT/'data'/'oran_kpis.csv'
if not data_path.exists():
    st.error('KPI dataset not found. In the VS Code terminal run: python src/generate_data.py')
    st.stop()
try:
    df=pd.read_csv(data_path); pred=forecast()
except Exception as e:
    st.error(str(e)); st.info('Run in the project terminal: python src\\train.py'); st.stop()
status,action=control_recommendation(pred)
last=df.iloc[-1]
cols=st.columns(5)
for col,feature,label,fmt in zip(cols, FEATURES, ['Throughput','Latency','Packet loss','PRB utilization','Active UEs'], ['.1f',' .1f',' .2f','.1f','.0f']):
    col.metric(label, f"{last[feature]:{fmt.strip()}}" + (' Mbps' if feature=='throughput_mbps' else ' ms' if feature=='latency_ms' else '%' if feature in ['packet_loss_pct','prb_utilization'] else ''))
if status=='NORMAL': st.success(f'{status}: {action}')
elif status=='MODERATE LOAD': st.warning(f'{status}: {action}')
else: st.error(f'⚠ {status}: {action}')
st.subheader('Future KPI forecasts (P10 / P50 / P90)')
selected=st.selectbox('Choose KPI to plot', FEATURES, format_func=lambda x: LABELS[x])
fig,ax=plt.subplots(figsize=(10,4))
future=pd.to_datetime(pred['timestamp'])
ax.plot(future,pred[selected+'_p50'],label='P50 forecast')
ax.fill_between(future,pred[selected+'_p10'],pred[selected+'_p90'],alpha=0.2,label='P10–P90 range')
ax.set_ylabel(LABELS[selected]); ax.set_xlabel('Future time'); ax.legend(); ax.grid(alpha=.25)
st.pyplot(fig); plt.close(fig)
st.subheader('Forecast table')
st.dataframe(pred, use_container_width=True)
st.subheader('Recent simulated KPI records')
st.dataframe(df.tail(10), use_container_width=True)
st.caption('Important: this demo uses simulated data. The warning is a threshold-based recommendation, not a live connection to a real O-RAN network.')
