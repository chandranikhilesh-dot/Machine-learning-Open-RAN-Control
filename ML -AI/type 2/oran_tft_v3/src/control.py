def control_recommendation(pred):
    prb = float(pred['prb_utilization_p90'].max())
    latency = float(pred['latency_ms_p90'].max())
    if prb >= 90 or latency >= 40:
        return 'HIGH LOAD PREDICTED', f'Forecast P90 reaches PRB {prb:.1f}% and latency {latency:.1f} ms. Consider load balancing or resource adjustment.'
    if prb >= 75 or latency >= 30:
        return 'MODERATE LOAD', f'Forecast P90 reaches PRB {prb:.1f}% and latency {latency:.1f} ms. Monitor closely.'
    return 'NORMAL', f'Forecast P90 remains below alert thresholds (PRB {prb:.1f}%, latency {latency:.1f} ms).'
