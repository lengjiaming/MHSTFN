import pandas as pd
import numpy as np

def detect_added_and_expanded_window(flow, station_id, window=288, expand_ratio=1.5, min_increase=5):
    """
    flow: 1D numpy array, 表示某个充电站的流量时间序列
    station_id: 站点编号
    window: 窗口大小（默认288=一天的5分钟点数）
    expand_ratio: 扩容判定阈值，新旧最大值比值
    min_increase: 扩容时最小绝对增长
    """
    T = len(flow)
    num_windows = T // window
    window_max = [flow[i*window:(i+1)*window].max() for i in range(num_windows)]
    window_max = np.array(window_max)

    cummax = np.maximum.accumulate(window_max)

    records = []

    # --- 检测增建 ---
    nonzero_idx = np.where(window_max > 0)[0]
    if len(nonzero_idx) > 0 and nonzero_idx[0] > 3:  # 前几个窗口全是0
        added_time = nonzero_idx[0] * window
        first_capacity = cummax[-1]
        records.append({
            "station": station_id,
            "event": "added",
            "time_idx": added_time,
            "old_capacity": 0,
            "new_capacity": first_capacity
        })

    # --- 检测扩容 ---
    last_max = cummax[0]
    expand_record = None
    for i in range(1, len(cummax)):
        if cummax[i] > last_max:
            if cummax[i] >= last_max * expand_ratio and cummax[i] - last_max >= min_increase and i >=30:
                expand_time = i * window
                records.append({
                    "station": station_id,
                    "event": "expanded",
                    "time_idx": expand_time,
                    "old_capacity": last_max,
                    "new_capacity": cummax[-1]
                })
            last_max = cummax[i]
    if expand_record is not None:
        records.append(expand_record)
    return records


# === 主程序 ===
df = pd.read_csv("data/hf-charge.csv", header=None)
T, N = df.shape
print(f"数据维度: {T} 行 (时间点), {N} 列 (充电站)")

all_records = []
for station in df.columns:
    flow = df[station].values
    recs = detect_added_and_expanded_window(flow, station)
    all_records.extend(recs)

results_df = pd.DataFrame(all_records)

# 保存结果
results_df.to_csv("station_events.csv", index=False)

print("事件总数:", len(results_df))
print("增建站数量:", (results_df["event"] == "added").sum())
print("扩容事件数量:", (results_df["event"] == "expanded").sum())
print(results_df.head(20))
