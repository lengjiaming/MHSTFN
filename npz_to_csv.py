import numpy as np
import pandas as pd

# 读取 .npz 文件
npz_file = np.load('data/hf-charge-increase.npz')
data = npz_file['data']  # shape: (26208, 817, 1)

# 去掉最后一个维度
data = data.squeeze(-1)  # shape: (26208, 817)

# 转成 DataFrame
df = pd.DataFrame(data.astype(np.int32))

# 保存为 CSV
df.to_csv('data/hf-charge-increase.csv', index=False, header=False)

print("✅ 已保存为 data/hf-charge-mask.csv！")
