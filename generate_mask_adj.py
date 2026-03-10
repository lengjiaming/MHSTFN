import pandas as pd
import numpy as np
import random

# 固定随机种子
random.seed(42)
np.random.seed(42)

# 1. 生成 10% 被 mask 的节点
num_nodes = 817
num_mask = num_nodes // 10  # 约82
all_nodes = np.arange(num_nodes)
masked_nodes = np.random.choice(all_nodes, size=num_mask, replace=False)
masked_nodes_set = set(masked_nodes)
print(f"Masked Nodes: {masked_nodes}")

# 2. 读取邻接矩阵 CSV 文件
edge_df = pd.read_csv("adj_matrix/hefei/sub_matrix_his_new.csv")  # 替换为你的实际文件名

# 3. 过滤掉 from 或 to 节点在 masked_nodes_set 中的行
filtered_df = edge_df[~edge_df['from'].isin(masked_nodes_set) & ~edge_df['to'].isin(masked_nodes_set)]

# 4. 保存新的 CSV 文件
filtered_df.to_csv("adj_matrix/hefei/sub_matrix_his_mask.csv", index=False)

print(f"原始边数: {len(edge_df)}")
print(f"保留边数: {len(filtered_df)}")
