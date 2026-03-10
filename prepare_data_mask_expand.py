import os
import numpy as np
import argparse
import configparser
import pandas as pd
from collections import defaultdict


def load_adj_matrix_from_csv(filenames, num_nodes):
    """从多个 csv 文件加载邻接矩阵，返回邻接信息字典"""
    adj_matrices = {}
    for name, filename in filenames.items():
        adj = defaultdict(list)
        df = pd.read_csv(filename)
        for _, row in df.iterrows():
            src, dst, cost = int(row['from']), int(row['to']), float(row['cost'])
            adj[src].append((dst, cost))
        adj_matrices[name] = adj
    return adj_matrices


# def redistribute_flow_per_sample(x_sample, node_max_capacity, adj_matrices, seed=None, max_trials=10):
#     """
#     对单个样本执行随机流量重分配，考虑节点容量上限
#     - 随机选出 5% 节点
#     - 被选节点每单位流量独立随机分配
#         - 90% 邻居，10% 全局
#         - 邻居类型随机：40% 距离/40% 偏好/10% 历史/10% 交通
#     - 输入 x 扩展为 3 个特征: [流量, 之前最大容量, 之后最大容量]
#         - 只有被选节点之前最大容量置为 0，其余节点为原始 node_max_capacity
#     - 分配时目标节点不能超过 node_max_capacity
#     """
#     rng = np.random.default_rng(seed)

#     num_nodes = x_sample.shape[1]
#     T = x_sample.shape[0]

#     # 初始化新特征
#     x_new = np.zeros((T, num_nodes, 3))
#     x_new[:, :, 0] = x_sample[..., 0]       # 原始流量
#     x_new[:, :, 1] = node_max_capacity      # 之前最大容量
#     x_new[:, :, 2] = node_max_capacity      # 之后最大容量

#     # 选取 5% 节点
#     num_selected = max(1, int(0.05 * num_nodes))
#     selected_nodes = rng.choice(num_nodes, size=num_selected, replace=False)

#     # x_new[:, selected_nodes, 0] = 0
#     x_new[:, selected_nodes, 1] = 0

#     # 生成邻居候选
#     neighbor_targets = {}
#     for node in selected_nodes:
#         neighbor_targets[node] = {}
#         for key in ['dis', 'poi', 'his', 'trans']:
#             neighbor_targets[node][key] = [n for n, _ in adj_matrices[key].get(node, []) if n != node]
#         neighbor_targets[node]['global'] = [n for n in range(num_nodes) if n != node]

#     for node in selected_nodes:
#         for t in range(T):
#             flow = int(round(x_sample[t, node, 0]))
#             if flow <= 0:
#                 continue

#             for _ in range(flow):
#                 # 分配尝试次数限制
#                 for trial in range(max_trials):
#                     if rng.random() < 0.9:
#                         r = rng.random()
#                         if r < 0.4 and neighbor_targets[node]['dis']:
#                             target = rng.choice(neighbor_targets[node]['dis'])
#                         elif r < 0.8 and neighbor_targets[node]['poi']:
#                             target = rng.choice(neighbor_targets[node]['poi'])
#                         elif r < 0.9 and neighbor_targets[node]['his']:
#                             target = rng.choice(neighbor_targets[node]['his'])
#                         elif neighbor_targets[node]['trans']:
#                             target = rng.choice(neighbor_targets[node]['trans'])
#                         else:
#                             target = rng.choice(neighbor_targets[node]['global'])
#                     else:
#                         target = rng.choice(neighbor_targets[node]['global'])

#                     # 检查容量是否允许
#                     if x_new[t, target, 0] < node_max_capacity[target]:
#                         x_new[t, target, 0] += 1
#                         # x_new[t, target, 2] = max(x_new[t, target, 2], x_new[t, target, 0])
#                         break  # 成功分配，跳出 trial 循环
#                     elif trial == max_trials - 1:
#                         # 尝试 max_trials 次仍未成功，则放弃
#                         pass

#             # 原节点流量减去分配出去的部分
#             x_new[t, node, 0] -= flow
#             x_new[t, node, 0] = max(x_new[t, node, 0], 0)

#     return x_new.transpose(1,2,0)

import numpy as np

def redistribute_flow_per_sample(x_sample, node_max_capacity, adj_matrices,
                                 seed=None, max_trials=10, gamma=0.8, horizon=4):
    """
    改进版：概率分配 + 衰减传递
    
    x_sample: (T, N, 1) 或 (T, N, ...)，只读取流量 x_sample[...,0]
    node_max_capacity: (N,) 每个节点的最大容量（标量/每时刻相同）
    adj_matrices: dict with keys 'dis','poi','his','trans' each mapping node -> list[(nbr, cost)]
    seed, max_trials: 随机种子与目标尝试次数上限
    gamma: 衰减系数 (0<gamma<1)，用于把一个节点分配出去的流量扩散到未来时刻
    horizon: 最多向未来扩展多少个时间步（包含当前 t）
    返回: x_new_transposed: shape (N, 3, T)  （与之前实现保持一致）
    """

    rng = np.random.default_rng(seed)

    # 取流量序列为浮点数
    flow_mat = np.array(x_sample[..., 0], dtype=float)  # shape (T, N)
    T, num_nodes = flow_mat.shape

    # 初始化 (T, N, 3)
    x_new = np.zeros((T, num_nodes, 3), dtype=float)
    x_new[:, :, 0] = flow_mat.copy()
    node_max_capacity = np.asarray(node_max_capacity, dtype=float).reshape(-1)
    x_new[:, :, 1] = node_max_capacity[np.newaxis, :]   # 之前最大容量
    x_new[:, :, 2] = node_max_capacity[np.newaxis, :]   # 之后最大容量

    # 选取 5% 节点作为要重分配的源
    num_selected = max(1, int(0.05 * num_nodes))
    selected_nodes = rng.choice(num_nodes, size=num_selected, replace=False)

    # 被选节点之前最大容量置 0
    x_new[:, selected_nodes, 1] = 0.0
    # —— 修复点 A：将 selected_nodes 标记为 disabled，避免它们被选为接收者 —— #
    disabled_mask = np.zeros(num_nodes, dtype=bool)
    disabled_mask[selected_nodes] = True
    # 预构造邻居候选列表
    neighbor_targets = {}
    for node in selected_nodes:
        neighbor_targets[node] = {}
        for key in ['dis', 'poi', 'his', 'trans']:
            neighbor_targets[node][key] = [n for n, _ in adj_matrices[key].get(node, []) if n != node and not disabled_mask[n]]
        neighbor_targets[node]['global'] = [n for n in range(num_nodes) if n != node and not disabled_mask[n]]

    # 主循环：对每个被选节点、每个时间步，整体分配流量
    for node in selected_nodes:
        for t in range(T):
            flow_value = int(round(flow_mat[t, node]))
            if flow_value <= 0:
                continue

            # 确定候选邻居
            if rng.random() < 0.9:
                # 90% 概率选局部邻居，分类型选择
                r = rng.random()
                if r < 0.4 and neighbor_targets[node]['dis']:
                    candidates = neighbor_targets[node]['dis']
                elif r < 0.8 and neighbor_targets[node]['poi']:
                    candidates = neighbor_targets[node]['poi']
                elif r < 0.9 and neighbor_targets[node]['his']:
                    candidates = neighbor_targets[node]['his']
                else:
                    candidates = neighbor_targets[node]['trans'] if neighbor_targets[node]['trans'] else neighbor_targets[node]['global']
            else:
                candidates = neighbor_targets[node]['global']

            if not candidates:
                continue

            # 随机抽样一部分候选作为接收者
            num_targets = min(len(candidates), rng.integers(1, min(4, len(candidates)) + 1))
            targets = rng.choice(candidates, size=num_targets, replace=False)

            # 按均匀分布或 Dirichlet 分布随机分配整数流量
            probs = rng.dirichlet(np.ones(num_targets))
            allocated = rng.multinomial(flow_value, probs)

            # 对每个目标，执行衰减传递
            for j, target in enumerate(targets):
                units = allocated[j]
                if units <= 0:
                    continue

                H = min(horizon, T - t)
                if H <= 0:
                    continue

                raw = np.array([gamma ** k for k in range(H)], dtype=float)
                weights = raw / raw.sum()
                delayed_flows = np.floor(units * weights + 1e-6).astype(int)

                # 保证总和等于 units（补偿误差）
                diff = units - delayed_flows.sum()
                if diff > 0:
                    idx = rng.choice(H, size=diff, replace=True)
                    for k in idx:
                        delayed_flows[k] += 1

                # 检查容量并写入
                feasible = True
                for k in range(H):
                    tt = t + k
                    if x_new[tt, target, 0] + delayed_flows[k] > node_max_capacity[target] + 1e-9:
                        feasible = False
                        break

                if feasible:
                    for k in range(H):
                        tt = t + k
                        add = delayed_flows[k]
                        x_new[tt, target, 0] += add
                        x_new[tt, target, 2] = max(x_new[tt, target, 2], x_new[tt, target, 0])
                    # 源节点扣减
                    x_new[t, node, 0] = 0
                    # x_new[t, node, 0] = max(x_new[t, node, 0], 0.0)

    return x_new.transpose(1, 2, 0)







def sliding_window_transform(data, num_his, num_pred, step=1):
    """把时间序列转成监督学习样本，支持步长"""
    x, y = [], []
    for i in range(0, len(data) - num_his - num_pred + 1, step):
        x.append(data[i:i + num_his])
        y.append(data[i + num_his:i + num_his + num_pred])
    return np.array(x), np.array(y)



def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    # 只对第一个特征（流量）做归一化
    mean = train[:, :, 0, :].mean()
    std = train[:, :, 0, :].std()
    # std[std == 0] = 1  # 防止除0
    print('mean:', mean)
    print('std:', std)

    def normalize(x):
        x_norm = x.copy()
        x_norm[:, :, 0, :] = (x[:, :, 0, :] - mean) / std  # 只标准化流量
        return x_norm

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm



def load_and_process_data(graph_signal_matrix_filename,
                          num_of_vertices,
                          points_per_hour,
                          num_for_predict,
                          adj_matrices,
                          save=True):
    """加载原始数据 -> 滑动窗口 -> 每个 sample 做流量重分配 -> 保存 npz"""
    data = np.load(graph_signal_matrix_filename)['data']  # (T, N)
    node_max_capacity = data.max(axis=0).squeeze()
    num_for_his = 12  # 例如使用 12 个历史点

    x, y = sliding_window_transform(data, num_for_his, num_for_predict, step=1)
    y = y.transpose(0,2,3,1).squeeze(2)
    # x: (num_samples, num_for_his, N), y: (num_samples, num_for_predict, N)

    # 对每个 sample 执行流量重分配
    x_new = []
    timestamps = []
    for i in range(len(x)):
        print(i)
        x_new.append(redistribute_flow_per_sample(x[i], node_max_capacity, adj_matrices))
        timestamps.append(np.array([i]))  # 时间戳为 sample 索引
    x_new = np.array(x_new)  # (num_samples, num_for_his, N, 3)
    timestamps = np.array(timestamps)  # (num_samples,1)

    # 简单划分 train/val/test
    num_samples = len(x_new)
    num_train = int(num_samples * 0.6)
    num_val = int(num_samples * 0.2)

    train_x, val_x, test_x = x_new[:num_train], x_new[num_train:num_train+num_val], x_new[num_train+num_val:]
    train_y, val_y, test_y = y[:num_train], y[num_train:num_train+num_val], y[num_train+num_val:]
    train_t, val_t, test_t = timestamps[:num_train], timestamps[num_train:num_train+num_val], timestamps[num_train+num_val:]

    # 标准化
    stats, train_x_norm, val_x_norm, test_x_norm = normalization(train_x, val_x, test_x)

    all_data = {
        'train': {'x': train_x_norm, 'target': train_y, 'timestamp': train_t},
        'val': {'x': val_x_norm, 'target': val_y, 'timestamp': val_t},
        'test': {'x': test_x_norm, 'target': test_y, 'timestamp': test_t},
        'stats': stats
    }

    if save:
        filename = os.path.splitext(graph_signal_matrix_filename)[0] + "-dataloader"
        np.savez_compressed(filename,
                            train_x=train_x_norm, train_target=train_y, train_timestamp=train_t,
                            val_x=val_x_norm, val_target=val_y, val_timestamp=val_t,
                            test_x=test_x_norm, test_target=test_y, test_timestamp=test_t,
                            mean=stats['_mean'], std=stats['_std'])
    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/my_hefei_increase.conf', type=str)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    data_config = config['Data']

    graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
    num_of_vertices = int(data_config['num_of_vertices'])
    points_per_hour = int(data_config['points_per_hour'])
    num_for_predict = int(data_config['num_for_predict'])

    dis_adj_filename = './adj_matrix/hefei/sub_matrix_dis.csv'
    poi_adj_filename = './adj_matrix/hefei/sub_matrix_poi.csv'
    his_adj_filename = './adj_matrix/hefei/sub_matrix_his_new.csv'
    trans_adj_filename = './adj_matrix/hefei/sub_matrix_trans_new.csv'

    adj_matrices = load_adj_matrix_from_csv({
        'dis': dis_adj_filename,
        'poi': poi_adj_filename,
        'his': his_adj_filename,
        'trans': trans_adj_filename
    }, num_of_vertices)

    all_data = load_and_process_data(
        graph_signal_matrix_filename,
        num_of_vertices,
        points_per_hour,
        num_for_predict,
        adj_matrices,
        save=True
    )

    print("数据处理完成，已保存 npz 文件！")
