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


import numpy as np

def redistribute_flow_per_sample(x_sample, node_max_capacity, adj_matrices,
                                 seed=None, max_trials=10, gamma=0.8, horizon=4):
    """
    改进版：扩容模拟 + 残留保留逻辑
    主要改动：
      - 把被选出的节点的之前最大容量设为原来的 1/2（而非 0）
      - 分配时只尝试把超过“之前最大容量”的部分（surplus）分配出去，
        若无法全部分配则尽可能多分配，剩余保留在源节点。
    其余逻辑（邻居选择、Dirichlet+multinomial、时间衰减、容量检查）保持一致。
    返回: x_new_transposed: shape (N, 3, T)
    """
    rng = np.random.default_rng(seed)

    # 取流量序列为浮点数
    flow_mat = np.array(x_sample[..., 0], dtype=float)  # shape (T, N)
    T, num_nodes = flow_mat.shape

    # 初始化 (T, N, 3)
    x_new = np.zeros((T, num_nodes, 3), dtype=float)
    x_new[:, :, 0] = flow_mat.copy()
    node_max_capacity = np.asarray(node_max_capacity, dtype=float).reshape(-1)
    x_new[:, :, 1] = node_max_capacity[np.newaxis, :]   # 之前最大容量（将针对 selected_nodes 调整为一半）
    x_new[:, :, 2] = node_max_capacity[np.newaxis, :]   # 之后最大容量（保持原值）

    # 选取 5% 节点作为要重分配的源（这里我们仍按原逻辑选节点）
    num_selected = max(1, int(0.05 * num_nodes))
    selected_nodes = rng.choice(num_nodes, size=num_selected, replace=False)

    # ==========  改动 A：把被选节点“之前最大容量”设为一半（不是 0）  ========== #
    x_new[:, selected_nodes, 1] = (x_new[:, selected_nodes, 1] * 0.5).astype(int)


    # 保持之前的 disabled_mask（如果你希望扩容站点仍然可以作为目标，将下面设为 False）
    disabled_mask = np.zeros(num_nodes, dtype=bool)
    disabled_mask[selected_nodes] = True

    # 预构造邻居候选列表（排除 disabled 节点）
    neighbor_targets = {}
    for node in selected_nodes:
        neighbor_targets[node] = {}
        for key in ['dis', 'poi', 'his', 'trans']:
            neighbor_targets[node][key] = [n for n, _ in adj_matrices[key].get(node, []) if n != node and not disabled_mask[n]]
        neighbor_targets[node]['global'] = [n for n in range(num_nodes) if n != node and not disabled_mask[n]]

    # helper: 计算给定 units，按 gamma 衰减得到的每个未来时刻的整数分配（确定性补偿）
    def compute_delayed_flows(units, H, gamma):
        if units <= 0 or H <= 0:
            return np.zeros((0,), dtype=int)
        raw = np.array([gamma ** k for k in range(H)], dtype=float)
        weights = raw / raw.sum()
        # 先 floor，然后把剩余按顺序补1（保证确定性、单调性，便于二分）
        delayed = np.floor(units * weights + 1e-12).astype(int)
        diff = int(units - delayed.sum())
        for i in range(diff):
            delayed[i % H] += 1
        return delayed

    # helper: 二分找到对某个 target 在时间窗 [t, t+H) 最多能放多少 units（<= units）
    def max_placeable_units(target, t, units, H, gamma):
        lo, hi = 0, units
        while lo < hi:
            mid = (lo + hi + 1) // 2
            delayed = compute_delayed_flows(mid, H, gamma)
            feasible = True
            for k in range(len(delayed)):
                tt = t + k
                if x_new[tt, target, 0] + delayed[k] > node_max_capacity[target] + 1e-9:
                    feasible = False
                    break
            if feasible:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # 主循环：对每个被选节点、每个时间步，分配超额（surplus）
    for node in selected_nodes:
        for t in range(T):
            flow_value = int(round(flow_mat[t, node]))
            if flow_value <= 0:
                continue

            # 计算该节点“之前最大容量”（取整）
            prev_cap = int(np.floor(x_new[t, node, 1] + 1e-9))
            # 只需分配超过之前容量的那部分
            surplus = max(flow_value - prev_cap, 0)
            if surplus <= 0:
                # 没有超额需要分配，保留原始流量
                continue

            surplus_remaining = surplus
            distributed_total = 0

            # 尝试多次分配（max_trials），尽可能将 surplus 放出去
            for trial in range(max_trials):
                if surplus_remaining <= 0:
                    break

                # 确定候选邻居（与原逻辑一致）
                if rng.random() < 0.9:
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
                    break

                # 预过滤：当前时刻已有剩余容量为 0 的节点可以先过滤（提高效率）
                candidates = [n for n in candidates if x_new[t, n, 0] + 1e-9 < node_max_capacity[n]]
                if not candidates:
                    break

                num_targets = min(len(candidates), rng.integers(1, min(4, len(candidates)) + 1))
                targets = list(rng.choice(candidates, size=num_targets, replace=False))

                # 按 Dirichlet 分配剩余的 surplus（整数化）
                probs = rng.dirichlet(np.ones(len(targets)))
                allocated = rng.multinomial(surplus_remaining, probs)

                # 对每个目标，尝试放置 allocated[j]（但允许放置 <= allocated[j]）
                any_placed_this_trial = False
                for j, target in enumerate(targets):
                    units_want = int(allocated[j])
                    if units_want <= 0:
                        continue

                    H = min(horizon, T - t)
                    if H <= 0:
                        continue

                    # 找到对 target 在该时间窗内最多能放多少（<= units_want）
                    m = max_placeable_units(target, t, units_want, H, gamma)
                    if m <= 0:
                        continue

                    # 计算 m 对应的各步 delayed_flows（确定性补偿）
                    delayed_flows = compute_delayed_flows(m, H, gamma)

                    # 写入 target（已保证可行）
                    for k in range(H):
                        tt = t + k
                        add = delayed_flows[k]
                        x_new[tt, target, 0] += add
                        # 更新之后最大容量记录（与原代码一致）
                        x_new[tt, target, 2] = max(x_new[tt, target, 2], x_new[tt, target, 0])

                    surplus_remaining -= m
                    distributed_total += m
                    any_placed_this_trial = True

                    if surplus_remaining <= 0:
                        break

                # 如果本次 trial 丝毫都放不进去，那么下次可以换一组候选（或直接结束）
                if not any_placed_this_trial:
                    continue

            # 最终：从源节点扣减已经成功分配出去的量（保留其余流量）
            x_new[t, node, 0] = max(x_new[t, node, 0] - distributed_total, 0.0)
            # （可选）保证至少保留不小于 prev_cap：这取决于你是否希望在分配失败时强制降到 prev_cap
            # 目前我们设计为：尽可能分配 surplus，未分配的留在源节点（即可能 > prev_cap）

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
