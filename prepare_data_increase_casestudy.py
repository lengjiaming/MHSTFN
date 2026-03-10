import os
import numpy as np
import argparse
import configparser
import pandas as pd
np.random.seed(42)
def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence,data_sequence2, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)
        target = np.concatenate([data_sequence2[i: j]
                                        for i, j in week_indices], axis=0)
    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)
        target = np.concatenate([data_sequence2[i: j]
                                        for i, j in day_indices], axis=0)
    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    # target = data_sequence[label_start_idx: label_start_idx + num_for_predict]
        target = np.concatenate([data_sequence2[i: j]
                                        for i, j in hour_indices], axis=0)
    return week_sample, day_sample, hour_sample, target


def read_and_generate_dataset(graph_signal_matrix_filename,graph_signal_matrix_filename2,
                                                     num_of_weeks, num_of_days,
                                                     num_of_hours, num_for_predict,
                                                     points_per_hour=12, save=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    data_seq = np.load(graph_signal_matrix_filename2)['data']  # (sequence_length, num_of_vertices, num_of_features)
    data_seq2 = np.load(graph_signal_matrix_filename)['data']
    max_count1 = np.max(data_seq, axis=0, keepdims=True)
    # max_count2 = np.max(data_seq2, axis=0, keepdims=True)
    max_count1 = max_count1.squeeze()  # 去掉维度，变成 (817,)
    num_nodes = max_count1.shape[0]

    # 计算需要修改的节点数量（20%）
    num_to_modify = int(num_nodes * 0.2)

    # 去除 max_count1 的多余维度，得到 (817,)
    max_capacity = max_count1.squeeze()

    # 计算 75% 阈值，用广播扩展为 (26208, 817)
    threshold = max_capacity * 0.90
    threshold = threshold.reshape(1, -1)  # shape: (1, 817)

    # 计算每个时间步中，哪些节点达到 >=75% 容量
    high_pressure_mask = (data_seq.squeeze() >= threshold)  # shape: (26208, 817), bool

    # 统计每个节点在多少个时间步处于高压状态，并归一化为比例
    high_pressure_ratio = high_pressure_mask.sum(axis=0) / data_seq.shape[0]  # shape: (817,)

    # 选出高压占用率最高的前 20% 节点索引
    num_to_modify = int(num_nodes * 0.2)
    selected_indices = np.argsort(high_pressure_ratio)[-num_to_modify:]
    # # 随机选择节点索引（不放回）
    # selected_indices = np.random.choice(num_nodes, size=num_to_modify, replace=False)

    # 创建拷贝以修改
    modified_max_count = max_count1.copy()

    # 扩大选中节点的最大桩数（扩大30%并四舍五入取整）
    modified_max_count[selected_indices] = np.round(modified_max_count[selected_indices] * 1.3).astype(int)

    # 保存为 CSV 文件，每行一个节点的最大桩数
    df = pd.DataFrame(modified_max_count, columns=["max_pile_count"])
    df.to_csv("modified_max_count_high.csv", index=False)

    print("保存成功，共修改了", num_to_modify, "个节点。")
    max_count = modified_max_count # 逐元素最大值，shape: (1, 817, 1)
    max_tensor = np.tile(max_count, (26208, 1, 1))
    merged_seq = np.concatenate([data_seq, max_tensor.transpose(0,2,1)], axis=-1)

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(merged_seq,data_seq2, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)
        #(1,170,3,12)
        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)

        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(
            sample)  # sample：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    #[(10699,170,3,12),(10699,170,12),(10699,1)]
    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    train_target = training_set[-2]  # (B,N,T)
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

    all_data = {
        'train': {
            'x': train_x_norm,
            # 'x': train_x,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            # 'x': val_x,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            # 'x': test_x,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        }
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data _mean :', stats['_mean'].shape, stats['_mean'])
    print('train data _std :', stats['_std'].shape, stats['_std'])

    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        # filename = os.path.join(dirpath, file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks))
        filename = os.path.join(dirpath, file +"-increase-casestudyhigh-dataloader")
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                            )
    return all_data


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

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0,1,3), keepdims=True)
    std = train.std(axis=(0,1,3), keepdims=True)
    print('mean.shape:',mean.shape)
    print('std.shape:',std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm


# prepare dataset
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations_aaai/my_hefei_increase.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

# global_adj_filename = data_config['global_adj_filename'] #./data/METR_LA/distance_LA.csv
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']#./data/METR_LA/METR_LA.npz
graph_signal_matrix_filename2 = data_config['graph_signal_matrix_filename2']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])#307
points_per_hour = int(data_config['points_per_hour'])#12
num_for_predict = int(data_config['num_for_predict'])#12
len_input = int(data_config['len_input'])#12
dataset_name = data_config['dataset_name']#METR_LA


data = np.load(graph_signal_matrix_filename)
data['data'].shape

all_data = read_and_generate_dataset(graph_signal_matrix_filename,graph_signal_matrix_filename2, 0, 0, 1, num_for_predict, points_per_hour=points_per_hour, save=False)

