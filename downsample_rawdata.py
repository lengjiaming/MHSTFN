import os
import numpy as np
import argparse
import configparser
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/my_hefei_increase.conf', type=str,
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
arr = data['data']

# 检查能否整除
assert arr.shape[0] % 6 == 0, "时间长度不是6的倍数，无法整齐降采样"

# reshape: (4368, 6, 817, 1)
arr_reshaped = arr.reshape(arr.shape[0] // 6, 6, arr.shape[1], arr.shape[2])

# 对6个时间点取平均，四舍五入取整
arr_downsampled = np.rint(arr_reshaped.mean(axis=1)).astype(int)  # (4368, 817, 1)

print("原始 shape:", arr.shape)
print("降采样后 shape:", arr_downsampled.shape)

# 保存 npz
np.savez_compressed("data/hf_downsample.npz", data=arr_downsampled)
print("已保存 npz 到 data/hf_downsample.npz")

# 保存 csv (展平最后一维)
arr_csv = arr_downsampled[:, :, 0]  # (4368, 817)
df = pd.DataFrame(arr_csv)
df.to_csv("data/hf_downsample.csv", index=False)
print("已保存 csv 到 data/hf_downsample.csv")