#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from model.HGT import make_model
from lib.utils_HGT import *
from tensorboardX import SummaryWriter
# from lib.metrics import masked_mape_np,  masked_mae,masked_mse,masked_rmse
import datetime
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/my_hefei.conf', type=str,
                    help="configuration file path")
# parser.add_argument("--config", default='configurations/my_shenzhen.conf', type=str,
#                     help="configuration file path")
# parser.add_argument("--config", default='configurations/PEMS08.conf', type=str,
#                     help="configuration file path")
# parser.add_argument("--month", default='10', type=str,choices=['10','11','12'])
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

dis_adj_filename = data_config['dis_adj_filename']
poi_adj_filename = data_config['poi_adj_filename']
his_adj_filename = data_config['his_adj_filename']
trans_adj_filename = data_config['trans_adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
num_of_graph = int(data_config['num_of_graph'])
dataset_name = data_config['dataset_name']
num_of_expert = int(data_config['num_of_expert'])
top_k = int(data_config['top_k'])

model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda:4')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
time_strides = int(training_config['time_strides'])
node_dim = int(training_config['node_dim'])
edge_dim = int(training_config['edge_dim'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
loss_function = training_config['loss_function']
lambda_contrastive = float(training_config['lambda_contrastive'])
num_prototypes  = int(training_config['num_prototypes'])
seed  = int(training_config['seed'])
folder_dir = '%s_channel%d_%e' % (model_name, in_channels, learning_rate)
print('folder_dir:', folder_dir)
# params_path = os.path.join('experiments', dataset_name, folder_dir,args.month)
params_path = os.path.join('experiments', dataset_name, folder_dir)

print('params_path:', params_path)

# edges_file_list = [dis_adj_filename, poi_adj_filename, his_adj_filename]
# edge_features = load_and_process_edges(edges_file_list)
# edge_features = torch.tensor(edge_features, dtype=torch.float32).to(DEVICE) 
# node_edge_matrix = torch.tensor(node_edge_matrix, dtype=torch.float32).to(DEVICE) 
dis_adj_mx = get_adjacency_matrix(dis_adj_filename, num_of_vertices)
poi_adj_mx = get_adjacency_matrix(poi_adj_filename, num_of_vertices)
his_adj_mx = get_adjacency_matrix(his_adj_filename, num_of_vertices)
trans_adj_mx = get_adjacency_matrix(trans_adj_filename, num_of_vertices)
# edge_features = np.stack([dis_adj_mx, poi_adj_mx, his_adj_mx,trans_adj_mx], axis=0)
edge_features = np.stack([dis_adj_mx,poi_adj_mx,his_adj_mx,trans_adj_mx], axis=0)
# dis_labels = balanced_partition(dis_adj_mx, n_clusters)
# poi_labels = balanced_partition(poi_adj_mx, n_clusters)
# his_labels = balanced_partition(his_adj_mx, n_clusters)
# trans_labels = balanced_partition(trans_adj_mx, n_clusters)
# dis_labels = graph_cut_partition(dis_adj_mx, n_clusters)
# poi_labels = graph_cut_partition(poi_adj_mx, n_clusters)
# his_labels = graph_cut_partition(his_adj_mx, n_clusters)
# trans_labels = graph_cut_partition(trans_adj_mx, n_clusters)
# node_labels =  np.stack([dis_labels, poi_labels, his_labels], axis=0)
# 转换为 PyTorch 张量
edge_features = torch.tensor(edge_features).to(DEVICE)
# adjacency_matrix = torch.sum(edge_features, dim=0)
# adjacency_matrix = (adjacency_matrix > 0).int()
adjacency_matrix = get_three_hop_adjacency(edge_features,2)
# node_labels = torch.tensor(node_labels).to(DEVICE)
# num_of_edges = node_edge_matrix.shape[1]
# subgraph_table = construct_subgraph_table(node_labels,num_of_subgraph)
net = make_model(DEVICE,nb_block, in_channels, node_dim, edge_dim, num_of_graph,  num_of_vertices,edge_features,adjacency_matrix,num_prototypes,len_input,num_for_predict)

# # 将模型发送到默认设备
# net.to(DEVICE)
train_loader, train_target_tensor, _mean, _std = load_graphdata_channel1(
    graph_signal_matrix_filename, DEVICE, batch_size)

val_loader, val_target_tensor, test_loader, test_target_tensor = load_graphdata_channel2(
    graph_signal_matrix_filename, DEVICE, batch_size)

def predict_main(global_step, data_loader, data_target_tensor, _mean, _std, type):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type)

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        loss1 = self.l1_loss(input, target)
        loss2 = self.mse_loss(input, target)
        return 0.2*loss1 + loss2
    
def train_main():
    torch.autograd.set_detect_anomaly(True)
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('node_dim\t', node_dim)
    print('edge_dim\t', edge_dim)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)
    masked_flag=0
    criterion = nn.L1Loss().to(DEVICE)
    
    if loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag= 0
    elif loss_function == 'mae_rmse':
        criterion = CombinedLoss().to(DEVICE)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()
    
    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    # train model
    for epoch in range(start_epoch, epochs):
        current_time = datetime.datetime.now()
        print(f"Epoch {epoch+1}/{epochs} at {current_time}")
        params_filename = os.path.join(params_path, f'epoch_{epoch}.params')

        val_loss = compute_val_loss_mstgcn(net, val_loader,lambda_contrastive,edge_features, criterion, masked_flag, sw, epoch)
        evaluate_on_test_mstgcn(net, test_loader, test_target_tensor,edge_features, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print(f'Save parameters to file: {params_filename}')

        net.train()  # Ensure dropout layers are in train mode
        epoch_loss = 0
        epoch_prediction_loss = 0
        epoch_contrastive_loss = 0
        batch_count = 0
        accumulation_steps = 3
        # Add a tqdm progress bar for batches
        for batch_index, batch_data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Batch Progress", leave=False):
            encoder_inputs, labels, timestamps = batch_data
            # optimizer.zero_grad()
            # outputs = net(encoder_inputs, timestamps)
            outputs= net(encoder_inputs)
            
            outputs = outputs.squeeze(2)
            prediction_loss = criterion(outputs, labels)
            loss = prediction_loss 

            loss.backward()
            # optimizer.step()
              # 每 accumulation_steps 次更新一次参数
            if (batch_index + 1) % accumulation_steps == 0 or (batch_index + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            training_loss = loss.item()
            global_step += 1
            sw.add_scalar('training_loss', training_loss, global_step)
            
            batch_count += 1
            epoch_loss += training_loss
            epoch_prediction_loss += prediction_loss.item()
            if global_step % 1000 == 0:
                elapsed_time = time() - start_time
                print(f'Global step: {global_step}, Training loss: {training_loss:.4f}, Time: {elapsed_time:.4f}s')
        train_loss_str = (
            '|Predicting Loss: {:.6f} | '
            'Total Loss: {:.6f}'
        ).format(
            epoch_prediction_loss / batch_count,
            epoch_loss / batch_count
        )
        with open('result_hefei_HGT/train_losses.txt', 'a', encoding='utf-8') as file:
            file.write(train_loss_str)
            file.write('\n')
        
    print('best epoch:', best_epoch)

    # apply the best model on the test set
    # predict_main(best_epoch, test_loader, test_target_tensor,metric_method ,_mean, _std, 'test')


# def predict_main(global_step, data_loader, data_target_tensor,metric_method, _mean, _std, type):
#     '''

#     :param global_step: int
#     :param data_loader: torch.utils.data.utils.DataLoader
#     :param data_target_tensor: tensor
#     :param mean: (1, 1, 3, 1)
#     :param std: (1, 1, 3, 1)
#     :param type: string
#     :return:
#     '''

#     params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
#     print('load weight from:', params_filename)

#     net.load_state_dict(torch.load(params_filename))

#     predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method,_mean, _std, params_path, type)


if __name__ == "__main__":

    train_main()

    # predict_main(70, test_loader, test_target_tensor, _mean, _std, 'test')














