import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import math
import time
def time_encoding(minutes_tensor):
    # 确保输入张量在 GPU 上
    
    mask = minutes_tensor > 8640
    minutes_tensor[mask] += 48

    num_dims = 12

    # 创建一个偏移量序列 [1, 2, ..., num_dims]
    offsets = torch.arange(0, num_dims).unsqueeze(0)  # (1, num_dims)
    offsets = offsets.to(minutes_tensor.device)

    # 对原始 (b, 1) 张量进行广播并添加偏移量序列
    expanded_tensor = minutes_tensor + offsets

    # 计算小时数和星期数
    # hours = (expanded_tensor // 12) % 24  # 假设 minutes_tensor 是从某一时间点开始的分钟数
    hours = expanded_tensor  % 48
    weekdays = (expanded_tensor // (48)) % 7
    hours = hours.int()
    weekdays = weekdays.int()
    # 2 * pi / 周期
    return hours,weekdays

class NodeEmbeddingNetwork(nn.Module):
    def __init__(self, DEVICE, in_channels, conv_dim, embed_dim):
        super(NodeEmbeddingNetwork, self).__init__()
        # self.embedding = nn.Embedding(max_value + 1, embed_dim)
        self.embedding = nn.Linear(1, embed_dim)
        self.conv1 = nn.Conv2d(in_channels, conv_dim // 2, kernel_size=(1, 3), padding=(0, 1))  # 第一层卷积
        # self.bn1 = nn.BatchNorm2d(conv_dim // 2)  # 第一层批归一化
        self.act1 = nn.LeakyReLU()  # 第一层激活函数
        
        self.conv2 = nn.Conv2d(conv_dim // 2, conv_dim, kernel_size=(1, 3), padding=(0, 1))  # 第二层卷积
        # self.bn2 = nn.BatchNorm2d(conv_dim)  # 第二层批归一化
        self.act2 = nn.LeakyReLU()  # 第二层激活函数

        self.fc = nn.Linear(conv_dim, embed_dim)  # 最后一层线性层
        self.DEVICE = DEVICE

        self.to(DEVICE)
    def forward(self, x):
        # 输入形状: (B, N, 1, T)
        B, N, C, T = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.permute(0,2,1,3).view(B,C,N,T)
        # x = x.view(x.shape[0], x.shape[1], x.shape[3], x.shape[2])  # 变为 (B, N, T, 1)
        # # x = self.embedding(x.long()).squeeze(3)
        # x = self.embedding(x)
        # x = x.view(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
        
        # 第一层卷积
        x = self.conv1(x)  # (B, N, conv_dim // 2, T)
        # x = self.bn1(x)  # 批归一化
        x = self.act1(x)  # 激活函数
        
        # 第二层卷积
        x = self.conv2(x)  # (B, N, conv_dim, T)
        # x = self.bn2(x)  # 批归一化
        x = self.act2(x)  # 激活函数

        x = x.permute(0, 2, 3, 1)  
        
        # 线性层
        x = self.fc(x) # (B, N, T, emb_dim)

        return x

class NodeEmbeddingNetwork2(nn.Module):
    def __init__(self, device, in_channels, node_feature_dim, max_value):
        super(NodeEmbeddingNetwork2, self).__init__()
        
        # 初始化 Embedding 层：输入维度为 1，输出维度为 node_feature_dim，最大特征值为 max_value
        self.embedding = nn.Linear(max_value + 1, node_feature_dim)
        self.device = device
        self.in_channels = in_channels
        self.node_feature_dim = node_feature_dim
        self.max_value = max_value

    def forward(self, x):
        # x 的形状为 [B, N, T, 1]
        
        # 将 x 转换为 one-hot 编码的形状 [B, N, T, max_value]
        x_int = x.permute(0, 1, 3, 2).squeeze(-1).long()  # 确保 x 是整数类型
        x_onehot = F.one_hot(x_int, num_classes=self.max_value + 1).float()
        
        # 使用 embedding 层将 one-hot 编码映射为目标维度 [B, N, T, OUTPUT_DIM]
        embedded = self.embedding(x_onehot).to(self.device)
        
        return embedded  
# class EdgeEmbeddingNetwork(nn.Module):
#     def __init__(self, DEVICE, input_dim=4, hidden_dim=32, output_dim=64):
#         super(EdgeEmbeddingNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.activation1 = nn.LeakyReLU()

#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.bn2 = nn.BatchNorm1d(output_dim)
#         self.activation2 = nn.LeakyReLU()
#         self.DEVICE = DEVICE

#         self.to(DEVICE)
#     def forward(self, x):
#         # x: N * 4
#         x = self.fc1(x)                 # N * hidden_dim
#         x = self.bn1(x)
#         x = self.activation1(x)

#         x = self.fc2(x)                 # N * output_dim
#         x = self.bn2(x)
#         x = self.activation2(x)

#         return x                        # N * output_dim

class EdgeUpdateNetwork(nn.Module):
    def __init__(self,DEVICE,
                 in_features,
                 num_features,
                 adjacency_matrix,
                 ratio=[2, 1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout
        self.adjacency_matrix = adjacency_matrix
        self.time_conv = nn.Conv2d(in_channels=12,
                                           out_channels=1,
                                           kernel_size=1)
        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                       out_channels=self.num_features_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            # layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
            #                                                 )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)
        # self.linear1 = torch.nn.Linear(64, 4)
        
        # if self.separate_dissimilarity:
        #     # layers
        #     layer_list = OrderedDict()
        #     for l in range(len(self.num_features_list)):
        #         # set layer
        #         layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
        #                                                    out_channels=self.num_features_list[l],
        #                                                    kernel_size=1,
        #                                                    bias=False)
        #         layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
        #                                                         )
        #         layer_list['relu{}'.format(l)] = nn.LeakyReLU()

        #         if self.dropout > 0:
        #             layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)

        #     layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
        #                                        out_channels=1,
        #                                        kernel_size=1)
        #     self.dsim_network = nn.Sequential(layer_list)
        self.DEVICE = DEVICE

        self.to(DEVICE)
    def forward(self, node_feat, edge_feat):
        # compute abs(x_i, x_j)
        Batch_size = node_feat.size(0)
        N_v = node_feat.size(1)
        T_len = node_feat.size(2)
        F_dim = node_feat.size(3)
        # start_time = time.time()
        # node_feat = node_feat.clone()
        # print(f"time1: {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        node_feat = self.time_conv(node_feat.permute(0,2,1,3)).squeeze(1)
        node_feat = node_feat.clone()
        # node_feat = node_feat.permute(0,2,1,3).reshape(Batch_size*T_len, N_v, F_dim)
        # node_feat = self.linear1(node_feat)
        # start_time = time.time()
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)
        # print(f"time2: {time.time() - start_time:.4f} seconds")
        # start_time = time.time()
        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = torch.sigmoid(self.sim_network(x_ij)).squeeze(1)
        # print(f"time3: {time.time() - start_time:.4f} seconds")
        # if self.separate_dissimilarity:
        #     dsim_val = F.sigmoid(self.dsim_network(x_ij))
        # else:
        #     dsim_val = 1.0 - sim_val

        # start_time = time.time()
        # 创建单位矩阵并直接在 DEVICE 上进行操作
        # eye_matrix1 = torch.eye(node_feat.size(1), device=self.DEVICE).unsqueeze(0)

        # # 计算 iag_mask，利用 expand 而非 repeat
        # diag_mask = (1.0 - eye_matrix1).expand(node_feat.size(0), node_feat.size(1), node_feat.size(1))
        diag_mask = self.adjacency_matrix.unsqueeze(0).expand(node_feat.size(0),-1,-1)
        # diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 4, 1, 1).to(self.DEVICE)
        # print(f"time4: {time.time() - start_time:.4f} seconds")
        # start_time = time.time()
        edge_feat = edge_feat * diag_mask
        merge_sum = torch.sum(edge_feat, -1, True)
        # print(f"time5: {time.time() - start_time:.4f} seconds")
        # set diagonal as zero and normalize
        # start_time = time.time()
        edge_feat = F.normalize(sim_val * edge_feat, p=1, dim=-1) * merge_sum
        # print(f"time6: {time.time() - start_time:.4f} seconds")
        # start_time = time.time()
        eye_matrix = torch.eye(node_feat.size(1), device=self.DEVICE).unsqueeze(0)

        # 拼接并复制一次
        force_edge_feat = eye_matrix.expand(node_feat.size(0),  -1, -1)

        # 如果不需要完全复制，可以考虑用 expand 而不是 repeat
        # force_edge_feat = force_edge_feat.expand(node_feat.size(0),  -1, -1)
        # force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0), torch.eye(node_feat.size(1)).unsqueeze(0),torch.eye(node_feat.size(1)).unsqueeze(0),torch.eye(node_feat.size(1)).unsqueeze(0)), 0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).to(self.DEVICE)
        # print(f"time7: {time.time() - start_time:.4f} seconds")
        # start_time = time.time()
        edge_feat = edge_feat + force_edge_feat
        # edge_feat = edge_feat + 1e-6
        # edge_feat = edge_feat / (torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 3, 1, 1))
        # print(f"time8: {time.time() - start_time:.4f} seconds")
        return edge_feat   

class MLP(nn.Module):
    def __init__(self, DEVICE,len_input, num_for_predict, input_dim, output_dim):
        """
        初始化 MLP 层
        
        参数:
        - input_dim: 输入特征的维度
        - hidden_dims: 一个列表，指定每个隐藏层的维度
        - output_dim: 输出特征的维度
        """
        super(MLP, self).__init__()
        
        # 定义全连接层列表
        layers = []
        self.final_conv = nn.Conv2d(len_input, num_for_predict, kernel_size=(1, 1))
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, int(input_dim/2)))
        layers.append(nn.ReLU()) 
        layers.append(nn.Linear(int(input_dim/2), output_dim))
        # layers.append(nn.ReLU())  # 激活函数
            
        # 输出层
        # layers.append(nn.Linear(input_dim/2, output_dim))
        
        # 将所有层组合成一个 Sequential 模块
        self.mlp = nn.Sequential(*layers)
        self.DEVICE = DEVICE

        self.to(DEVICE)
    def forward(self, x):
        x = self.final_conv(x.permute(0,2,1,3)).permute(0,2,1,3)
        return self.mlp(x)

class NodeUpdateNetwork(nn.Module):
    def __init__(self,DEVICE,
                 in_features,
                 num_features,number_of_graph,adjacency_matrix,
                 ratio=[2, 1],
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 2,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            # layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
            #                                                 )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)
        self.DEVICE = DEVICE

        self.to(DEVICE)
    def forward(self, node_feat, edge_feat):
        # get size
        #[B,N,T,F]
    
        Batch_size = node_feat.size(0)
        N_v = node_feat.size(1)
        T_len = node_feat.size(2)
        F_dim = node_feat.size(3)
        # num_edge = edge_feat.size(1)
        # get eye matrix (batch_size x 2 x node_size x node_size)
        # diag_mask = 1.0 - torch.eye(N_v).unsqueeze(0).unsqueeze(0).repeat(Batch_size,num_edge, 1, 1).to(self.DEVICE)
        diag_mask = (1.0 - torch.eye(N_v, device=self.DEVICE).unsqueeze(0))
        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)
        # compute attention and aggregate
        t1 = edge_feat.unsqueeze(1).expand(node_feat.size(0), node_feat.size(2), N_v ,N_v)
        t1 = t1.reshape(t1.shape[0]*t1.shape[1],t1.shape[2],t1.shape[3])
        t2 = node_feat.permute(0, 2, 1, 3).reshape(node_feat.size(0) * node_feat.size(2), node_feat.size(1), node_feat.size(3))
        # t1 = torch.cat(torch.split(edge_feat, 1, 1), 2).expand(node_feat.size(0), node_feat.size(2), num_edge*N_v ,N_v)
        # t1 = t1.reshape(t1.shape[0]*t1.shape[1],t1.shape[2],t1.shape[3])
        # t2 = node_feat.permute(0, 2, 1, 3).reshape(node_feat.size(0) * node_feat.size(2), node_feat.size(1), node_feat.size(3))
        aggr_feat = torch.bmm(t1, t2)
    
        aggr_feat = aggr_feat.reshape(node_feat.size(0), node_feat.size(2), aggr_feat.shape[1],aggr_feat.shape[2]).permute(0, 2, 1, 3)
        # node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(N_v, 1), -1)], -1).permute(0,2, 3, 1)
        node_feat = torch.cat([node_feat, aggr_feat], -1).permute(0,2, 3, 1)
        node_feat = self.network(node_feat.permute(0,2,1,3))
        node_feat = node_feat.permute(0,3,2,1)
        # non-linear transform
        # node_feat = self.network(node_feat.reshape(node_feat.shape[0]*node_feat.shape[1],node_feat.shape[2],node_feat.shape[3]).unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        # node_feat = node_feat.reshape(Batch_size,T_len,N_v,F_dim).permute(0,2,1,3)
        return node_feat

class TemporalHourEncoding(nn.Module):
    def __init__(self,DEVICE, feature_dim, max_len=100):
        super(TemporalHourEncoding, self).__init__()
        self.feature_dim = feature_dim

        # 创建一个 (max_len, feature_dim) 形状的张量以存储位置编码
        pe = torch.zeros(max_len, feature_dim)

        # 生成一个位置索引序列 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 定义一个缩放参数，控制频率
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(10000.0) / feature_dim))

        # 对偶数和奇数位置分别使用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 在第0维增加一个批次维度以匹配输入
        pe = pe.unsqueeze(0).unsqueeze(1)  # (1, 1, max_len, feature_dim)

        # 将位置编码设为不可训练的常量
        self.register_buffer('pe', pe)

        self.DEVICE = DEVICE

        # self.to(DEVICE)
    def forward(self, x, hours):
        # `x` 形状为 (batch_size, N, T, feature_dim)
        batch_size, num_nodes, num_timesteps, _ = x.shape

        hours = hours.long()
        # 通过 `hours` 张量筛选 `pe`
        # pe shape: (1, 1, max_len, feature_dim)
        # Select based on the hours index and expand to the (batch_size, num_nodes, T, feature_dim)
        position_encoded = self.pe[0, 0, hours].unsqueeze(1).expand(batch_size, num_nodes, num_timesteps, self.feature_dim)

        return position_encoded

class TemporalDayEncoding(nn.Module):
    def __init__(self, DEVICE,feature_dim, max_len=100):
        super(TemporalDayEncoding, self).__init__()
        self.feature_dim = feature_dim

        # 创建一个 (max_len, feature_dim) 形状的张量以存储位置编码
        pe = torch.zeros(max_len, feature_dim)

        # 生成一个位置索引序列 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 定义一个缩放参数，控制频率
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(20000.0) / feature_dim))

        # 对偶数和奇数位置分别使用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 在第0维增加一个批次维度以匹配输入
        pe = pe.unsqueeze(0).unsqueeze(1)  # (1, 1, max_len, feature_dim)

        # 将位置编码设为不可训练的常量
        self.register_buffer('pe', pe)
        self.DEVICE = DEVICE

        # self.to(DEVICE)
    def forward(self, x, hours):
        # `x` 形状为 (batch_size, N, T, feature_dim)
        batch_size, num_nodes, num_timesteps, _ = x.shape
        hours = hours.long()
        # 获取位置编码，并匹配节点数量和批次大小
        position_encoded = self.pe[0, 0, hours].unsqueeze(1).expand(batch_size, num_nodes, num_timesteps, self.feature_dim)

        return position_encoded

class MemoryPoolAttention(nn.Module):
    def __init__(self, DEVICE, num_prototypes, embedding_dim):
        super(MemoryPoolAttention, self).__init__()
        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim

        # 初始化 memory pool
        self.s_memory_pool = nn.Parameter(torch.rand(num_prototypes, embedding_dim))
        self.s_keyMatrix = nn.Parameter(torch.rand(num_prototypes, embedding_dim))  # M,C
        self.s_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.t_memory_pool = nn.Parameter(torch.rand(num_prototypes, embedding_dim))
        self.t_keyMatrix = nn.Parameter(torch.rand(num_prototypes, embedding_dim))  # M,C
        self.t_proj = nn.Linear(embedding_dim, embedding_dim)
        self.DEVICE = DEVICE

        # layer_list = OrderedDict()
        # for l in range(1):
        #     layer_list['conv{}'.format(l)] = nn.Conv2d(
        #         in_channels= embedding_dim*2,
        #         out_channels= embedding_dim,
        #         kernel_size=1,
        #         bias=False)
        #     layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=embedding_dim,
        #                                                     )
        #     layer_list['relu{}'.format(l)] = nn.LeakyReLU()

        # self.network = nn.Sequential(layer_list)
        self.to(DEVICE)
    def forward(self, x):
        """
        x: [B, N, T, D] 形状的输入节点特征
        """
        xs = x.permute(0,2,1,3)
        s_query = torch.tanh(self.s_proj(xs))

        att_weight = F.linear(input=s_query, weight=self.s_keyMatrix)# [N,D] by [M,D]^T --> [N,M]
        
        attention_weights = F.softmax(att_weight, dim=-1)  # 对P维度做softmax

        s_output = F.linear(attention_weights, self.s_memory_pool.permute(1, 0))  # [N,M] by [M,D]  --> [N,D]

        s_output = s_output.permute(0,2,1,3)
        
        t_query = torch.tanh(self.t_proj(x))

        t_att_weight = F.linear(input=t_query, weight=self.t_keyMatrix)# [T,D] by [M,D]^T --> [T,M]
        
        t_attention_weights = F.softmax(t_att_weight, dim=-1)  # 对P维度做softmax

        t_output = F.linear(t_attention_weights, self.t_memory_pool.permute(1, 0))  # [T,M] by [M,D]  --> [T,D]

        output = s_output + t_output +x
        # output = torch.cat([x,output],-1)
        # output = self.network(output.permute(0,3,1,2)).permute(0,2,3,1)
        return output

# class SubgraphPooling(nn.Module):
#     def __init__(self,DEVICE, num_angles, num_subgraphs):
#         super(SubgraphPooling, self).__init__()
#         self.num_angles = num_angles
#         self.num_subgraphs = num_subgraphs  # 子图数量，即 K
#         self.DEVICE = DEVICE

#         self.to(DEVICE)
#     def forward(self, output, node_labels):
#         """
#         输入:
#         - output: [B, N, T, F]，原始节点表示
#         - node_labels: [num_angles, N]，表示每个角度上节点所属的子图标签

#         输出:
#         - subgraph_representation: [B, num_angles, K, T, F]，子图级别的聚合表示
#         """
#         B, N, T, F = output.shape

#         # 初始化子图聚合表示张量
#         subgraph_representation = torch.zeros(B, self.num_angles, self.num_subgraphs, T, F, device=output.device)

#         # 遍历每个角度，进行子图聚合
#         for angle in range(self.num_angles):
#             # 获取当前角度的节点标签，大小为 [N]
#             labels = node_labels[angle]
            
#             # 根据每个子图标签进行聚合
#             for k in range(self.num_subgraphs):
#                 # 找到属于第 k 个子图的节点索引
#                 mask = (labels == k)  # [N] 的布尔掩码
#                 if mask.sum() == 0:  # 如果该子图没有任何节点，跳过
#                     continue
#                 selected_nodes = output[:, mask, :, :]  # [B, sum(mask), T, F]

#                 # 将节点聚合为子图表示，可以使用 mean 或 sum 聚合
#                 subgraph_representation[:, angle, k, :, :] = selected_nodes.mean(dim=1)

#         return subgraph_representation

class FusionLayer(nn.Module):
    def __init__(self, DEVICE, num_angles, feature_dim):
        """
        初始化FusionLayer.
        
        参数:
        - num_angles: int, 表示角度数量（如4）
        - feature_dim: int, 表示节点特征维度F
        """
        super(FusionLayer, self).__init__()
        self.num_angles = num_angles
        self.feature_dim = feature_dim
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, output, subgraph_representation, node_labels):
        """
        执行节点和子图特征的融合.
        
        参数:
        - output: [B, N, T, F]，原始节点特征
        - subgraph_representation: [B, num_angles, K, T, F]，子图特征
        - node_labels: [num_angles, N]，每个角度上节点所属的子图标签

        返回:
        - fused_representation: [B, N, T, F + 4 * F]，融合后的特征表示
        """
        B, N, T, F = output.shape

        # 初始化存储四种角度子图特征的张量 [B, N, T, 4 * F]
        subgraph_features = torch.zeros(B, N, T, self.num_angles * F, device=output.device)

        # 遍历每个角度，提取每个节点对应的子图特征
        for angle in range(self.num_angles):
            # 获取当前角度的子图编号，形状为 [N]
            node_subgraph_indices = node_labels[angle]  # [N]

            # 调整 node_subgraph_indices 的形状以便与 subgraph_representation 匹配
            node_subgraph_indices_expanded = node_subgraph_indices.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            node_subgraph_indices_expanded = node_subgraph_indices_expanded.expand(B, -1, T, F)

            # 从 subgraph_representation 中提取当前角度的子图特征，结果为 [B, N, T, F]
            angle_subgraph_features = torch.gather(subgraph_representation[:, angle], 1, node_subgraph_indices_expanded.to(torch.int64))

            # 将当前角度的子图特征放入 subgraph_features
            subgraph_features[:, :, :, angle * F: (angle + 1) * F] = angle_subgraph_features

        # 将原始节点特征 output 与子图特征 subgraph_features 拼接
        fused_representation = torch.cat([output, subgraph_features], dim=3)  # [B, N, T, F + 4 * F]

        return fused_representation
    
class FusionLayer2(nn.Module):
    def __init__(self, num_angles, num_subgraphs):
        """
        初始化融合层
        
        参数:
        - num_angles: 子图角度数，通常是 4
        - num_subgraphs: 子图的数量，通常是 10
        - feature_dim: 每个节点和子图的特征维度 F
        - hidden_dim: MLP 隐藏层的维度
        - output_dim: 输出的维度，即最终预测结果的维度
        """
        super(FusionLayer2, self).__init__()
        
        self.num_angles = num_angles
        self.num_subgraphs = num_subgraphs
    
    def forward(self, node_level_features, subgraph_level_features):
        """
        执行特征融合和预测
        
        参数:
        - node_level_features: [B, N, T, F]，节点级别的特征表示
        - subgraph_level_features: [B, 4, 10, T, F]，子图级别的特征表示
        
        返回:
        - prediction: [B, N, T]，预测的交通流量
        """
        B, N, T, F = node_level_features.shape
        B_sub, angle_num, subgraph_num, T_sub, F_sub = subgraph_level_features.shape
        
        # 确保时间维度一致
        assert T == T_sub, "Time dimensions should match between node and subgraph features."
        
        # 1. 展平子图级别的特征 [B, 4, 10, T, F] -> [B, N + 4 * 10, T, F]
        subgraph_level_features_flat = subgraph_level_features.view(B, self.num_angles * self.num_subgraphs, T, F)
        
        # 2. 拼接节点级别的特征和子图级别的特征 [B, N + 4 * 10, T, F]
        fused_features = torch.cat([node_level_features, subgraph_level_features_flat], dim=1)
        
        return fused_features
class OutputLayer(nn.Module):
    def __init__(self, num_nodes, feature_dim, hidden_dim):
        """
        初始化输出层
        
        参数:
        - num_nodes: 节点数 N
        - feature_dim: 特征维度 F
        - hidden_dim: MLP 隐藏层的维度
        """
        super(OutputLayer, self).__init__()
        
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        
        # MLP 层1: 将节点数 N + 4 * 10 映射回 N
        self.mlp1 = nn.Sequential(
            nn.Linear((num_nodes + 10 + 4 * 10), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes)  # 输出维度为 [B, N, T, F]
        )

        # MLP 层2: 将特征维度 F 映射成 1
        self.mlp2 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, fused_features):
        """
        执行输出层操作并产生最终预测
        
        参数:
        - fused_features: [B, N + 4 * 10, T, F]，融合后的特征
        
        返回:
        - prediction: [B, N, T]，交通流量预测
        """
        B, N_plus_subgraph, T, F = fused_features.shape
        
        # 1. 将 N + 4 * 10 的维度展平为 [B, (N + 4 * 10) * T, F]
        fused_features_permuted = fused_features.permute(0, 2, 3, 1)  # 交换后的形状: [B, T, F, N + 4 * 10]
        
        # 2. 展平，将 [B, T, N + 4 * 10, F] 变为 [B * N * T, F]
        # fused_features_flat = fused_features_permuted.reshape(B * N_plus_subgraph * T, F)  # [B * N * T, F]
        
        # 3. 通过 MLP1 映射回 [B * N * T, F]
        node_level_features = self.mlp1(fused_features_permuted)  # [B * N * T, F]
        
        # 4. 使用 MLP2 将每个时间步的 F 映射到 1，得到最终预测
        prediction = self.mlp2(node_level_features.permute(0, 3, 1, 2)).squeeze(3)  # [B * N * T, 1]
        
        return prediction

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        '''Align the input and output.
        '''
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1), similar to fc

    def forward(self, x):  # x: (n,c,l,v)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, self.c_out - self.c_in,  0, 0, 0, 0])
        return x  

class TemporalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        """
        Temporal GRU for modeling temporal dependencies.

        Args:
            input_dim (int): Number of input features (F).
            hidden_dim (int): Number of hidden units in GRU.
            num_layers (int): Number of GRU layers.
            dropout (float): Dropout rate.
        """
        super(TemporalGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer to project back to input_dim
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, T, F).

        Returns:
            Tensor: Output tensor of shape (B, N, T, F).
        """
        B, N, T, F = x.shape

        # Reshape to (B * N, T, F) for GRU layers
        x = x.reshape(B * N, T, F)

        # Pass through GRU
        output, _ = self.gru(x)  # output shape: (B * N, T, hidden_dim)

        # Project back to input_dim using a fully connected layer
        output = self.fc(output)  # shape: (B * N, T, F)

        # Reshape back to (B, N, T, F)
        output = output.reshape(B, N, T, F)

        return output

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers, dropout=0.1,max_len=100):
        """
        Temporal Transformer for modeling temporal dependencies.

        Args:
            input_dim (int): Number of input features (F).
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimension of feedforward layers.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
        """
        super(TemporalTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.positional_encoding = self._generate_positional_encoding(input_dim, max_len)
        # Transformer Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
    def _generate_positional_encoding(self,dim, max_len):
        """Generates sinusoidal positional encoding."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, max_len, dim)
        return pe

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, T, F).

        Returns:
            Tensor: Output tensor of shape (B, N, T, F).
        """
        B, N, T, F = x.shape

        # Add positional encoding along the temporal dimension
        positional_encoding = self.positional_encoding[:, :, :T, :].to(x.device)
        x = x + positional_encoding

        # Reshape to (B * N, T, F) for Transformer layers
        x = x.reshape(B * N, T, F)

        # Pass through Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Reshape back to (B, N, F, T)
        x = x.reshape(B, N, T, F)

        return x
    
# class TimeSeriesTransformer(nn.Module):
#     def __init__(self,DEVICE, in_channels, out_channels, num_heads, num_layers, dropout=0.1):
#         super(TimeSeriesTransformer, self).__init__()

#         # 输入特征维度，输出特征维度，以及 Transformer 配置
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.DEVICE = DEVICE

#         self.to(DEVICE)
#         # 线性层用于映射输入的特征维度到Transformer的特征维度
#         self.input_linear = nn.Linear(in_channels, out_channels)
#         # Transformer层
#         self.transformer = nn.Transformer(
#             d_model=out_channels,  # Transformer的特征维度
#             nhead=num_heads,       # 注意力头数
#             num_encoder_layers=num_layers,  # 编码器层数
#             num_decoder_layers=num_layers,  # 解码器层数
#             dropout=dropout        # Dropout率
#         )
#         # self.transformer2 = nn.Transformer(
#         #     d_model=out_channels,  # Transformer的特征维度
#         #     nhead=num_heads,       # 注意力头数
#         #     num_encoder_layers=num_layers,  # 编码器层数
#         #     num_decoder_layers=num_layers,  # 解码器层数
#         #     dropout=dropout        # Dropout率
#         # )
#         # 最后的线性层用于映射 Transformer 输出到目标输出维度
#         self.output_linear = nn.Linear(out_channels, in_channels)

#     def forward(self, x):
#         """
#         :param x: 输入形状 (B, N, T, F)
#         :return: 输出形状 (B, N, T, F)
#         """
#         B, N, T, F = x.shape
        
#         # 将输入 x 转换为 (B * N, T, F)，即 [批次大小 * 节点数, 时间步数, 特征维度]
#         x = x.permute(2, 0, 1, 3).reshape(T, B * N, F)

#         # 输入到线性层，映射到Transformer期望的特征维度
#         x = self.input_linear(x)  # 输出形状 (T, B * N, out_channels)

#         # Transformer 需要的形状是 [序列长度，批次大小，特征维度]，这里的 B * N 是批次大小
#         # 然后进行 Transformer 前向传播
#         x = self.transformer(x, x)  # Transformer的输入和输出通常是相同的（自注意力）
#         # x = self.transformer2(x, x)
#         # 将 Transformer 输出的形状还原回 (B, N, T, out_channels)
#         x = x.reshape(T, B, N, self.out_channels).permute(1, 2, 0, 3)

#         # 最后通过输出线性层映射回原始的特征维度
#         x = self.output_linear(x)  # 输出形状 (B, N, T, F)

#         return x
    

class CrossAttentionFusion(nn.Module):
    def __init__(self, DEVICE, input_dim, embed_dim, num_heads, dropout, num_representations=3):
        super(CrossAttentionFusion, self).__init__()
        self.num_representations = num_representations
        
        # 定义线性层用于生成 Q, K, V
        self.to_q = nn.Linear(input_dim, embed_dim)
        self.to_k = nn.Linear(input_dim, embed_dim)
        self.to_v = nn.Linear(input_dim, embed_dim)
        
        # 定义多头注意力
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # # 可训练权重，用于加权不同表示的注意力输出
        # self.fusion_weights = nn.Parameter(torch.ones(num_representations, dtype=torch.float32) / num_representations)
        
        # 最后的线性层
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, node_memory, subgraph_memory):
        """
        Args:
            node_memory: Tensor of shape [B, N, T, F], node embeddings.
            subgraph_memory: Tensor of shape [B, N, T, F, num_representations], subgraph embeddings.

        Returns:
            output: Tensor of shape [B, N, T, embed_dim], fused representation.
        """
        B, N, T, F = node_memory.shape
        num_representations = subgraph_memory.shape[-1] + 1
        
        # 生成 Q (来自 node_memory)
        Q = self.to_q(node_memory)  # [B, N, T, embed_dim]

        # 生成 K 和 V (来自 subgraph_memory)
        subgraph_memory = subgraph_memory.permute(0, 1, 2, 4, 3)  # [B, N, T, num_representations, F]
        subgraph_memory = torch.cat([node_memory.unsqueeze(3),subgraph_memory],-2)
        K = self.to_k(subgraph_memory)  # [B, N, T, num_representations, embed_dim]
        V = self.to_v(subgraph_memory)  # [B, N, T, num_representations, embed_dim]

        # 调整形状以适配多头注意力
        Q = Q.view(B * N * T, 1, -1)  # [B*N*T, 1, embed_dim]
        K = K.view(B * N * T, num_representations, -1)  # [B*N*T, num_representations, embed_dim]
        V = V.view(B * N * T, num_representations, -1)  # [B*N*T, num_representations, embed_dim]

        # 计算多头注意力输出
        attn_output, _ = self.multihead_attn(Q, K, V)  # [B*N*T, 1, embed_dim]
        attn_output = attn_output.squeeze(1)  # [B*N*T, embed_dim]

        # 恢复形状并通过线性层
        attn_output = attn_output.view(B, N, T, -1)  # [B, N, T, embed_dim]
        output = self.fc(attn_output)  # [B, N, T, embed_dim]

        return output
    
class CrossAttentionFusion1(nn.Module):
    def __init__(self, DEVICE, input_dim, embed_dim, num_heads, dropout, num_representations=4):
        super(CrossAttentionFusion1, self).__init__()
        self.num_representations = num_representations
        
        # 定义线性层用于生成 Q, K, V
        self.to_q = nn.Linear(input_dim, embed_dim)
        self.to_k = nn.Linear(input_dim, embed_dim)
        self.to_v = nn.Linear(input_dim, embed_dim)
        
        # 定义多头注意力
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # # 可训练权重，用于加权不同表示的注意力输出
        self.fusion_weights = nn.Parameter(torch.ones(num_representations, dtype=torch.float32) / num_representations)
        
        # 最后的线性层
        self.fc = nn.Linear(4*embed_dim, embed_dim)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, node_memory, subgraph_memory):
        """
        Args:
            node_memory: Tensor of shape [B, N, T, F], node embeddings.
            subgraph_memory: Tensor of shape [B, N, T, F, num_representations], subgraph embeddings.

        Returns:
            output: Tensor of shape [B, N, T, embed_dim], fused representation.
        """
        B, N, T, F = node_memory.shape
        num_representations = subgraph_memory.shape[-1]+1
        combined_emb = node_memory.unsqueeze(3)
        subgraph_memory = subgraph_memory.permute(0, 1, 2, 4, 3)  # [B, N, T, num_representations, F]
        combined_emb = torch.cat([combined_emb,subgraph_memory],-2)
        # 生成 Q (来自 node_memory)
        Q = self.to_q(combined_emb)  # [B, N, T, embed_dim]

        # 生成 K 和 V (来自 subgraph_memory)
        # subgraph_memory = subgraph_memory.permute(0, 1, 2, 4, 3)  # [B, N, T, num_representations, F]
        K = self.to_k(combined_emb)  # [B, N, T, num_representations, embed_dim]
        V = self.to_v(combined_emb)  # [B, N, T, num_representations, embed_dim]

        # 调整形状以适配多头注意力
        Q = Q.view(B * N * T, num_representations, -1)  # [B*N*T, 1, embed_dim]
        K = K.view(B * N * T, num_representations, -1)  # [B*N*T, num_representations, embed_dim]
        V = V.view(B * N * T, num_representations, -1)  # [B*N*T, num_representations, embed_dim]

        # 计算多头注意力输出
        attn_output, _ = self.multihead_attn(Q, K, V)  # [B*N*T, 1, embed_dim]
        # attn_output = attn_output.squeeze(1)  # [B*N*T, embed_dim]

        # 恢复形状并通过线性层
        attn_output = attn_output.reshape(B, N, T, num_representations*F)  # [B, N, T, embed_dim]
        output = self.fc(attn_output)  # [B, N, T, embed_dim]

        return output
class WeightedSubgraphAggregator(nn.Module):
    def __init__(self, DEVICE,adjacency_matrices, num_nodes, num_edgetype,embed_dim,len_input):
        super(WeightedSubgraphAggregator, self).__init__()
        # adjacency_matrices 是一个 [3, N, N] 的常量，表示 3 种类型的边
        self.adjacency_matrices = torch.tensor(adjacency_matrices, dtype=torch.float32, device=DEVICE)
        self.num_edgetype = num_edgetype
        # 定义一个可学习的线性权重矩阵，用于每个节点的三种异质子图的加权
        # self.weight = nn.Parameter(torch.rand(num_nodes, num_edgetype), requires_grad=True)  # [N, 3] 的可学习权重参数
        self.edge_embeddings = nn.Parameter(torch.rand(num_edgetype, num_nodes, len_input,  embed_dim), requires_grad=True)  # [3, embed_dim]
        # 最后的非线性激活函数
        # self.memory_pools = nn.ModuleList([
        #     MemoryPoolAttention(DEVICE, num_prototypes, embed_dim) 
        #     for _ in range(num_edgetype+1)
        # ])
        self.Linears = nn.ModuleList([
            nn.Linear(embed_dim,embed_dim)
            for _ in range(num_edgetype)
        ])
        # self.activation = nn.ReLU()
        self.DEVICE = DEVICE
        self.node_norm = nn.LayerNorm(embed_dim)
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_edgetype)
        ])
        self.to(DEVICE)
    def forward(self, node_feat, node_weights,edgepre_his,edgepre_pref):
        """
        node_feat: [B, N, T, F] -> 节点特征
        node_weights: [B, N, N] -> 节点间的权重
        返回每个节点的子图聚合表示: [B, N, T, F]
        """
        # 获取输入的形状
        B, N, T, C = node_feat.shape
        node_feat = self.node_norm(node_feat)
        t2 = node_feat.permute(0, 2, 1, 3).reshape(B * T, N, C)
        # 初始化子图特征表示列表
        subgraph_features = []

        # 遍历 3 种边类型，计算每种类型下的子图表示
        for edge_type in range(self.num_edgetype):
            # 取当前边类型的邻接矩阵
            if(edge_type<self.adjacency_matrices.shape[0]):
                adj_matrix = self.adjacency_matrices[edge_type]  # [N, N]
            elif(edge_type==self.adjacency_matrices.shape[0]+1):
                adj_matrix = edgepre_his
            else:
                adj_matrix = edgepre_pref
            # 将邻接矩阵扩展到批次维度，并对非邻居节点进行掩码
            # adj_masked [B, N, N] 0 表示非邻居，1 表示邻居
            adj_masked = adj_matrix.unsqueeze(0).expand(B, -1, -1)  # 扩展到批次维度

            # 使用邻接矩阵掩码 node_weights，掩去非邻居和自身
            masked_weights = node_weights * adj_masked  # [B, N, N]
            normalized_weights = F.normalize(masked_weights, p=1, dim=-1, eps=1e-6)
            # 对邻居特征加权求和
            t1 = normalized_weights.unsqueeze(1).expand(B, T, N ,N)
            t1 = t1.reshape(B * T, N, N)
            weighted_neighbors = torch.bmm(t1,t2)  # [B*T, N,F]
            weighted_neighbors = weighted_neighbors.view(B, T, N, C).permute(0,2,1,3)  # 恢复形状 [B, N, T, F]
            # subgraph_features.append(weighted_neighbors)
            # weighted_neighbors = self.network(weighted_neighbors).permute(0,2,3,1)
            edge_embedding = self.edge_embeddings[edge_type]
            edge_embedding = self.Linears[edge_type](edge_embedding)
            fused_features = weighted_neighbors + edge_embedding.unsqueeze(0)
            # 保存每种类型的子图表示
            # fused_memory = self.memory_pools[edge_type](fused_features)
            # fused_memory = fused_features
            fused_features = self.norms[edge_type](fused_features)
            subgraph_features.append(fused_features)  # 列表中每个元素是 [B, N, T, F]

        # 聚合三种异质子图
        subgraph_features = torch.stack(subgraph_features, dim=-1)  # [B, N, T, F, 3]
        # origin_memory = self.memory_pools[edge_type+1](node_feat)
        origin_memory = node_feat
        # 根据每个节点的可学习权重进行加权
        # 使用权重 [N, 3] 对子图特征进行加权和
        # weights = F.softmax(self.weight, dim=-1).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, 1, 1, N, 3]
        # weighted_subgraphs = torch.sum(subgraph_features * weights, dim=-1)  # [B, N, T, F]

        # # 应用非线性激活函数
        # subgraph_aggregate = self.activation(weighted_subgraphs)

        return origin_memory,subgraph_features
    
class ContrastiveLossModule(torch.nn.Module):
    def __init__(self, DEVICE,N, num_subgraph,edge_features,temperature=0.5):
        super(ContrastiveLossModule, self).__init__()
        self.temperature = temperature
        self.DEVICE = DEVICE
        self.neighbor_mask = self._create_neighbor(N, num_subgraph,edge_features).to(self.DEVICE)
        # self.to(DEVICE)
        self.pos_mask = self._create_pos_mask(N, num_subgraph).to(self.DEVICE)
        self.neg_mask = self._create_neg_mask(N, num_subgraph, self.neighbor_mask, self.pos_mask).to(self.DEVICE)
    def _create_neighbor(self, N, num_subgraph, edge_features):
        # pos_mask = torch.zeros(N, num_subgraph * N)
        pos_mask = edge_features.permute(1,0,2).reshape(N , num_subgraph*N)
        return pos_mask
    def _create_pos_mask(self, N, num_subgraph):
        """
        创建正样本掩码，形状为 [N, num_subgraph * N]。
        """
        pos_mask = torch.zeros(N, num_subgraph * N)
        for n in range(N):
            for s1 in range(num_subgraph):
                idx_s1 = n
                idx_s2 = n + s1 * N
                pos_mask[idx_s1, idx_s2] = 1
        return pos_mask
    def _create_neg_mask(self, N, num_subgraph, neighbor_mask, pos_mask):
        """
        预先构建负样本掩码
        """
        neg_mask = torch.zeros_like(pos_mask)
        
        # 计算负样本掩码
        for b in range(N):
            # 邻居负样本掩码
            neighbor_indices = neighbor_mask[b].nonzero(as_tuple=True)
            num_neighbor_negatives = num_subgraph*7  # 假设选择的负样本数量为10
            selected_neighbor_negatives = torch.randperm(len(neighbor_indices[0]))[:num_neighbor_negatives]
            neg_mask[b][neighbor_indices[0][selected_neighbor_negatives]] = 1

            # 非邻居负样本掩码
            non_neighbor_mask = 1 - neighbor_mask[b] - pos_mask[b]
            non_neighbor_indices = non_neighbor_mask.nonzero(as_tuple=True)
            num_non_neighbor_negatives = num_subgraph*3  # 假设选择的负样本数量为10
            selected_non_neighbor_negatives = torch.randperm(len(non_neighbor_indices[0]))[:num_non_neighbor_negatives]
            neg_mask[b][non_neighbor_indices[0][selected_non_neighbor_negatives]] = 1
            
        return neg_mask
    def forward(self, node_memory,subgraph_embeddings):
        """
        subgraph_embeddings: 张量形状为 [B, N, T, F, num_subgraph]
        adjacency_matrices: 每个子图的邻接矩阵，形状为 [num_subgraph, N, N]
        """
        B, N, T, C, num_subgraph = subgraph_embeddings.size()

        # 合并 B 和 T 维度，得到新的批次维度 B*T
        embeddings = subgraph_embeddings.permute(0,2,1,3,4).reshape(B * T, N, C, num_subgraph)

        # 调整维度 [B*T, num_subgraph, N, F] 方便后续操作
        embeddings = embeddings.permute(0, 3, 1, 2)

        # 合并成 [B*T, num_subgraph*N, F]
        embeddings = embeddings.reshape(B * T, N *num_subgraph, C)

        # 沿特征维度进行L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        node_memory = node_memory.permute(0,2,1,3).reshape(B * T, N, C)
        # 计算相似性矩阵，形状为 [B*T, num_subgraph * N, num_subgraph * N]
        similarity_matrix = torch.matmul(node_memory, embeddings.transpose(1, 2))

        # 创建正样本掩码，形状为 [num_subgraph * N*T, num_subgraph * N*T]
        # 同一时间步内同一节点在不同子图的表示为正样本
        pos_mask = self.pos_mask.unsqueeze(0).expand(B * T, -1, -1)
        neg_mask = self.neg_mask.unsqueeze(0).expand(B * T, -1, -1)
        
        # 计算相似性的指数并除以温度参数
        exp_sim = torch.exp(similarity_matrix / self.temperature)

        # 计算分母：对每行进行求和
        denominator = (exp_sim * (pos_mask + neg_mask)).sum(dim=-1)
        # 计算对比损失：基于正样本掩码和负样本掩码
        numerator = (exp_sim * pos_mask).sum(dim=-1)
        contrastive_loss = -torch.log(numerator / (denominator + 1e-12)).mean()

        # # 添加负样本损失（可选择性添加，但建议负样本对不应该太相似）
        # contrastive_loss += torch.log(exp_sim / (denominator + 1e-10)) * neg_mask

        # # 平均损失
        # contrastive_loss = contrastive_loss.sum() / (B * N * T)

        return contrastive_loss

class SubgraphMOE(nn.Module):
    def __init__(self, DEVICE,feature_dim, num_prototypes, num_experts, top_k):
        """
        MOE 模块
        :param feature_dim: 输入特征维度 F
        :param num_subgraphs: 子图数量 C
        :param num_experts: 专家数量 M
        :param top_k: Top-k 分流数量
        """
        super(SubgraphMOE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(feature_dim , num_experts)
        self.routing_norm = nn.LayerNorm(self.num_experts)  # 初始化时添加
        # self.experts = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(num_experts)])  # M 个专家
        self.memory_pools = nn.ModuleList([
            MemoryPoolAttention(DEVICE, num_prototypes, feature_dim) 
            for _ in range(num_experts)
        ])
        self.to(DEVICE)
    def forward(self, maingraph,subgraph):
        """
        :param subgraph: 子图表征 [B, N, T, F, C]
        :return: 分流后的表征 [B, N, T, F, M]
        """
        B, N, T, D, C = subgraph.shape

        # Step 1: 展平子图维度并拼接
        subgraph_flattened = subgraph.permute(0, 4, 1, 2, 3).reshape(B, C * N, T, D)  # [B, C*N, T, F]
        input = torch.cat([maingraph, subgraph_flattened], dim=1)  # [B, (C+1)N, T, F]
        # Step 2: 通过 Router 计算分流概率
        routing_logits = self.router(input)  # [B, (C+1)N, T, M]
        routing_logits = self.routing_norm(routing_logits)
        routing_probs = F.softmax(routing_logits, dim=-1)  # [B, (C+1)N, T, M]

        # Step 3: 按照 Top-k 分流
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)  # [B, (C+1)N, T, k]
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) +1e-8) # [B, (C+1)N, T, k]
        # Step 4: 获取专家输出
        expert_outputs = torch.stack([expert(input) for expert in self.memory_pools], dim=-1)  # [B, (C+1)N, T, F, M]

        # Step 5: 选择 Top-k 专家并加权聚合
        top_k_expert_outputs = torch.gather(
            expert_outputs, -1, top_k_indices.unsqueeze(-2).expand(-1, -1, -1, D, -1)
        )  # [B, (C+1)N, T, F, k]
        weighted_outputs = (top_k_expert_outputs * top_k_probs.unsqueeze(-2)).sum(dim=-1)  # [B, (C+1)N, T, F]

        # Step 6: 分离主图和子图的输出
        maingraph_embeddings = weighted_outputs[:, :N]  # [B, N, T, F]
        subgraph_embeddings = weighted_outputs[:, N:]  # [B, C*N, T, F]
        subgraph_embeddings = subgraph_embeddings.reshape(B, C, N, T, D).permute(0, 2, 3, 4, 1)  # [B, N, T, F, C]


        return maingraph_embeddings,subgraph_embeddings # [B, (C+1)N, T, F]

class LinkPredictor(nn.Module):
    def __init__(self, in_feat, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feat * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, nodefeat):
        """
        nodefeat: [B, N, T, F]
        returns: [B, N, N] logits
        """
        B, N, T, F = nodefeat.shape
        # Step 1: Aggregate temporal info (e.g., average)
        h = nodefeat.mean(dim=2)  # [B, N, F]
        
        # Step 2: Construct all pairs of node embeddings
        h_i = h.unsqueeze(2).expand(B, N, N, F)  # [B, N, N, F]
        h_j = h.unsqueeze(1).expand(B, N, N, F)  # [B, N, N, F]
        h_pair = torch.cat([h_i, h_j], dim=-1)   # [B, N, N, 2F]
        
        # Step 3: Predict link logits
        logits = self.mlp(h_pair).squeeze(-1)    # [B, N, N]
        return logits
        
class MVSTFN(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, node_feature_dim, edge_feature_dim, num_of_graph, num_of_global_vertices,edge_features,adjacency_matrix,num_prototypes,len_input,num_for_predict,num_of_expert,top_k):
        super(MVSTFN, self).__init__()
        
        # Initialize the four MVSTFN submodules
        # Embedding、节点和边传播模块、子图聚合、超节点聚合、transformer模块、MLP层
        self.node_embedding1 = NodeEmbeddingNetwork(DEVICE,in_channels,node_feature_dim,node_feature_dim)
        # self.node_embedding2 = NodeEmbeddingNetwork2(DEVICE,in_channels,node_feature_dim,max_value)
        # self.node_embedding3 = Align(in_channels, node_feature_dim)
        # self.edge_embedding = EdgeEmbeddingNetwork(DEVICE,input_dim=4, hidden_dim=32, output_dim=64)
        self.linkpre1 = LinkPredictor(in_feat=node_feature_dim, hidden_dim=node_feature_dim*2)
        self.linkpre2 = LinkPredictor(in_feat=node_feature_dim, hidden_dim=node_feature_dim*2)
        self.submodule1 = MVSTFN_submodule(DEVICE, nb_block, node_feature_dim, edge_feature_dim, num_of_global_vertices,num_of_graph,adjacency_matrix)
        self.time_encode_hour = TemporalHourEncoding(DEVICE,node_feature_dim, max_len=48)
        self.time_encode_week = TemporalDayEncoding(DEVICE,node_feature_dim, max_len=7)
        # self.subgraph_poolling = SubgraphPooling(DEVICE,num_of_graph,num_of_subgraph,node_labels,dropout=0.1)
        self.cross_attn_layer_hour = CrossAttentionFusion(DEVICE,input_dim=node_feature_dim,embed_dim=node_feature_dim, num_heads=4, dropout=0.1)
        self.cross_attn_layer_day = CrossAttentionFusion(DEVICE,input_dim=node_feature_dim,embed_dim=node_feature_dim, num_heads=4, dropout=0.1)
        # self.cross_attn_layer2 = CrossAttention(DEVICE,embed_dim=node_feature_dim, num_heads=4, dropout=0.1)
        # self.cross_attn_layer1 = CrossAttentionFusion1(DEVICE,input_dim=node_feature_dim,embed_dim=node_feature_dim, num_heads=4, dropout=0.1)
        # self.supernode_pooling = SupernodeAssignmentLayer(DEVICE,feature_dim=node_feature_dim, dropout=0.1, latend_num=10)
        self.subgraph_aggregator = WeightedSubgraphAggregator(DEVICE,edge_features, num_of_global_vertices,num_of_graph,node_feature_dim,len_input)
        self.subgraph_moe = SubgraphMOE(DEVICE,node_feature_dim, num_prototypes, num_of_expert, top_k)
        self.contrastiveLoss = ContrastiveLossModule(DEVICE,num_of_global_vertices,num_of_graph,edge_features,temperature=0.5)
        
        # self.memorylearning = MemoryPoolAttention(DEVICE, num_prototypes, node_feature_dim)
        # self.fusion_layer = FusionLayer(DEVICE, num_angles=num_of_graph, feature_dim=node_feature_dim)
        # self.fusion_layer2 = FusionLayer2(num_angles=num_of_graph, num_subgraphs=num_of_subgraph)
        # self.gru = TemporalGRU(input_dim=node_feature_dim, hidden_dim=128, num_layers=2, dropout=0.1)
        # self.transformer1 = TimeSeriesTransformer(DEVICE,in_channels=node_feature_dim, out_channels=128, num_heads=4, num_layers=4)
        self.transformer = TemporalTransformer(input_dim= node_feature_dim, num_heads=4, ff_dim=128, num_layers=3)
        self.final_mlp = MLP(DEVICE, len_input, num_for_predict,node_feature_dim, 1)
        # MLP层1: 将节点和子图特征拼接后从 N + 4 * 10 映射到 N
        # self.output_layer = OutputLayer(num_nodes=num_of_global_vertices, feature_dim= node_feature_dim, hidden_dim= node_feature_dim*2)
        # self.edge_features = nn.Parameter(data=torch.rand(num_of_graph, num_of_global_vertices, num_of_global_vertices),requires_grad=True)
        self.edge_features = nn.Parameter(
            data=torch.rand(num_of_global_vertices, num_of_global_vertices),
            requires_grad=True
        )
        self.adj_matrix = adjacency_matrix
        # self.edge_features.data = F.normalize(self.edge_features.data, p=2, dim=-1)
        self.t_proj = nn.Linear(node_feature_dim*2, node_feature_dim)
        # self.t_proj = nn.Linear(node_feature_dim*10, node_feature_dim)
        self.to(DEVICE)
    def forward(self, data,  timestamps):       
        # Step 1: Node embedding
        # start_time = time.time()
        node_emb = self.node_embedding1(data)  # [B,N,T,F]
        # print(f"Node embedding time: {time.time() - start_time:.4f} seconds")
        # node_emb = self.node_embedding3(data).permute(0,1,3,2)
        # edge_features = F.relu(self.edge_features*self.adj_matrix).unsqueeze(0).repeat(node_emb.size(0), 1, 1)
        norm_edge = F.normalize(self.edge_features * self.adj_matrix + 1e-6, p=2, dim=-1)
        edge_features = F.relu(norm_edge).unsqueeze(0).repeat(node_emb.size(0), 1, 1)

        # Step 3: Submodule 1
        # start_time = time.time()
        nodefeat,edgefeat = self.submodule1(node_emb, edge_features)

        edgepre_his = self.linkpre1(nodefeat)
        edgepre_pref = self.linkpre2(nodefeat)
        # hours, weekdays = time_encoding(timestamps)
        # position_encoded_hour = self.time_encode_hour(nodefeat, hours)
     
        # position_encoded_day = self.time_encode_week(nodefeat, weekdays)
        # fused_output = nodefeat + position_encoded_day +position_encoded_hour
        # print(f"Submodule1 time: {time.time() - start_time:.4f} seconds")
        # fused_output = nodefeat
        # start_time = time.time()
        maingraph_feat,subgraph_feat = self.subgraph_aggregator(nodefeat,edgefeat,edgepre_his,edgepre_pref)
        # maingraph_moefeat, subgraph_moefeat = maingraph_feat,subgraph_feat
        maingraph_moefeat, subgraph_moefeat =self.subgraph_moe(maingraph_feat,subgraph_feat)
        # fused_output = node_memory
        # print(f"subgraph_aggregator time: {time.time() - start_time:.4f} seconds")
        # start_time = time.time()
        # start_time = time.time()
        contrastive_loss = self.contrastiveLoss(maingraph_moefeat, subgraph_moefeat)
        # contrastive_loss = torch.tensor(0.001)
        # print(f"contrastiveLoss time: {time.time() - start_time:.4f} seconds")
        # start_time = time.time()
        hours, weekdays = time_encoding(timestamps)     

        position_encoded_hour = self.time_encode_hour(maingraph_moefeat, hours)
     
        position_encoded_day = self.time_encode_week(maingraph_moefeat, weekdays)

        node_memory_hour = maingraph_moefeat + position_encoded_hour
        node_memory_day = maingraph_moefeat + position_encoded_day
        subgraph_memory_hour = subgraph_moefeat + position_encoded_hour.unsqueeze(-1)
        subgraph_memory_day = subgraph_moefeat + position_encoded_day.unsqueeze(-1)
        # B=data.size(0)
        # fused_output = torch.tanh(self.t_proj(torch.cat([node_memory_hour.unsqueeze(-1),node_memory_day.unsqueeze(-1),subgraph_memory_hour,subgraph_memory_day],-1).reshape(B,817,12,64*10)))
        subgraph_memory = torch.cat([subgraph_memory_hour,subgraph_memory_day],-1)
        fused_output_hour = self.cross_attn_layer_hour(node_memory_hour, subgraph_memory)
        fused_output_day = self.cross_attn_layer_day(node_memory_day, subgraph_memory)
        fused_output = torch.tanh(self.t_proj(torch.cat([fused_output_hour,fused_output_day],-1)))
        # print(f"st-fusion time: {time.time() - start_time:.4f} seconds")
        # fused_output = self.cross_attn_layer_hour(maingraph_moefeat, subgraph_moefeat)
        # start_time = time.time()
        trans_output = self.transformer(fused_output)
        # print(f"transformer time: {time.time() - start_time:.4f} seconds")
        # trans_output = self.gru(fused_output)
        # print(f"contrastiveLoss time: {time.time() - start_time:.4f} seconds")
        # # print(f"Submodule1 time: {time.time() - start_time:.4f} seconds")
        # # start_time = time.time()
        # # subgraph_representation = self.subgraph_poolling(output)
        # # # # print(f"Subgraph pooling time: {time.time() - start_time:.4f} seconds")
        # fused_output = self.cross_attn_layer1(output,subgraph_representation)
        # # start_time = time.time()
        # latentnode = self.memorylearning(nodefeat)
        # latentnode = self.supernode_pooling(nodefeat)
        # # # # # print(f"Latent node pooling time: {time.time() - start_time:.4f} seconds")
        # # # # # # fused_output = self.cross_attn_layer2(output,latentnode)
        # fused_output = torch.cat((node_memory, subgraph_memory), dim=3)
        # start_time = time.time()
        # fused_output = self.cross_attn_layer(node_memory, subgraph_memory)
        # print(f"cross_attn time: {time.time() - start_time:.4f} seconds")
        # start_time = time.time()
        # latentnode = self.supernode_pooling(output)
        # print(f"Meta-node pooling time: {time.time() - start_time:.4f} seconds")
        # fusion_node = torch.cat([output,subgraph_representation.reshape(size(0),size(1)*size(2),size(3),size(4)),latentnode.permute(0,1,3,2)], dim=1)
        # _, N, _, _ = output.shape
        # # start_time = time.time()
        # # gru_output, hidden_output = self.gru(output.permute(2, 0, 1, 3).reshape(T, B * N, F))
        # # print(f"GRU time: {time.time() - start_time:.4f} seconds")
        # # Step 4: Time encoding
        # start_time = time.time()
       
        # print(f"Transformer time: {time.time() - start_time:.4f} seconds")

        # # Step 9: Transformer 2
        # start_time = time.time()
        # trans_output = self.transformer2(trans_output)
        # print(f"Transformer2 time: {time.time() - start_time:.4f} seconds")

        # Step 10: Final MLP
        # start_time = time.time()
        final_output = self.final_mlp(trans_output).squeeze(3)    
        # print(f"Output layer time: {time.time() - start_time:.4f} seconds")
        # print(" ")
        return final_output,contrastive_loss,edgepre_his,edgepre_pref

class MVSTFN_submodule(nn.Module):

    def __init__(self, DEVICE, nb_block, node_feature_dim, edge_feature_dim, num_of_vertices,num_of_graph,adjacency_matrix):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        super(MVSTFN_submodule, self).__init__()

        self.BlockList = nn.ModuleList([MVSTFN_block(DEVICE, node_feature_dim, edge_feature_dim, num_of_vertices,num_of_graph,0.1,adjacency_matrix)])

        self.BlockList.extend([MVSTFN_block(DEVICE, node_feature_dim, edge_feature_dim, num_of_vertices,num_of_graph, 0.0,adjacency_matrix) for _ in range(nb_block-1)])

        # self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, edge_features):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        for block in self.BlockList:
            x , edge_features = block(x, edge_features)
        
        # output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return x, edge_features

class MVSTFN_block(nn.Module):

    def __init__(self, DEVICE, node_feature_dim, edge_feature_dim, num_of_vertices,number_of_graph,dropout,adjacency_matrix):
        super(MVSTFN_block, self).__init__()
        self.edge2node_net = NodeUpdateNetwork(DEVICE, in_features=node_feature_dim,
                                              num_features=node_feature_dim,number_of_graph=number_of_graph,
                                              adjacency_matrix=adjacency_matrix,
                                              dropout=dropout)
        # self.node2edge_net = EdgeUpdateNetwork(DEVICE,in_features=node_feature_dim,
        #                                       num_features=edge_feature_dim,
        #                                       dropout=dropout)
        self.node2edge_net = EdgeUpdateNetwork(DEVICE,in_features=node_feature_dim,
                                              num_features=edge_feature_dim,
                                              adjacency_matrix=adjacency_matrix,
                                              dropout=dropout)
        self.residual_conv = nn.Conv2d(node_feature_dim, node_feature_dim, kernel_size=(1, 1), stride=(1, 1))
        self.ln = nn.LayerNorm(node_feature_dim)  #需要将channel放到最后一个维度上

    def forward(self, x, edge_features):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_timesteps,num_of_features = x.shape
        # start_time = time.time()
        edge_feat = self.node2edge_net(x, edge_features)
        # print(f"Edge update time: {time.time() - start_time:.4f} seconds")
        # start_time = time.time()
        node_feat = self.edge2node_net(x, edge_feat) #(b,N,T,F)
        # print(f"Node update time: {time.time() - start_time:.4f} seconds")
        # edge_feat = self.node2edge_net(node_feat, edge_features)
        x_residual = self.residual_conv(x.permute(0, 3, 1, 2))  # (b,N,T,F)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)
        x_residual = self.ln(F.relu(x_residual + node_feat.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)).permute(0, 2, 1, 3)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,T,F)
        return x_residual, edge_feat
        # return node_feat, edge_feat
        
def make_model(DEVICE, seed, nb_block, in_channels, node_dim, edge_dim, num_of_graph, num_of_vertices,edge_features,adjacency_matrix,num_prototypes,len_input,num_for_predict,num_of_expert,top_k):

    model = MVSTFN(DEVICE, nb_block, in_channels, node_dim, edge_dim, num_of_graph, num_of_vertices,edge_features,adjacency_matrix,num_prototypes,len_input,num_for_predict,num_of_expert,top_k)
    seed = seed
    torch.manual_seed(seed)  # 固定 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)
    for p in model.parameters():
        if p.dim() > 1 and p is not model.edge_features:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model
