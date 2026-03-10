import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import function as fn
from dgl.nn import HeteroGraphConv, edge_softmax
from dgl.utils import expand_as_pair


############################################################
# 1. 原始 HGT 模块：Attention / Layer / Model
############################################################

class HGTAttention(nn.Module):
    def __init__(self, out_dim, num_heads, k_linear, q_linear, v_linear,
                 w_att, w_msg, mu):
        """
        :param out_dim: 输出特征维数 d_out
        :param num_heads: 注意力头数 K
        :param k_linear: nn.Linear(d_in, d_out)
        :param q_linear: nn.Linear(d_in, d_out)
        :param v_linear: nn.Linear(d_in, d_out)
        :param w_att: nn.Parameter(K, d_out/K, d_out/K)
        :param w_msg: nn.Parameter(K, d_out/K, d_out/K)
        :param mu: nn.Parameter(K,)
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads

        self.k_linear = k_linear
        self.q_linear = q_linear
        self.v_linear = v_linear
        self.w_att = w_att
        self.w_msg = w_msg
        self.mu = mu

    def forward(self, g, feat):
        """
        :param g: DGLGraph 二分图（只包含一种关系）
        :param feat:
            - tensor(N_src, d_in)
            或
            - (tensor(N_src, d_in), tensor(N_dst, d_in))
        :return: tensor(N_dst, d_out)
        """
        with g.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, g)
            # feat_src: (N_src, d_in), feat_dst: (N_dst, d_in)

            # 线性映射 + 多头拆分
            k = self.k_linear(feat_src).view(-1, self.num_heads, self.d_k)  # (N_src, K, d_k)
            v = self.v_linear(feat_src).view(-1, self.num_heads, self.d_k)  # (N_src, K, d_k)
            q = self.q_linear(feat_dst).view(-1, self.num_heads, self.d_k)  # (N_dst, K, d_k)

            # 关系特定变换
            k = torch.einsum('nhi,hij->nhj', k, self.w_att)  # (N_src, K, d_k)
            v = torch.einsum('nhi,hij->nhj', v, self.w_msg)  # (N_src, K, d_k)

            g.srcdata.update({'k': k, 'v': v})
            g.dstdata['q'] = q

            # 边上点积注意力分数
            g.apply_edges(fn.v_dot_u('q', 'k', 't'))  # g.edata['t']: (E, K, 1)
            attn = g.edata.pop('t').squeeze(-1)       # (E, K)
            attn = attn * self.mu / math.sqrt(self.d_k)
            attn = edge_softmax(g, attn)              # (E, K)
            g.edata['t'] = attn.unsqueeze(-1)         # (E, K, 1)

            # 消息传递：v * attn，按 dst 聚合
            g.update_all(fn.u_mul_e('v', 't', 'm'), fn.sum('m', 'h'))
            out = g.dstdata['h'].view(-1, self.out_dim)  # (N_dst, d_out)
            return out


class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads,
                 ntypes, etypes, dropout=0.2, use_norm=True):
        """
        :param in_dim: 输入特征维数
        :param out_dim: 输出特征维数
        :param num_heads: 注意力头数
        :param ntypes: 顶点类型列表
        :param etypes: List[(str, str, str)] 规范边类型列表
        """
        super().__init__()
        d_k = out_dim // num_heads

        # 按节点类型的 K/Q/V 线性变换
        k_linear = {ntype: nn.Linear(in_dim, out_dim) for ntype in ntypes}
        q_linear = {ntype: nn.Linear(in_dim, out_dim) for ntype in ntypes}
        v_linear = {ntype: nn.Linear(in_dim, out_dim) for ntype in ntypes}

        # 按关系类型的参数
        w_att = {r[1]: nn.Parameter(torch.Tensor(num_heads, d_k, d_k)) for r in etypes}
        w_msg = {r[1]: nn.Parameter(torch.Tensor(num_heads, d_k, d_k)) for r in etypes}
        mu = {r[1]: nn.Parameter(torch.ones(num_heads)) for r in etypes}

        self.reset_parameters(w_att, w_msg)

        # 每种关系一个 HGTAttention，聚合方式为 mean
        self.conv = HeteroGraphConv({
            etype: HGTAttention(
                out_dim, num_heads,
                k_linear[stype], q_linear[dtype], v_linear[stype],
                w_att[etype], w_msg[etype], mu[etype]
            )
            for stype, etype, dtype in etypes
        }, 'mean')

        self.a_linear = nn.ModuleDict({
            ntype: nn.Linear(out_dim, out_dim) for ntype in ntypes
        })
        self.skip = nn.ParameterDict({
            ntype: nn.Parameter(torch.ones(1)) for ntype in ntypes
        })
        self.drop = nn.Dropout(dropout)

        self.use_norm = use_norm
        if use_norm:
            self.norms = nn.ModuleDict({
                ntype: nn.LayerNorm(out_dim) for ntype in ntypes
            })

    def reset_parameters(self, w_att, w_msg):
        for etype in w_att:
            nn.init.xavier_uniform_(w_att[etype])
            nn.init.xavier_uniform_(w_msg[etype])

    def forward(self, g, feats):
        """
        :param g: 异构图
        :param feats: {ntype: tensor(N_i, d_in)}
        :return: {ntype: tensor(N_i, d_out)}
        """
        if g.is_block:
            feats_dst = {
                ntype: feats[ntype][:g.num_dst_nodes(ntype)]
                for ntype in feats
            }
        else:
            feats_dst = feats

        with g.local_scope():
            # 1) 异构互注意力 + 消息传递
            hs = self.conv(g, (feats, feats))  # {ntype: (N_i, d_out)}

            # 2) 残差 + 线性 + Dropout (+ LayerNorm)
            out_feats = {}
            for ntype in g.dsttypes:
                if g.num_dst_nodes(ntype) == 0:
                    continue
                alpha = torch.sigmoid(self.skip[ntype])  # (1,)
                trans_out = self.drop(self.a_linear[ntype](hs[ntype]))
                out = alpha * trans_out + (1 - alpha) * feats_dst[ntype]
                out_feats[ntype] = self.norms[ntype](out) if self.use_norm else out

            return out_feats


class HGT(nn.Module):
    def __init__(self, in_dims, hidden_dim, out_dim, num_heads,
                 ntypes, etypes, predict_ntype,
                 num_layers, dropout=0.2, use_norm=True):
        """
        :param in_dims: Dict[str, int] 顶点类型到输入特征维数
        :param hidden_dim: 隐含特征维数
        :param out_dim: 输出特征维数
        :param num_heads: 注意力头数
        :param ntypes: 顶点类型列表
        :param etypes: 规范边类型列表
        :param predict_ntype: 需要输出的顶点类型
        :param num_layers: HGT 层数
        """
        super().__init__()
        self.predict_ntype = predict_ntype

        self.adapt_fcs = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim)
            for ntype, in_dim in in_dims.items()
        })

        self.layers = nn.ModuleList([
            HGTLayer(hidden_dim, hidden_dim, num_heads,
                     ntypes, etypes, dropout, use_norm)
            for _ in range(num_layers)
        ])

        self.predict = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, feats):
        """
        :param g: DGL 异构图
        :param feats: {ntype: (N_i, d_in)}
        :return: (N_pred, out_dim)
        """
        # 维度对齐
        hs = {
            ntype: F.gelu(self.adapt_fcs[ntype](feats[ntype]))
            for ntype in feats
        }  # {ntype: (N_i, hidden_dim)}

        # 多层 HGT
        for layer in self.layers:
            hs = layer(g, hs)

        # 只对 predict_ntype 做输出头
        out = self.predict(hs[self.predict_ntype])  # (N_pred, out_dim)
        return out


############################################################
# 2. HGT 时空预测 baseline，接口与 MVSTFN 一致
############################################################

class HGTBaseline(nn.Module):
    """
    只用 HGT 做空间编码的时空预测 baseline：
      - __init__ 参数与 MVSTFN 一致（未使用的参数会忽略）
      - forward(data, timestamps) -> final_output
      - final_output: (B, N, num_for_predict)
      - 不返回对比学习 loss
    """
    def __init__(self, DEVICE, nb_block, in_channels, node_feature_dim,
                 edge_feature_dim, num_of_graph, num_of_global_vertices,
                 edge_features, adjacency_matrix, num_prototypes,
                 len_input, num_for_predict):
        super(HGTBaseline, self).__init__()

        self.DEVICE = DEVICE
        self.num_nodes = num_of_global_vertices   # N
        self.num_of_graph = num_of_graph          # 关系数（你现在是 4）
        self.len_input = len_input                # 历史长度 T
        self.in_channels = in_channels
        self.num_for_predict = num_for_predict

        # 1) 构建只有一个节点类型 'node'，多关系 ('node', 'relX', 'node') 的异构图
        self.ntypes = ['node']
        self.g_single, self.etypes = self.build_heterograph(edge_features)

        # 2) HGT 模型设置
        in_dim_node = in_channels * len_input      # 把时间维 T 展平作为特征
        self.hidden_dim = node_feature_dim         # HGT 隐含/输出维度

        # 头数：尽量用 4，否则退化为 1 保证能整除
        if self.hidden_dim % 4 == 0:
            num_heads = 4
        else:
            num_heads = 1

        self.hgt = HGT(
            in_dims={'node': in_dim_node},
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            num_heads=num_heads,
            ntypes=self.ntypes,
            etypes=self.etypes,
            predict_ntype='node',
            num_layers=nb_block,
            dropout=0.2,
            use_norm=True
        )

        # 3) 简单预测头：每个节点 embedding -> num_for_predict 步
        self.predict_head = nn.Linear(self.hidden_dim, num_for_predict)

        self.to(DEVICE)

    def build_heterograph(self, adjacency_matrix):
        """
        adjacency_matrix: 期望形状为 [num_of_graph, N, N]
        为每个关系 r 建一个 ('node', f'rel{r}', 'node') 类型的边，
        非零位置视为有边（只用拓扑，不用边权）。
        """
        if isinstance(adjacency_matrix, torch.Tensor):
            adj = adjacency_matrix.detach().cpu()
        else:
            adj = torch.tensor(adjacency_matrix)

        num_rel, N, _ = adj.shape
        data_dict = {}
        etypes = []

        for r in range(num_rel):
            etype_name = f'rel{r}'
            etypes.append(('node', etype_name, 'node'))

            # 使用 >0 位置作为边（你也可以改成 >= 某个阈值）
            src, dst = (adj[r] > 0).nonzero(as_tuple=True)
            src = src.numpy()
            dst = dst.numpy()
            data_dict[('node', etype_name, 'node')] = (src, dst)

        g = dgl.heterograph(data_dict, num_nodes_dict={'node': N})
        return g, etypes

    def forward(self, data, timestamps=None):
        """
        :param data: tensor(B, N, T, in_channels)
        :param timestamps: tensor(B, T)（这里只是占位，不使用）
        :return: final_output: tensor(B, N, num_for_predict)
        """
        # data: (B, N, T, C)
        B, N, C, T = data.shape
        assert N == self.num_nodes, "num_of_global_vertices 与 data.shape[1] 不一致"
        assert T == self.len_input, "len_input 与 data.shape[2] 不一致"
        assert C == self.in_channels, "in_channels 与 data.shape[3] 不一致"
        device = data.device
        # 1) 节点特征展平时间维： (B, N, T, C) -> (B, N, T*C)
        x = data.permute(0,1,3,2).view(B, N, T * C)          # (B, N, in_dim_node)
        # 再展平 batch 维 + 节点维： (B, N, in_dim) -> (B*N, in_dim)
        x_flat = x.view(B * N, T * C)

        # 2) 复制图结构 B 份：dgl.batch
        g_batch = dgl.batch([self.g_single] * B).to(device)

        # 3) HGT 编码：输入 {ntype: (num_nodes_total, in_dim)}
        h_flat = self.hgt(g_batch, {'node': x_flat})   # (B*N, hidden_dim)

        # 4) reshape 回 (B, N, hidden_dim)
        h = h_flat.view(B, N, self.hidden_dim)

        # 5) 节点级预测头： (B, N, hidden_dim) -> (B, N, num_for_predict)
        final_output = self.predict_head(h)

        # baseline：不再返回对比学习 loss
        return final_output

        
def make_model(DEVICE, nb_block, in_channels, node_dim, edge_dim, num_of_graph, num_of_vertices,edge_features,adjacency_matrix,num_prototypes,len_input,num_for_predict):

    model = HGTBaseline(DEVICE, nb_block, in_channels, node_dim, edge_dim, num_of_graph, num_of_vertices,edge_features,adjacency_matrix,num_prototypes,len_input,num_for_predict)
    seed = 2025
    torch.manual_seed(seed)  # 固定 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)
    for p in model.parameters():
        if p.dim() > 1 :
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model
