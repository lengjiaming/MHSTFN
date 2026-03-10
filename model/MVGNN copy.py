import math
import torch
import torch.nn as nn
import torch.nn.functional as F


############################################################
# 1. 简单 GCN 层：基于稠密归一化邻接矩阵
############################################################

class GraphConvLayer(nn.Module):
    """
    简单的 GCN 层：Z^{(l)} = A_hat @ (Z^{(l-1)} W^{(l)})
    输入:
        Z: (B, N, F_in)
        A_hat: (N, N)  归一化邻接矩阵
    输出:
        Z_next: (B, N, F_out)
    """
    def __init__(self, in_dim, out_dim, activation=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.activation = activation

    def forward(self, Z, A_hat):
        # Z: (B, N, F_in)
        ZW = self.lin(Z)  # (B, N, F_out)
        # A_hat @ ZW，批处理方式用 einsum
        Z_next = torch.einsum('ij,bjd->bid', A_hat, ZW)  # (B, N, F_out)
        if self.activation:
            Z_next = F.relu(Z_next)
        return Z_next


############################################################
# 2. 轻量版 MVGNN：多图融合 + 多层 GCN（无虚拟节点）
############################################################

class MVGNNBaseline(nn.Module):
    """
    轻量版 MVGNN 时空预测 baseline：
      - __init__ 参数与 HGTBaseline / MVSTFN 一致（未用参数直接忽略）
      - forward(data, timestamps=None) -> final_output
      - final_output: (B, N, num_for_predict)
      - 不使用虚拟节点增强，显存开销远小于原版 MVGNN
    """

    def __init__(self, DEVICE, nb_block, in_channels, node_feature_dim,
                 edge_feature_dim, num_of_graph, num_of_global_vertices,
                 edge_features, adjacency_matrix, num_prototypes,
                 len_input, num_for_predict):
        super(MVGNNBaseline, self).__init__()

        self.DEVICE = DEVICE
        self.num_nodes = num_of_global_vertices   # N
        self.len_input = len_input                # T
        self.in_channels = in_channels            # C
        self.num_for_predict = num_for_predict
        self.hidden_dim = node_feature_dim        # GCN 隐层维度

        # =======================
        # 1) 处理多图 adjacency: AO, AF, AS
        # =======================
        if isinstance(edge_features, torch.Tensor):
            adj = edge_features.detach().float()
        else:
            adj = torch.tensor(edge_features, dtype=torch.float32)

        if adj.dim() == 3:
            # 形状: (num_of_graph, N, N)
            assert adj.size(1) == self.num_nodes and adj.size(2) == self.num_nodes, \
                "adjacency_matrix 维度与 num_of_global_vertices 不一致"
            A_O = adj[0]  # 原始图
            if adj.size(0) > 1:
                A_F = adj[1]  # 第二张图当作 feature graph
            else:
                A_F = A_O.clone()
        elif adj.dim() == 2:
            # 只有一张图，就同时当作 AO 和 AF
            assert adj.size(0) == self.num_nodes and adj.size(1) == self.num_nodes, \
                "adjacency_matrix 维度与 num_of_global_vertices 不一致"
            A_O = adj
            A_F = adj.clone()
        else:
            raise ValueError("adjacency_matrix 必须是 2D 或 3D 张量")

        # 注册为 buffer（不参与训练）
        self.register_buffer("A_O", A_O)  # (N, N)
        self.register_buffer("A_F", A_F)  # (N, N)

        # 自适应图 AS：可学习参数，初始化为 (AO + AF) / 2
        A_S_init = (A_O + A_F) / 2.0
        self.A_S = nn.Parameter(A_S_init)  # (N, N)

        # =======================
        # 2) 多图 attention 权重模块（节点级）
        # =======================
        # 使用线性层: R^{N} -> R^{d_att}，再用 q 向量打分
        att_dim = min(64, self.num_nodes)  # 可以调，先取一个较小的
        self.att_O_W = nn.Linear(self.num_nodes, att_dim)
        self.att_F_W = nn.Linear(self.num_nodes, att_dim)
        self.att_S_W = nn.Linear(self.num_nodes, att_dim)
        self.att_O_q = nn.Parameter(torch.randn(att_dim))
        self.att_F_q = nn.Parameter(torch.randn(att_dim))
        self.att_S_q = nn.Parameter(torch.randn(att_dim))

        # =======================
        # 3) 输入投影 + 多层 GCN
        # =======================
        in_dim_node = in_channels * len_input     # flatten 时间维度：T*C
        self.input_proj = nn.Linear(in_dim_node, self.hidden_dim)

        # nb_block 层 GCN
        self.gcn_layers = nn.ModuleList()
        for layer_id in range(nb_block):
            self.gcn_layers.append(GraphConvLayer(self.hidden_dim,
                                                  self.hidden_dim,
                                                  activation=True))

        # =======================
        # 4) 输出头：每个节点 -> num_for_predict
        # =======================
        self.predict_head = nn.Linear(self.hidden_dim, num_for_predict)

        self.to(DEVICE)

    # =========================================================
    # 多图 attention 融合，得到 fused adjacency A (N, N)
    # =========================================================
    def fuse_graphs(self):
        """
        返回融合后的邻接矩阵 A: (N, N)
        """
        A_O = self.A_O  # (N, N)
        A_F = self.A_F  # (N, N)
        A_S = F.softplus(self.A_S)


        device = A_S.device
        A_O = A_O.to(device)
        A_F = A_F.to(device)

        # 对每个节点 i，用其邻接行 Ai: (N,) 计算 attention 分数
        h_O = torch.tanh(self.att_O_W(A_O))  # (N, att_dim)
        h_F = torch.tanh(self.att_F_W(A_F))  # (N, att_dim)
        h_S = torch.tanh(self.att_S_W(A_S))  # (N, att_dim)

        # ψ_i = q^T h_i
        psi_O = (h_O * self.att_O_q).sum(dim=-1)  # (N,)
        psi_F = (h_F * self.att_F_q).sum(dim=-1)  # (N,)
        psi_S = (h_S * self.att_S_q).sum(dim=-1)  # (N,)

        # 按节点 i 对三个图做 softmax
        psi_stack = torch.stack([psi_O, psi_F, psi_S], dim=-1)  # (N, 3)
        alpha = F.softmax(psi_stack, dim=-1)                    # (N, 3)

        alpha_O = alpha[:, 0].unsqueeze(-1)  # (N,1)
        alpha_F = alpha[:, 1].unsqueeze(-1)  # (N,1)
        alpha_S = alpha[:, 2].unsqueeze(-1)  # (N,1)

        # 融合时对每个节点 i 的行进行加权
        A = alpha_O * A_O + alpha_F * A_F + alpha_S * A_S  # (N, N)

        # 对称化，防止数值不对称
        A = 0.5 * (A + A.t())

        return A

    # =========================================================
    # 邻接矩阵对称归一化: A_hat = D^{-1/2} (A + I) D^{-1/2}
    # =========================================================
    @staticmethod
    def normalize_adjacency(A):
        """
        A: (N, N)
        返回: A_hat: (N, N)
        """
        device = A.device
        N = A.size(0)
        A_tilde = A + torch.eye(N, device=device, dtype=A.dtype)
        deg = A_tilde.sum(dim=1)  # (N,)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D_inv_sqrt = deg_inv_sqrt.view(-1, 1)
        A_hat = D_inv_sqrt * A_tilde * D_inv_sqrt.t()
        return A_hat

    # =========================================================
    # 前向传播：时空预测 baseline
    # =========================================================
    def forward(self, data, timestamps=None):
        """
        :param data: tensor(B, N, C, T)
        :param timestamps: 占位符，不使用，仅保持接口一致
        :return: final_output: tensor(B, N, num_for_predict)
        """
        # data: (B, N, C, T)
        B, N, C, T = data.shape
        assert N == self.num_nodes, "num_of_global_vertices 与 data.shape[1] 不一致"
        assert T == self.len_input, "len_input 与 data.shape[-1] 不一致"
        assert C == self.in_channels, "in_channels 与 data.shape[2] 不一致"

        device = data.device

        # 1) 展平时间维度： (B, N, C, T) -> (B, N, T, C) -> (B, N, T*C)
        x = data.permute(0, 1, 3, 2).contiguous()  # (B, N, T, C)
        x = x.view(B, N, T * C)                   # (B, N, in_dim_node)

        # 2) 输入投影到隐藏维度：Z0: (B, N, hidden_dim)
        Z = self.input_proj(x)  # (B, N, hidden_dim)

        # 3) 多图融合得到融合邻接矩阵 A: (N, N)
        A = self.fuse_graphs().to(device)   # (N, N)

        # 4) 对 A 做对称归一化，得到 A_hat: (N, N)
        A_hat = self.normalize_adjacency(A)  # (N, N)

        # 5) 多层 GCN
        for gcn in self.gcn_layers:
            Z = gcn(Z, A_hat)  # (B, N, hidden_dim)

        # 6) 节点预测头：(B, N, hidden_dim) -> (B, N, num_for_predict)
        final_output = self.predict_head(Z)

        return final_output


############################################################
# 3. 工厂函数：与 HGT.py 的 make_model 接口一致
############################################################

def make_model(DEVICE, nb_block, in_channels, node_dim, edge_dim,
               num_of_graph, num_of_vertices, edge_features,
               adjacency_matrix, num_prototypes, len_input, num_for_predict):

    model = MVGNNBaseline(DEVICE, nb_block, in_channels, node_dim,
                          edge_dim, num_of_graph, num_of_vertices,
                          edge_features, adjacency_matrix, num_prototypes,
                          len_input, num_for_predict)
    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
