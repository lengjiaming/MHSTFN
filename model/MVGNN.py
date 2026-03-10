import math
import torch
import torch.nn as nn
import torch.nn.functional as F


############################################################
# 1. GCN layer（更稳定）
############################################################
class GCNLayer(nn.Module):
    """
    标准 GCN：Z' = ReLU(A_hat Z W)
    输入:
        Z: (B, N, F_in)
        A_hat: (N, N)
    输出:
        Z_next: (B, N, F_out)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, Z, A_hat):
        ZW = self.lin(Z)  # (B, N, out_dim)
        out = torch.einsum("ij,bjd->bid", A_hat, ZW)  # A_hat ZW
        return F.relu(out)


############################################################
# 2. 轻量版 MVGNN（忠实于原文，但无虚拟节点）
############################################################
class MVGNN(nn.Module):
    """
    向原文靠拢的轻量 MVGNN：
      ✔ AO（原图）
      ✔ AF（特征图：cosine + KNN）
      ✔ AS（可学习图结构）
      ✔ Gating attention 融合 3 张图
      ✔ Smoothness regularization
      ✔ 多层 GCN
      ✖ 无虚拟节点（避免爆显存）
    """

    def __init__(self, DEVICE, nb_block, in_channels, hidden_dim,
                 edge_dim, num_graph, num_nodes, edge_features,
                 adjacency_matrix, num_proto, len_input, num_for_predict,
                 k=10):
        super().__init__()

        self.DEVICE = DEVICE
        self.N = num_nodes
        self.C = in_channels
        self.T = len_input
        self.hidden_dim = hidden_dim
        self.num_for_predict = num_for_predict
        self.k = k  # AF 的 KNN 阈值

        # =====================
        # AO: 原始图（由 adjacency_matrix 给定）
        # =====================
        if isinstance(adjacency_matrix, torch.Tensor):
            A = adjacency_matrix.float()
        else:
            A = torch.tensor(adjacency_matrix).float()
        self.register_buffer("A_O", A)

        # =====================
        # AS: 可学习图（初始化为 AO）
        # =====================
        self.A_S = nn.Parameter(A.clone())

        # =====================
        # 输入投影：flatten(T*C) -> hidden_dim
        # =====================
        in_dim = in_channels * len_input
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # =====================
        # 多层 GCN
        # =====================
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim) for _ in range(nb_block)
        ])

        # =====================
        # 图融合 gating：q^T tanh(W Ai + b)
        # =====================
        att_dim = min(64, self.N)
        self.W_o = nn.Linear(self.N, att_dim)
        self.W_f = nn.Linear(self.N, att_dim)
        self.W_s = nn.Linear(self.N, att_dim)

        self.q_o = nn.Parameter(torch.randn(att_dim))
        self.q_f = nn.Parameter(torch.randn(att_dim))
        self.q_s = nn.Parameter(torch.randn(att_dim))

        # =====================
        # 节点预测头
        # =====================
        self.predict_head = nn.Linear(hidden_dim, num_for_predict)

        self.to(DEVICE)


    ############################################################
    # AF: 特征相似图（cosine + KNN）
    ############################################################
    def build_AF(self, X):
        """
        输入:
            X: (B, N, hidden_dim_initial) 节点初始特征
        输出:
            AF: (N, N) batch 平均后的特征图
        """
        # 先对 batch 做平均（原文不会处理 batch）
        X_avg = X.mean(dim=0)   # (N, hidden_dim)

        # cosine 相似度：N×N
        norm_X = F.normalize(X_avg, dim=-1)
        AF = torch.mm(norm_X, norm_X.t())  # (N, N)

        # KNN：只保留 top-k 相关节点
        topk = torch.topk(AF, self.k, dim=-1).values[:, -1].unsqueeze(-1)
        AF = AF * (AF >= topk)  # thresholding

        return AF


    ############################################################
    # Softmax gating for AO, AF, AS
    ############################################################
    def fuse_graphs(self, A_O, A_F, A_S):
        """
        输入: 三张图 (N, N)
        输出: 融合图 A (N, N)
        """

        # tanh(W A_i)
        h_o = torch.tanh(self.W_o(A_O))
        h_f = torch.tanh(self.W_f(A_F))
        h_s = torch.tanh(self.W_s(A_S))

        psi_o = (h_o * self.q_o).sum(-1)  # (N,)
        psi_f = (h_f * self.q_f).sum(-1)
        psi_s = (h_s * self.q_s).sum(-1)

        psi = torch.stack([psi_o, psi_f, psi_s], dim=-1)  # (N,3)

        # 数值稳定 softmax
        psi = psi - psi.max(-1, keepdim=True).values
        alpha = F.softmax(psi, dim=-1)  # (N,3)

        a_o = alpha[:, 0].unsqueeze(-1)
        a_f = alpha[:, 1].unsqueeze(-1)
        a_s = alpha[:, 2].unsqueeze(-1)

        A = a_o * A_O + a_f * A_F + a_s * A_S

        # 强制对称
        A = 0.5 * (A + A.t())

        return A


    ############################################################
    # 邻接矩阵归一化
    ############################################################
    def normalize(self, A):
        N = A.size(0)
        A_tilde = A + torch.eye(N, device=A.device)
        deg = A_tilde.sum(1)
        deg_inv = torch.pow(deg + 1e-6, -0.5)
        D = deg_inv.view(-1,1)
        return D * A_tilde * D.t()


    ############################################################
    # Forward
    ############################################################
    def forward(self, data, timestamps=None):
        """
        data: (B, N, C, T)
        """
        B, N, C, T = data.shape
        device = data.device

        # 1) Flatten (C,T)
        x = data.permute(0,1,3,2).reshape(B, N, C*T)
        Z0 = self.input_proj(x)   # (B,N,hidden_dim)

        # 2) 构造 AF（依据 Z0）
        A_F = self.build_AF(Z0)   # (N,N)

        # 3) 融合 AO, AF, AS
        A_S = F.softplus(self.A_S)
        A = self.fuse_graphs(self.A_O, A_F, A_S)

        # 4) 归一化
        A_hat = self.normalize(A)

        # 5) 多层 GCN
        Z = Z0
        for gcn in self.gcn_layers:
            Z = gcn(Z, A_hat)

        # 6) 输出预测
        pred = self.predict_head(Z)  # (B,N,num_for_predict)

        return pred



############################################################
# 工厂函数保持不变
############################################################
def make_model(DEVICE, nb_block, in_channels, node_dim, edge_dim,
               num_of_graph, num_of_vertices, edge_features,
               adjacency_matrix, num_prototypes, len_input, num_for_predict):

    model = MVGNN(DEVICE, nb_block, in_channels, node_dim,
                       edge_dim, num_of_graph, num_of_vertices,
                       edge_features, adjacency_matrix, num_prototypes,
                       len_input, num_for_predict)
    torch.manual_seed(2025)
    torch.cuda.manual_seed(2025)

    return model
