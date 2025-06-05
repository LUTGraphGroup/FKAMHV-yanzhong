import random
import os
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import *
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_torch(seed=1234)



class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        return G @ x

class HGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(HGCN, self).__init__()
        self.hgc_layer = HGNN_conv(in_dim, out_dim)

    def forward(self, x, G):
        x = self.hgc_layer(x, G)
        x = F.leaky_relu(x, 0.25)
        return x


# ========= 创新型 多头 HGNN Layer ==========
class MultiHeadInnovative_HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, num_heads=8, bias=True, use_edge_attention=True, use_feature_recalibration=True):
        super(MultiHeadInnovative_HGNN_conv, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.num_heads = num_heads
        self.use_edge_attention = use_edge_attention
        self.use_feature_recalibration = use_feature_recalibration

        # 多头权重
        self.weights = nn.ParameterList([
            Parameter(torch.Tensor(in_ft, out_ft)) for _ in range(num_heads)
        ])

        # 每个 head 独立邻接矩阵（动态调整）
        self.head_edge_attn = nn.Parameter(torch.Tensor(num_heads)) if use_edge_attention else None

        # SE Block 风格的特征重标定
        if self.use_feature_recalibration:
            self.fc_recalib = nn.Sequential(
                nn.Linear(out_ft, out_ft // 4),
                nn.ReLU(),
                nn.Linear(out_ft // 4, out_ft),
                nn.Sigmoid()
            )

        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)

        self.norm = nn.LayerNorm(out_ft)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            stdv = 1. / math.sqrt(weight.size(1))
            weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        if self.head_edge_attn is not None:
            self.head_edge_attn.data.fill_(1.0)

    def forward(self, x, G):
        """
        x: [N, in_ft]
        G: [N, N] adjacency matrix of hypergraph
        """
        head_outputs = []

        for i, weight in enumerate(self.weights):
            # Head-specific 邻接矩阵
            if self.head_edge_attn is not None:
                G_head = G * self.head_edge_attn[i]  # 动态调节
            else:
                G_head = G

            # 卷积操作
            out = G_head.matmul(x.matmul(weight))
            head_outputs.append(out)

        # 多头平均
        x_conv = sum(head_outputs) / self.num_heads

        if self.bias is not None:
            x_conv += self.bias

        # 特征重标定
        if self.use_feature_recalibration:
            recalib_weight = self.fc_recalib(x_conv)
            x_conv = x_conv * recalib_weight

        x_conv = self.norm(x_conv)

        return x_conv

# ========= 创新型 HGNN 网络 ==========
class Innovative_HGCN(nn.Module):
    def __init__(self, in_dim, hidden_list, num_layers=15, num_heads=8, dropout=0.5, use_edge_attention=True, use_feature_recalibration=True, use_jk=True):
        super(Innovative_HGCN, self).__init__()
        self.dropout = dropout
        self.use_jk = use_jk

        self.layers = nn.ModuleList()
        prev_dim = in_dim

        for _ in range(num_layers):
            for hidden_dim in hidden_list:
                self.layers.append(MultiHeadInnovative_HGNN_conv(
                    prev_dim, hidden_dim, num_heads=num_heads,
                    use_edge_attention=use_edge_attention,
                    use_feature_recalibration=use_feature_recalibration
                ))
                prev_dim = hidden_dim

        if self.use_jk:
            self.jk_proj = nn.Linear(len(self.layers) * prev_dim, prev_dim)

    def forward(self, x, G):
        x_embed = x
        all_outputs = []

        for layer in self.layers:
            x_embed_new = layer(x_embed, G)
            x_embed_new = F.leaky_relu(x_embed_new, 0.25)
            x_embed_new = F.dropout(x_embed_new, self.dropout, training=self.training)

            x_embed = x_embed_new

            if self.use_jk:
                all_outputs.append(x_embed_new)

        if self.use_jk:
            x_out = torch.cat(all_outputs, dim=-1)
            x_out = self.jk_proj(x_out)
        else:
            x_out = x_embed

        return x_out


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret




class FKAMHV(nn.Module):
    def __init__(self, mi_num, dis_num, hidd_list, num_proj_hidden, hyperpm, num_in_node=374, num_in_edge=788,
                 num_hidden1=512, num_out=64, num_layers=15):
        super(FKAMHV, self).__init__()


        # 使用FastKAN改进的特征预处理层
        self.mi_fastkan = FastKANLayer(
            input_dim=mi_num + dis_num,
            output_dim=hidd_list[0],
            num_grids=8,
            grid_min=-2,
            grid_max=2,
            use_base_update=True
        )

        self.dis_fastkan = FastKANLayer(
            input_dim=dis_num + mi_num,
            output_dim=hidd_list[0],
            num_grids=8,
            grid_min=-2,
            grid_max=2,
            use_base_update=True
        )

        self.hgcn_mi = Innovative_HGCN(
            in_dim=hidd_list[0],
            hidden_list=hidd_list,
            dropout=0.5,
            use_edge_attention=True,
            use_feature_recalibration=True,
            use_jk=True,
            num_layers=num_layers
        )

        self.hgcn_dis = Innovative_HGCN(
            in_dim=hidd_list[0],
            hidden_list=hidd_list,
            dropout=0.5,
            use_edge_attention=True,
            use_feature_recalibration=True,
            use_jk=True,
            num_layers=num_layers
        )


        # 保留原始VAE组件
        self.node_encoders1 = node_encoder(num_in_edge, num_hidden1, 0.3).to(device)
        self.hyperedge_encoders1 = hyperedge_encoder(num_in_node, num_hidden1, 0.3).to(device)
        self.decoder2 = decoder2(act=lambda x: x).to(device)
        self._enc_mu_node = node_encoder1(num_hidden1, num_out, 0.3, act=lambda x: x).to(device)
        self._enc_log_sigma_node = node_encoder1(num_hidden1, num_out, 0.3, act=lambda x: x).to(device)
        self._enc_mu_hedge = hyperedge_encoder1(num_hidden1, num_out, 0.3, act=lambda x: x).to(device)
        self._enc_log_sigma_hyedge = hyperedge_encoder1(num_hidden1, num_out, 0.3, act=lambda x: x).to(device)

        # 改进的特征转换层（保持维度不变）
        self.linear_x = nn.Sequential(
            nn.Linear(hidd_list[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.linear_y = nn.Sequential(
            nn.Linear(hidd_list[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def sample_latent(self, z_node, z_hyperedge):
        # 保持原有实现不变
        self.z_node_mean = self._enc_mu_node(z_node)
        self.z_node_log_std = self._enc_log_sigma_node(z_node)
        self.z_node_std = torch.exp(self.z_node_log_std)
        z_node_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_node_std.size())).double().to(device)
        self.z_node_ = self.z_node_mean + self.z_node_std.mul(Variable(z_node_std_, requires_grad=True))

        self.z_edge_mean = self._enc_mu_hedge(z_hyperedge)
        self.z_edge_log_std = self._enc_log_sigma_hyedge(z_hyperedge)
        self.z_edge_std = torch.exp(self.z_edge_log_std)
        z_edge_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_edge_std.size())).double().to(device)
        self.z_hyperedge_ = self.z_edge_mean + self.z_edge_std.mul(Variable(z_edge_std_, requires_grad=True))

        return self.z_node_, self.z_hyperedge_ if self.training else (self.z_node_mean, self.z_edge_mean)

    def forward(self, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_dis_Kn, AT, A):
        # 使用FastKAN进行特征增强
        mi_embedded = self.mi_fastkan(concat_mi_tensor)
        dis_embedded = self.dis_fastkan(concat_dis_tensor)

        # 通过超图卷积
        mi_feature = self.hgcn_mi(mi_embedded, G_mi_Kn)
        dis_feature = self.hgcn_dis(dis_embedded, G_dis_Kn)


        # 特征转换
        x = self.linear_x(mi_feature)
        y = self.linear_y(dis_feature)

        # 计算异构结果
        result_h = x.mm(y.t())  # float32

        # VAE部分保持不变
        z_node_encoder = self.node_encoders1(concat_mi_tensor)  # 注意使用原始输入
        z_hyperedge_encoder = self.hyperedge_encoders1(concat_dis_tensor)
        self.z_node_s, self.z_hyperedge_s = self.sample_latent(z_node_encoder, z_hyperedge_encoder)

        reconstruction_en = self.decoder2(self.z_node_mean, self.z_edge_mean)
        result = self.z_node_mean.mm(self.z_edge_mean.t()).float()  # 确保类型一致

        # 将结果压缩到0-1范围
        result_h = torch.sigmoid(result_h)  # 使用sigmoid标准化到(0,1)
        result = torch.sigmoid(result)  # 使用sigmoid标准化到(0,1)


        # 结果融合（保持原有权重）
        score = 0.7 * result + 0.3 * result_h



        return score, reconstruction_en