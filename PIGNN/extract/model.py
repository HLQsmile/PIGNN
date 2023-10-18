from torch import nn
import torch as t
from torch_geometric.nn import SAGEConv, GATConv
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
import math


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)
# Channel Attention
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(8, 8 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(8 // reduction, 8, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.permute(1, 2, 0,3)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=False, bias=True, activate=False, alphas=[0, 1],
                 shared_weight=False, aggr='mean',
                 **kwargs):
        super(SAGEConv, self).__init__(aggr=aggr, **kwargs)
        self.shared_weight = shared_weight
        self.activate = activate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.channel_attention = CALayer(channel=out_channels)
        self.alphas = Parameter(torch.Tensor(alphas))
        self.bias = Parameter(torch.Tensor(self.out_channels)) if bias else None

        if self.shared_weight:
            self.self_weight = self.weight
        else:
            self.self_weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)
        uniform(self.in_channels, self.self_weight)

    def forward(self, x, edge_index, edge_weight=None, size=None):

        out = torch.matmul(x, self.self_weight)
        out2 = self.propagate(edge_index, size=size, x=x,edge_weight=edge_weight)
        out = self.channel_attention(out).squeeze(-1)
        out2 = self.channel_attention(out2).squeeze(-1)
        return self.alphas[0] * out + self.alphas[1] * out2

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1, 1) * x_j

    def matmul_with_weight(self, tensor):
        if tensor is not None:
            return torch.matmul(tensor, self.weight)
        else:
            return None

    def update(self, aggr_out):

        if self.activate:
            aggr_out = F.relu(aggr_out)

        if torch.is_tensor(aggr_out):
            aggr_out = torch.matmul(aggr_out, self.weight)
        else:
            aggr_out = (self.matmul_with_weight(aggr_out[0]), self.matmul_with_weight(aggr_out[1]))
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
def init_weights(m):
    if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class CA_SAGE(nn.Module):
    def __init__(self, in_channel=1, mid_channel=8, out_channel=2, num_nodes=2207, edge_num=151215,
                 **args):
        super(CA_SAGE, self).__init__()
        self.in_channels = in_channel
        self.mid_channel = mid_channel
        self.dropout_ratio = args.get('dropout_ratio', 0.3)
        n_out_nodes = num_nodes
        self.conv1 = SAGEConv(in_channel, mid_channel, )
        self.act1 = nn.ReLU()
        self.bn1 = torch.nn.LayerNorm((num_nodes, mid_channel))
        self.global_conv1_dim = 4 * 3
        self.global_conv2_dim = args.get('global_conv2_dim', 4)
        self.global_conv1 = t.nn.Conv2d(mid_channel * 1, self.global_conv1_dim, [1, 1])
        self.global_bn1 = torch.nn.BatchNorm2d(self.global_conv1_dim)
        self.global_act1 = nn.ReLU()
        self.global_conv2 = t.nn.Conv2d(self.global_conv1_dim, self.global_conv2_dim, [1, 1])
        self.global_bn2 = torch.nn.BatchNorm2d(self.global_conv2_dim)
        self.global_act2 = nn.ReLU()
        self.weight_edge_flag = True

        last_feature_node = 1024
        channel_list = [self.global_conv2_dim * n_out_nodes, 2048, 1024]

        self.nn = []
        for idx, num in enumerate(channel_list[:-1]):
            self.nn.extend([
                nn.Linear(channel_list[idx], channel_list[idx + 1]),
                nn.BatchNorm1d(channel_list[idx + 1]),
                nn.Dropout(self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity(),
                nn.ReLU()
            ])
            self.global_fc_nn = nn.Sequential(*self.nn)
            self.fc1 = nn.Linear(last_feature_node, out_channel)
        self.edge_num = edge_num
        self.edge_weight = nn.Parameter(t.ones(edge_num).float() * 0.01)
        self.reset_parameters()

    def reset_parameters(self, ):
        # uniform(self.mid_channel, self.global_conv1.weight
        #
        self.conv1.apply(init_weights)
        nn.init.kaiming_normal_(self.global_conv1.weight, mode='fan_out')
        uniform(self.mid_channel, self.global_conv1.bias)

        nn.init.kaiming_normal_(self.global_conv2.weight, mode='fan_out')
        uniform(self.global_conv1_dim, self.global_conv2.bias)

        self.global_fc_nn.apply(init_weights)
        self.fc1.apply(init_weights)
        pass

    def get_gcn_weight_penalty(self, mode='L2'):

        if mode == 'L1':
            func = lambda x: t.sum(t.abs(x))
        elif mode == 'L2':
            func = lambda x: t.sqrt(t.sum(x ** 2))

        loss = 0

        tmp = getattr(self.conv1, 'weight', None)
        if tmp is not None:
            loss += func(tmp)

        tmp = getattr(self.conv1, 'self_weight', None)
        if tmp is not None:
            loss += 1 * func(tmp)

        tmp = getattr(self.global_conv1, 'weight', None)
        if tmp is not None:
            loss += func(tmp)
        tmp = getattr(self.global_conv2, 'weight', None)
        if tmp is not None:
            loss += func(tmp)

        return loss

    def forward(self, data, get_latent_varaible=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.weight_edge_flag:
            one_graph_edge_weight = torch.sigmoid(self.edge_weight)  # *self.edge_num
            edge_weight = one_graph_edge_weight
        else:
            edge_weight = None

        x = self.act1(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = x.permute(2, 0, 1)
        x = x.permute(1, 0, 2)
        x = self.bn1(x)
        x = x.permute(1, 0, 2)
        if self.dropout_ratio > 0:
            x = F.dropout(x, p=0.1, training=self.training)
        x = x.permute(1, 2, 0)
        x = x.unsqueeze(dim=-1)
        x = self.global_conv1(x)
        x = self.global_act1(x)
        x = self.global_bn1(x)
        if self.dropout_ratio > 0:
            x = F.dropout(x, p=0.3, training=self.training)
        x = self.global_conv2(x)
        x = self.global_act1(x)
        x = self.global_bn2(x)
        if self.dropout_ratio > 0:
            x = F.dropout(x, p=0.3, training=self.training)
        x = x.squeeze(dim=-1)  # #samples  x #features  x #nodes  [64,4,42625]
        num_samples = x.shape[0]
        x = x.view(num_samples, -1)
        x = self.global_fc_nn(x)
        if get_latent_varaible:
            return x
        else:
            x = self.fc1(x)
            return F.softmax(x, dim=-1)
