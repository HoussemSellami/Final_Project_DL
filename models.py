import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import GATConv, JumpingKnowledge, GCNConv
from torch.autograd import Variable
import math


class GCNNet(torch.nn.Module):
    def __init__(self, dropout, input_dim, hidden_dim=64, num_outputs=6, num_layers=3, aggr='add'):
        super(GCNNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv1.aggr = aggr
        self.convs = []
        for _ in range(num_layers - 2):
            conv = GCNConv(hidden_dim, hidden_dim)
            conv.aggr = aggr
            self.convs.append(conv)
        self.convout = GCNConv(hidden_dim, num_outputs)
        self.convout.aggr = aggr

    def forward(self, data):
        x, edge_index = data.x, data.edge_index.type(torch.LongTensor)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.convout(x, edge_index)

        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, dropout, input_dim, hidden_dim=64, heads=2, num_outputs=6, num_layers=3, aggr='add'):
        super(GATNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv1.aggr = aggr
        self.convs = []
        for _ in range(num_layers - 2):
            conv = GATConv(heads * hidden_dim, hidden_dim, heads=heads)
            conv.aggr = aggr
            self.convs.append(conv)

        self.convout = GATConv(heads * hidden_dim, num_outputs)
        self.convout.aggr = aggr

    def forward(self, data):
        x, edge_index = data.x, data.edge_index.type(torch.LongTensor)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.convout(x, edge_index)

        return F.log_softmax(x, dim=1)


class GATWithJK(torch.nn.Module):
    def __init__(self, input_dim, num_layers, dropout, hidden_dim, num_outputs=6, heads=1, mode='lstm',
                 aggr='add'):
        super(GATWithJK, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, hidden_dim, heads)
        self.conv1.aggr = aggr

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            tmp_conv = GATConv(heads * hidden_dim, hidden_dim, heads)
            tmp_conv.aggr = aggr
            self.convs.append(tmp_conv)
        self.jump = JumpingKnowledge(mode, channels=heads*hidden_dim, num_layers=2)
        if mode == 'cat':
            self.lin1 = Linear(heads * num_layers * hidden_dim, hidden_dim)
        else:
            self.lin1 = Linear(heads * hidden_dim, hidden_dim)

        self.lin2 = Linear(input_dim + hidden_dim, hidden_dim)
        self.lin_out = Linear(hidden_dim, num_outputs)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.task_1.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(self, data):
        x_in, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x_in, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            xs += [x]
        x = self.jump(xs)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin2(torch.cat([x, x_in], axis=1))
        x = F.relu(x)
        x = self.lin_out(x)

        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=20):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


class Norm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = Parameter(torch.ones(self.size))
        self.bias = Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff=4, dropout=0.1):
        super().__init__()
        self.linear_1 = Linear(d_model, d_model*d_ff)
        self.dropout = dropout
        self.linear_2 = Linear(d_model*d_ff, d_model)

    def forward(self, x):
        x = F.dropout(F.relu(self.linear_1(x)), p=self.dropout)
        x = self.linear_2(x)
        return x


def attention(q, k, v, d_k, mask=None):
    att_w = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        att_w = att_w.masked_fill(mask == 0, -1e9)
    output = F.softmax(att_w, dim=-1)

    output = torch.matmul(output, v)
    return output, att_w


class Attention_network(torch.nn.Module):
    def __init__(self, heads, d_model,  dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)
        self.dropout = dropout
        self.out = Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.d_k)

        scores, att_w = attention(q, k, v, self.d_k, mask)
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output, att_w


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_layers, heads=1, num_outputs=6, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.pe = PositionalEncoder(d_model, num_layers)
        self.attn = Attention_network(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout = dropout
        self.num_layers = num_layers
        self.regression = Linear(d_model, num_outputs)

    def forward(self, x, mask=None):
        # add positional encoding
        x = self.pe(x)
        # batch normalization
        x2 = self.norm_1(x)
        att, scores = self.attn(x2, x2, x2, mask)
        x = x + F.dropout(att, p=self.dropout)
        x2 = self.norm_2(x)
        x = x + F.dropout(self.ff(x2), p=self.dropout)
        # last layer as output (in principle arbitrary)
        x = x[:, self.num_layers-1, :] + F.dropout(self.ff(x2[:, self.num_layers-1, :]), p=self.dropout)

        x = F.log_softmax(self.regression(x), dim=1)
        return x, scores

    def get_attention(self, x, mask=None):
        x2 = self.norm_1(x)
        _, scores = self.attn(x2, x2, x2, mask)
        return scores


class TransfEnc(torch.nn.Module):
    def __init__(self, input_dim, num_layers, dropout, hidden_dim, num_outputs=6, heads=1,
                 loads=False, aggr='add'):
        super(TransfEnc, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv1.aggr = aggr
        self.loads = loads
        self.num_outputs = num_outputs
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            tmp_conv = GATConv(hidden_dim, hidden_dim)
            tmp_conv.aggr = aggr
            self.convs.append(tmp_conv)

        self.encoder = EncoderLayer(d_model=hidden_dim,
                                    num_layers=num_layers,
                                    heads=1,
                                    num_outputs=num_outputs,
                                    dropout=dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.encoder.reset_parameters()

    def forward(self, data):
        x_in, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x_in, edge_index)
        xs = torch.unsqueeze(x, 1)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            xs = torch.cat([xs, torch.unsqueeze(x, 1)], dim=1)
        x = self.encoder(xs)
        return x

    def get_att(self, data):
        x_in, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x_in, edge_index)
        xs = torch.unsqueeze(x, 1)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            xs = torch.cat([xs, torch.unsqueeze(x, 1)], dim=1)

        attentions = self.encoder.get_attention(xs)
        return attentions

    def __repr__(self):
        return self.__class__.__name__