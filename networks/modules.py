import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim),nn.Sigmoid())
        self.clf.apply(self.xavier_init)

    def xavier_init(self,m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    def forward(self, x):
        x = self.clf(x)
        return x

class IIP(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )
        # self.norm = nn.BatchNorm1d(in_dim)
        # self.dropout = nn.Dropout(dropout)
    def forward(self, attention_ft, ft,attention_mask):
        w = self.attention(attention_ft).float()
        w[attention_mask == 0] = float('-inf')
        w = torch.softmax(w, 1)
        # w = torch.sigmoid(w)
        attention_embeddings = torch.sum(w * ft, dim=1)
        # loss = torch.mean(w)
        # attention_embeddings = torch.sum(w * ft, dim=1)
        # attention_embeddings = self.dropout(self.norm(attention_embeddings))
        return w,attention_embeddings

class ASP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask == 0] = float('-inf')
        w = torch.softmax(w, 1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return w,attention_embeddings

class CNNBlock(nn.Module):
    def __init__(self, in_channel,out_channels=64):
        super(CNNBlock, self).__init__()
        #output = (input - kenal_size + 2Padding) / s + 1
        self.conv = nn.Conv1d(in_channels=in_channel, out_channels=out_channels, kernel_size=3, stride=1,
                              padding=1)  # out_channels+1
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1,padding=1)  # out_channels-1
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.active = nn.ReLU()

    def forward(self, protein_ft):
        '''
        :param protein_ft: batch*len*amino_dim
        :return:
        '''
        protein_ft = protein_ft.transpose(1, 2)
        protein_ft = self.conv(protein_ft)  # B,out_channels,max_len
        protein_ft = self.active(protein_ft)
        protein_ft = self.bn1(protein_ft)
        conv_ft = self.pool(protein_ft)
        return conv_ft.transpose(1, 2)

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1,dropout=0.5):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)
        self.drop_out=nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x=self.drop_out(x)
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.drop_out(x)
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.drop_out(x)
        x = self.fc4(x)
        return x

class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ELU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k #3
        self.v_dim = v_dim #128
        self.q_dim = q_dim #128
        self.h_dim = h_dim #256
        self.h_out = h_out #head 2

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        """
        :param v: drug batch*N*embeddim
        :param q: protein batch*M*embeddim
        :param softmax:
        :return:
        """
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()
        #dims=[v_dim, h_dim * self.k]=[128,256*3]
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)