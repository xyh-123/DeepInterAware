
import torch.nn as nn
import torch.nn.functional as F
import torch
from munch import Munch
from torch.nn.parameter import Parameter
from torch.nn.utils.weight_norm import weight_norm

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

class GCNConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 dropout=0.6,
                 bias=True
                 ):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.bias = bias
        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels)
        )
        nn.init.xavier_normal_(self.weight)
        if bias is True:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, node_ft, adj_mat):
        laplace_mat = get_laplace_mat(adj_mat, type='sym')
        node_state = torch.mm(laplace_mat, node_ft)
        node_state = torch.mm(node_state, self.weight)
        if self.bias is not None:
            node_state = node_state + self.bias

        return node_state

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class BasicBlock2D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, res_connect=True):
        super(BasicBlock2D, self).__init__()
        self.res_connect = res_connect
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel//2, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        if self.res_connect is True:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        residual = self.residual_function(x)
        if self.res_connect:
            residual += self.shortcut(x)
        residual = F.relu(residual)
        return residual

class ProteinCNN(nn.Module):
    def __init__(self, num_vocabulary=21):
        super(ProteinCNN, self).__init__()
        #embedding_dim, num_filters, kernel_size, mini = False, padding = True
        embedding_dim=128 #128
        num_filters=[128, 128, 128] #[128, 128, 128]
        kernel_size=[3,6,9]
        padding = True

        if padding:
            self.embedding = nn.Embedding(num_vocabulary, embedding_dim, padding_idx=0) #
        else:
            self.embedding = nn.Embedding(num_vocabulary, embedding_dim)
        in_ch = [embedding_dim] + num_filters #[128,128,128,128]
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0]) #3
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        # if not self.mini:
        #     # 后面的是否可以考虑空洞卷积
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])  # 6
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2]) #9
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        #v:batch_size,max_length
        v = self.embedding(v.long()) #batch*max_length*emebd_dim
        v = v.transpose(2, 1) #batch*emebd_dim*max_length
        v = self.bn1(F.relu(self.conv1(v))) #batch*emebd_dim*(max_length-kernel_size+1) batch,128,1198
        v = self.bn2(F.relu(self.conv2(v)))  # batch,128,1193
        v = self.bn3(F.relu(self.conv3(v))) # batch,128,1185
        v = v.view(v.size(0), v.size(2), -1) #batch,1185,128

        return v

class ESM2AbLang(nn.Module):
    def __init__(self, cfg):
        super(ESM2AbLang, self).__init__()
        # self.encoder=model = BALMForMaskedLM.from_pretrained("networks/pretrained-BALM/")
        # self.encoder=EsmModel.from_pretrained("networks/pretrained-ESM2/")

        # self.encoder = EsmModel.from_pretrained("networks/pretrained-ESM2-small/")
        # self.encoder = EsmModel.from_pretrained("/root/autodl-tmp/pretrain_model/pretrained-ESM2/")
        if cfg.set.dataset == 'AVIDa_hIL6':
            self.out_dim = 480 + 768
        else:
            self.out_dim = 480+768+768

        self.predict_layer = MLPDecoder(self.out_dim, 256, 256 // 2, binary=2)

        self.cls_func = torch.nn.CrossEntropyLoss(reduction='mean')
    def forward(self, inputs):
        # ag = inputs.batch.ag_tokenizer
        ab_ft = inputs.batch.ab_ft
        ag_ft = inputs.batch.ag_ft

        pair_ft = torch.cat([ab_ft, ag_ft], dim=-1).float()
        score = self.predict_layer(pair_ft)
        loss = self.cls_func(score, inputs.batch.label)

        outputs = Munch(
            feature=pair_ft,
            score = score,
            loss = loss

        )
        return outputs

class ESM2AntiBERTy(nn.Module):
    def __init__(self, cfg):
        super(ESM2AntiBERTy, self).__init__()
        self.out_dim = 480 + 512
        self.predict_layer = MLPDecoder(self.out_dim, 256, 256 // 2, binary=1)
        # self.cls_func = torch.nn.CrossEntropyLoss(reduction='mean')
        self.cls_func = nn.BCELoss()
        # self.norm = nn.BatchNorm1d(self.out_dim)


    def forward(self, inputs):

        # ab_ft = inputs.batch.ab_ft
        # ag_ft = inputs.batch.ag_ft
        # ab_ft = self.ab_pool(ab_ft)
        # ag_ft = self.ag_pool(ag_ft)

        ab_token_ft = inputs.batch.ab_token_ft
        ag_token_ft = inputs.batch.ag_token_ft
        ab_ft = torch.mean(ab_token_ft,dim=1)
        ag_ft = torch.mean(ag_token_ft,dim=1)

        pair_ft = torch.cat([ab_ft, ag_ft], dim=-1).float()
        pair_ft = F.relu(pair_ft)
        # pair_ft = self.norm(pair_ft)
        pair_ft = F.dropout(pair_ft,p=0.6)
        score = self.predict_layer(pair_ft)
        score = torch.sigmoid(score)
        score = score.view(-1)
        loss = self.cls_func(score, inputs.batch.label.float())
        outputs = Munch(
            feature=pair_ft,
            score = score,
            loss = loss
        )
        return outputs

class DeepAAI(nn.Module):
    def __init__(self, cfg):
        super(DeepAAI, self).__init__()
        self.kmer_dim = 8420
        self.h_dim = 512
        self.out_dim = self.h_dim
        self.dropout = 0.4
        self.adj_loss_coef=1e-4
        self.param_l2_coef=5e-4
        # self.amino_embedding_dim = param_dict['amino_embedding_dim']

        self.task = 'cls'

        # self.kernel_cfg = param_dict['kernel_cfg']
        # self.channel_cfg = param_dict['channel_cfg']
        # self.dilation_cfg = param_dict['dilation_cfg']

        self.antibody_kmer_linear = nn.Linear(self.kmer_dim, self.h_dim)
        self.virus_kmer_linear = nn.Linear(self.kmer_dim, self.h_dim)
        self.antibody_pssm_linear = nn.Linear(0, self.h_dim)
        self.virus_pssm_linear = nn.Linear(0, self.h_dim)
        # self.antibody_pssm_linear = nn.Linear(param_dict['pssm_antibody_dim'], self.h_dim)
        # self.virus_pssm_linear = nn.Linear(param_dict['pssm_virus_dim'], self.h_dim)

        self.share_linear = nn.Linear(self.h_dim, self.h_dim)

        self.share_gcn1 = GCNConv(self.h_dim, self.h_dim)
        self.share_gcn2 = GCNConv(self.h_dim, self.h_dim)

        self.antibody_adj_trans = nn.Linear(self.h_dim, self.h_dim)
        self.virus_adj_trans = nn.Linear(self.h_dim, self.h_dim)

        self.cross_scale_merge = nn.Parameter(
            torch.ones(1)
        )

        # self.amino_embedding_layer = nn.Embedding(param_dict['amino_type_num'], self.amino_embedding_dim)
        # self.channel_cfg.insert(0, self.amino_embedding_dim)
        # self.local_linear = nn.Linear(self.channel_cfg[-1] * 2, self.h_dim)
        self.global_linear = nn.Linear(self.h_dim * 2, self.h_dim)
        self.pred_linear = nn.Linear(self.h_dim, 1)

        self.activation = nn.ELU()
        for m in self.modules():
            self.weights_init(m)

        self.max_virus_len = cfg.set.max_antigen_len
        self.max_antibody_len = cfg.set.max_antibody_len

        self.cnnmodule = CNNmodule(in_channel=21,l=self.max_antibody_len)
        self.cnnmodule2 = CNNmodule(in_channel=21,l=self.max_virus_len)

        self.local_linear1 = nn.Linear(1024, 512)
        self.local_linear2 = nn.Linear(512, 512)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)

    def forward(self, inputs):
        '''
        :param ft_dict:
                ft_dict = {
                'antibody_graph_node_ft': FloatTensor  node_num * kmer_dim
                'virus_graph_node_ft': FloatTensor  node_num * kmer_dim,
                'antibody_amino_ft': LongTensor  batch * max_antibody_len * 1
                'virus_amino_ft': LongTensor  batch * max_virus_len * 1,
                'antibody_idx': LongTensor  batch
                'virus_idx': LongTensor  batch
            }
        :return:
        '''

        device = inputs.device
        ab, ag = inputs.batch.ab, inputs.batch.ag
        ab_id, ag_id = inputs.batch.ab_id, inputs.batch.ag_id
        antibody_graph_map_arr,antigen_graph_map_arr = inputs.batch.antibody_graph_map_arr, inputs.batch.antigen_graph_map_arr
        antibody_node_idx_in_graph = antibody_graph_map_arr[ab_id]
        antigen_node_idx_in_graph = antigen_graph_map_arr[ag_id]


        unique_ab_kmer = inputs.batch.unique_ab_kmer
        unique_ag_kmer = inputs.batch.unique_ag_kmer

        batch_size = ab.size()[0]

        ab = F.one_hot(ab.long(), num_classes=21).type(torch.float)
        ag = F.one_hot(ag.long(), num_classes=21).type(torch.float)

        antibody_ft = self.cnnmodule(ab).view(batch_size, -1)
        virus_ft = self.cnnmodule2(ag).view(batch_size, -1)

        local_pair_ft = torch.cat([virus_ft, antibody_ft], dim=-1).view(batch_size, -1)
        local_pair_ft = self.activation(local_pair_ft)
        local_pair_ft = self.local_linear1(local_pair_ft)
        local_pair_ft = self.activation(local_pair_ft)
        local_pair_ft = self.local_linear2(local_pair_ft)


        antibody_graph_node_num = unique_ab_kmer.size()[0] #所有训练集已知的抗体全局特征
        virus_graph_node_num = unique_ag_kmer.size()[0] #所有训练集已知的抗原全局特征

        antibody_res_mat = torch.zeros(antibody_graph_node_num, self.h_dim).to(device)
        virus_res_mat = torch.zeros(virus_graph_node_num, self.h_dim).to(device)

        antibody_node_kmer_ft = self.antibody_kmer_linear(unique_ab_kmer)
        antibody_node_ft=antibody_node_kmer_ft

        virus_node_kmer_ft = self.virus_kmer_linear(unique_ag_kmer)
        virus_node_ft = virus_node_kmer_ft


        antibody_node_ft = self.activation(antibody_node_ft)
        antibody_node_ft = F.dropout(antibody_node_ft, p=self.dropout, training=self.training)

        # share
        antibody_node_ft = self.share_linear(antibody_node_ft)
        antibody_res_mat = antibody_res_mat + antibody_node_ft
        antibody_node_ft = self.activation(antibody_node_ft)
        antibody_node_ft = F.dropout(antibody_node_ft, p=self.dropout, training=self.training)

        # generate antibody adj
        antibody_trans_ft = self.antibody_adj_trans(antibody_node_ft)
        antibody_trans_ft = torch.tanh(antibody_trans_ft)
        w = torch.norm(antibody_trans_ft, p=2, dim=-1).view(-1, 1)
        w_mat = w * w.t()
        antibody_adj = torch.mm(antibody_trans_ft, antibody_trans_ft.t()) / w_mat

        antibody_node_ft = self.share_gcn1(antibody_node_ft, antibody_adj)
        antibody_res_mat = antibody_res_mat + antibody_node_ft

        antibody_node_ft = self.activation(antibody_res_mat)  # add
        antibody_node_ft = F.dropout(antibody_node_ft, p=self.dropout, training=self.training)
        antibody_node_ft = self.share_gcn2(antibody_node_ft, antibody_adj)
        antibody_res_mat = antibody_res_mat + antibody_node_ft


        virus_node_ft = self.activation(virus_node_ft)
        virus_node_ft = F.dropout(virus_node_ft, p=self.dropout, training=self.training)

        # share
        virus_node_ft = self.share_linear(virus_node_ft)
        virus_res_mat = virus_res_mat + virus_node_ft
        virus_node_ft = self.activation(virus_node_ft)
        virus_node_ft = F.dropout(virus_node_ft, p=self.dropout, training=self.training)

        # generate antibody adj
        virus_trans_ft = self.virus_adj_trans(virus_node_ft)
        virus_trans_ft = torch.tanh(virus_trans_ft)
        w = torch.norm(virus_trans_ft, p=2, dim=-1).view(-1, 1)
        w_mat = w * w.t()
        virus_adj = torch.mm(virus_trans_ft, virus_trans_ft.t()) / w_mat
        # virus_adj = eye_adj

        virus_node_ft = self.share_gcn1(virus_node_ft, virus_adj)
        virus_res_mat = virus_res_mat + virus_node_ft

        virus_node_ft = self.activation(virus_res_mat)  # add
        virus_node_ft = F.dropout(virus_node_ft, p=self.dropout, training=self.training)
        virus_node_ft = self.share_gcn2(virus_node_ft, virus_adj)
        virus_res_mat = virus_res_mat + virus_node_ft

        antibody_res_mat = self.activation(antibody_res_mat)
        virus_res_mat = self.activation(virus_res_mat)

        antibody_res_mat = antibody_res_mat[antibody_node_idx_in_graph.long()]
        virus_res_mat = virus_res_mat[antigen_node_idx_in_graph.long()]

        # cross
        global_pair_ft = torch.cat([virus_res_mat, antibody_res_mat], dim=1)
        global_pair_ft = self.activation(global_pair_ft)
        global_pair_ft = F.dropout(global_pair_ft, p=self.dropout, training=self.training)
        global_pair_ft = self.global_linear(global_pair_ft)


        pair_ft = global_pair_ft + local_pair_ft + (global_pair_ft * local_pair_ft) * self.cross_scale_merge




        pair_ft = self.activation(pair_ft)
        pair_ft = F.dropout(pair_ft, p=self.dropout, training=self.training)

        score = self.pred_linear(pair_ft)
        if self.task == 'cls':
            score = torch.sigmoid(score)
        loss_func = nn.BCELoss()

        score = score.view(-1)

        c_loss = loss_func(score, inputs.batch.label.float())

        if antibody_adj != None:
            param_l2_loss = 0
            param_l1_loss = 0
            for name, param in self.named_parameters():
                if 'bias' not in name:
                    param_l2_loss += torch.norm(param, p=2)
                    param_l1_loss += torch.norm(param, p=1)

            param_l2_loss = self.param_l2_coef * param_l2_loss
            # param_l1_loss = self.param_dict['param_l1_coef'] * param_l1_loss
            adj_l1_loss = self.adj_loss_coef * torch.norm(virus_adj) + \
                          self.adj_loss_coef* torch.norm(antibody_adj)
            loss = c_loss + adj_l1_loss + param_l2_loss
        else:
            loss = c_loss

        output = Munch(
            feature=pair_ft,
            score=score,
            loss=loss
        )
        return output

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

class DrugBAN(nn.Module):
    """for ppi test"""
    def __init__(self, cfg):
        super(DrugBAN, self).__init__()

        self.h_dim = 512
        self.max_antibody_length = cfg.set.max_antibody_len
        self.max_antigen_length = cfg.set.max_antigen_len
        self.dropout = 0.4
        self.out_dim=self.h_dim

        self.task = 'cls'
        self.activation = nn.ELU()
        # self.bcn = weight_norm(
        #     BANLayer(v_dim=64, q_dim=64, h_dim=self.h_dim, h_out=2),
        #     name='h_mat', dim=None)

        self.bcn = weight_norm(
            BANLayer(v_dim=128, q_dim=128, h_dim=self.h_dim, h_out=2),
            name='h_mat', dim=None)

        self.ab_extractor = ProteinCNN(num_vocabulary=21)
        self.ag_extractor = ProteinCNN(num_vocabulary=21)

        for m in self.modules():
            self.weights_init(m)

        # self.pred_linear = MLPDecoder(self.h_dim, 512, 128, binary=1)
        self.pred_linear = MLPDecoder(self.h_dim, 256, 128, binary=1)
        self.loss_func = nn.BCELoss()


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)


    def forward(self, inputs):
        ab, ag = inputs.batch.ab, inputs.batch.ag

        # ab = F.one_hot(batch_antibody_ft.long(), num_classes=21).type(torch.float)
        # ag = F.one_hot(batch_virus_ft.long(), num_classes=21).type(torch.float)

        ab = self.ab_extractor(ab)  # batch*embeddim,batch*max_length*out_channels
        ag = self.ag_extractor(ag)  # batch*embeddim,batch*max_length*out_channels
        #interaction expert
        pair_ft, att = self.bcn(ab, ag)  # batch*self.h_dim,batch*head*max_ab_length*max_ag_length

        score = self.pred_linear(pair_ft)
        if self.task == 'cls':
            score = torch.sigmoid(score)
        # loss_func = nn.BCELoss()

        # score = score.view(-1)

        # loss = loss_func(score, inputs.batch.label.float())
        score = score.view(-1)
        loss = self.loss_func(score, inputs.batch.label.float())
        output = Munch(
            feature=pair_ft,
            score=score,
            loss=loss
        )
        return output

class AbAgIntPre(nn.Module):
    def __init__(self,cfg):
        super(AbAgIntPre, self).__init__()
        self.out_dim=5120*2
        self.task = 'cls'
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=10, kernel_size=3, stride=1), #4,10,18,18
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.Conv2d(10, 20, 3, 1),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(),
            Flatten(),
        )
        self.fc = MLPDecoder(self.out_dim, 256, 256 // 2, binary=1)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.out_dim, 64),
        #     nn.Linear(64, 1),
        #     # nn.Softmax()
        # )
    def inference(self,ab, ag):
        ab_feature = self.forward_once(ab)
        ag_feature = self.forward_once(ag)
        pair_ft = torch.cat((ab_feature, ag_feature), 1)
        score = self.fc(pair_ft)
        output = Munch(
            feature=pair_ft,
            score=score,
        )
        return output

    def forward_once(self,x):
        output = self.cnn(x)
        return output

    def forward(self, input):
        ab, ag = input.batch.ab_CKSAAP, input.batch.ag_CKSAAP
        # ab, ag = batch_antibody_ft, batch_virus_ft
        ab_feature = self.forward_once(ab)
        ag_feature = self.forward_once(ag)
        pair_ft = torch.cat((ab_feature,ag_feature),1)
        score = self.fc(pair_ft)
        if self.task == 'cls':
            score = torch.sigmoid(score)
        loss_func = nn.BCELoss()

        score = score.view(-1)

        loss = loss_func(score, input.batch.label.float())

        output = Munch(
            feature=pair_ft,
            score=score,
            loss=loss
        )
        return output

class MasonsCNNmodule(nn.Module):
    def __init__(self, in_channel,  l=0):
        super(MasonsCNNmodule, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channel, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.out_linear = nn.Linear(l * 64, 512)
        self.dropout = nn.Dropout(0.5)

    def forward(self, protein_ft):
        '''
        :param protein_ft: batch*len*amino_dim
        :return:
        '''
        batch_size = protein_ft.size()[0]
        protein_ft = protein_ft.transpose(1, 2)

        conv_ft = self.conv(protein_ft)
        conv_ft = self.dropout(conv_ft)
        conv_ft = self.pool(conv_ft).view(batch_size, -1)
        conv_ft = self.out_linear(conv_ft)
        return conv_ft

class CNNmodule(nn.Module):
    def __init__(self, in_channel, l=0):
        super(CNNmodule, self).__init__()
        # self.kernel_width = kernel_width
        self.conv = nn.Conv1d(in_channels=in_channel, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.out_linear = nn.Linear(l * 64, 512)
        self.dropout = nn.Dropout(0.5)

    def forward(self, protein_ft):
        '''
        :param protein_ft: batch*len*amino_dim
        :return:
        '''
        batch_size = protein_ft.size()[0]
        protein_ft = protein_ft.transpose(1, 2)

        conv_ft = self.conv(protein_ft)
        conv_ft = self.dropout(conv_ft)
        conv_ft = self.pool(conv_ft).view(batch_size, -1)
        conv_ft = self.out_linear(conv_ft)
        return conv_ft

class MasonsCNN(nn.Module):
    def __init__(self,cfg):
        super(MasonsCNN, self).__init__()
        self.amino_ft_dim = 21
        self.h_dim = 512
        self.out_dim=self.h_dim
        self.dropout = 0.1 #0.1
        self.inception_out_channel = 3

        self.max_virus_len = cfg.set.max_antigen_len
        self.max_antibody_len = cfg.set.max_antibody_len
        self.task = 'cls'

        # self.cnnmodule = MasonsCNNmodule(in_channel=21,l=self.max_antibody_len)
        # self.cnnmodule2 = MasonsCNNmodule(in_channel=21, l=self.max_virus_len)

        self.cnnmodule = CNNmodule(in_channel=21, l=self.max_antibody_len)
        self.cnnmodule2 = CNNmodule(in_channel=21, l=self.max_virus_len)

        self.out_linear1 = nn.Linear(512 * 2, 512)
        # self.out_linear2 = nn.Linear(512, 2)
        self.out_linear2 = nn.Linear(512, 1)
        # self.predict_layer = nn.Linear(512, 1)

        self.activation = nn.ELU()

    def forward(self, inputs):
        batch_antibody_ft, batch_virus_ft = inputs.batch.ab, inputs.batch.ag
        '''
        :param batch_antibody_ft:   tensor    batch, len, amino_ft_dim
        :param batch_virus_ft:     tensor    batch, amino_ft_dim
        :return:
        '''

        batch_size = batch_antibody_ft.size()[0]

        batch_virus_ft = F.one_hot(batch_virus_ft.long(), num_classes=21).type(torch.float)
        batch_antibody_ft = F.one_hot(batch_antibody_ft.long(), num_classes=21).type(torch.float)

        antibody_ft = self.cnnmodule(batch_antibody_ft).view(batch_size, -1)
        virus_ft= self.cnnmodule2(batch_virus_ft).view(batch_size, -1)

        pair_ft = torch.cat([virus_ft, antibody_ft], dim=-1).view(batch_size, -1)
        pair_ft = self.activation(pair_ft)
        pair_ft = self.out_linear1(pair_ft)
        pair_ft = self.activation(pair_ft)
        score = self.out_linear2(pair_ft)

        loss_func = nn.BCELoss()
        if self.task == 'cls':
            score = torch.sigmoid(score)
        score = score.view(-1)

        if inputs.mode == 'train':
            loss = loss_func(score, inputs.batch.label.float())

            output = Munch(
                feature=pair_ft,
                score=score,
                loss=loss
            )

        else:
            output = Munch(
                feature=pair_ft,
                score=score,
            )
        return output

class ResPPI(nn.Module):
    def __init__(self,cfg):
        super(ResPPI, self).__init__()


        self.amino_ft_dim = 21  # 21

        self.max_virus_len = cfg.set.max_antigen_len
        self.max_antibody_len = cfg.set.max_antibody_len
        self.h_dim = 512  # 512
        self.task = 'cls'


        self.dropout = 0.3 #0.3
        self.mid_channels = 16
        self.out_dim = self.mid_channels * self.amino_ft_dim * 2

        self.predict_layer = MLPDecoder(self.out_dim, 256, 256 // 2, binary=1)

        # self.out_linear1 = nn.Linear(self.mid_channels * self.amino_ft_dim * 2, self.h_dim)
        # # self.out_dim=self.h_dim
        # self.out_linear2 = nn.Linear(self.h_dim, 1)


        self.res_net = nn.Sequential(
            BasicBlock2D(in_channel=1, out_channel=self.mid_channels, res_connect=True),
            BasicBlock2D(in_channel=self.mid_channels, out_channel=self.mid_channels, res_connect=False),
            BasicBlock2D(in_channel=self.mid_channels, out_channel=self.mid_channels, res_connect=True),
            # BasicBlock2D(in_channel=64, out_channel=64, res_connect=False),
            # BasicBlock2D(in_channel=64, out_channel=64, res_connect=True),
        )
        self.activation = nn.ELU()

    def forward(self, inputs):
        batch_antibody_ft, batch_virus_ft = inputs.batch.ab, inputs.batch.ag
        '''
        :param batch_antibody_ft:   tensor    batch, antibody_dim
        :param batch_virus_ft:     tensor    batch, virus_dim
        :return:
        '''
        batch_size = batch_antibody_ft.size()[0]

        batch_virus_ft = F.one_hot(batch_virus_ft.long(), num_classes=21).type(torch.float)
        batch_antibody_ft = F.one_hot(batch_antibody_ft.long(), num_classes=21).type(torch.float)

        batch_virus_ft = batch_virus_ft.unsqueeze(1)
        batch_antibody_ft = batch_antibody_ft.unsqueeze(1)

        virus_ft = self.res_net(batch_virus_ft)
        antibody_ft = self.res_net(batch_antibody_ft)

        virus_ft = F.max_pool2d(virus_ft, kernel_size=[self.max_virus_len, 1]).view(batch_size, -1)
        antibody_ft = F.max_pool2d(antibody_ft, kernel_size=[self.max_antibody_len, 1]).view(batch_size, -1)

        pair_ft = torch.cat([virus_ft, antibody_ft], dim=-1)

        # pair_ft = self.out_linear1(pair_ft)
        # pair_ft = self.activation(pair_ft)
        # score = self.out_linear2(pair_ft)
        score = self.predict_layer(pair_ft)
        if self.task == 'cls':
            score = torch.sigmoid(score)
        score = score.view(-1)
        loss_func = nn.BCELoss()
        loss = loss_func(score, inputs.batch.label.float())
        output = Munch(
            feature=pair_ft,
            score=score,
            loss=loss
        )
        return output

class PIPR(nn.Module):
    def __init__(self, cfg):
        super(PIPR, self).__init__()
        self.protein_ft_dim = 21
        self.hidden_num = 50
        self.kernel_size = 3
        self.pool_size = 3
        self.conv_layer_num = 2
        self.task = 'cls'

        self.conv1d_layer_list = nn.ModuleList()
        for idx in range(self.conv_layer_num):
            in_channels = self.hidden_num * 2 + self.hidden_num // self.pool_size
            if idx == 0:
                in_channels = self.protein_ft_dim
            self.conv1d_layer_list.append(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=self.hidden_num,
                          kernel_size=self.kernel_size,
                          padding=0)
            )

        self.gru_list = nn.ModuleList()
        for idx in range(self.conv_layer_num - 1):
            self.gru_list.append(
                nn.GRU(input_size=self.hidden_num // self.pool_size, hidden_size=self.hidden_num, batch_first=True,
                       bidirectional=True)
            )

        self.linear_pred = nn.Linear(self.hidden_num, 1)

    def block_cnn_rnn(self, cnn_layer, rnn_layer, ft_mat, kernel_wide=2):
        '''
        :param cnn_layer:
        :param rnn_layer:
        :param ft_mat:     (batch, features, steps)
        :param kernel_wide:
        :return:
        '''

        # print("in ", ft_mat.size())
        ft_mat = ft_mat.transpose(1, 2)
        ft_mat = cnn_layer(ft_mat)
        ft_mat = ft_mat.transpose(1, 2)

        ft_mat = F.max_pool1d(ft_mat, kernel_wide)
        h0 = torch.randn(2, ft_mat.size()[0], self.hidden_num).to(ft_mat.device)
        output, hn = rnn_layer(ft_mat, h0)
        output = torch.cat([ft_mat, output], dim=-1)
        return output

    def forward(self, inputs):
        batch_antibody_ft, batch_virus_ft = inputs.batch.ab, inputs.batch.ag
        '''
        :param batch_antibody_ft:   tensor    batch, len, amino_ft_dim
        :param batch_virus_ft:     tensor    batch, amino_ft_dim
        :return:
        '''

        batch_size = batch_antibody_ft.size()[0]

        batch_virus_ft = F.one_hot(batch_virus_ft.long(), num_classes=21).type(torch.float)
        batch_antibody_ft = F.one_hot(batch_antibody_ft.long(), num_classes=21).type(torch.float)
        for gru_layer in self.gru_list:
            gru_layer.flatten_parameters()
        # batch  * seq_len * feature
        # print('1 antibody_ft = ', antibody_ft[:, :, 1])

        for idx in range(self.conv_layer_num - 1):
            batch_antibody_ft = self.block_cnn_rnn(
                cnn_layer=self.conv1d_layer_list[idx],
                rnn_layer=self.gru_list[idx],
                ft_mat=batch_antibody_ft,
                kernel_wide=self.pool_size
            )
        batch_antibody_ft = self.conv1d_layer_list[-1](
            batch_antibody_ft.transpose(1, 2))  # (batch, features, steps)

        batch_antibody_ft = F.max_pool2d(
            batch_antibody_ft, kernel_size=(1, batch_antibody_ft.size()[-1])).squeeze()  # (batch, features)

        for idx in range(self.conv_layer_num - 1):
            batch_virus_ft = self.block_cnn_rnn(
                cnn_layer=self.conv1d_layer_list[idx],
                rnn_layer=self.gru_list[idx],
                ft_mat=batch_virus_ft,
                kernel_wide=self.pool_size
            )
        batch_virus_ft = self.conv1d_layer_list[-1](
            batch_virus_ft.transpose(1, 2))  # (batch, features, steps)
        batch_virus_ft = F.max_pool2d(
            batch_virus_ft, kernel_size=(1, batch_virus_ft.size()[-1])).squeeze()  # (batch, features)

        pair_ft = batch_antibody_ft + batch_virus_ft
        score = self.linear_pred(pair_ft)

        if self.task == 'cls':
            score = torch.sigmoid(score)
        score = score.view(-1)
        loss_func = nn.BCELoss()
        loss = loss_func(score, inputs.batch.label.float())
        output = Munch(
            feature=pair_ft,
            score=score,
            loss=loss
        )

        return output

def get_laplace_mat(adj_mat, type='sym', add_i=False, degree_version='v2'):
    if type == 'sym':
        # Symmetric normalized Laplacian
        if add_i is True:
            adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        else:
            adj_mat_hat = adj_mat
        # adj_mat_hat = adj_mat_hat[adj_mat_hat > 0]
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-0.5, degree_version=degree_version)
        # print(degree_mat_hat.dtype, adj_mat_hat.dtype)
        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)
        # print(laplace_mat)
        laplace_mat = torch.mm(laplace_mat, degree_mat_hat)
        return laplace_mat
    elif type == 'rw':
        # Random walk normalized Laplacian
        adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-1)
        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)
        return laplace_mat
def get_degree_mat(adj_mat, pow=1, degree_version='v1'):
    degree_mat = torch.eye(adj_mat.size()[0]).to(adj_mat.device)

    if degree_version == 'v1':
        degree_list = torch.sum((adj_mat > 0), dim=1).float()
    elif degree_version == 'v2':
        # adj_mat_hat = adj_mat.data
        # adj_mat_hat[adj_mat_hat < 0] = 0
        adj_mat_hat = F.relu(adj_mat)
        degree_list = torch.sum(adj_mat_hat, dim=1).float()
    elif degree_version == 'v3':
        degree_list = torch.sum(adj_mat, dim=1).float()
        degree_list = F.relu(degree_list)
    else:
        exit('error degree_version ' + degree_version)
    degree_list = torch.pow(degree_list, pow)
    degree_mat = degree_mat * degree_list
    # degree_mat = torch.pow(degree_mat, pow)
    # degree_mat[degree_mat == float("Inf")] = 0
    # degree_mat.requires_grad = False
    # print('degree_mat = ', degree_mat)
    return degree_mat
def return_model(cfg):
    return eval(cfg.set.model_name + "(cfg)", globals(), {"cfg": cfg})