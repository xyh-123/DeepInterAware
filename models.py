import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from configs import get_cfg_defaults
from networks.modules import *
from munch import Munch
import yaml

class DeepInterAwar(nn.Module):
    """best"""
    def __init__(self, cfg):
        super(DeepInterAwar, self).__init__()
        self.h_dim = cfg.predict.hidden_dim
        self.channel = cfg.protein.channel
        self.out_dim = self.h_dim
        self.max_antibody_length = cfg.protein.max_antibody_len
        self.max_antigen_length = cfg.protein.max_antigen_len
        self.phy=cfg.set.phy

        if cfg.set.ab_model == 'antiberty':
            ab_embed_dim = 512
        else:
            ab_embed_dim = 768

        ag_embed_dim=480
        self.head = 2
        self.activation = nn.ELU()

        self.use_bn = cfg.protein.use_bn


        self.iil_tcp = LinearLayer(self.h_dim, 1)
        self.sil_tcp = LinearLayer(self.h_dim, 1)

        drop_out=cfg.protein.dropout
        self.dropout = nn.Dropout(drop_out)
        self.ab_extractor = CNNBlock(in_channel=ab_embed_dim,out_channels=self.channel)
        self.ag_extractor = CNNBlock(in_channel=ag_embed_dim,out_channels=self.channel)

        if cfg.set.phy:
            embed_dim = self.channel + 5
        else:
            embed_dim = self.channel

        self.ab_iil_pool =IIP(self.channel)
        self.ag_iil_pool =IIP(self.channel)
        self.bcn = weight_norm(
            BANLayer(v_dim=embed_dim, q_dim=embed_dim, h_dim=self.h_dim, h_out=self.head),
            name='h_mat', dim=None)
        self.ab_sil_pool = ASP(self.channel)
        self.ag_sil_pool = ASP(self.channel)

        if self.use_bn:
            self.ab_norm = nn.BatchNorm1d(self.channel)
            self.ag_norm = nn.BatchNorm1d(self.channel)

        self.iil_project = nn.Sequential(
            nn.Linear(self.channel * 2, self.h_dim),nn.ELU(),nn.BatchNorm1d(self.out_dim),nn.Dropout(drop_out)
        )
        self.sil_project = nn.Sequential(
            nn.Linear(self.channel * 2, self.h_dim),nn.ELU(), nn.BatchNorm1d(self.out_dim), nn.Dropout(drop_out)
        )

        # self.iil_linear = nn.Linear(self.channel * 2, self.h_dim)
        # self.sil_linear = nn.Linear(self.channel * 2, self.h_dim)
        # self.iil_norm=nn.BatchNorm1d(self.out_dim)
        # self.sil_norm=nn.BatchNorm1d(self.out_dim)

        for m in self.modules():
            self.weights_init(m)

        if cfg.predict.simple:
            self.iil_predict_layer = LinearLayer(self.out_dim,2)
            self.sil_predict_layer = LinearLayer(self.out_dim,2)
            self.predict_layer = LinearLayer(self.out_dim*2,2)
        else:
            self.iil_predict_layer = MLPDecoder(self.out_dim, 256, 256 // 2, binary=2)
            self.sil_predict_layer = MLPDecoder(self.out_dim, 256, 256 // 2, binary=2)
            self.predict_layer = MLPDecoder(self.out_dim*2, 256, 256 // 2, binary=2)
        # if cfg.set.dataset == 'AVIDa_hIL6':
        #     self.cls_func = ClsFocalLoss()
        # else:
        self.cls_func = torch.nn.CrossEntropyLoss(reduction='mean')
        self.tcp_func = nn.MSELoss()

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)

    def iilmodule(self,ab,ag,ab_mask,ag_mask,ab_phy=None,ag_phy=None,label=None):
        if self.phy:
            ab_ = torch.cat([ab,ab_phy],dim=-1)
            ag_ = torch.cat([ag,ag_phy],dim=-1)
            iil_pair_ft, att = self.bcn(ab_, ag_)
        else:
            iil_pair_ft, att = self.bcn(ab, ag)  # batch*self.h_dim,batch*head*max_ab_length*max_ag_length
        att = torch.mean(att, dim=1)  # batch*max_ab_length*max_ag_length
        att_ab = torch.einsum('bnm, bmc -> bnc', att, ag)  #
        att_ag = torch.einsum('bnm, bnc -> bmc', att, ab)

        # ab = att_ab * ab
        # ag = att_ag * ag

        w_ab, iil_ab_ft = self.ab_iil_pool(att_ab, ab, ab_mask)
        w_ag, iil_ag_ft = self.ag_iil_pool(att_ag, ag, ag_mask)
        if self.use_bn:
            iil_ab_ft = self.dropout(self.ab_norm(iil_ab_ft))
            iil_ag_ft = self.dropout(self.ag_norm(iil_ag_ft))

        iil_pair_ft = torch.cat((iil_ab_ft, iil_ag_ft), dim=-1)
        # print(iil_pair_ft.shape)
        iil_pair_ft = self.iil_project(iil_pair_ft)
        iil_pred = self.iil_predict_layer(iil_pair_ft)
        out = Munch(
            score = iil_pred,
            feature = iil_pair_ft,
            att=att,
            # w_ab=w_ab,
            # w_ag=w_ag

        )
        if label != None:
            loss = self.cls_func(iil_pred,label)
            out['loss'] = loss
        return out

    def silmodule(self,ab,ag,ab_mask,ag_mask,ab_phy=None,ag_phy=None,label=None):
        # if self.phy:
        #     ab = torch.cat([ab,ab_phy],dim=-1)
        #     ag = torch.cat([ag,ag_phy],dim=-1)
        w_ab,sil_ab_ft = self.ab_sil_pool(ab, ab_mask)
        # sil_ab_ft = self.dropout(sil_ab_ft)
        w_ag,sil_ag_ft = self.ag_sil_pool(ag, ag_mask)
        # sil_ag_ft = self.dropout(sil_ag_ft)

        sil_pair_ft = torch.cat([sil_ab_ft, sil_ag_ft], dim=-1)
        sil_pair_ft = self.sil_project(sil_pair_ft)
        sil_pred = self.sil_predict_layer(sil_pair_ft)

        out = Munch(
            score=sil_pred,
            feature=sil_pair_ft,
            w_ab=w_ab,
            w_ag=w_ag

        )
        if label != None:
            loss = self.cls_func(sil_pred, label)
            out['loss'] = loss
        return out


    def inference(self,ag_token_ft,ab_token_ft,ag_mask,ab_mask,ab_phy=None,ag_phy=None,label = None):
        ab = self.ab_extractor(ab_token_ft)  # batch*max_length*out_channels,batch*max_length*out_channels
        ag = self.ag_extractor(ag_token_ft)  # batch*max_length*out_channels,batch*max_length*out_channels

        iil_out = self.iilmodule(ab, ag, ab_mask, ag_mask,ab_phy,ag_phy)
        sil_out = self.silmodule(ab, ag, ab_mask, ag_mask)

        iil_feature = iil_out.feature
        sil_feature = sil_out.feature

        # iil_pred = iil_out.score
        # sil_pred = sil_out.score

        iil_tcp = self.iil_tcp(iil_feature)
        sil_tcp = self.sil_tcp(sil_feature)

        tcp_feature = [iil_feature * iil_tcp, sil_feature * sil_tcp]
        feature = torch.cat(tcp_feature, dim=-1)
        predict = self.predict_layer(feature)

        n = F.softmax(predict, dim=1)[:, 1]
        pred = n.detach().cpu()

        outputs = Munch(
            score=pred,
            feature=feature,
            iil_out = iil_out,
            sil_out = sil_out,
        )

        return outputs

    def forward(self,inputs):
        label = inputs.batch.label
        ab_token, ag_token=inputs.batch.ab_token_ft,inputs.batch.ag_token_ft
        ab_mask, ag_mask = inputs.batch.ab_mask, inputs.batch.ag_mask

        ab = self.ab_extractor(ab_token)  # batch*max_length*out_channels,batch*max_length*out_channels
        ag = self.ag_extractor(ag_token)  #batch*max_length*out_channels,batch*max_length*out_channels

        if self.phy:
            iil_out = self.iilmodule(ab,ag,ab_mask,ag_mask,ab_phy=inputs.batch.ab_phy,ag_phy=inputs.batch.ag_phy,label=label)
        else:
            iil_out = self.iilmodule(ab, ag, ab_mask,ag_mask,label=label)

        # if self.phy:
        #     sil_out = self.globalmodule(ab,ag,ab_mask,ag_mask,ab_phy=inputs.batch.ab_phy,ag_phy=inputs.batch.ag_phy,label=label)
        # else:
        sil_out = self.silmodule(ab, ag, ab_mask,ag_mask,label=label)

        iil_feature = iil_out.feature
        sil_feature = sil_out.feature

        iil_pred = iil_out.score
        sil_pred = sil_out.score

        iil_loss = iil_out.loss
        sil_loss = sil_out.loss

        # loss = iil_loss + sil_loss

        if inputs.stage == 1:
            outputs = Munch(
                feature=torch.cat([iil_feature, sil_feature], dim=-1),
                iil_feature=iil_feature,
                sil_feature=sil_feature,
                iil_pred=iil_pred,
                sil_pred=sil_pred,
                iil_loss=iil_loss,
                sil_loss=sil_loss,
                # loss = loss
                # ratio=ratio_local
            )
        else:
            iil_pred_soft = F.softmax(iil_pred, dim=1)
            sil_pred_soft = F.softmax(sil_pred, dim=1)

            iil_tcp = self.iil_tcp(iil_feature)
            sil_tcp = self.sil_tcp(sil_feature)

            iil_p_target = torch.gather(input=iil_pred_soft, dim=1,
                                          index=inputs.batch.label.unsqueeze(dim=1)).view(
                -1)  # 置信度
            sil_p_target = torch.gather(input=sil_pred_soft, dim=1,
                                           index=inputs.batch.label.unsqueeze(dim=1)).view(-1)  # 置信度
            iil_tcp_loss = self.tcp_func(iil_tcp.squeeze(-1), iil_p_target.detach())
            sil_tcp_loss = self.tcp_func(sil_tcp.squeeze(-1), sil_p_target.detach())

            # iil_tcp = iil_tcp / (iil_tcp + sil_tcp)
            # sil_tcp = sil_tcp / (iil_tcp + sil_tcp)
            # predict = iil_tcp * iil_pred + sil_tcp * sil_pred

            tcp_feature = [iil_feature * iil_tcp, sil_feature * sil_tcp]
            feature = torch.cat(tcp_feature, dim=-1)
            predict = self.predict_layer(feature)
            jcl_loss = self.cls_func(predict, inputs.batch.label)

            jcl_loss = iil_tcp_loss + sil_tcp_loss + jcl_loss

            outputs = Munch(
                score=predict,
                feature=feature,
                iil_feature=iil_feature,
                sil_feature=sil_feature,
                iil_pred=iil_pred,
                sil_pred=sil_pred,
                iil_loss=iil_loss,
                sil_loss=sil_loss,
                # loss =loss,
                jcl_loss=jcl_loss,
                iil_tcp=iil_tcp,
                sil_tcp=sil_tcp,
                iil_out=iil_out,
                sil_out=sil_out,
                # ratio=ratio_local
            )
        return outputs

def load_model(yml,model_path,gpu=0):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    cfg = get_cfg_defaults()
    cfg.merge_from_file(yml)
    model = DeepInterAwar(cfg)
    state_dict = torch.load(model_path)
    # print(state_dict['model_state_dict'].keys())
    model.load_state_dict(state_dict['model_state_dict'])
    model = model.to(device)
    return model
