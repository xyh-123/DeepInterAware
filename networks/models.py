from configs import get_cfg_defaults
from networks.modules import *
from munch import Munch

class DeepInterAware(nn.Module):
    """best"""
    def __init__(self, cfg):
        super(DeepInterAware, self).__init__()
        self.h_dim = cfg.predict.hidden_dim
        self.channel = cfg.protein.channel
        self.out_dim = self.h_dim
        self.max_antibody_length = cfg.protein.max_antibody_len
        self.max_antigen_length = cfg.protein.max_antigen_len
        self.phy=cfg.set.phy

        if cfg.set.ab_model == 'antiberty':
            ab_embed_dim = 512
        elif cfg.set.ab_model == 'esm2':
            ab_embed_dim = 480
        else:
            ab_embed_dim = 768

        ag_embed_dim=480
        self.head = 2
        self.activation = nn.ELU()

        self.use_bn = cfg.protein.use_bn


        self.iil_tcp = LinearLayer(self.h_dim, 1)
        # self.iil_tcp = nn.Sequential(MLPDecoder(self.h_dim, 256, 256 // 2, binary=1),nn.Sigmoid())
        self.sil_tcp = LinearLayer(self.h_dim, 1)
        # self.sil_tcp = nn.Sequential(MLPDecoder(self.h_dim, 256, 256 // 2, binary=1),nn.Sigmoid())

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
            w_ab=w_ab,
            w_ag=w_ag

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

class DeepInterAwareSite(nn.Module):
    """best"""
    def __init__(self, cfg):
        super(DeepInterAwareSite, self).__init__()
        self.h_dim = cfg.predict.hidden_dim
        self.channel = cfg.protein.channel
        self.out_dim = self.h_dim
        self.max_antibody_length = cfg.protein.max_antibody_len
        self.max_antigen_length = cfg.protein.max_antigen_len
        self.phy=cfg.set.phy

        if cfg.set.ab_model == 'antiberty':
            ab_embed_dim = 512
        elif cfg.set.ab_model == 'esm2':
            ab_embed_dim = 480
        else:
            ab_embed_dim = 768

        ag_embed_dim=480
        self.head = 2
        self.activation = nn.ELU()

        self.use_bn = cfg.protein.use_bn
        self.iil_ag_tcp = nn.Sequential(MLPDecoder(self.channel, 256, 256 // 2, binary=1),nn.Sigmoid())
        self.iil_ab_tcp = nn.Sequential(MLPDecoder(self.channel, 256, 256 // 2, binary=1),nn.Sigmoid())
        self.sil_ag_tcp = nn.Sequential(MLPDecoder(self.channel, 256, 256 // 2, binary=1),nn.Sigmoid())
        self.sil_ab_tcp = nn.Sequential(MLPDecoder(self.channel, 256, 256 // 2, binary=1),nn.Sigmoid())

        drop_out=cfg.protein.dropout
        self.dropout = nn.Dropout(drop_out)
        self.ab_extractor = CNNBlock(in_channel=ab_embed_dim,out_channels=self.channel)
        self.ag_extractor = CNNBlock(in_channel=ag_embed_dim,out_channels=self.channel)

        if cfg.set.phy:
            embed_dim = self.channel + 5
        else:
            embed_dim = self.channel


        self.bcn = weight_norm(
            BANLayer(v_dim=embed_dim, q_dim=embed_dim, h_dim=self.h_dim, h_out=self.head),
            name='h_mat', dim=None)


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
            self.iil_ag_predict_layer = LinearLayer(self.out_dim,2)
            self.iil_ab_predict_layer = LinearLayer(self.out_dim,2)
            self.sil_ag_predict_layer = LinearLayer(self.out_dim,2)
            self.sil_ab_predict_layer = LinearLayer(self.out_dim,2)

            self.ab_predict_layer = LinearLayer(self.out_dim*2,2)
            self.ag_predict_layer = LinearLayer(self.out_dim*2,2)
        else:
            self.iil_ag_predict_layer = MLPDecoder(self.channel, 256, 256 // 2, binary=2)
            self.iil_ab_predict_layer = MLPDecoder(self.channel, 256, 256 // 2, binary=2)
            self.sil_ag_predict_layer = MLPDecoder(self.channel, 256, 256 // 2, binary=2)
            self.sil_ab_predict_layer = MLPDecoder(self.channel, 256, 256 // 2, binary=2)

            self.ab_predict_layer = MLPDecoder(self.channel*2, 256, 256 // 2, binary=2)
            self.ag_predict_layer = MLPDecoder(self.channel*2, 256, 256 // 2, binary=2)
        # if cfg.set.dataset == 'AVIDa_hIL6':
        #     self.cls_func = ClsFocalLoss()
        # else:
        self.cls_func = torch.nn.CrossEntropyLoss(reduction='mean')
        self.tcp_func = nn.MSELoss()

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)

    def iilmodule(self,ab,ag,ab_mask_expanded,ag_mask_expanded,ab_label=None,ag_label=None):
        iil_pair_ft, att = self.bcn(ab, ag)  # batch*self.h_dim,batch*head*max_ab_length*max_ag_length
        att = torch.mean(att, dim=1)  # batch*max_ab_length*max_ag_length
        att_ab = torch.einsum('bnm, bmc -> bnc', att, ag)  # batch*max_ab_length
        att_ag = torch.einsum('bnm, bnc -> bmc', att, ab)  # batch*max_ag_length

        ab_token_features = att_ab[ab_mask_expanded.bool()].reshape(-1, self.channel)
        ag_token_features = att_ag[ag_mask_expanded.bool()].reshape(-1, self.channel)

        ab_pred = self.iil_ab_predict_layer(ab_token_features)
        ag_pred = self.iil_ag_predict_layer(ag_token_features)

        # ab = att_ab * ab
        # ag = att_ag * ag
        out = Munch(
            iil_ab = ab_token_features,
            iil_ag = ag_token_features,
            ab_pred = ab_pred,
            ag_pred = ag_pred,
            att = att
        )

        if ag_label != None:
            loss = self.cls_func(ag_pred,ag_label)+self.cls_func(ab_pred,ab_label)
            out['loss'] = loss
        return out

    def silmodule(self,ab,ag,ab_mask_expanded,ag_mask_expanded,ab_label=None,ag_label=None):
        ab_token_features = ab[ab_mask_expanded.bool()].reshape(-1, self.channel)
        ag_token_features = ag[ag_mask_expanded.bool()].reshape(-1, self.channel)

        ab_pred = self.sil_ab_predict_layer(ab_token_features)
        ag_pred = self.sil_ag_predict_layer(ag_token_features)

        out = Munch(
            sil_ab = ab_token_features,
            sil_ag = ag_token_features,
            ab_pred = ab_pred,
            ag_pred = ag_pred,

        )

        if ag_label != None:
            loss = self.cls_func(ag_pred, ag_label) + self.cls_func(ab_pred, ab_label)
            out['loss'] = loss
        return out


    def inference(self,ag_token_ft,ab_token_ft,ag_mask,ab_mask):
        ab_mask_expanded = ab_mask.unsqueeze(-1).expand(-1, -1, self.channel)
        ag_mask_expanded = ag_mask.unsqueeze(-1).expand(-1, -1, self.channel)


        ab = self.ab_extractor(ab_token_ft)  # batch*max_length*out_channels,batch*max_length*out_channels
        ag = self.ag_extractor(ag_token_ft)  # batch*max_length*out_channels,batch*max_length*out_channels

        iil_out = self.iilmodule(ab, ag, ab_mask_expanded, ag_mask_expanded)
        sil_out = self.silmodule(ab, ag, ab_mask_expanded, ag_mask_expanded)

        iil_ab = iil_out.iil_ab
        iil_ag = iil_out.iil_ag

        sil_ab = sil_out.sil_ab
        sil_ag = sil_out.sil_ag

        iil_ag_tcp = self.iil_ag_tcp(iil_ag)
        iil_ab_tcp = self.iil_ab_tcp(iil_ab)

        sil_ab_tcp = self.sil_ag_tcp(sil_ab)
        sil_ag_tcp = self.sil_ab_tcp(sil_ag)

        ab_feature = [iil_ab * iil_ab_tcp, sil_ab * sil_ab_tcp]
        ag_feature = [iil_ag * iil_ag_tcp, sil_ag * sil_ag_tcp]

        ab_feature = torch.cat(ab_feature, dim=-1)
        ag_feature = torch.cat(ag_feature, dim=-1)

        ab_predict = self.ab_predict_layer(ab_feature)
        ag_predict = self.ag_predict_layer(ag_feature)

        outputs = Munch(
            ab_score = ab_predict,
            ag_score = ag_predict,
            iil_out = iil_out,
            sil_out = sil_out,
        )
        return outputs

    def forward(self,inputs):
        ag_label = inputs.batch.ag_label
        ab_label = inputs.batch.ab_label

        ab_token, ag_token=inputs.batch.ab_token_ft,inputs.batch.ag_token_ft
        ab_mask, ag_mask = inputs.batch.ab_mask, inputs.batch.ag_mask

        ab_mask_expanded = ab_mask.unsqueeze(-1).expand(-1, -1, self.channel)
        ag_mask_expanded = ag_mask.unsqueeze(-1).expand(-1, -1, self.channel)

        ag_label = ag_label[ag_mask.bool()]  # (number_res,)
        ab_label = ab_label[ab_mask.bool()]  # (number_res,)


        ab = self.ab_extractor(ab_token)  # batch*max_length*out_channels,batch*max_length*out_channels
        ag = self.ag_extractor(ag_token)  #batch*max_length*out_channels,batch*max_length*out_channels


        iil_out = self.iilmodule(ab, ag, ab_mask_expanded,ag_mask_expanded,ab_label,ag_label)

        # if self.phy:
        #     sil_out = self.globalmodule(ab,ag,ab_mask,ag_mask,ab_phy=inputs.batch.ab_phy,ag_phy=inputs.batch.ag_phy,label=label)
        # else:
        sil_out = self.silmodule(ab, ag,ab_mask_expanded,ag_mask_expanded,ab_label,ag_label)

        iil_ab = iil_out.iil_ab
        iil_ag = iil_out.iil_ag

        sil_ab = sil_out.sil_ab
        sil_ag = sil_out.sil_ag

        iil_ab_pred = iil_out.ab_pred
        iil_ag_pred = iil_out.ag_pred

        sil_ab_pred = sil_out.ab_pred
        sil_ag_pred = sil_out.ag_pred

        iil_loss = iil_out.loss
        sil_loss = sil_out.loss

        if inputs.stage == 1:
            outputs = Munch(
                iil_ab_pred=iil_ab_pred,
                iil_ag_pred=iil_ag_pred,
                sil_ab_pred=sil_ab_pred,
                sil_ag_pred=sil_ag_pred,
                ag_label = ag_label,
                ab_label = ab_label,
                iil_loss = iil_loss,
                sil_loss = sil_loss
            )
        else:
            iil_ab_pred_soft = F.softmax(iil_ab_pred, dim=1)
            iil_ag_pred_soft = F.softmax(iil_ag_pred, dim=1)

            sil_ab_pred_soft = F.softmax(sil_ab_pred, dim=1)
            sil_ag_pred_soft = F.softmax(sil_ag_pred, dim=1)

            iil_ag_tcp = self.iil_ag_tcp(iil_ag)
            iil_ab_tcp = self.iil_ab_tcp(iil_ab)

            sil_ag_tcp = self.sil_ag_tcp(sil_ag)
            sil_ab_tcp = self.sil_ab_tcp(sil_ab)

            iil_p_ab_target = torch.gather(input=iil_ab_pred_soft, dim=1,
                                          index=ab_label.unsqueeze(dim=1)).view(
                -1)  # 置信度
            iil_p_ag_target = torch.gather(input=iil_ag_pred_soft, dim=1,
                                        index=ag_label.unsqueeze(dim=1)).view(
                -1)  # 置信度

            sil_p_ab_target = torch.gather(input=sil_ab_pred_soft, dim=1,
                                           index=ab_label.unsqueeze(dim=1)).view(-1)  # 置信度
            sil_p_ag_target = torch.gather(input=sil_ag_pred_soft, dim=1,
                                           index=ag_label.unsqueeze(dim=1)).view(-1)  # 置信度


            iil_ab_tcp_loss = self.tcp_func(iil_ab_tcp.squeeze(-1), iil_p_ab_target.detach())

            iil_ag_tcp_loss = self.tcp_func(iil_ag_tcp.squeeze(-1), iil_p_ag_target.detach())

            sil_ab_tcp_loss = self.tcp_func(sil_ab_tcp.squeeze(-1), sil_p_ab_target.detach())
            sil_ag_tcp_loss = self.tcp_func(sil_ag_tcp.squeeze(-1), sil_p_ag_target.detach())

            iil_tcp_loss = iil_ab_tcp_loss+iil_ag_tcp_loss
            sil_tcp_loss = sil_ab_tcp_loss+sil_ag_tcp_loss

            ab_feature = [iil_ab * iil_ab_tcp, sil_ab * sil_ab_tcp]
            ag_feature = [iil_ag * iil_ag_tcp, sil_ag * sil_ag_tcp]

            ab_feature = torch.cat(ab_feature, dim=-1)
            ag_feature = torch.cat(ag_feature, dim=-1)

            ab_predict = self.ab_predict_layer(ab_feature)
            ag_predict = self.ag_predict_layer(ag_feature)

            jcl_loss = self.cls_func(ab_predict, ab_label) + self.cls_func(ag_predict, ag_label)
            loss = iil_tcp_loss + sil_tcp_loss + jcl_loss

            outputs = Munch(
                iil_ab_pred=iil_ab_pred,
                iil_ag_pred=iil_ag_pred,
                sil_ab_pred=sil_ab_pred,
                sil_ag_pred=sil_ag_pred,
                ab_score=ab_predict,
                ag_score=ag_predict,
                iil_loss=iil_loss,
                sil_loss=sil_loss,
                jcl_loss=loss,
                iil_out=iil_out,
                sil_out=sil_out,
                ag_label=ag_label,
                ab_label=ab_label,
                # ratio=ratio_local
            )
        return outputs

class DeepInterAware_AM(nn.Module):
    """best"""
    def __init__(self, pair_encoder):
        super(DeepInterAware_AM, self).__init__()
        self.pair_encoder = pair_encoder
        # self.pair_encoder = DeepInterAware(cfg)
        self.out_dim = self.pair_encoder.out_dim
        # self.predict_layer = AffinityDecoder(self.out_dim*4, 256, 256/2, binary=1)
        if self.operation=='diff':
            self.predict_layer = MLPDecoder(self.out_dim*2, 256, 256 // 2, binary=1)
        else:
            self.predict_layer = MLPDecoder(self.out_dim*4, 256, 256//2, binary=1)
        self.reg_func = nn.MSELoss()

    def inference(self,wt,mu):
        """
        :return:
        """
        wt_ag_token_ft, wt_ab_token_ft, wt_ag_mask, wt_ab_mask = wt
        mu_ag_token_ft, mu_ab_token_ft, mu_ag_mask, mu_ab_mask = mu

        wt_output = self.pair_encoder.inference(wt_ag_token_ft, wt_ab_token_ft, wt_ag_mask, wt_ab_mask)
        wt_feature = wt_output.feature
        mu_output = self.pair_encoder.inference(mu_ag_token_ft, mu_ab_token_ft, mu_ag_mask, mu_ab_mask)
        mu_feature = mu_output.feature

        feature = torch.cat([mu_feature,wt_feature],dim=-1)

        n = self.predict_layer(feature)
        pred = n.detach().cpu()
        outputs = Munch(
            score=pred,
            feature=feature,
            wt_output=wt_output,
            mu_output=mu_output
        )

        return outputs

    def forward(self,wt_inputs,mu_inputs,label):

        wt_ab_token_ft, wt_ag_token_ft = wt_inputs.ab_token_ft,wt_inputs.ag_token_ft
        wt_ab_mask, wt_ag_mask = wt_inputs.ab_mask, wt_inputs.ag_mask

        mu_ab_token_ft, mu_ag_token_ft = mu_inputs.ab_token_ft, mu_inputs.ag_token_ft
        mu_ab_mask, mu_ag_mask = mu_inputs.ab_mask, mu_inputs.ag_mask

        wt_output = self.pair_encoder.inference(wt_ag_token_ft, wt_ab_token_ft, wt_ag_mask, wt_ab_mask)
        wt_feature = wt_output.feature

        mu_output = self.pair_encoder.inference(mu_ag_token_ft, mu_ab_token_ft, mu_ag_mask, mu_ab_mask)
        mu_feature = mu_output.feature
        if self.operation=='diff':
            feature = mu_feature-wt_feature
        else:
            feature = torch.cat([mu_feature,wt_feature],dim=-1)
        n = self.predict_layer(feature)

        label = label.to(torch.float32)
        loss = self.reg_func(n, label)
        # print(loss,loss.dtype)

        outputs = Munch(
            score=n,
            feature=feature,
            loss=loss
        )

        return outputs

def load_model(model_name,yml,model_path,gpu=0):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    cfg = get_cfg_defaults()
    cfg.merge_from_file(yml)
    state_dict = torch.load(model_path,map_location=device)
    if model_name=='DeepInterAwareAM':
        encoder = DeepInterAware(cfg)
        model = DeepInterAware_AM(encoder)
        model.load_state_dict(state_dict)
    else:
        model = eval(model_name + "(cfg)", globals(), {"cfg": cfg})
        model.load_state_dict(state_dict)
        # try:
        #     model.load_state_dict(state_dict)
        # except:
        #     torch.save(state_dict['model_state_dict'],model_path)
        #     state_dict = torch.load(model_path, map_location=device)
        #     # model.load_state_dict(state_dict['model_state_dict'])
        #     model.load_state_dict(state_dict)

    # print(state_dict['model_state_dict'].keys())
    # model.load_state_dict(state_dict['model_state_dict'])
    model = model.to(device)
    return model

def load_AM_model(yml,model_path,gpu=0,model_name='DeepInterAware'):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    cfg = get_cfg_defaults()
    cfg.merge_from_file(yml)
    encoder = DeepInterAware(cfg)
    state_dict = torch.load(model_path, map_location=f'cuda:{gpu}')
    # print(state_dict['model_state_dict'].keys())
    encoder.load_state_dict(state_dict['model_state_dict'])
    model = DeepInterAware_AM(encoder)
    model = model.to(device)
    return model


