
import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
from .loss import RankingLoss, CosineLoss, KLDivLoss
import torch.nn.functional as F

class Fusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model, device):
	
        super(Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model
        self.device = device

        target_classes = self.args.num_classes
        lstm_in = self.ehr_model.feats_dim
        lstm_out = self.cxr_model.feats_dim
        projection_in = self.cxr_model.feats_dim

        

        if self.args.labels_set == 'radiology':
            target_classes = self.args.vision_num_classes
            lstm_in = self.cxr_model.feats_dim
            projection_in = self.ehr_model.feats_dim

        # import pdb; pdb.set_trace()
        self.projection = nn.Linear(projection_in, lstm_in)
        feats_dim = 2 * self.ehr_model.feats_dim
        # feats_dim = self.ehr_model.feats_dim + self.cxr_model.feats_dim

        self.fused_cls = nn.Sequential(
            nn.Linear(feats_dim, self.args.num_classes),
            nn.Sigmoid()
        )

        self.align_loss = CosineLoss()
        self.kl_loss = KLDivLoss()

        self.lstm_fused_cls =  nn.Sequential(
            nn.Linear(lstm_out, target_classes),
            nn.Sigmoid()
        ) 

        self.lstm_fusion_layer = nn.LSTM(
            lstm_in, lstm_out,
            batch_first=True,
            dropout = 0.0)

        self.sigma = torch.Tensor([1.01])
        self.D = 512
        self.time_embedding = nn.Sequential(nn.Linear(1, self.D), nn.Tanh()).to(self.device)

        M = 1024
        D = self.D
        nnet = nn.Sequential(nn.Linear(D, M), nn.SiLU(),
                     nn.Linear(M, M), nn.SiLU(),
                     nn.Linear(M, M), nn.SiLU(),
                     nn.Linear(M, D), nn.Hardtanh(min_val=-3., max_val=3.))
        self.nnet = nnet.to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.T = 2
        self.EPS = 1.e-5

    def sample_p_t(self, x_0, x_1, t):
        # sampling from p_0t(x_t|x_0)
        # x_0 ~ data, x_1 ~ noise
        x = x_0 + self.sigma_fun(t) * x_1
        
        return x

    def lambda_t(self, t):
        # the loss weighting
        return self.sigma_fun(t)**2

    def sigma_fun(self, t):
        # the sigma function (dependent on t), it is the std of the distribution
        # print("t", t)
        # print("self.sigma", self.sigma)
        sigma = self.sigma
        sigma = sigma.to(self.device)
        return torch.sqrt((1./(2. * torch.log(sigma))) * (sigma**(2.*t) - 1.))

    def diffusion_coeff(self, t):
        # the diffusion coefficient in the SDE
        return self.sigma.to(self.device)**t

    def sample_base(self, x_0):
        # sampling from the base distribution
        return self.base.rsample(sample_shape=torch.Size([x_0.shape[0]]))

    def sample(self, x_0, batch_size=64):
        # 1) sample x_0 ~ Normal(0,1/(2log sigma) * (sigma**2 - 1))
        # x_t = self.sample_base(torch.empty(batch_size, self.D))
        x_t = x_0.to(self.device)
        
        # Apply Euler's method
        # NOTE: x_0 - data, x_1 - noise
        #       Therefore, we must use BACKWARD Euler's method! This results in the minus sign! 
        ts = torch.linspace(1e-3, self.EPS, self.T).to(self.device)
        delta_t = ts[0] - ts[1]
        
        for t in ts[1:]:
            tt = torch.Tensor([t]).to(self.device)
            u = 0.5 * self.diffusion_coeff(tt).to(self.device) * self.nnet(x_t.to(self.device) + self.time_embedding(tt).to(self.device))
            x_t = x_t - delta_t * u
        
        # x_t = torch.tanh(x_t)
        return x_t

    def re_sample(self, x_0, batch_size=64):
        # 1) sample x_0 ~ Normal(0,1/(2log sigma) * (sigma**2 - 1))
        # x_t = self.sample_base(torch.empty(batch_size, self.D))
        x_t = x_0.to(self.device)
        
        # Apply Euler's method
        # NOTE: x_0 - data, x_1 - noise
        #       Therefore, we must use BACKWARD Euler's method! This results in the minus sign! 
        ts = torch.linspace(1e-3, self.EPS, self.T_p).to(self.device)
        delta_t = ts[0] - ts[1]
        
        for t in ts[1:]:
            tt = torch.Tensor([t]).to(self.device)
            u = 0.5 * self.diffusion_coeff(tt).to(self.device) * self.nnet(x_t.to(self.device) + self.time_embedding(tt).to(self.device))
            x_t = x_t + delta_t * u
        
        # x_t = torch.tanh(x_t)
        return x_t


    def score_based_loss(self, patient_emb, aggr='sum', test_mode=False):
        x_1 = torch.randn_like(patient_emb).to(self.device)   
        # print("x_1", x_1)
        t = torch.rand(size=(patient_emb.shape[0], 1))  * (1. - 1.e-5) + 1.e-5 
        if test_mode == True:
            # print("test_mode=True")
            t = torch.ones(size=(patient_emb.shape[0], 1)) 
        t = t.to(self.device)
        x_0 = patient_emb.to(self.device)
        x_t = self.sample_p_t(x_0, x_1, t)
        # print("x_t", x_t)
        t_embd = self.time_embedding(t).to(self.device)

        # print("temp", nnet(x_t+t_embd))

        nnet = self.nnet.to(self.device)

        # hazard = 1/(1+torch.exp(-nnet(x_t+t_embd)))
        hazard = nnet(x_t+t_embd)
        x_pred = -self.sigma_fun(t) * hazard
        # x_pred = -hazard

        final_embed = self.sample(patient_emb)


        score_matching_loss = 0.1 * self.lambda_t(t) * torch.pow(x_pred + x_1, 2).mean(-1) 
        
        if aggr == 'sum':
            sm_loss = score_matching_loss.sum()
        else:
            sm_loss = score_matching_loss.mean()

        return final_embed, sm_loss

            
    def forward_uni_cxr(self, x, seq_lengths=None, img=None ):
        cxr_preds, _ , feats = self.cxr_model(img)
        return {
            'uni_cxr': cxr_preds,
            'cxr_feats': feats
            }
    # 
    def forward(self, x, seq_lengths=None, img=None, pairs=None, test_mode=False ):
        if self.args.fusion_type == 'uni_cxr':
            return self.forward_uni_cxr(x, seq_lengths=seq_lengths, img=img)
        elif self.args.fusion_type in ['joint',  'early', 'late_avg', 'unified']:
            return self.forward_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs )
        elif self.args.fusion_type == 'uni_ehr':
            return self.forward_uni_ehr(x, seq_lengths=seq_lengths, img=img)
        elif self.args.fusion_type == 'lstm':
            return self.forward_lstm_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs, test_mode=test_mode )

        elif self.args.fusion_type == 'uni_ehr_lstm':
            return self.forward_lstm_ehr(x, seq_lengths=seq_lengths, img=img, pairs=pairs )

    def forward_uni_ehr(self, x, seq_lengths=None, img=None ):
        ehr_preds , feats = self.ehr_model(x, seq_lengths)
        return {
            'uni_ehr': ehr_preds,
            'ehr_feats': feats
            }
    def forward_fused(self, x, seq_lengths=None, img=None, pairs=None ):

        ehr_preds , ehr_feats = self.ehr_model(x, seq_lengths)
        cxr_preds, _ , cxr_feats = self.cxr_model(img)
        projected = self.projection(cxr_feats)


        feats = torch.cat([ehr_feats, projected], dim=1)
        fused_preds = self.fused_cls(feats)
        # late_avg = (cxr_preds + ehr_preds)/2
        return {
            'early': fused_preds, 
            'joint': fused_preds, 
            # 'late_avg': late_avg,
            # 'align_loss': loss,
            'ehr_feats': ehr_feats,
            'cxr_feats': projected,
            'unified': fused_preds
            }

    all_preds = []
    def forward_lstm_fused(self, x, seq_lengths=None, img=None, pairs=None, test_mode=False ):
        if self.args.labels_set == 'radiology':
            _ , ehr_feats = self.ehr_model(x, seq_lengths)
            
            _, _ , cxr_feats = self.cxr_model(img)

            feats = cxr_feats[:,None,:]

            ehr_feats = self.projection(ehr_feats)

            ehr_feats[list(~np.array(pairs))] = 0
            feats = torch.cat([feats, ehr_feats[:,None,:]], dim=1)
        else:

            _ , ehr_feats = self.ehr_model(x, seq_lengths)

            _, _ , cxr_feats = self.cxr_model(img)
            cxr_feats = self.projection(cxr_feats)

            cxr_feats[list(~np.array(pairs))] = 0
            if len(ehr_feats.shape) == 1:
                # print(ehr_feats.shape, cxr_feats.shape)
                # import pdb; pdb.set_trace()
                feats = ehr_feats[None,None,:]
                feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
            else:
                feats = ehr_feats[:,None,:]
                feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
        seq_lengths = np.array([1] * len(seq_lengths))
        seq_lengths[pairs] = 2
        
        feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True, enforce_sorted=False)

    # We parameterize a multi-modal fusion network, ffusion, as a single LSTM layer with input dimension of 256 
    # and a hidden dimension of 512, that aggregates the multi-modal sequence through recurrence. 
    # The motivation for using an LSTM is two-fold. 

        x, (ht, _) = self.lstm_fusion_layer(feats)

        out = ht.squeeze()

        # print('out', out, out.shape)
        # out, sm_loss, pre_embed = self.score_based_loss(out, aggr='mean', test_mode=test_mode)
        # out, sm_loss = self.score_based_loss(out, aggr='mean', test_mode=test_mode)

        fused_preds = self.lstm_fused_cls(out)
        # pre_preds = self.lstm_fused_cls(pre_embed)
        # print('fused_preds', fused_preds, fused_preds.shape)

        return {
            'lstm': fused_preds,
            # 'sm': sm_loss,
            # 'pre_preds': pre_preds,
            'ehr_feats': ehr_feats,
            'cxr_feats': cxr_feats,
            'out': out,
        }
    
    def forward_lstm_ehr(self, x, seq_lengths=None, img=None, pairs=None ):
        _ , ehr_feats = self.ehr_model(x, seq_lengths)
        feats = ehr_feats[:,None,:]
        
        
        seq_lengths = np.array([1] * len(seq_lengths))
        
        feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True, enforce_sorted=False)

        x, (ht, _) = self.lstm_fusion_layer(feats)
        out = ht.squeeze()
        
        fused_preds = self.lstm_fused_cls(out)

        return {
            'uni_ehr_lstm': fused_preds,
        }




class Fusionnew(nn.Module):
    def __init__(self, args, ehr_model, cxr_model, device):
	
        super(Fusionnew, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model
        self.device = device

        target_classes = self.args.num_classes
        lstm_in = self.ehr_model.feats_dim
        lstm_out = self.cxr_model.feats_dim
        projection_in = self.cxr_model.feats_dim

        

        if self.args.labels_set == 'radiology':
            target_classes = self.args.vision_num_classes
            lstm_in = self.cxr_model.feats_dim
            projection_in = self.ehr_model.feats_dim

        # import pdb; pdb.set_trace()
        self.projection = nn.Linear(projection_in, lstm_in)
        feats_dim = 2 * self.ehr_model.feats_dim
        # feats_dim = self.ehr_model.feats_dim + self.cxr_model.feats_dim

        self.fused_cls = nn.Sequential(
            nn.Linear(feats_dim, self.args.num_classes),
            nn.Sigmoid()
        )

        self.align_loss = CosineLoss()
        self.kl_loss = KLDivLoss()

        self.lstm_fused_cls =  nn.Sequential(
            nn.Linear(lstm_out, target_classes),
            nn.Sigmoid()
        ) 

        self.lstm_fusion_layer = nn.LSTM(
            lstm_in, lstm_out,
            batch_first=True,
            dropout = 0.0)

        self.sigma = torch.Tensor([1.01])
        self.D = 512
        self.time_embedding = nn.Sequential(nn.Linear(1, self.D), nn.Tanh()).to(self.device)

        M = 1024
        D = self.D
        nnet = nn.Sequential(nn.Linear(D, M), nn.SiLU(),
                     nn.Linear(M, M), nn.SiLU(),
                     nn.Linear(M, M), nn.SiLU(),
                     nn.Linear(M, D), nn.Hardtanh(min_val=-3., max_val=3.))
        self.nnet = nnet.to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.T = 2
        self.T_p = 10
        self.EPS = 1.e-5

    def sample_p_t(self, x_0, x_1, t):
        # sampling from p_0t(x_t|x_0)
        # x_0 ~ data, x_1 ~ noise
        x = x_0 + self.sigma_fun(t) * x_1
        
        return x

    def lambda_t(self, t):
        # the loss weighting
        return self.sigma_fun(t)**2

    def sigma_fun(self, t):
        # the sigma function (dependent on t), it is the std of the distribution
        # print("t", t)
        # print("self.sigma", self.sigma)
        sigma = self.sigma
        sigma = sigma.to(self.device)
        return torch.sqrt((1./(2. * torch.log(sigma))) * (sigma**(2.*t) - 1.))

    def diffusion_coeff(self, t):
        # the diffusion coefficient in the SDE
        return self.sigma.to(self.device)**t

    def sample_base(self, x_0):
        # sampling from the base distribution
        return self.base.rsample(sample_shape=torch.Size([x_0.shape[0]]))

    def sample(self, x_0, batch_size=64):
        # 1) sample x_0 ~ Normal(0,1/(2log sigma) * (sigma**2 - 1))
        # x_t = self.sample_base(torch.empty(batch_size, self.D))
        x_t = x_0.to(self.device)
        
        # Apply Euler's method
        # NOTE: x_0 - data, x_1 - noise
        #       Therefore, we must use BACKWARD Euler's method! This results in the minus sign! 
        ts = torch.linspace(1e-3, self.EPS, self.T).to(self.device)
        delta_t = ts[0] - ts[1]
        
        for t in ts[1:]:
            tt = torch.Tensor([t]).to(self.device)
            u = 0.5 * self.diffusion_coeff(tt).to(self.device) * self.nnet(x_t.to(self.device) + self.time_embedding(tt).to(self.device))
            x_t = x_t - delta_t * u
        
        # x_t = torch.tanh(x_t)
        return x_t

    def re_sample(self, x_0, batch_size=64):
        # 1) sample x_0 ~ Normal(0,1/(2log sigma) * (sigma**2 - 1))
        # x_t = self.sample_base(torch.empty(batch_size, self.D))
        x_t = x_0.to(self.device)
        
        # Apply Euler's method
        # NOTE: x_0 - data, x_1 - noise
        #       Therefore, we must use BACKWARD Euler's method! This results in the minus sign! 
        ts = torch.linspace(1e-3, self.EPS, self.T_p).to(self.device)
        delta_t = ts[0] - ts[1]
        
        for t in ts[1:]:
            tt = torch.Tensor([t]).to(self.device)
            u = 0.5 * self.diffusion_coeff(tt).to(self.device) * self.nnet(x_t.to(self.device) + self.time_embedding(tt).to(self.device))
            x_t = x_t + delta_t * u
        
        # x_t = torch.tanh(x_t)
        return x_t


    def score_based_loss(self, patient_emb, aggr='sum', test_mode=False):
        x_1 = torch.randn_like(patient_emb).to(self.device)   
        # print("x_1", x_1)
        t = torch.rand(size=(patient_emb.shape[0], 1))  * (1. - 1.e-5) + 1.e-5 
        if test_mode == True:
            # print("test_mode=True")
            t = torch.ones(size=(patient_emb.shape[0], 1)) 
        t = t.to(self.device)
        x_0 = patient_emb.to(self.device)
        x_t = self.sample_p_t(x_0, x_1, t)

        t_embd = self.time_embedding(t).to(self.device)

        # print("temp", nnet(x_t+t_embd))

        nnet = self.nnet.to(self.device)

        hazard = nnet(x_t+t_embd)
        x_pred = -self.sigma_fun(t) * hazard

        final_embed = self.sample(patient_emb)


        score_matching_loss = 0.1 * self.lambda_t(t) * torch.pow(x_pred + x_1, 2).mean(-1) 
        
        if aggr == 'sum':
            sm_loss = score_matching_loss.sum()
        else:
            sm_loss = score_matching_loss.mean()

        return final_embed, sm_loss

            
    def forward_uni_cxr(self, x, seq_lengths=None, img=None ):
        cxr_preds, _ , feats = self.cxr_model(img)
        return {
            'uni_cxr': cxr_preds,
            'cxr_feats': feats
            }
    # 
    def forward(self, x, seq_lengths=None, img=None, pairs=None, test_mode=False ):
        if self.args.fusion_type == 'uni_cxr':
            return self.forward_uni_cxr(x, seq_lengths=seq_lengths, img=img)
        elif self.args.fusion_type in ['joint',  'early', 'late_avg', 'unified']:
            return self.forward_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs )
        elif self.args.fusion_type == 'uni_ehr':
            return self.forward_uni_ehr(x, seq_lengths=seq_lengths, img=img)
        elif self.args.fusion_type == 'lstm':
            return self.forward_lstm_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs, test_mode=test_mode )

        elif self.args.fusion_type == 'uni_ehr_lstm':
            return self.forward_lstm_ehr(x, seq_lengths=seq_lengths, img=img, pairs=pairs )

    def forward_uni_ehr(self, x, seq_lengths=None, img=None ):
        ehr_preds , feats = self.ehr_model(x, seq_lengths)
        return {
            'uni_ehr': ehr_preds,
            'ehr_feats': feats
            }
    def forward_fused(self, x, seq_lengths=None, img=None, pairs=None ):

        ehr_preds , ehr_feats = self.ehr_model(x, seq_lengths)
        cxr_preds, _ , cxr_feats = self.cxr_model(img)
        projected = self.projection(cxr_feats)

        # loss = self.align_loss(projected, ehr_feats)
        feats = torch.cat([ehr_feats, projected], dim=1)
        fused_preds = self.fused_cls(feats)
        # late_avg = (cxr_preds + ehr_preds)/2
        return {
            'early': fused_preds, 
            'joint': fused_preds, 
            # 'late_avg': late_avg,
            # 'align_loss': loss,
            'ehr_feats': ehr_feats,
            'cxr_feats': projected,
            'unified': fused_preds
            }
    def forward_lstm_fused(self, x, seq_lengths=None, img=None, pairs=None, test_mode=False ):
        if self.args.labels_set == 'radiology':
            _ , ehr_feats = self.ehr_model(x, seq_lengths)
            
            _, _ , cxr_feats = self.cxr_model(img)

            feats = cxr_feats[:,None,:]

            ehr_feats = self.projection(ehr_feats)

            ehr_feats[list(~np.array(pairs))] = 0
            feats = torch.cat([feats, ehr_feats[:,None,:]], dim=1)
        else:

            _ , ehr_feats = self.ehr_model(x, seq_lengths)

            _, _ , cxr_feats = self.cxr_model(img)
            cxr_feats = self.projection(cxr_feats)


            cxr_feats[list(~np.array(pairs))] = 0
            if len(ehr_feats.shape) == 1:
                # print(ehr_feats.shape, cxr_feats.shape)
                # import pdb; pdb.set_trace()
                feats = ehr_feats[None,None,:]
                feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
            else:
                feats = ehr_feats[:,None,:]
                feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
        seq_lengths = np.array([1] * len(seq_lengths))
        seq_lengths[pairs] = 2
        
        feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True, enforce_sorted=False)

    # We parameterize a multi-modal fusion network, ffusion, as a single LSTM layer with input dimension of 256 
    # and a hidden dimension of 512, that aggregates the multi-modal sequence through recurrence. 
    # The motivation for using an LSTM is two-fold. 

        x, (ht, _) = self.lstm_fusion_layer(feats)

        out = ht.squeeze()


        out, sm_loss = self.score_based_loss(out, aggr='mean', test_mode=test_mode)

        fused_preds = self.lstm_fused_cls(out)


        return {
            'lstm': fused_preds,
            'sm': sm_loss,
            # 'pre_preds': pre_preds,
            'ehr_feats': ehr_feats,
            'cxr_feats': cxr_feats,
            'out': out,
        }
    
    
    def forward_lstm_ehr(self, x, seq_lengths=None, img=None, pairs=None ):
        _ , ehr_feats = self.ehr_model(x, seq_lengths)
        feats = ehr_feats[:,None,:]
        
        
        seq_lengths = np.array([1] * len(seq_lengths))
        
        feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True, enforce_sorted=False)

        x, (ht, _) = self.lstm_fusion_layer(feats)
        out = ht.squeeze()
        
        fused_preds = self.lstm_fused_cls(out)

        return {
            'uni_ehr_lstm': fused_preds,
        }