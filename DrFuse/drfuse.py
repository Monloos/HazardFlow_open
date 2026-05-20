import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights
import pickle
from .ehr_transformer import EHRTransformer

class DrFuseModel(nn.Module):
    def __init__(self, hidden_size, num_classes, ehr_dropout, ehr_n_layers, ehr_n_head, device, task,
                 cxr_model='swin_s', logit_average=False):
        super().__init__()
        self.num_classes = num_classes
        self.logit_average = logit_average
        self.ehr_model = EHRTransformer(input_size=76, num_classes=num_classes,
                                        d_model=hidden_size, n_head=ehr_n_head,
                                        n_layers_feat=1, n_layers_shared=ehr_n_layers,
                                        n_layers_distinct=ehr_n_layers,
                                        dropout=ehr_dropout)

        resnet = resnet50()
        self.cxr_model_feat = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        resnet = resnet50()
        self.cxr_model_shared = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )
        self.cxr_model_shared.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=hidden_size)

        resnet = resnet50()
        self.cxr_model_spec = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )
        self.cxr_model_spec.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=hidden_size)

        self.shared_project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.ehr_model_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.cxr_model_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.fuse_model_shared = nn.Linear(in_features=hidden_size, out_features=num_classes)

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        self.attn_proj = nn.Linear(hidden_size, (2+num_classes)*hidden_size)
        self.final_pred_fc = nn.Linear(hidden_size, num_classes)

        # # cqy
        self.device = device
        self.task = task
        print("self.device", self.device)
        self.sigma = torch.Tensor([1.01])
        self.D = hidden_size
        if self.task == 'phe':
            self.time_embedding = nn.Sequential(nn.Linear(1, self.D), nn.Tanh())
        else:
            self.time_embedding = nn.Sequential(nn.Linear(1, self.D), nn.Tanh())

        self.time_embedding = self.time_embedding.to(self.device)
        # print("self.time_embedding", self.time_embedding.device)
        D = self.D


        M = 512
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

        sigma_t = self.sigma_fun(t)
        

        if self.task == 'phe':
            sigma_t = sigma_t.unsqueeze(-1)
            # x_0 = x_0.squeeze(1)
            # x_1 = x_1.squeeze(1)
        x = x_0 + sigma_t * x_1
        
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

    def sample(self, x_0, batch_size=64, store=False):
        # 1) sample x_0 ~ Normal(0,1/(2log sigma) * (sigma**2 - 1))
        # x_t = self.sample_base(torch.empty(batch_size, self.D))
        x_t = x_0.to(self.device)
        
        # Apply Euler's method
        # NOTE: x_0 - data, x_1 - noise
        #       Therefore, we must use BACKWARD Euler's method! This results in the minus sign! 
        ts = torch.linspace(1, self.EPS, self.T).to(self.device)
        delta_t = ts[0] - ts[1]
        
        cnt=1
        for t in ts[1:]:
            tt = torch.Tensor([t]).to(self.device)
            # print("tt", t)
            # print("self.time_embedding(tt)", self.time_embedding(tt))
            # if self.task == 'phe':
            #     tt = tt.repeat(1,25)
            u = 0.5 * self.diffusion_coeff(tt).to(self.device) * self.nnet(x_t.to(self.device) + self.time_embedding(tt).to(self.device))
            x_t = x_t - delta_t * u
            # if store==True:
            #     with open('sample'+str(cnt)+'.pkl', 'wb') as f:
            #         pickle.dump(x_t, f)
            #     cnt+=1
        # x_t = torch.tanh(x_t)s
        return x_t

    def re_sample(self, x_0, batch_size=64, store=False):
        # 1) sample x_0 ~ Normal(0,1/(2log sigma) * (sigma**2 - 1))
        # x_t = self.sample_base(torch.empty(batch_size, self.D))
        x_t = x_0.to(self.device)
        
        # Apply Euler's method
        # NOTE: x_0 - data, x_1 - noise
        #       Therefore, we must use BACKWARD Euler's method! This results in the minus sign! 
        ts = torch.linspace(1, self.EPS, self.T_p).to(self.device)
        delta_t = ts[0] - ts[1]
        
        for t in ts[1:]:
            tt = torch.Tensor([t]).to(self.device)
            u = 0.5 * self.diffusion_coeff(tt).to(self.device) * self.nnet(x_t.to(self.device) + self.time_embedding(tt).to(self.device))
            x_t = x_t + delta_t * u
        # x_t = torch.tanh(x_t)
        return x_t


    def score_based_loss(self, patient_emb, aggr='sum', test_mode=False):
        ori_patient_emb = patient_emb
        x_1 = torch.randn_like(patient_emb).to(self.device)   

        t = torch.rand(size=(patient_emb.shape[0], 1))  * (1. - 1.e-5) + 1.e-5 

        t = t.to(self.device)
        # print("t device", self.device)
        x_0 = patient_emb.to(self.device)


        x_t = self.sample_p_t(x_0, x_1, t)


        t_embd = self.time_embedding(t).to(self.device)


        if self.task == 'phe':
            t_embd = t_embd.unsqueeze(1)


        nnet = self.nnet.to(self.device)

        x = x_t + t_embd 
        if self.task == 'phe':
            x_pooled = x.mean(dim=1) 
            x_1 = x_1.mean(dim=1)
            hazard = self.nnet(x_pooled)  
        else:
            hazard = self.nnet(x)  


        x_pred = -self.sigma_fun(t) * hazard

        final_embed = self.sample(ori_patient_emb, store=test_mode)

        a = self.lambda_t(t)
        b = torch.pow(x_pred + x_1, 2)
        c = a*b 
        score_matching_loss = 0.1 * c.mean(-1) 

        
        if aggr == 'sum':
            sm_loss = score_matching_loss.sum()
        else:
            sm_loss = score_matching_loss.mean()

        return final_embed, sm_loss


    def forward(self, x, img, seq_lengths, pairs, grl_lambda, test_mode=False):
        feat_ehr_shared, feat_ehr_distinct, pred_ehr = self.ehr_model(x, seq_lengths)
        feat_cxr = self.cxr_model_feat(img)
        feat_cxr_shared = self.cxr_model_shared(feat_cxr)
        feat_cxr_distinct = self.cxr_model_spec(feat_cxr)

        # get shared feature
        pred_cxr = self.cxr_model_linear(feat_cxr_distinct).sigmoid()

        feat_ehr_shared = self.shared_project(feat_ehr_shared)
        feat_cxr_shared = self.shared_project(feat_cxr_shared)

        pairs = pairs.unsqueeze(1)

        h1 = feat_ehr_shared
        h2 = feat_cxr_shared
        term1 = torch.stack([h1+h2, h1+h2, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)

        feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * feat_ehr_shared
        pred_shared = self.fuse_model_shared(feat_avg_shared).sigmoid()

        # Disease-wise Attention
        attn_input = torch.stack([feat_ehr_distinct, feat_avg_shared, feat_cxr_distinct], dim=1)
        # attn_input, sm_loss= self.score_based_loss(attn_input)

        qkvs = self.attn_proj(attn_input)
        q, v, *k = qkvs.chunk(2+self.num_classes, dim=-1)

        # compute query vector
        q_mean = pairs * q.mean(dim=1) + (1-pairs) * q[:, :-1].mean(dim=1)

        # compute attention weighting
        ks = torch.stack(k, dim=1)
        attn_logits = torch.einsum('bd,bnkd->bnk', q_mean, ks)
        attn_logits = attn_logits / math.sqrt(q.shape[-1])

        # filter out non-paired
        attn_mask = torch.ones_like(attn_logits)
        attn_mask[pairs.squeeze()==0, :, -1] = 0
        attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)

        # get final class-specific representation and prediction
        # v, sm_loss= self.score_based_loss(v)
        feat_final = torch.matmul(attn_weights, v)


        feat_final, sm_loss = self.score_based_loss(feat_final, test_mode=test_mode)

        pred_final = self.final_pred_fc(feat_final)
        pred_final = torch.diagonal(pred_final, dim1=1, dim2=2).sigmoid()

        outputs = {
            'feat_ehr_shared': feat_ehr_shared,
            'feat_cxr_shared': feat_cxr_shared,
            'feat_ehr_distinct': feat_ehr_distinct,
            'feat_cxr_distinct': feat_cxr_distinct,
            'feat_final': feat_final,
            'pred_final': pred_final,
            'pred_shared': pred_shared,
            'pred_ehr': pred_ehr,
            'pred_cxr': pred_cxr,
            'attn_weights': attn_weights,
            'sm': sm_loss,
            # 'pre_embed': pre_embed,
        }

        return outputs
