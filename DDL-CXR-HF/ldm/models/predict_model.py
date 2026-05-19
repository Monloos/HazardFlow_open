import math
from typing import Any
import pandas as pd
import torch
import os
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
import torchvision
from ldm.modules.diffusionmodules.openaimodel import QKVAttention, AttentionBlock
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
import pytorch_lightning as pl
from sklearn.metrics import f1_score,roc_auc_score, average_precision_score
# import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_score, recall_score
from torch.distributions.normal import Normal
import math
import pickle

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 4, patch_size: int = 4, emb_size: int = 128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x


# CNN, padding, Transformer
class ResnetMultiCXR(pl.LightningModule):
    def __init__(self,  
                 task,  
                 VAE_config=None,
                 vision_backbone='resnet34',    
                 pretrained=True,    
                 ehr_encoder_config=None,     
                 fusion_config=None,    
                 hidden_size=128,    
                 hid_dim_1=1024,    
                 dropout=0.1,    
                 drop_cxr=0.0,    
                 ): 
        pass

class ResnetCXR(pl.LightningModule):
    def __init__(self,
                 task,
                 VAE_config=None,
                 vision_backbone='resnet34',
                 pretrained=True,
                 ehr_encoder_config=None,     
                 fusion_config=None, 
                 hidden_size=128,
                 hid_dim_1=1024,
                 dropout=0.1,
                 drop_cxr=0.0,
                 ):
        
        super().__init__(task)

        self.task = task

class ResnetMultiCxrEHR(pl.LightningModule):
    def __init__(self,
                 task,
                 VAE_config=None,
                 vision_backbone='resnet34',
                 pretrained=True,
                 ehr_encoder_config=None,     
                 fusion_config=None, 
                 hidden_size=128,
                 hid_dim_1=1024,
                 dropout=0.1,
                 drop_cxr=0.0,
                 ):
        
        super().__init__(task)

        self.task = task
        if self.task=='mortality':    
            pos_weight = torch.tensor([5.89])    
            num_classes=1    
        if self.task=='phenotype':    
            pos_weight = torch.tensor([1.0])    
            num_classes=1    
          

class ResnetCxrEHR(pl.LightningModule):
    def __init__(self,
                 task,
                 fusion_way,
                 latent_cxr=False,
                 VAE_config=None,
                 vision_backbone='resnet34',
                 pretrained=True,
                 ehr_encoder_config=None,     
                 fusion_config=None, 
                 hidden_size=128,
                 hid_dim_1=1024,
                 dropout=0.1,
                 drop_cxr=0.0,
                 
                 mode='max',          
                 max_epoch=100,
                 ckpt_path=None,
                 ignore_keys=[]
                 ):
        
        super().__init__(task)

        self.fusion_way=fusion_way
        self.mode = mode
        self.task = task
        self.max_epoch=max_epoch
        self.monitor='val/pr_auc'


        if self.task=='mortality':
            pos_weight = torch.tensor([5.89])
            num_classes=1
            
        if self.task=='phenotype':
            pos_weight = torch.tensor([1.66, 10.01, 10.45, 1.51, 3.09, 4.98, 3.22, 7.85, 2.07, 2.2, 6.96, 3.81, 1.51, 1.28, 0.92, 13.53, 3.55, 4.72, 6.16, 12.68, 8.3, 3.48, 1.95, 2.71, 3.44])
            # pos_weight =torch.ones(25)
            num_classes=25
            
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.num_classes = num_classes

        # self.subgroup=[(0,12),(12,19),(19,32),(32,47),(47,71)]
        # self.test_df=pd.read_csv(test_pred_path)

        
        self.ehr_encoder=instantiate_from_config(ehr_encoder_config)

        self.vision_backbone = getattr(torchvision.models, vision_backbone)(pretrained=pretrained)
        classifiers = [ 'classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features
            setattr(self.vision_backbone, classifier, nn.Identity(d_visual))
            break
        
        self.cxr_feat_project=nn.Sequential(nn.Linear(d_visual, hidden_size),
                            nn.GELU(),nn.Dropout(drop_cxr))
 
        if fusion_way=='attention':
            self.fusion_tf=instantiate_from_config(fusion_config)

        
        
        
        if fusion_way=='concat':
            self.concat_linear=nn.Sequential(nn.Linear(hidden_size*2, hidden_size),
                            nn.GELU(),nn.Dropout(drop_cxr))


        self.mlp_head  = nn.Sequential(nn.Linear(hidden_size, hid_dim_1),
                              nn.GELU(),
                              nn.Dropout(dropout),
                              nn.Linear(hid_dim_1,  self.num_classes)
                              )
        # self.mlp_head = nn.Linear(hidden_size, self.num_classes)
        self.latent_cxr=latent_cxr
        if latent_cxr:
            assert VAE_config is not None
            self.instantiate_decoder(VAE_config)


        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def instantiate_decoder(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_input(self, batch):

        ehr = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        ehr = ehr.to(self.device)

        
        y= torch.stack(list(map(lambda x: x[1], batch)),dim=0)
        y = y.to(self.device).float()

        x = torch.stack(list(map(lambda x: x[2], batch)), dim=0)
        x = x.to(memory_format=torch.contiguous_format)
        x = x.to(self.device)

        if self.latent_cxr:
            x=self.first_stage_model.decode(x)

        sample_id=torch.tensor(list(map(lambda x: x[3], batch)))
        sample_id = sample_id.to(self.device)

        return ehr, y, x, sample_id

    def forward(self, ehr=None, x=None ) -> Any:

        # encode ehr
        cls_ehr,_=self.ehr_encoder.encode(ehr)
        cls_ehr=cls_ehr.squeeze()
        # encode cxr
        visual_feats = self.vision_backbone(x)
        cls_cxr = self.cxr_feat_project(visual_feats)

        # fusion
        cls = torch.cat([cls_ehr, cls_cxr], dim=-1)
        assert self.fusion_way=='concat'
        fused_cls=self.concat_linear(cls)

        # predict
        ret = self.mlp_head(fused_cls).squeeze(dim=1)
        return ret
            
       


       
    def training_step(self, batch,batch_idx):
            
        ehr, target, x,_= self.get_input(batch)
        output = self(ehr=ehr, x=x)
        
        loss = self.loss(output, target)

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                batch_size=target.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):

        ehr, target, x,_= self.get_input(batch)
        output = self(ehr=ehr, x=x)
        
        
        CEloss = self.loss(output, target)

        
        preds = torch.sigmoid(output)

        return {'val_loss': CEloss, 'target': target, 'preds': preds}

    def test_step(self, batch, batch_idx) :

        ehr, target, x,sample_id= self.get_input(batch)
        output = self(ehr=ehr, x=x)
        
        CEloss = self.loss(output, target)

       
        preds = torch.sigmoid(output)

        return {'test_loss': CEloss, 'target': target, 'preds': preds, 'sample_id':sample_id}


class FusionTokens3inputAttnFuse(pl.LightningModule):
    """
    The prediciton model of DDL-CXR, using Attention as the final fusion.
    """
    def __init__(self,
                 task,
                 sbs,
                 z1_embed_type='linear',
                 ehr_encoder_config=None,
                 cxr_transformer_config=None,
                 fusion_config=None,
                 vision_backbone='resnet34',
                 hidden_size=128,
                 hid_dim_1=128,
                 dropout=0.1,
                 conv_in_chan=4,
                 conv_out_chan=49,
                 z0_view_size=3136,
                 z0_view_size_chan1=784,
                 ehr_modal=False,
                 x0_modal=False,
                 z1_modal=False,
                 use_pos_weight=True,
                 fusion_way='na',
                 monitor='val/pr_auc',
                 mode='max',
                 max_epoch=100,
                 ckpt_path=None,

                 ignore_keys=[]
                 ):
        super().__init__()

       
        self.vision_backbone = getattr(torchvision.models, vision_backbone)(pretrained=True)
        classifiers = [ 'classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features
            setattr(self.vision_backbone, classifier, nn.Identity(d_visual))
            break

        # self.cxr_feat_project=nn.Sequential(nn.Linear(d_visual, hidden_size),
        #                     nn.GELU(),nn.Dropout(dropout))
        self.cxr_feat_project=nn.Linear(d_visual, hidden_size)


        self.ehr_modal=ehr_modal
        self.z1_modal = z1_modal
        self.x0_modal=x0_modal  
        self.sbs = sbs


        self.mode = mode

        self.task = task
        self.max_epoch=max_epoch
        self.monitor=monitor
        
        
        if self.task=='mortality':
            pos_weight = torch.tensor([5.89])
            num_classes=1
            
        if self.task=='phenotype':
            if use_pos_weight:
                pos_weight = torch.tensor([1.66, 10.01, 10.45, 1.51, 3.09, 4.98, 3.22, 7.85, 2.07, 2.2, 6.96, 3.81, 1.51, 1.28, 0.92, 13.53, 3.55, 4.72, 6.16, 12.68, 8.3, 3.48, 1.95, 2.71, 3.44])
            else:
                pos_weight =torch.ones(25)
            num_classes=25


        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.num_classes = num_classes


        if self.ehr_modal:
            self.ehr_encoder=instantiate_from_config(ehr_encoder_config)


        self.z1_embed_type=z1_embed_type
        if self.z1_modal:
            # # encode image
            if z1_embed_type=='patchembedding':
               
                self.cxr_embdedding=PatchEmbedding()
                
                assert cxr_transformer_config is not None
                self.cxr_transformer=instantiate_from_config(cxr_transformer_config)

            elif z1_embed_type=='conv_trans':
                self.image_conv = nn.Conv2d(conv_in_chan, conv_out_chan, kernel_size=3, stride=1, padding=1)
                cxr_transformer_config['params']['feat_dim']=z0_view_size_chan1
                self.cxr_encoder=instantiate_from_config(cxr_transformer_config)

            elif self.z1_embed_type == 'conv_linear':
                self.image_conv = nn.Conv2d(conv_in_chan, conv_out_chan, kernel_size=3, stride=1, padding=1)
                self.image_fc=nn.Sequential(nn.Linear(z0_view_size_chan1, hidden_size),
                                nn.GELU())

            else:
                assert z1_embed_type=='linear'
                # (b,4,28,28) -> (b,3136) -> (b,3136) -> (b,128)
                self.cxr_linear=nn.Sequential(nn.Linear(z0_view_size, hidden_size),
                                nn.GELU())  #,nn.Dropout(dropout))


        self.fusion_way = fusion_way


        if fusion_way=='attention':
            # assert 'linear' not in cxr_embed_type, "linear cxr embedding cannot do attention"
            self.fusion_tf=instantiate_from_config(fusion_config)


        n_modal=int(ehr_modal+z1_modal+x0_modal)
        if fusion_way=='concat':
            self.concat_linear=nn.Sequential(nn.Linear(hidden_size*n_modal, hidden_size),
                            nn.GELU(),nn.Dropout(dropout))


        self.mlp_head  = nn.Sequential(nn.Linear(hidden_size, hid_dim_1),
                              nn.GELU(),
                            #   nn.Dropout(dropout),
                              nn.Linear(hid_dim_1,  self.num_classes)
                              )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        ''' The initialization below is from HazardFlow '''
        self.sigma = torch.Tensor([1.01])
        self.sigma = self.sigma.to(self.device)
        self.D = hidden_size
        # self.D = 128
        self.time_embedding = nn.Sequential(nn.Linear(1, self.D), nn.Tanh())
        # self.device = device

        M = 512
        # self.B = 128
        # self.nnet = HazardAttentionNet(emb_dim=self.D)
        self.sigmoid = nn.Sigmoid()

        ''' This is the score network'''
        self.nnet = nn.Sequential(nn.Linear(self.D, M), nn.SiLU(),
                     nn.Linear(M, M), nn.SiLU(),
                     nn.Linear(M, M), nn.SiLU(),
                     nn.Linear(M, self.D), nn.Hardtanh(min_val=-3., max_val=3.))
        # self.nnet = self.nnet.to(self.device)
        self.mu = torch.nn.Parameter(torch.randn(1, self.D)).to(self.device)

        self.T = 10
        self.T_p = 5
        self.EPS = 1.e-5
        # self.base = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(self.D), var * torch.eye(self.D))


    def init_from_ckpt(self, path, ignore_keys=list()):

        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @torch.no_grad()
    def get_input(self, batch):
        # (ehr, y, x_0, sample_id,z1)
        ehr = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        ehr = ehr.to(self.device)


        y= torch.stack(list(map(lambda x: x[1], batch)),dim=0)
        y = y.to(self.device).float()
       

        z = torch.stack(list(map(lambda x: x[4], batch)), dim=0)
        z = z.to(memory_format=torch.contiguous_format)
        z = z.to(self.device)

        x0 = torch.stack(list(map(lambda x: x[2], batch)), dim=0)
        x0 = x0.to(memory_format=torch.contiguous_format)
        x0 = x0.to(self.device)

        sample_id=torch.tensor(list(map(lambda x: x[3], batch)))
        sample_id = sample_id.to(self.device)

        return ehr, y, z, sample_id,x0





    def calculate_f1_score(self, predictions, target):

        preds = torch.argmax(predictions, dim=1).cpu()
        target = target.cpu()
        f1_micro = f1_score(target.numpy(), preds.numpy(), average='micro')
        f1_macro = f1_score(target.numpy(), preds.numpy(), average='macro')
        return torch.tensor(f1_micro).to(self.device), torch.tensor(f1_macro).to(self.device)

    def calculate_metrics(self, predictions, target):


        preds = predictions.numpy()
        target = target.numpy()

        roc_auc = roc_auc_score(target, preds)
        pr_auc = average_precision_score(target, preds)

        return torch.tensor(roc_auc).to(self.device), torch.tensor(pr_auc).to(self.device)


    ''' The functions below are from HazardFlow '''
    def sample_p_t(self, x_0, x_1, t):
        # sampling from p_0t(x_t|x_0)
        # x_0 ~ data, x_1 ~ noise
        x = x_0 + self.sigma_fun(t) * x_1
        
        return x

    def lambda_t(self, t):
        return self.sigma_fun(t)**2

    def sigma_fun(self, t):
        sigma = self.sigma
        sigma = sigma.to(self.device)
        val = (1./(2. * torch.log(sigma))) * (sigma**(2.*t) - 1.)
        val = torch.clamp(val, min=1e-5)  

        return torch.sqrt(val)

    def diffusion_coeff(self, t):
        # the diffusion coefficient in the SDE
        return self.sigma.to(self.device)**t.to(self.device)

    def sample_base(self, x_0):
        # sampling from the base distribution
        return self.base.rsample(sample_shape=torch.Size([x_0.shape[0]]))

    ''' This is hazard escalation '''
    def sample(self, x_0, batch_size=64, test_mode=False):
        x_t = x_0.to(self.device)
        
        # Apply Euler's method
        # NOTE: x_0 - data, x_1 - noise
        #       Therefore, we must use BACKWARD Euler's method! This results in the minus sign! 
        ts = torch.linspace(1., self.EPS, self.T).to(self.device)
        delta_t = ts[0] - ts[1]
        
        cnt=1
        for t in ts[1:]:
            tt = torch.Tensor([t]).to(self.device)
            u = 0.5 * self.diffusion_coeff(tt).to(self.device) * self.nnet(x_t + self.time_embedding(tt).to(self.device))
            x_t = x_t - delta_t * u
        return x_t

    ''' This is hazard reduction '''
    def re_sample(self, x_0, batch_size=64):

        x_t = x_0.to(self.device)
        
        # Apply Euler's method
        # NOTE: x_0 - data, x_1 - noise
        #       Therefore, we must use BACKWARD Euler's method! This results in the minus sign! 
        ts = torch.linspace(1., self.EPS, self.T_p).to(self.device)
        delta_t = ts[0] - ts[1]
        
        for t in ts[1:]:
            tt = torch.Tensor([t]).to(self.device)
            u = 0.5 * self.diffusion_coeff(tt).to(self.device) * self.nnet(x_t + self.time_embedding(tt).to(self.device))
            x_t = x_t + delta_t * u
        
        # x_t = torch.tanh(x_t)
        return x_t

    # def score_based_loss(self, patient_emb, y_true, aggr='sum'):
    ''' This is the score matching loss'''
    def score_based_loss(self, patient_emb, aggr='mean', test_mode=False):
        x_1 = torch.randn_like(patient_emb).to(self.device)   
        # print("x_1", x_1)
        t = torch.rand(size=(patient_emb.shape[0], 1))  * (1. - 1.e-5) + 1.e-5 
        if test_mode == True:
            t = torch.ones(size=(patient_emb.shape[0],1))
        t = t.to(self.device)
            
        x_0 = patient_emb.to(self.device)
        x_t = self.sample_p_t(x_0, x_1, t)

        t_embd = self.time_embedding(t)

        hazard = self.nnet(x_t+t_embd)
        x_pred = -self.sigma_fun(t) * hazard

        final_embed = self.sample(patient_emb, test_mode=test_mode)

        score_matching_loss = 0.1*self.lambda_t(t) * torch.pow(x_pred + x_1, 2).mean(-1) 
        
        if aggr == 'sum':
            sm_loss = score_matching_loss.sum()
        else:
            sm_loss = score_matching_loss.mean()

        return sm_loss, final_embed



    def forward(self, ehr=None, z=None, x0=None, test_mode=False ) -> Any:

        ehr_cls=None
        z1_cls=None
        x0_cls=None

        multimodal_reps = []

        if self.ehr_modal:
            # ehr_cls:(b,1,d_model)  encoded_ehr:(b,49,d_model)
            ehr_cls, encoded_ehr=self.ehr_encoder.encode(ehr)
            # ehr_cls:(b,1,d_model)
            encoded_ehr = encoded_ehr[:, 1:]
            multimodal_reps.append(encoded_ehr)

        if self.z1_modal:
            if self.z1_embed_type=='patchembedding':
                patched_z = self.cxr_embdedding(z)
                # z1_cls:(b,1,d_model)  encoded_cxr:(b,49,d_model)
                z1_cls, encoded_cxr = self.cxr_transformer.encode(patched_z)
                # z1_cls:(b,1,d_model)
                encoded_cxr = encoded_cxr[:, 1:]
                multimodal_reps.append(encoded_cxr)

            elif self.z1_embed_type == 'conv_trans':
                z=self.image_conv(z)
                b,out_chan,*spatial=z.shape
                #·Flatten the image tensor
                z=z.view(b, out_chan,-1)
                #cls=self.image_fc(z).mean(dim=1,keepdim=True)
                z1_cls, encoded_cxr=self.cxr_encoder.encode(z)
                z1_cls=z1_cls
                multimodal_reps.append(encoded_cxr)

            elif self.z1_embed_type == 'conv_linear':
                # (b,conv_out_channel,28,28)
                z=self.image_conv(z)
                b,out_chan,*spatial=z.shape
                #·Flatten the image tensor
                # (b,conv_out_channel, 28*28)
                z=z.view(b, out_chan,-1)
                # (b,49,128)
                z1_cls=self.image_fc(z).mean(dim=1,keepdim=True)
                # z1_cls, encoded_cxr=self.cxr_encoder.encode(z)
                # z1_cls=z1_cls.squeeze(dim=1)
                multimodal_reps.append(z1_cls)

            elif self.z1_embed_type=='linear':
                b = z.shape[0]
                z=z.view(b,-1)
                # z1_cls:(b,d_model)
                z1_cls=self.cxr_linear(z).unsqueeze(dim=1)
                multimodal_reps.append(z1_cls)
            else:
                raise NotImplementedError('unknown z1_emb_type')


        # encode cxr
        visual_feats = self.vision_backbone(x0)
        x0_cls = self.cxr_feat_project(visual_feats).unsqueeze(dim=1)
        multimodal_reps.append(x0_cls)

        seq = torch.cat(multimodal_reps, dim=1)

        if self.fusion_way == 'attention':
            fused_cls,_=self.fusion_tf.encode(seq)
            fused_cls=fused_cls.squeeze(dim=1)
       
        else:
            raise NotImplementedError('not implemented')

        ''' We plug the HazardFlow here '''
        if self.sbs == True:       
            # print("true")
            # sm_loss, fused_cls, pre_output = self.score_based_loss(fused_cls, aggr='mean', test_mode=test_mode)
            sm_loss, fused_cls = self.score_based_loss(fused_cls, aggr='mean', test_mode=test_mode)
        
        # print("fused_cls, fused_cls.shape", fused_cls, fused_cls.shape)
        output = self.mlp_head(fused_cls)
        # print("self.mlp_head", self.mlp_head)
        pre_output = self.mlp_head(fused_cls)
        # print("output", output, output.shape)
        # if test_mode==True:
        #     torch.save(self.mlp_head.state_dict(), 'mlp_head.pth')

        if self.task == 'mortality':
            output = output.squeeze(1)
            pre_output = pre_output.squeeze(1)
        # print("output_sq", output, output.shape)

        if self.sbs == True:
            return output, sm_loss
            # return output, sm_loss, pre_output
        else:
            return output

    def training_step(self, batch,batch_idx):

        ehr, y, z, sample_id,x0= self.get_input(batch)

        if self.sbs == True:
            # output, sm_loss, pre_output = self(ehr=ehr, z=z, x0=x0, test_mode=False)
            # loss = self.loss(output, y) + sm_loss + self.loss(pre_output, torch.zeros_like(y))
            output, sm_loss = self(ehr=ehr, z=z, x0=x0, test_mode=False)
            loss = self.loss(output, y) + sm_loss
        else:
            output = self(ehr=ehr, z=z, x0=x0, test_mode=False)
            loss = self.loss(output, y)

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=y.shape[0])
       

        return loss

    def validation_step(self, batch, batch_idx):


        ehr, y, z, sample_id,x0= self.get_input(batch)

        if self.sbs == True:
            # output, sm_loss, pre_output = self(ehr=ehr, z=z, x0=x0, test_mode=True)
            output, sm_loss = self(ehr=ehr, z=z, x0=x0, test_mode=False)
            loss = self.loss(output, y) + sm_loss
            # loss = self.loss(output, y) + sm_loss + self.loss(pre_output, torch.zeros_like(y))
        else:
            output = self(ehr=ehr, z=z, x0=x0, test_mode=False)
            loss = self.loss(output, y)

        preds = torch.sigmoid(output)
        

        return {'val_loss': loss, 'target': y, 'preds': preds}


    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_target = torch.cat([x['target'] for x in outputs]).cpu()
        all_preds = torch.cat([x['preds'] for x in outputs]).cpu()

        roc_auc, pr_auc = self.calculate_metrics(all_preds, all_target)
        self.log('val/loss', avg_loss, prog_bar=True, on_epoch=True)
        self.log('val/roc_auc', roc_auc, prog_bar=True, on_epoch=True)
        self.log('val/pr_auc', pr_auc, prog_bar=True, on_epoch=True)

        
        del avg_loss,all_target,all_preds



    def test_step(self, batch, batch_idx) :



        ehr, y, z, sample_id,x0= self.get_input(batch)

        if self.sbs == True:
            # output, sm_loss, pre_output = self(ehr=ehr, z=z, x0=x0, test_mode=True)
            # loss = self.loss(output, y) + sm_loss + self.loss(pre_output, torch.zeros_like(y))
            output, sm_loss = self(ehr=ehr, z=z, x0=x0, test_mode=True)
            loss = self.loss(output, y) + sm_loss
        else:
            output = self(ehr=ehr, z=z, x0=x0, test_mode=False)
            loss = self.loss(output, y)
        # with open('23y_gt.pkl', 'wb') as f:
        #     pickle.dump(y, f)
        preds = torch.sigmoid(output)

        return {'test_loss': loss, 'target': y, 'preds': preds, 'sample_id':sample_id}


    def test_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        all_target = torch.cat([x['target'] for x in outputs]).cpu()
        all_preds = torch.cat([x['preds'] for x in outputs]).cpu()
        all_sample_id = torch.cat([x['sample_id'] for x in outputs]).cpu()


        roc_auc, pr_auc = self.calculate_metrics(all_preds, all_target)
        self.log('test/loss', avg_loss, prog_bar=True, on_epoch=True)
        self.log('test/roc_auc', roc_auc, prog_bar=True, on_epoch=True)
        self.log('test/pr_auc', pr_auc, prog_bar=True, on_epoch=True)

        
        if self.task=='mortality':
            data={'sample_id':all_sample_id,'target':all_target,'pred':all_preds}

            logdir=self.trainer.logger.save_dir
            df_results=pd.DataFrame(data=data)
            df_results.to_csv(os.path.join(logdir,'results.csv'))

            df_metrics=pd.DataFrame(data={'roc_auc':[roc_auc.item()],'pr_auc':[pr_auc.item()]})
            df_metrics.to_csv(os.path.join(logdir,'metrics.csv'))

        if self.task=='phenotype':


            namelist = ['sample_id']+["gt_" + str(i) for i in range(25)]+["gen_" + str(i) for i in range(25)]
            concatenated_tensor = np.concatenate((all_sample_id.unsqueeze(dim=1),all_target,all_preds), axis=1)

            df_results = pd.DataFrame(concatenated_tensor, columns=namelist)
            logdir=self.trainer.logger.save_dir

            df_results.to_csv(os.path.join(logdir,'results.csv'))



    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        schedular=CosineAnnealingLR(opt,T_max=self.max_epoch,eta_min=1e-7)
        return [opt],[schedular]


