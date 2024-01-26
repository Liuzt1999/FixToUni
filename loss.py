import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import numpy as np
from torch.distributions.normal import Normal

def l2_norm(input):
    if len(input.shape) == 1:  # 方式出现训练最后一个step时，出现v是一维的情况
        input = torch.unsqueeze(input, 0)
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output
    
class RankingLoss(torch.nn.Module):
    def __init__(self, ranking_way, gen_anchor, embedding_size=128, scale = 32, margin_latent = 0.5, margin_embedding = 0.5, margin_anchor = 0.5, N = 2, r = 1.0):
        torch.nn.Module.__init__(self)
        self.margin_latent = margin_latent # marin in latent space
        self.margin_embedding = margin_embedding # margin in embedding
        self.margin_anchor = margin_anchor
        self.N = N # the number of generation
        self.r = r # the strength of generation
        self.embedding_size = embedding_size # the size of latent space
        self.directions = torch.nn.Parameter(torch.randn(self.N, self.embedding_size).cuda())
        nn.init.kaiming_normal_(self.directions, mode = 'fan_out')
        self.scale = scale
        self.ranking_way = ranking_way # chose ranking loss
        self.gen_anchor = gen_anchor # chose gen_anchor loss
    
    def get_loss_in_latent(self, X_w, X_s1, X_s2):
        satisfy_size = X_w.size(0)
        loss = list()
        for i in range(satisfy_size):
          weak = X_w[i]
          strong1 = X_s1[i]
          strong2 = X_s2[i]
          cos_w_s1 = F.linear(l2_norm(weak), l2_norm(strong1))
          cos_w_s2 = F.linear(l2_norm(weak),l2_norm(strong2))
          loss1 = torch.log(1 + torch.exp(cos_w_s2 - cos_w_s1 + self.margin_latent))
          loss.append(loss1)
        loss = sum(loss) / satisfy_size
        return loss
      
    def get_loss_after_latent(self, X_w, target_X_w, model):
        satisfy_size = X_w.size(0)
        sementic_changes = self.directions
        sementic_changes = l2_norm(sementic_changes)
        # get random directions
        for i in range(self.N):
            sementic_changes[i] = sementic_changes[i] * (i + 1) * self.r
        sementic_changes = l2_norm(sementic_changes)
        loss_ranking = list() 
        loss_anchor = list() 
        for i in range(satisfy_size):
            label = target_X_w[i]
            if len(label.shape) == 0:  # fix dimension problem
                label = torch.unsqueeze(label, 0)
            gen_samples = torch.empty(self.N, X_w.size(1)).cuda()
            for j in range(self.N):
                gen_samples[j] = X_w[i] + sementic_changes[j]
            gen_samples = l2_norm(gen_samples)
            feat_X_w = model.fc(X_w[i])
            feat_gen = model.fc(gen_samples)
            cos_gx = F.linear(feat_gen, feat_X_w)
            
            if (self.ranking_way == 'triplet'):
              loss_gen_ranking = torch.log(1 + torch.exp(cos_gx[1] - cos_gx[0] + self.margin_embedding))
            
            elif (self.ranking_way == 'ranking1'):
              loss_gen_temp = list()
              for i in range(self.N - 1):
                loss_temp = torch.exp(cos_gx[i + 1] - cos_gx[i] + self.margin_embedding)
                loss_gen_temp.append(loss_temp) 
              loss_gen_ranking = torch.log(1 + sum(loss_gen_temp))            
            
            elif (self.ranking_way == 'ranking2'):
              loss_gen_temp = list()
              g_loss_l_mid = list() 
              g_loss_r_mid = list() 
              for k in range(self.N):
                g_loss_l_inner = list()
                g_loss_r_inner = list()
                if k > 0:
                    for h in range(k):
                        loss_l = torch.exp(self.scale * (cos_gx[k] - cos_gx[h] + self.margin_embedding))
                        g_loss_l_inner.append(loss_l)
                    g_loss_l_inner_ = sum(g_loss_l_inner)
                    g_loss_l_mid.append(g_loss_l_inner_)

                if k < self.N - 1:
                    for j in range(k + 1, self.N):
                        loss_r = torch.exp(self.scale * (cos_gx[j] - cos_gx[k] + self.margin_embedding))
                        g_loss_r_inner.append(loss_r)
                    g_loss_r_inner_ = sum(g_loss_r_inner)
                    g_loss_r_mid.append(g_loss_r_inner_)
              g_loss_l_mid_ = sum(g_loss_l_mid)
              g_loss_r_mid_ = sum(g_loss_r_mid)
              g_loss_l_mid_ = (1 / self.scale) * torch.log(1 + g_loss_l_mid_)
              g_loss_r_mid_ = (1 / self.scale) * torch.log(1 + g_loss_r_mid_)
              loss_gen_ranking = g_loss_l_mid_ + g_loss_r_mid_
             
            
            loss_ranking.append(loss_gen_ranking)
            
            feat_gen_0 = torch.unsqueeze(feat_gen[0],0)
            feat_gen_1 = torch.unsqueeze(feat_gen[1],0)
            
            loss_gen_anchor_metric = 0.5 * torch.log(1 + torch.exp(self.margin_anchor - cos_gx[0])) + 0.5 * torch.log(1 + torch.exp(self.margin_anchor - cos_gx[1]))
                                
            loss_gen_anchor_ce = 0.5 * (F.cross_entropy(feat_gen_0, label, reduction='none') + F.cross_entropy(feat_gen_1, label, reduction='none'))
            
            if (self.gen_anchor == 'metric'): # metric learning way
              loss_anchor.append(loss_gen_anchor_metric)
            elif (self.gen_anchor == 'CE'): # cross entropy
              loss_anchor.append(loss_gen_anchor_ce)
            elif (self.gen_anchor == 'both'): # both
              loss_anchor.append(0.5 * loss_gen_anchor_metric)
              loss_anchor.append(0.5 * loss_gen_anchor_ce)
            
        loss = (sum(loss_ranking) + sum(loss_anchor)) / satisfy_size
        return loss
   
    def forward(self, X_w_latent, X_s1_latent, X_s2_latent, target_X_w, model):
        loss1 = self.get_loss_in_latent(X_w_latent, X_s1_latent, X_s2_latent)
        loss2 = self.get_loss_after_latent(X_w_latent, target_X_w, model)
        return loss1 + loss2
        