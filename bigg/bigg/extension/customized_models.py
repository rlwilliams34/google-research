# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from bigg.model.tree_model import RecurTreeGen
# import torch
# from bigg.common.pytorch_util import glorot_uniform, MLP
# import torch.nn as nn

from bigg.model.tree_model import RecurTreeGen
from bigg.extension.gcn_build import *
import torch
from bigg.common.pytorch_util import glorot_uniform, MLP, MultiLSTMCell, BinaryTreeLSTMCell
import torch.nn as nn
import numpy 
from torch.nn.parameter import Parameter
from datetime import datetime

# pylint: skip-file
class BiggWithEdgeLen(RecurTreeGen):
    def __init__(self, args):
        super().__init__(args)
        cmd_args.wt_drop = -1
        self.method = args.method
        self.sampling_method = cmd_args.sampling_method
        self.update_left = args.update_left
        
        assert self.sampling_method in ['gamma', 'lognormal', 'softplus', 'vae']
        assert self.method in ['Test', 'MLP-Repeat', 'MLP-Multi', 'MLP-Double', 'LSTM', 'MLP-Leaf']
        
        self.nodelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])
        self.nodelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, 1])
        
        self.edgelen_mean = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
        self.edgelen_lvar = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)
        #self.update_wt = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
        self.update_wt = nn.LSTMCell(args.weight_embed_dim, args.embed_dim)
        self.topdown_update_wt = nn.LSTMCell(args.weight_embed_dim, args.embed_dim)
        #self.topdown_update_wt = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
        
        if self.method == "Test":
            self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = cmd_args.wt_drop)
        
        if self.method == "MLP-Repeat":
            self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])#, dropout = cmd_args.wt_drop)
            
        if self.method == "MLP-2":
            self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])#, dropout = cmd_args.wt_drop)
            self.leaf_h0_top = Parameter(torch.Tensor(1, args.embed_dim))
            self.leaf_c0_top = Parameter(torch.Tensor(1, args.embed_dim))
        
        if self.method == "MLP-Multi":
            self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim * args.rnn_layers], dropout = cmd_args.wt_drop)
        
        if self.method == "MLP-Double":
            self.edgelen_encoding_h = MLP(1, [2 * args.embed_dim, args.embed_dim * args.rnn_layers], dropout = cmd_args.wt_drop)
            self.edgelen_encoding_c = MLP(1, [2 * args.embed_dim, args.embed_dim * args.rnn_layers], dropout = cmd_args.wt_drop)
        
        if self.method == "LSTM":
            self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = cmd_args.wt_drop)
            #self.edgeLSTM = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
            self.edgeLSTM = nn.LSTMCell(args.weight_embed_dim, args.embed_dim)
            
            self.leaf_h0_wt = Parameter(torch.Tensor(1, args.embed_dim))
            self.leaf_c0_wt = Parameter(torch.Tensor(1, args.embed_dim))
            
            self.edgelen_mean = MLP(2 * args.embed_dim, [3 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
            self.edgelen_lvar = MLP(2 * args.embed_dim, [3 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
        
        if self.method == "LSTM2":
            self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = cmd_args.wt_drop)
            #self.edgeLSTM = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
            self.edgeLSTM = nn.LSTMCell(args.weight_embed_dim, args.embed_dim)
            
            self.leaf_h0_wt = Parameter(torch.Tensor(1, args.embed_dim))
            self.leaf_c0_wt = Parameter(torch.Tensor(1, args.embed_dim))
            
            self.edgelen_mean = MLP(2 * args.embed_dim, [3 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
            self.edgelen_lvar = MLP(2 * args.embed_dim, [3 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
            
        
        if self.method == "MLP-Leaf":
            self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = cmd_args.wt_drop)
            self.wt_h0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim // 2))
            self.wt_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim // 2))
            self.leaf_h0_2 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
            self.leaf_c0_2 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
            
            self.edgeLSTM = MultiLSTMCell(args.weight_embed_dim, args.embed_dim // 2, args.rnn_layers)
            
            self.edgelen_mean = MLP(int(1.5 * args.embed_dim), [2 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
            self.edgelen_lvar = MLP(int(1.5 * args.embed_dim), [2 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
        
        self.embed_dim = args.embed_dim
        self.num_layers = args.rnn_layers
        
        mu_wt = torch.tensor(0, dtype = float)
        var_wt = torch.tensor(1, dtype = float)
        n_obs = torch.tensor(0, dtype = int)
        min_wt = torch.tensor(torch.inf, dtype = float)
        max_wt = torch.tensor(-torch.inf, dtype = float)
        epoch_num = torch.tensor(0, dtype = int)
        
        self.register_buffer("mu_wt", mu_wt)
        self.register_buffer("var_wt", var_wt)
        self.register_buffer("n_obs", n_obs)
        self.register_buffer("min_wt", min_wt)
        self.register_buffer("max_wt", max_wt)
        self.register_buffer("epoch_num", epoch_num)
        self.mode = args.wt_mode
        
        self.log_wt = False
        self.sm_wt = False
        self.wt_range = 1.0
        self.wt_scale = 1.0
        
        glorot_uniform(self)
        
#         if args.sampling_method == "vae":
#             self.embed_wt_vae = MLP(1, [2 * args.embed_dim, args.embed_dim])
#             self.vae_mu = MLP(2 * args.embed_dim, [4 * args.embed_dim, 8])
#             self.vae_sig = MLP(2 * args.embed_dim, [4 * args.embed_dim, 8])
#             self.weight_out = MLP(args.embed_dim + 8, [2 * args.embed_dim + 16, 1])

    def standardize_edge_feats(self, edge_feats): 
      if self.log_wt:
        edge_feats = torch.log(edge_feats)
      
      elif self.sm_wt:
        edge_feats = torch.log(torch.special.expm1(edge_feats))
      
      if self.mode == "none":
        return edge_feats
      
      if self.epoch_num == 1:
        self.update_weight_stats(edge_feats)
      
      if self.mode == "score":
        edge_feats = (edge_feats - self.mu_wt) / (self.var_wt**0.5 + 1e-15)
          
      elif self.mode == "normalize":
        edge_feats = -1 + 2 * (edge_feats - self.min_wt) / (self.max_wt - self.min_wt + 1e-15)
        edge_feats = self.wt_range * edge_feats
      
      elif self.mode == "scale":
        edge_feats = edge_feats * self.wt_scale
      
      elif self.mode == "exp":
        edge_feats = torch.exp(-1/edge_feats)
      
      elif self.mode == "exp-log":
        edge_feats = torch.exp(-1 / torch.log(1 + edge_feats))
        
      return edge_feats
  
    def update_weight_stats(self, edge_feats):
      '''
      Updates necessary global statistics (mean, variance, min, max) of edge_feats per batch
      if standardizing edge_feats prior to MLP embedding. Only performed during the
      first epoch of training.
      
      Args Used:
        edge_feats: edge_feats from current iteration batch
      '''
      
      ## Current training weight statistics
      with torch.no_grad():
        if self.mode == "score":
          mu_n = self.mu_wt
          var_n = self.var_wt
          n = self.n_obs
          
          ## New weight statistics
          m = len(edge_feats)
          
          if m > 1:
            var_m = torch.var(edge_feats)
          else:
            var_m = 0.0
            
          mu_m = torch.mean(edge_feats)
          tot = n + m
          
          if tot == 1:
            self.mu_wt = mu_m
            self.n_obs = tot
         
          else:
            ## Update weight statistics
            new_mu = (n * mu_n + m * mu_m) / tot
            
            new_var_avg = (max(n - 1, 0) * var_n + (m - 1) * var_m)/(tot - 1)
            new_var_resid = n * m * (mu_n - mu_m)**2 / (tot * (tot - 1))
            new_var = new_var_avg + new_var_resid
            
            ## Save
            self.mu_wt = new_mu
            self.var_wt = new_var
            self.n_obs += m
        
        elif self.mode == "normalize":
          batch_max = edge_feats.max()
          batch_min = edge_feats.min()
          self.min_wt = torch.min(batch_min, self.min_wt)
          self.max_wt = torch.max(batch_max, self.max_wt)
    
    def embed_node_feats(self, node_feats):
        return self.nodelen_encoding(node_feats)

    def embed_edge_feats(self, edge_feats, noise=0.0, prev_state=None, as_list=False):
        noise = 0.0
        if not torch.is_tensor(edge_feats): 
            edge_feats_normalized = []
            for edge_feats in edge_feats:
                edge_feats_normalized_i = self.standardize_edge_feats(edge_feats)
                edge_feats_normalized.append(edge_feats_normalized_i)
        
        else:
            edge_feats_normalized = self.standardize_edge_feats(edge_feats) + noise
        
        if self.method == "Test":
            edge_embed = self.edgelen_encoding(edge_feats_normalized)
            #print(edge_embed.shape)
            return edge_embed
        
        if self.method == "MLP-Repeat" or self.method == "MLP-2":
            edge_embed = self.edgelen_encoding(edge_feats_normalized)
            #edge_embed = edge_embed.unsqueeze(0).repeat(self.num_layers, 1, 1)
            edge_embed = (edge_embed, edge_embed)
            return edge_embed
        
        if self.method == "MLP-Multi":
            edge_embed = self.edgelen_encoding(edge_feats_normalized)
            edge_embed = edge_embed.reshape(edge_feats.shape[0], self.num_layers, self.embed_dim).movedim(0, 1)
            edge_embed = (edge_embed, edge_embed)
            return edge_embed
        
        if self.method == "MLP-Double":
            edge_embed_h = self.edgelen_encoding_h(edge_feats_normalized)
            edge_embed_h = edge_embed_h.reshape(edge_feats.shape[0], self.num_layers, self.embed_dim).movedim(0, 1)
            
            edge_embed_c = self.edgelen_encoding_c(edge_feats_normalized)
            edge_embed_c = edge_embed_c.reshape(edge_feats.shape[0], self.num_layers, self.embed_dim).movedim(0, 1)
            
            edge_embed = (edge_embed_h, edge_embed_c)
            return edge_embed
        
        def LSTM_pad(edge_feats_normalized):
            lens = [len(x) for x in edge_feats_normalized]
            max_len = max(lens)
            edge_feats_normalized_pad = []
            for i, edge in enumerate(edge_feats_normalized):
                edge_feats_normalized_pad.append(torch.nn.functional.pad(edge, (0, 0, 0, max_len - lens[i]), value = np.inf))
            
            edge_feats_normalized = torch.cat(edge_feats_normalized_pad, dim = -1)
            return edge_feats_normalized
            
        
        if self.method == "LSTM":
            if prev_state is None:
                states_h = []
                states_c = []
                
                #edge_feats_normalized = torch.cat(edge_feats_normalized, dim = -1)
                edge_feats_normalized = LSTM_pad(edge_feats_normalized)
                #edge_embed = self.edgelen_encoding(edge_feats_normalized.unsqueeze(-1))
                
                
                print(edge_feats_normalized)
                print(edge_feats_normalized.shape)
                
                B = edge_feats_normalized.shape[1]
                cur_state = (self.leaf_h0_wt.repeat(B, 1), self.leaf_c0_wt.repeat(B, 1))
                #prev_states_h = [[] for _ in range(B)]
                #states_h = [[] for _ in range(B)]
                #states_c = [[] for _ in range(B)]
                prev_states_h = []
                prev_idx = None
                for edge in edge_feats_normalized:
                    prev_states_h.append(cur_state[0])
                    idx = torch.isfinite(edge)
                    if prev_idx is None:
                        prev_idx = idx
                        state_idx = idx
                    else:
                        state_idx = idx[prev_idx]
                        prev_idx = idx
                    
                    if torch.sum(idx) != B:
                        print(idx)
                        print(prev_idx)
                        print(state_idx)
                    
                    edge = edge[idx]
                    if torch.sum(idx) != B:
                        print(edge)
                    edge = self.edgelen_encoding(edge.unsqueeze(-1))
                    if torch.sum(idx) != B:
                        print(edge)
                    cur_state = self.edgeLSTM(edge, (cur_state[0][state_idx], cur_state[1][state_idx]))
                    states_h.append(cur_state[0])
                    #print(cur_state[0].shape)
                    states_c.append(cur_state[1])
                print(prev_states_h)
                state_h = torch.cat(states_h, 0)
                state_c = torch.cat(states_c, 0)
                #print(state_h.shape)
                prev_h = torch.cat(prev_states_h, dim = 0)#.view(state_h.shape[0], state_h.shape[1])
                #print(prev_h.shape)
                state = (state_h, state_c) 
                print(STOP)
                return state, prev_h
                
            else:
                 edge_embed = self.edgelen_encoding(edge_feats_normalized)
                 state = self.edgeLSTM(edge_embed, prev_state)   
                 return state
        
        if self.method == "LSTM2":
            edge_feats_normalized = torch.cat(edge_feats_normalized, dim = -1)
            edge_embed = self.edgelen_encoding(edge_feats_normalized)
            edge_embed = self.edgeLSTM(edge_embed, (self.leaf_h0_wt.repeat(edge_embed.shape[1], 1), self.leaf_c0_wt.repeat(edge_embed.shape[1], 1)))
            return edge_embed
            
        
        if self.method == "MLP-Leaf":
            if prev_state is None:
                states_h = []
                states_c = []
                
                edge_feats_normalized = torch.cat(edge_feats_normalized, dim = -1)
                
                edge_embed = self.edgelen_encoding(edge_feats_normalized.unsqueeze(-1))
                
                B = edge_feats_normalized.shape[1]
                cur_state = (self.wt_h0.repeat(1, B, 1), self.wt_c0.repeat(1, B, 1))
                for edge in edge_embed:
                    cur_state = self.edgeLSTM(edge, cur_state)
                    states_h.append(cur_state[0])
                    states_c.append(cur_state[1])       
                state_h = torch.cat(states_h, 1)
                state_c = torch.cat(states_c, 1) 
                out = (state_h, state_c) 
                
                out_h = torch.cat([self.leaf_h0_2.repeat(1, state_h.shape[1], 1), out[0]], dim = -1)
                out_c = torch.cat([self.leaf_c0_2.repeat(1, state_h.shape[1], 1), out[1]], dim = -1)
                return (out_h, out_c)
                
            else:
                edge_embed = self.edgelen_encoding(edge_feats_normalized)
                out = self.edgeLSTM(edge_embed, prev_state)
                out_h = torch.cat([self.leaf_h0_2.repeat(1, edge_feats.shape[0], 1), out[0]], dim = -1)
                out_c = torch.cat([self.leaf_c0_2.repeat(1, edge_feats.shape[0], 1), out[1]], dim = -1)
                return (out_h, out_c), out

    def compute_softminus(self, edge_feats, threshold = 20):
      '''
      Computes 'softminus' of weights: log(exp(w) - 1). For numerical stability,
      reverts to linear function if w > 20.
      
      Args Used:
        x_adj: adjacency vector at this iteration
        threshold: threshold value to revert to linear function
      
      Returns:
        x_sm: adjacency vector with softminus applied to weight entries
      '''
      x_thresh = (edge_feats <= threshold).float()
      x_sm = torch.log(torch.special.expm1(edge_feats))
      x_sm = torch.mul(x_sm, x_thresh)
      x_sm = x_sm + torch.mul(edge_feats, 1 - x_thresh)
      return x_sm
    
    
    def predict_node_feats(self, state, node_feats=None):
        """
        Args:
            state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
            node_feats: N x feat_dim or None
        Returns:
            new_state,
            likelihood of node_feats under current state,
            and, if node_feats is None, then return the prediction of node_feats
            else return the node_feats as it is
        """
        h, _ = state
        pred_node_len = self.nodelen_pred(h[-1])
        state_update = self.embed_node_feats(pred_node_len) if node_feats is None else self.embed_node_feats(node_feats)
        new_state = self.node_state_update(state_update, state)
        if node_feats is None:
            ll = 0
            node_feats = pred_node_len
        else:
            ll = -(node_feats - pred_node_len) ** 2
            ll = torch.sum(ll)
        return new_state, ll, node_feats

    def predict_edge_feats(self, state, edge_feats=None,prev_state=None):
        """
        Args:
            state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
            edge_feats: N x feat_dim or None
        Returns:
            likelihood of edge_feats under current state,
            and, if edge_feats is None, then return the prediction of edge_feats
            else return the edge_feats as it is
        """
        #h = h[-1]
        
        if self.method == "MLP-2":
            B = state[0].shape[0]
            h, _ = self.l2r_cell(state, (self.leaf_h0_top.repeat(B, 1), self.leaf_c0_top.repeat(B, 1)))
        
        else:
            h, _ = state
        
        if prev_state is not None:
            h = torch.cat([h, prev_state], dim = -1)
        
        mus, lvars = self.edgelen_mean(h), self.edgelen_lvar(h)
        
        if edge_feats is None:
            ll = 0
            
            if self.sampling_method == "softplus": 
                pred_mean = mus
                pred_lvar = lvars
                pred_sd = torch.exp(0.5 * pred_lvar)
                edge_feats = torch.normal(pred_mean, pred_sd)
                #edge_feats = edge_feats * (self.var_wt**0.5 + 1e-15) + self.mu_wt
                edge_feats = torch.nn.functional.softplus(edge_feats)
            
            elif self.sampling_method  == "lognormal":
                pred_mean = mus
                pred_lvar = lvars
                pred_sd = torch.exp(0.5 * pred_lvar)
                edge_feats = torch.normal(pred_mean, pred_sd)
                #edge_feats = edge_feats * (self.var_wt**0.5 + 1e-15) + self.mu_wt
                edge_feats = torch.exp(edge_feats)
            
            elif self.sampling_method  == "gamma": 
                loga = mus
                logb = lvars
                a = torch.exp(loga)
                b = torch.exp(logb)
                
                edge_feats = torch.distributions.gamma.Gamma(a, b).sample()
            
            elif self.sampling_method == "vae":
                z = torch.randn(1, 8).to(h.device)
                edge_feats, _ = self.decode_weight(z, h)
                
        else:
            if self.sampling_method  == "softplus":
                ### Update log likelihood with weight prediction
                #print("Hi")
                ### Trying with softplus parameterization...
                edge_feats_invsp = edge_feats #self.compute_softminus(edge_feats)
                
                ### Standardize
                #edge_feats_invsp = self.standardize_edge_feats(edge_feats_invsp)
                
                ## MEAN AND VARIANCE OF LOGNORMAL
                var = torch.exp(lvars) 
                
                ## diff_sq = (mu - softminusw)^2
                diff_sq = torch.square(torch.sub(mus, edge_feats_invsp))
                
                ## diff_sq2 = v^-1*diff_sq
                diff_sq2 = torch.div(diff_sq, var)
                
                ## add to ll
                ll = - torch.mul(lvars, 0.5) - torch.mul(diff_sq2, 0.5) #+ edge_feats - edge_feats_invsp - 0.5 * np.log(2*np.pi)
                ll = torch.sum(ll)
            
            elif self.sampling_method  == "lognormal":
                ### Trying with softplus parameterization...
                log_edge_feats = torch.log(edge_feats)
                var = torch.exp(lvars) 
                
                ll = torch.sub(log_edge_feats - mus)
                ll = torch.square(ll)
                ll = torch.div(ll, var)
                ll = ll + np.log(2 * np.pi) + lvars
                ll = -0.5 * ll - log_edge_feats 
                ll = torch.sum(ll)
                #ll = torch.mean(ll)
            
            elif self.sampling_method  == "gamma":
                loga = mus
                logb = lvars
                a = torch.exp(loga)
                b = torch.exp(logb)
                log_edge_feats = torch.log(edge_feats)
                
                ll = torch.mul(a, logb)
                ll = ll - torch.lgamma(a)
                ll = ll + torch.mul(a - 1, log_edge_feats)
                ll = ll - torch.mul(b, edge_feats)
                ll = torch.sum(ll)
            
#             elif self.sampling_method == "vae":
#                 z, ll_kl = self.encode_weight(edge_feats, h)
#                 _, ll = self.decode_weight(z, h, edge_feats)
#                 ll = ll + ll_kl
        return ll, edge_feats
    
#     def encode_weight(self, edge_feats, h):
#         edge_feats = self.standardize_edge_feats(edge_feats)
#         vae_embed = self.embed_wt_vae(edge_feats)
#         input_ = torch.cat([vae_embed, h[-1]], -1)
#         mu = self.vae_mu(input_)
#         logvar = self.vae_sig(input_)
#         eps = torch.randn_like(mu)
#         z = mu + torch.exp(0.5 * logvar) * eps
#         ll_kl = 0.5 * torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar))
#         return z, ll_kl
#     
#     def decode_weight(self, z, h, edge_feats=None):
#         input_ = torch.cat([z, h[-1]], -1)
#         w_star = self.weight_out(input_)
#         ll = 0
#         if edge_feats is not None:
#             ll = -torch.square(w_star - edge_feats)
#             ll = torch.sum(ll)
#         return w_star, ll


 
class BiggWithGCN(RecurTreeGen):

    def __init__(self, args):
        super().__init__(args)
        self.gcn_mod = GCN_Generate(args)
        
        
    def forward_train2(self, batch_indices, feat_idx, edge_list, batch_weight_idx):
        ll_top, _, _ = self.forward_train(batch_indices)
        ll_wt = self.gcn_mod.forward(feat_idx, edge_list, batch_weight_idx)
        return ll_top, ll_wt
    
    def sample2(self, num_nodes, display=None):
        init = datetime.now()
        _, pred_edges, _, _, _ = self.forward(node_end = num_nodes, display=display)
        cur = datetime.now() - init
        
        
        fix_edges = []
        for e1, e2 in pred_edges:
            if e1 > e2:
                fix_edges.append((e2, e1))
            else:
                fix_edges.append((e1, e2))
                    
        pred_edge_tensor = torch.tensor(fix_edges).to(cmd_args.device)
        init = datetime.now()
        pred_weighted_tensor = self.gcn_mod.sample(num_nodes, pred_edge_tensor)
        cur = datetime.now() - init
        return pred_edges, pred_weighted_tensor
    
    
