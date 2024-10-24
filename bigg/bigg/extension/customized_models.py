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

from bigg.model.tree_model import RecurTreeGen
from bigg.extension.gcn_build import *
import torch
from bigg.common.pytorch_util import glorot_uniform, MLP, MultiLSTMCell
import torch.nn as nn
import numpy 
from torch.nn.parameter import Parameter

# pylint: skip-file

## Widen LSTM
## Check MLP for embeddings
## Multi Layers

class BiggWithEdgeLen(RecurTreeGen):

    def __init__(self, args):
        super().__init__(args)
        self.method = args.method
        
        self.nodelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])
        self.nodelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, 1])
        
        self.edgelen_mean = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
        self.edgelen_lvar = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)
        
        if self.method == "MLP-Repeat":
            self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim], dropout = cmd_args.wt_drop)
            
        if self.method == "MLP-Multi":
            self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim * args.rnn_layers], dropout = cmd_args.wt_drop)
        
        if self.method == "MLP-Double":
            self.edgelen_encoding_h = MLP(1, [2 * args.embed_dim, args.embed_dim * args.rnn_layers], dropout = cmd_args.wt_drop)
            self.edgelen_encoding_c = MLP(1, [2 * args.embed_dim, args.embed_dim * args.rnn_layers], dropout = cmd_args.wt_drop)
        
        if self.method == "LSTM":
            self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = cmd_args.wt_drop)
            self.edgeLSTM = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers, dropout = cmd_args.wt_drop)
            #self.edgelen_encoding = MLP(1, [32, 16])
            
            #self.leaf_h0_wt = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim // 2))
            #self.leaf_c0_wt = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim // 2))
        
        if self.method == "MLP-Leaf":
            self.edgelen_encoding = MLP(1, [args.embed_dim, args.embed_dim * args.rnn_layers // 2], dropout = cmd_args.wt_drop)
            self.leaf_h0_wt = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
            self.leaf_c0_wt = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
            
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

    # to be customized
    
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

    def embed_edge_feats(self, edge_feats):
        #edge_feats = self.compute_softminus(edge_feats)
        #if self.epoch_num == 0:
        #    self.update_weight_stats(edge_feats)
        edge_feats_normalized = self.standardize_edge_feats(edge_feats)
        
        if self.method == "MLP-Repeat":
            edge_embed = self.edgelen_encoding(edge_feats_normalized)
            edge_embed = edge_embed.unsqueeze(0).repeat(self.num_layers, 1, 1)
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
        
        if self.method == "LSTM":
            edge_embed = self.edgelen_encoding(edge_feats_normalized)
            state = self.edgeLSTM(edge_embed, (self.leaf_h0.repeat(1, edge_embed.shape[0], 1), self.leaf_c0.repeat(1, edge_embed.shape[0],1)))
            return state
        
        if self.method == "MLP-Leaf":
            edge_embed = self.edgelen_encoding(edge_feats_normalized)
            out = edge_embed.reshape(edge_feats.shape[0], self.num_layers, self.embed_dim // 2).movedim(0, 1)
            out_h = torch.cat([self.leaf_h0_wt.repeat(1, edge_feats.shape[0], 1), out], dim = -1)
            out_c = torch.cat([self.leaf_c0_wt.repeat(1, edge_feats.shape[0], 1), out], dim = -1)
            return (out_h, out_c) 
            
    
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

    def predict_edge_feats(self, state, edge_feats=None):
        """
        Args:
            state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
            edge_feats: N x feat_dim or None
        Returns:
            likelihood of edge_feats under current state,
            and, if edge_feats is None, then return the prediction of edge_feats
            else return the edge_feats as it is
        """
        h, _ = state
        mus, lvars = self.edgelen_mean(h[-1]), self.edgelen_lvar(h[-1])
        
        if edge_feats is None:
            ll = 0
            pred_mean = mus
            pred_lvar = lvars
            pred_sd = torch.exp(0.5 * pred_lvar)
            edge_feats = torch.normal(pred_mean, pred_sd)
            #edge_feats = edge_feats * (self.var_wt**0.5 + 1e-15) + self.mu_wt
            edge_feats = torch.nn.functional.softplus(edge_feats)
            
        else:
            ### Update log likelihood with weight prediction
            
            ### Trying with softplus parameterization...
            edge_feats_invsp = self.compute_softminus(edge_feats)
            
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
        return ll, edge_feats




class BiggWithGCN(RecurTreeGen):

    def __init__(self, args):
        super().__init__(args)
        self.gcn_mod = GCN_Generate(args)
        
    def forward_train2(self, batch_indices, feat_idx, edge_list, batch_weight_idx):
        ll_top, _, _ = self.forward_train(batch_indices)
        ll_wt = self.gcn_mod.forward(feat_idx, edge_list, batch_weight_idx)
        return ll_top, ll_wt
    
    def sample2(self, num_nodes, display):
        _, pred_edges, _, _, _ = self.forward(node_end = num_nodes, display=display)
        fix_edges = []
        for e1, e2 in pred_edges:
            if e1 > e2:
                fix_edges.append((e2, e1))
            else:
                fix_edges.append((e1, e2))
                    
        pred_edge_tensor = torch.tensor(fix_edges).to(cmd_args.device)
        pred_weighted_tensor = self.gcn_mod.sample(num_nodes, pred_edge_tensor)
        return pred_edges, pred_weighted_tensor
    
    
