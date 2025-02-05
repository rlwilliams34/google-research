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

# 
# 
# import numpy as np
# import scipy.stats as stats 
# from matplotlib import pyplot as plt
# 
# data = stats.gamma.rvs(alpha, loc = 0, scale = 1/beta, size = 10000)
# y1 = sns.kdeplot(data)
# y2 = sns.kdeplot(weights)
# #stats.gamma.pdf(sorted(weights), a=alpha, scale=1/beta)
# # plt.plot(x, y1, "y-", label=(r'Gamma Dist')) 
# # plt.plot(weights, y2, "y-", label=(r'Weights')) 
# # plt.xlim([0,1])
# plt.show()
# 
# 
# 
# 
# # Generate some data
# stats.gamma.sample(x, a=alpha, scale=1/beta)
# data = np.random.normal(0, 1, 1000)
# 
# # Fit a normal distribution to the data
# mu, std = stats.norm.fit(data)
# 
# # Create a histogram of the data
# plt.hist(data, bins=20, density=True, alpha=0.6)
# 
# # Plot the PDF of the fitted normal distribution
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = stats.norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# 
# plt.show()
# 








from bigg.model.tree_model import RecurTreeGen, FenwickTree
from bigg.extension.gcn_build import *
import torch
from bigg.common.pytorch_util import glorot_uniform, MLP, MultiLSTMCell, BinaryTreeLSTMCell
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.nn.parameter import Parameter
from datetime import datetime
from bigg.torch_ops import PosEncoding #, PosEncoding2D
from torch.nn import Module
from bigg.common.consts import t_float
torch.set_printoptions(threshold=1000000)
np.set_printoptions(threshold=1000000)


class PosEncoding2D(Module):
    def __init__(self, dim, device, base=10000, bias=0):
        super(PosEncoding2D, self).__init__()
        p = []
        sft = []
        assert dim % 2 == 0 ## Dimension needs to be even for this to work
        for i in range(dim // 2):
            b = 2 * (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)
            else:
                sft.append(bias)
        self.device = device
        self.sft = torch.tensor(sft, dtype=t_float).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=t_float).view(1, -1).to(device)
    
    def forward(self, row, col):
        with torch.no_grad():
            #if isinstance(row, list):
            if not torch.is_tensor(row):
                row = torch.tensor(row, dtype=t_float).to(self.device)
                col = torch.tensor(col, dtype=t_float).to(self.device)
            row = row.view(-1, 1)
            col = col.view(-1, 1)
            x = row / self.base + self.sft
            y = col / self.base + self.sft
            out = torch.cat([x, y], dim = -1)
            return torch.sin(out)


# pylint: skip-file
class BiggWithEdgeLen(RecurTreeGen):
    def __init__(self, args):
        super().__init__(args)
        cmd_args.wt_drop = -1
        self.method = args.method
        self.sampling_method = cmd_args.sampling_method
        self.row_LSTM = args.row_LSTM
        self.wt_mlp = args.wt_mlp
        self.num_edge = 0
        self.g_type = args.g_type
        self.embed_dim = args.embed_dim
        self.weight_embed_dim = args.weight_embed_dim
        self.num_layers = args.rnn_layers
        self.sigma = args.noise
        self.wt_one_layer = args.wt_one_layer
        self.add_states = args.add_states
        self.penalty = args.penalty
        
        assert self.sampling_method in ['gamma', 'lognormal', 'softplus']
        assert self.method in ['Test9', 'Test10', 'Test11', 'Test12', 'MLP-Repeat', 'Test285', 'Test286', 'Test287', 'Test75', 'Test85', 'None', 'Leaf-LSTM']
        if self.method == "None":
           assert self.has_egde_feats == 0
        
        if args.has_node_feats:
            self.nodelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])
            self.nodelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, 1])
            self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)
        
        self.edgelen_mean = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout = args.wt_drop)
        self.edgelen_lvar = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout = args.wt_drop)
        
        if self.method not in ["Test75", "Test85"]:
            self.leaf_embed = Parameter(torch.Tensor(1, args.weight_embed_dim))
            self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = args.wt_drop)
            self.leaf_LSTM = MultiLSTMCell(2 * args.weight_embed_dim, args.embed_dim, args.rnn_layers)
        
        if self.method == "Leaf-LSTM":
            self.leaf_embed = Parameter(torch.Tensor(1, args.weight_embed_dim))
            self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = args.wt_drop)
            self.leaf_LSTM = MultiLSTMCell(2 * args.weight_embed_dim, args.embed_dim, args.rnn_layers)
        
        elif self.method == "MLP-Repeat":
            self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim], dropout = args.wt_drop)
        
        elif self.method == "Test75" or self.method == "Test85":
            self.merge_top_wt = BinaryTreeLSTMCell(args.embed_dim)
            self.update_wt = BinaryTreeLSTMCell(args.embed_dim)
            self.leaf_LSTM = MultiLSTMCell(1, args.embed_dim, args.rnn_layers)
            if args.row_LSTM:
                self.row_LSTM = MultiLSTMCell(1, args.embed_dim, args.rnn_layers)
                self.leaf_h0_wt = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
                self.leaf_c0_wt = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
            else:
                self.weight_tree = FenwickTree(args)
                
            if self.wt_mlp:
                self.leaf_LSTM = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
                self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = args.wt_drop)
                self.edge_pos_enc = PosEncoding(args.weight_embed_dim, args.device, args.pos_base)
            
            if self.method == "Test85":
                self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = args.wt_drop)
                #self.edge_pos_enc = PosEncoding(args.weight_embed_dim, args.device, args.pos_base)
                self.edge_pos_enc = PosEncoding2D(args.weight_embed_dim, args.device, args.pos_base)
                self.leaf_LSTM = MultiLSTMCell(2 * args.weight_embed_dim, args.embed_dim, args.rnn_layers)
            
            if self.add_states:
                self.scale_tops = Parameter(torch.Tensor(1))
                self.scale_wts = Parameter(torch.Tensor(1))
            
        
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

    def embed_edge_feats(self, edge_feats, sigma = 0, prev_state=None, list_num_edges=None, db_info=None, edge_feats_lstm=None, rc=None):
        if not self.row_LSTM: 
            B = edge_feats.shape[0]
            edge_feats_normalized = self.standardize_edge_feats(edge_feats)
        
        else:
            if prev_state is None:
                edge_feats_lstm = edge_feats_lstm.float()
                L = edge_feats_lstm.shape[0]
                B = edge_feats_lstm.shape[1]
                tot_edges = torch.sum(edge_feats_lstm > 0).item()
                Z = torch.cumsum((edge_feats_lstm > 0).int(), 1)
                idx_to = torch.sum(F.pad(Z[:,:-1], (1,0,0,0), mode='constant',value=0),dim=0)
                edge_feats_normalized = edge_feats_lstm.clone()
                edge_idx = (edge_feats_normalized > -1)
                edge_feats_normalized[edge_idx] = self.standardize_edge_feats(edge_feats_normalized[edge_idx])
                if sigma > 0:
                    edge_feats_normalized[edge_idx] = edge_feats_normalized[edge_idx] + sigma * torch.randn(edge_idx.shape).to(edge_feats.device)
                
            else:
                B = edge_feats.shape[0]
                edge_feats_normalized = self.standardize_edge_feats(edge_feats)
        
        if self.method == "MLP-Repeat":
            edge_embed = self.edgelen_encoding(edge_feats_normalized)
            edge_embed = edge_embed.unsqueeze(0).repeat(self.num_layers, 1, 1)
            edge_embed = (edge_embed, edge_embed)
            return edge_embed
        
        elif self.method == "Leaf-LSTM":
            edge_embed = self.edgelen_encoding(edge_feats_normalized)
            edge_embed = torch.cat([edge_embed, self.leaf_embed.repeat(B, 1)], dim = -1)
            edge_embed = self.leaf_LSTM(edge_embed)
            return edge_embed

        else:
            if self.row_LSTM:
                if prev_state is not None:
                    if self.wt_mlp:
                        edge_embed = self.edgelen_encoding(edge_feats_normalized)
                        edge_embed = self.row_LSTM(edge_embed, prev_state) 
                    else:
                        edge_embed = self.row_LSTM(edge_feats_normalized, prev_state)  
                    return edge_embed
                
                else:
                    prev_state = (self.leaf_h0_wt.repeat(1, B, 1), self.leaf_c0_wt.repeat(1, B, 1))
                    edge_embed_h = torch.zeros(self.num_layers, tot_edges, self.embed_dim).to(edge_feats.device)
                    edge_embed_c = torch.zeros(self.num_layers, tot_edges, self.embed_dim).to(edge_feats.device)
                    for i in range(L):
                        if self.wt_mlp:
                            x_in = self.edgelen_encoding(edge_feats_normalized[i, :].unsqueeze(-1))
                            next_state = self.row_LSTM(x_in, prev_state)
                        else:
                            next_state = self.row_LSTM(edge_feats_normalized[i, :].unsqueeze(-1), prev_state)
                        prev_state = next_state
                        cur_edge_feats = edge_feats_lstm[i, :]
                        mask = (cur_edge_feats > 0)
                        if i == 0:
                            edge_embed_h[:, idx_to] = prev_state[0]
                            edge_embed_c[:, idx_to] = prev_state[1]
                        else: 
                            idx_to_cur = idx_to[mask] + i
                            edge_embed_h[:, idx_to_cur] = prev_state[0][:, mask]
                            edge_embed_c[:, idx_to_cur] = prev_state[1][:, mask]
                        
                    edge_embed = (edge_embed_h, edge_embed_c)
                    return edge_embed
            
            else:
                if self.method == "Test85":
                    edge_embed = self.edgelen_encoding(edge_feats_normalized)
                    edge_row = rc[:, 0]
                    edge_col = rc[:, 1]
                    edge_pos = self.edge_pos_enc(edge_row, edge_col)
                    edge_embed = torch.cat([edge_embed, edge_pos], dim = -1)
                    edge_embed = self.leaf_LSTM(edge_embed)
                    
                    #row_pos = self.edge_pos_enc(edge_row.tolist())
                    #col_pos = self.edge_pos_enc(edge_col.tolist())
                    #edge_embed = torch.cat([edge_embed, row_pos, col_pos], dim = -1)
                    #edge_embed = self.leaf_LSTM(edge_embed)
                    #rc_pos = (row_pos + col_pos) / 2
                    #edge_embed = [rc_pos + x for x in edge_embed]
                else:
                    if self.wt_mlp:
                        edge_embed = self.edgelen_encoding(edge_feats_normalized)
                        edge_embed = self.leaf_LSTM(edge_embed)
                    else:
                        edge_embed = self.leaf_LSTM(edge_feats_normalized)
            if self.wt_one_layer:
                edge_embed = (edge_embed[0][-1:], edge_embed[1][-1:])
            
            if list_num_edges is None:
                edge_embed = self.weight_tree(edge_embed)
            else:
                edge_embed = self.weight_tree.forward_train_weights(edge_embed, list_num_edges, db_info)
            return edge_embed 
        
    
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

    def predict_edge_feats(self, state, edge_feats=None,batch_idx=None,ll_batch_wt=None):
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
            ll_batch_wt = 0
            
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
            
            return ll, ll_batch_wt, edge_feats
                
        else:
            if self.sampling_method  == "softplus":
                ### Update log likelihood with weight prediction
                ### Trying with softplus parameterization...
                edge_feats_invsp = self.compute_softminus(edge_feats)
                
                ### Standardize
                #edge_feats_invsp = self.standardize_edge_feats(edge_feats_invsp)
                
                ## MEAN AND VARIANCE OF LOGNORMAL
                var = torch.exp(lvars) 
                
                diff_sq = torch.square(torch.sub(mus, edge_feats_invsp))
                
                ## diff_sq2 = v^-1*diff_sq
                diff_sq = torch.div(diff_sq, var)
                
                ## add to ll
                ll = - torch.mul(lvars, 0.5) - torch.mul(diff_sq, 0.5) #+ edge_feats - edge_feats_invsp - 0.5 * np.log(2*np.pi)
            
            elif self.sampling_method  == "lognormal":
                log_edge_feats = torch.log(edge_feats)
                var = torch.exp(lvars)
                
                ll = torch.sub(log_edge_feats - mus)
                ll = torch.square(ll)
                ll = torch.div(ll, var)
                ll = ll + np.log(2 * np.pi) + lvars
                ll = -0.5 * ll - log_edge_feats
            
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
#                 print("edge feats", edge_feats)
#                 print("loga: ", loga)
#                 print("logb: ", logb)
#                 print("mu: ", a / b)
#                 print("sigma: ", (a / b * 1 / b)**0.5)
#                 print("============================")
                if self.penalty:
                    ll = ll - 1e-4 * a - 1e-4 * b
            
            if batch_idx is not None:
                i = 0
                for B in np.unique(batch_idx):
                    ll_batch_wt[i] = ll_batch_wt[i] + torch.sum(ll[batch_idx == B])
                    i = i + 1
            
            ll = torch.sum(ll)
        return ll, ll_batch_wt, edge_feats


 
class BiggWithGCN(RecurTreeGen):
    def __init__(self, args):
        super().__init__(args)
        self.gcn_mod = GCN_Generate(args)
        self.method = "None"
        self.update_wt = None
        
        
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
    
    





























# from bigg.model.tree_model import RecurTreeGen, FenwickTree
# from bigg.extension.gcn_build import *
# import torch
# from bigg.common.pytorch_util import glorot_uniform, MLP, MultiLSTMCell, BinaryTreeLSTMCell
# import torch.nn as nn
# from torch.nn import functional as F
# import numpy 
# from torch.nn.parameter import Parameter
# from datetime import datetime
# from bigg.torch_ops import PosEncoding
# 
# # pylint: skip-file
# class BiggWithEdgeLen(RecurTreeGen):
#     def __init__(self, args):
#         super().__init__(args)
#         cmd_args.wt_drop = -1
#         self.method = args.method
#         self.sampling_method = cmd_args.sampling_method
#         self.row_LSTM = args.row_LSTM
#         self.test_topdown = args.test_topdown
#         self.wt_mlp = args.wt_mlp
#         self.test = args.test
#         self.test2 = args.test2
#         self.test3 = args.test3
#         self.num_edge = 0
#         self.g_type = args.g_type
#         self.test_sep = args.test_sep
#         
#         assert self.sampling_method in ['gamma', 'lognormal', 'softplus']
#         assert self.method in ['Test9', 'Test10', 'Test11', 'Test12', 'MLP-Repeat', 'Test285', 'Test286', 'Test287', 'Test75', 'Test85']
#         
#         if args.has_node_feats:
#             self.nodelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])
#             self.nodelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, 1])
#             self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)
#         
#         self.edgelen_mean = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout = args.wt_drop)
#         self.edgelen_lvar = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout = args.wt_drop)
#         
#         if self.method != "Test75":
#             self.leaf_embed = Parameter(torch.Tensor(1, args.weight_embed_dim))
#             self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = args.wt_drop)
#             self.leaf_LSTM = MultiLSTMCell(2 * args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#         
#         if self.method == "MLP-Repeat":
#             self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim], dropout = args.wt_drop)
#         
#         if self.method == "Test9":
#             self.empty_embed = Parameter(torch.Tensor(1, args.weight_embed_dim))
#             self.test_h0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             self.test_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#         
#         if self.method == "Test10":
#             self.edge_pos_enc = PosEncoding(args.weight_embed_dim, args.device, args.pos_base)
#             self.leaf_LSTM = MultiLSTMCell(2 * args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#         
#         if self.method == "Test11":
#             self.leaf_LSTM = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#         
#         if self.method == "Test285":
#             self.weight_tree = FenwickTree(args)
#             self.leaf_LSTM = MultiLSTMCell(2 * args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#             self.leaf_embed = Parameter(torch.Tensor(1, args.weight_embed_dim))
#         
#         if self.method == "Test286":
#             self.weight_tree = FenwickTree(args)
#             self.leaf_LSTM = MultiLSTMCell(1, args.embed_dim, args.rnn_layers)
# 
#         if self.method == "Test287":
#             self.weight_tree = FenwickTree(args)
#             self.leaf_LSTM = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#             self.leaf_embed = Parameter(torch.Tensor(1, args.weight_embed_dim))  
#             self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim], dropout = args.wt_drop)
#         
#         if self.method == "Test288":
#             self.weight_tree = FenwickTree(args)
#             self.leaf_LSTM = MultiLSTMCell(2 * args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#             self.leaf_embed = Parameter(torch.Tensor(1, args.weight_embed_dim))  
#             self.empty_embed = Parameter(torch.Tensor(1, args.weight_embed_dim))
#             self.test_h0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             self.test_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = args.wt_drop)
#         
#         if self.method == "Test75" or self.method == "Test85":
#             self.merge_top_wt = BinaryTreeLSTMCell(args.embed_dim)
#             self.update_wt = BinaryTreeLSTMCell(args.embed_dim)
#             self.leaf_LSTM = MultiLSTMCell(1, args.embed_dim, args.rnn_layers)
#             if args.row_LSTM:
#                 self.row_LSTM = MultiLSTMCell(1, args.embed_dim, args.rnn_layers)
#                 self.leaf_h0_wt = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#                 self.leaf_c0_wt = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             else:
#                 self.weight_tree = FenwickTree(args)
#                 
#             if self.wt_mlp:
#                 self.leaf_LSTM = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#                 self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = args.wt_drop)
#             
#             if self.method == "Test85":
#                 self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = args.wt_drop)
#                 self.edge_pos_enc = PosEncoding(args.weight_embed_dim, args.device, args.pos_base)
#                 self.leaf_LSTM = MultiLSTMCell(3 * args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#         
#         self.embed_dim = args.embed_dim
#         self.weight_embed_dim = args.weight_embed_dim
#         self.num_layers = args.rnn_layers
#         self.sigma = args.noise
#         
#         mu_wt = torch.tensor(0, dtype = float)
#         var_wt = torch.tensor(1, dtype = float)
#         n_obs = torch.tensor(0, dtype = int)
#         min_wt = torch.tensor(torch.inf, dtype = float)
#         max_wt = torch.tensor(-torch.inf, dtype = float)
#         epoch_num = torch.tensor(0, dtype = int)
#         
#         self.register_buffer("mu_wt", mu_wt)
#         self.register_buffer("var_wt", var_wt)
#         self.register_buffer("n_obs", n_obs)
#         self.register_buffer("min_wt", min_wt)
#         self.register_buffer("max_wt", max_wt)
#         self.register_buffer("epoch_num", epoch_num)
#         self.mode = args.wt_mode
#         
#         self.log_wt = False
#         self.sm_wt = False
#         self.wt_range = 1.0
#         self.wt_scale = 1.0
#         
#         glorot_uniform(self)
# 
#     def standardize_edge_feats(self, edge_feats): 
#       if self.log_wt:
#         edge_feats = torch.log(edge_feats)
#       
#       elif self.sm_wt:
#         edge_feats = torch.log(torch.special.expm1(edge_feats))
#       
#       if self.mode == "none":
#         return edge_feats
#       
#       if self.epoch_num == 1:
#         self.update_weight_stats(edge_feats)
#       
#       if self.mode == "score":
#         edge_feats = (edge_feats - self.mu_wt) / (self.var_wt**0.5 + 1e-15)
#           
#       elif self.mode == "normalize":
#         edge_feats = -1 + 2 * (edge_feats - self.min_wt) / (self.max_wt - self.min_wt + 1e-15)
#         edge_feats = self.wt_range * edge_feats
#       
#       elif self.mode == "scale":
#         edge_feats = edge_feats * self.wt_scale
#       
#       elif self.mode == "exp":
#         edge_feats = torch.exp(-1/edge_feats)
#       
#       elif self.mode == "exp-log":
#         edge_feats = torch.exp(-1 / torch.log(1 + edge_feats))
#         
#       return edge_feats
#   
#     def update_weight_stats(self, edge_feats):
#       '''
#       Updates necessary global statistics (mean, variance, min, max) of edge_feats per batch
#       if standardizing edge_feats prior to MLP embedding. Only performed during the
#       first epoch of training.
#       
#       Args Used:
#         edge_feats: edge_feats from current iteration batch
#       '''
#       
#       ## Current training weight statistics
#       with torch.no_grad():
#         if self.mode == "score":
#           mu_n = self.mu_wt
#           var_n = self.var_wt
#           n = self.n_obs
#           
#           ## New weight statistics
#           m = len(edge_feats)
#           
#           if m > 1:
#             var_m = torch.var(edge_feats)
#           else:
#             var_m = 0.0
#             
#           mu_m = torch.mean(edge_feats)
#           tot = n + m
#           
#           if tot == 1:
#             self.mu_wt = mu_m
#             self.n_obs = tot
#          
#           else:
#             ## Update weight statistics
#             new_mu = (n * mu_n + m * mu_m) / tot
#             
#             new_var_avg = (max(n - 1, 0) * var_n + (m - 1) * var_m)/(tot - 1)
#             new_var_resid = n * m * (mu_n - mu_m)**2 / (tot * (tot - 1))
#             new_var = new_var_avg + new_var_resid
#             
#             ## Save
#             self.mu_wt = new_mu
#             self.var_wt = new_var
#             self.n_obs += m
#         
#         elif self.mode == "normalize":
#           batch_max = edge_feats.max()
#           batch_min = edge_feats.min()
#           self.min_wt = torch.min(batch_min, self.min_wt)
#           self.max_wt = torch.max(batch_max, self.max_wt)
#     
#     def embed_node_feats(self, node_feats):
#         return self.nodelen_encoding(node_feats)
#     
#     def LSTM_pad(self, list_feats):
#         lens = [len(x) for x in list_feats]
#         max_len = max(lens)
#         list_feats_pad = []
#         for i, feat in enumerate(list_feats):
#             if not isinstance(feat, torch.Tensor):
#                 list_feats_pad.append(np.concatenate([feat, np.full((max_len - lens[i], 1, 2), np.inf)], axis = 0))
#                 cat = False
#             else:
#                 list_feats_pad.append(torch.nn.functional.pad(feat, (0, 0, 0, max_len - lens[i]), value = np.inf))
#                 cat = True
#         
#         if cat:
#             feats_pad = torch.cat(list_feats_pad, dim = -1)
#         
#         else:
#             feats_pad = np.concatenate(list_feats_pad, axis = 1)
#         return feats_pad
# 
#     def embed_edge_feats(self, edge_feats, sigma=0.0, prev_state=None, list_num_edges=None, db_info=None, edge_feats_lstm=None, rc=None):
#         
#         if self.method != "Test12" and not self.row_LSTM: 
#             B = edge_feats.shape[0]
#             edge_feats_normalized = self.standardize_edge_feats(edge_feats)
#             if sigma > 0:
#                 edge_feats_normalized = edge_feats_normalized + sigma * torch.randn(edge_feats.shape).to(edge_feats.device)
#         
#         elif self.row_LSTM:
#             if prev_state is None:
#                 edge_feats_lstm = edge_feats_lstm.float()
#                 L = edge_feats_lstm.shape[0]
#                 B = edge_feats_lstm.shape[1]
#                 tot_edges = torch.sum(edge_feats_lstm > 0).item()
#                 Z = torch.cumsum((edge_feats_lstm > 0).int(), 1)
#                 idx_to = torch.sum(F.pad(Z[:,:-1], (1,0,0,0), mode='constant',value=0),dim=0)
#                 edge_feats_normalized = edge_feats_lstm.clone()
#                 edge_idx = (edge_feats_normalized > -1)
#                 edge_feats_normalized[edge_idx] = self.standardize_edge_feats(edge_feats_normalized[edge_idx])
#                 
#                 if sigma > 0:
#                     edge_feats_normalized[edge_idx] = edge_feats_normalized[edge_idx] + sigma * torch.randn(edge_idx.shape).to(edge_feats.device)
#                 
#             else:
#                 B = edge_feats.shape[0]
#                 edge_feats_normalized = self.standardize_edge_feats(edge_feats)
#                 #edge_feats_normalized = edge_feats_normalized + sigma * torch.randn(edge_feats.shape).to(edge_feats.device)
#         
#         if self.method == "MLP-Repeat":
#             edge_embed = self.edgelen_encoding(edge_feats_normalized)
#             edge_embed = edge_embed.unsqueeze(0).repeat(self.num_layers, 1, 1)
#             edge_embed = (edge_embed, edge_embed)
#             return edge_embed
# 
#         elif self.method in ["Test9", "Test10", "Test11"]:
#             edge_embed = self.edgelen_encoding(edge_feats_normalized)
#             
#             if self.method == "Test9":
#                 x_in = torch.cat([self.leaf_embed.repeat(B, 1), edge_embed], dim = -1)
#             
#             if self.method == "Test10":
#                 x_in = torch.cat([self.leaf_embed.repeat(B, 1), edge_embed], dim = -1)
#             
#             s_in = (self.test_h0.repeat(1, B, 1), self.test_c0.repeat(1, B, 1))
#             edge_embed = self.leaf_LSTM(x_in, s_in)
#             return edge_embed
# 
#         elif self.method in ["Test285", "Test286", "Test287", "Test75", "Test85"]:
#             if self.method == "Test285":
#                 #Encode weight in MLP; concatenate leaf embeddings; run through empty state LSTM
#                 edge_embed = self.edgelen_encoding(edge_feats_normalized)
#                 x_in = torch.cat([self.leaf_embed.repeat(B, 1), edge_embed], dim = -1)
#                 edge_embed = self.leaf_LSTM(x_in)
#             
#             elif self.method in ["Test286", "Test75", "Test85"]:
#                 if self.row_LSTM:
#                     if prev_state is not None:
#                         if self.wt_mlp:
#                             edge_embed = self.edgelen_encoding(edge_feats_normalized)
#                             edge_embed = self.row_LSTM(edge_embed, prev_state) 
#                         
#                         else:
#                             edge_embed = self.row_LSTM(edge_feats_normalized, prev_state)  
#                         return edge_embed
#                     
#                     else:
#                         prev_state = (self.leaf_h0_wt.repeat(1, B, 1), self.leaf_c0_wt.repeat(1, B, 1))
#                         edge_embed_h = torch.zeros(self.num_layers, tot_edges, self.embed_dim).to(edge_feats.device)
#                         edge_embed_c = torch.zeros(self.num_layers, tot_edges, self.embed_dim).to(edge_feats.device)
#                         
#                         for i in range(L):
#                             if self.wt_mlp:
#                                 x_in = self.edgelen_encoding(edge_feats_normalized[i, :].unsqueeze(-1))
#                                 next_state = self.row_LSTM(x_in, prev_state)
#                             else:
#                                 next_state = self.row_LSTM(edge_feats_normalized[i, :].unsqueeze(-1), prev_state)
#                             
#                             prev_state = next_state
#                             cur_edge_feats = edge_feats_lstm[i, :]
#                             mask = (cur_edge_feats > 0)
# 
#                             if i == 0:
#                                 edge_embed_h[:, idx_to] = prev_state[0]
#                                 edge_embed_c[:, idx_to] = prev_state[1]
#                             else: 
#                                 idx_to_cur = idx_to[mask] + i
#                                 edge_embed_h[:, idx_to_cur] = prev_state[0][:, mask]
#                                 edge_embed_c[:, idx_to_cur] = prev_state[1][:, mask]
#                             
#                         edge_embed = (edge_embed_h, edge_embed_c)
#                         return edge_embed
#                 
#                 else:
#                     if self.method == "Test85":
#                         edge_embed = self.edgelen_encoding(edge_feats_normalized)
#                         edge_row = rc[:, 0]
#                         edge_col = rc[:, 1]
#                         row_pos = self.edge_pos_enc(edge_row.tolist())
#                         col_pos = self.edge_pos_enc(edge_col.tolist())
#                         edge_embed = torch.cat([edge_embed, row_pos, col_pos], dim = -1)
#                         edge_embed = self.leaf_LSTM(edge_embed)
#                     
#                     else:
#                         if self.wt_mlp:
#                             edge_embed = self.edgelen_encoding(edge_feats_normalized)
#                             edge_embed = self.leaf_LSTM(edge_embed)
#                         
#                         else:
#                             edge_embed = self.leaf_LSTM(edge_feats_normalized)
#             
#             elif self.method == "Test287":
#                 # Just use MLP for init state"
#                 edge_embed = self.edgelen_encoding(edge_feats_normalized)
#                 edge_embed = edge_embed.unsqueeze(0).repeat(self.num_layers, 1, 1)
#                 edge_embed = (edge_embed, edge_embed)
#             
#             elif self.method == "Test288":
#                 edge_embed = self.edgelen_encoding(edge_feats_normalized)
#                 x_in = torch.cat([self.leaf_embed.repeat(B, 1), edge_embed], dim = -1)
#                 s_in = (self.test_h0.repeat(1, B, 1), self.test_c0.repeat(1, B, 1))
#                 edge_embed = self.leaf_LSTM(x_in, s_in)
#             
#             else:
#                 # Encode Weight in MLP; concatenate leaf embedding; run through init leaf-state LSTM
#                 edge_embed = self.edgelen_encoding(edge_feats_normalized)
#                 x_in = torch.cat([self.leaf_embed.repeat(B, 1), edge_embed], dim = -1)
#                 s_in = (self.leaf_h0.repeat(1, B, 1), self.leaf_c0.repeat(1, B, 1))
#                 edge_embed = self.leaf_LSTM(x_in, s_in)
#             
#             if list_num_edges is None:
#                 edge_embed = self.weight_tree(edge_embed)
#             
#             else:
#                 edge_embed = self.weight_tree.forward_train_weights(edge_embed, list_num_edges, db_info)
#             
#             return edge_embed
#         
# #         elif self.method == "Test75":
# #             if prev_state is None:
#                 
#         
#         elif self.method == "Test12": 
#             if prev_state is None:
#                 edge_feats_normalized = []
#                 for e in edge_feats:
#                     edge_feats_normalized_i = self.standardize_edge_feats(e) +  sigma * torch.randn(e.shape).to(e.device)
#                     edge_feats_normalized.append(edge_feats_normalized_i)
#                 
#                 states_h = []
#                 states_c = []
#                 
#                 edge_feats_normalized = self.LSTM_pad(edge_feats_normalized)
#                 
#                 B = edge_feats_normalized.shape[1]
#                 cur_state = (self.leaf_h0.repeat(1, B, 1), self.leaf_c0.repeat(1, B, 1))
#                 prev_states_h = []
#                 prev_idx = None
#                 
#                 edge_feats_idx = torch.zeros(edge_feats_normalized.shape).to(edge_feats_normalized.device)
#                 i = 0
#                 for k in range(edge_feats_normalized.shape[1]):
#                     for idx in range(edge_feats_normalized.shape[0]):
#                         if torch.isfinite(edge_feats_normalized[:, k][idx]): 
#                             edge_feats_idx[:, k][idx] = i
#                             i = i + 1
#                         else:
#                             edge_feats_idx[:, k][idx] = np.inf
#                 
#                 L = torch.sum(torch.isfinite(edge_feats_normalized))
#                 
#                 states_h = torch.zeros(self.num_layers, L, self.embed_dim).to(edge_feats_normalized.device)
#                 states_c = torch.zeros(self.num_layers, L, self.embed_dim).to(edge_feats_normalized.device)
#                 
#                 for i, edge in enumerate(edge_feats_normalized):
#                     idx = torch.isfinite(edge)
#                     if prev_idx is None:
#                         prev_idx = idx
#                         state_idx = idx
#                     else:
#                         state_idx = idx[prev_idx]
#                         prev_idx = idx
#                     
#                     cur_idx = edge_feats_idx[i][torch.isfinite(edge_feats_idx[i])]
#                     
#                     edge = edge[idx]
#                     edge = self.edgelen_encoding(edge.unsqueeze(-1))
#                     embed_edge = torch.cat([self.leaf_embed.repeat(edge.shape[0], 1), edge], dim = -1)
#                     
#                     cur_state = self.leaf_LSTM(embed_edge, (cur_state[0][:, state_idx], cur_state[1][:, state_idx]))
#                     
#                     states_h[:, cur_idx.long()] = cur_state[0]
#                     states_c[:, cur_idx.long()] = cur_state[1]
#                 
#                 state = (states_h, states_c)
#                 return state
#                 
#             else:
#                 edge_feats_normalized = self.standardize_edge_feats(edge_feats)
#                 edge_embed = self.edgelen_encoding(edge_feats_normalized)
#                 edge_embed = torch.cat([self.leaf_embed, edge_embed], dim = -1)
#                 state = self.leaf_LSTM(edge_embed, prev_state)   
#                 return state        
#         
#     
#     def compute_softminus(self, edge_feats, threshold = 20):
#       '''
#       Computes 'softminus' of weights: log(exp(w) - 1). For numerical stability,
#       reverts to linear function if w > 20.
#       
#       Args Used:
#         x_adj: adjacency vector at this iteration
#         threshold: threshold value to revert to linear function
#       
#       Returns:
#         x_sm: adjacency vector with softminus applied to weight entries
#       '''
#       x_thresh = (edge_feats <= threshold).float()
#       x_sm = torch.log(torch.special.expm1(edge_feats))
#       x_sm = torch.mul(x_sm, x_thresh)
#       x_sm = x_sm + torch.mul(edge_feats, 1 - x_thresh)
#       return x_sm
#     
#     
#     def predict_node_feats(self, state, node_feats=None):
#         """
#         Args:
#             state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
#             node_feats: N x feat_dim or None
#         Returns:
#             new_state,
#             likelihood of node_feats under current state,
#             and, if node_feats is None, then return the prediction of node_feats
#             else return the node_feats as it is
#         """
#         h, _ = state
#         pred_node_len = self.nodelen_pred(h[-1])
#         state_update = self.embed_node_feats(pred_node_len) if node_feats is None else self.embed_node_feats(node_feats)
#         new_state = self.node_state_update(state_update, state)
#         if node_feats is None:
#             ll = 0
#             node_feats = pred_node_len
#         else:
#             ll = -(node_feats - pred_node_len) ** 2
#             ll = torch.sum(ll)
#         return new_state, ll, node_feats
# 
#     def predict_edge_feats(self, state, edge_feats=None,batch_idx=None,ll_batch_wt=None):
#         """
#         Args:
#             state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
#             edge_feats: N x feat_dim or None
#         Returns:
#             likelihood of edge_feats under current state,
#             and, if edge_feats is None, then return the prediction of edge_feats
#             else return the edge_feats as it is
#         """
#         h, _ = state
#         mus, lvars = self.edgelen_mean(h[-1]), self.edgelen_lvar(h[-1])
#         
#         if edge_feats is None:
#             ll = 0
#             ll_batch_wt = 0
#             
#             if self.sampling_method == "softplus": 
#                 pred_mean = mus
#                 pred_lvar = lvars
#                 pred_sd = torch.exp(0.5 * pred_lvar)
#                 edge_feats = torch.normal(pred_mean, pred_sd)
#                 #edge_feats = edge_feats * (self.var_wt**0.5 + 1e-15) + self.mu_wt
#                 edge_feats = torch.nn.functional.softplus(edge_feats)
#             
#             elif self.sampling_method  == "lognormal":
#                 pred_mean = mus
#                 pred_lvar = lvars
#                 pred_sd = torch.exp(0.5 * pred_lvar)
#                 edge_feats = torch.normal(pred_mean, pred_sd)
#                 #edge_feats = edge_feats * (self.var_wt**0.5 + 1e-15) + self.mu_wt
#                 edge_feats = torch.exp(edge_feats)
#             
#             elif self.sampling_method  == "gamma": 
#                 loga = mus
#                 logb = lvars
#                 a = torch.exp(loga)
#                 b = torch.exp(logb)
#                 
#                 edge_feats = torch.distributions.gamma.Gamma(a, b).sample()
#             
#             return ll, ll_batch_wt, edge_feats
#                 
#         else:
#             if self.sampling_method  == "softplus":
#                 ### Update log likelihood with weight prediction
#                 ### Trying with softplus parameterization...
#                 edge_feats_invsp = self.compute_softminus(edge_feats)
#                 
#                 ### Standardize
#                 #edge_feats_invsp = self.standardize_edge_feats(edge_feats_invsp)
#                 
#                 ## MEAN AND VARIANCE OF LOGNORMAL
#                 var = torch.exp(lvars) 
#                 
#                 diff_sq = torch.square(torch.sub(mus, edge_feats_invsp))
#                 
#                 ## diff_sq2 = v^-1*diff_sq
#                 diff_sq = torch.div(diff_sq, var)
#                 
#                 ## add to ll
#                 ll = - torch.mul(lvars, 0.5) - torch.mul(diff_sq, 0.5) #+ edge_feats - edge_feats_invsp - 0.5 * np.log(2*np.pi)
#             
#             elif self.sampling_method  == "lognormal":
#                 log_edge_feats = torch.log(edge_feats)
#                 var = torch.exp(lvars)
#                 
#                 ll = torch.sub(log_edge_feats - mus)
#                 ll = torch.square(ll)
#                 ll = torch.div(ll, var)
#                 ll = ll + np.log(2 * np.pi) + lvars
#                 ll = -0.5 * ll - log_edge_feats
#             
#             elif self.sampling_method  == "gamma":
#                 loga = mus
#                 logb = lvars
#                 a = torch.exp(loga)
#                 b = torch.exp(logb)
#                 log_edge_feats = torch.log(edge_feats)
#                 
#                 ll = torch.mul(a, logb)
#                 ll = ll - torch.lgamma(a)
#                 ll = ll + torch.mul(a - 1, log_edge_feats)
#                 ll = ll - torch.mul(b, edge_feats)
#             
#             if batch_idx is not None:
#                 i = 0
#                 for B in np.unique(batch_idx):
#                     ll_batch_wt[i] = ll_batch_wt[i] + torch.sum(ll[batch_idx == B])
#                     i = i + 1
#             
#             ll = torch.sum(ll)
#         return ll, ll_batch_wt, edge_feats
# 
# 
#  
# class BiggWithGCN(RecurTreeGen):
#     def __init__(self, args):
#         super().__init__(args)
#         self.gcn_mod = GCN_Generate(args)
#         self.method = "None"
#         self.update_wt = None
#         
#         
#     def forward_train2(self, batch_indices, feat_idx, edge_list, batch_weight_idx):
#         ll_top, _, _ = self.forward_train(batch_indices)
#         ll_wt = self.gcn_mod.forward(feat_idx, edge_list, batch_weight_idx)
#         return ll_top, ll_wt
#     
#     def sample2(self, num_nodes, display=None):
#         init = datetime.now()
#         _, pred_edges, _, _, _ = self.forward(node_end = num_nodes, display=display)
#         cur = datetime.now() - init
#         
#         
#         fix_edges = []
#         for e1, e2 in pred_edges:
#             if e1 > e2:
#                 fix_edges.append((e2, e1))
#             else:
#                 fix_edges.append((e1, e2))
#                     
#         pred_edge_tensor = torch.tensor(fix_edges).to(cmd_args.device)
#         init = datetime.now()
#         pred_weighted_tensor = self.gcn_mod.sample(num_nodes, pred_edge_tensor)
#         cur = datetime.now() - init
#         return pred_edges, pred_weighted_tensor


# 
# 
# 
# from bigg.model.tree_model import RecurTreeGen
# from bigg.extension.gcn_build import *
# import torch
# from bigg.common.pytorch_util import glorot_uniform, MLP, MultiLSTMCell, BinaryTreeLSTMCell
# import torch.nn as nn
# import numpy 
# from torch.nn.parameter import Parameter
# from datetime import datetime
# from bigg.torch_ops import PosEncoding
# 
# # pylint: skip-file
# class BiggWithEdgeLen(RecurTreeGen):
#     def __init__(self, args):
#         super().__init__(args)
#         cmd_args.wt_drop = -1
#         self.method = args.method
#         self.sampling_method = cmd_args.sampling_method
#         self.update_left = args.update_left
#         
#         assert self.sampling_method in ['gamma', 'lognormal', 'softplus', 'vae']
#         assert self.method in ['Test', 'MLP-Repeat', 'MLP-Multi', 'MLP-Double', 'LSTM', 'MLP-Leaf', 'Test2', 'Test3', 'Test4', 'Test5', 'Test6', 'Test7', 'Test8', 'Test9', 'Test10', 'Test11']
#         
#         self.nodelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])
#         self.nodelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, 1])
#         
#         self.edgelen_mean = MLP(args.embed_dim, [2 * args.embed_dim, 4 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
#         self.edgelen_lvar = MLP(args.embed_dim, [2 * args.embed_dim, 4 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
#         #self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)
#         self.update_wt = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#         self.topdown_update_wt = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#         self.weight_embed_dim = args.weight_embed_dim
#         
#         if self.method in ["Test9", "Test10", "Test11"]:
#             self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = cmd_args.wt_drop)
#             self.leaf_LSTM = MultiLSTMCell(3 * args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#             self.leaf_embed = Parameter(torch.Tensor(1, 2 * args.weight_embed_dim))
#             self.empty_embed = Parameter(torch.Tensor(1, 2 * args.weight_embed_dim))
#             self.test_h0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             self.test_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             if self.method == "Test10":
#                 self.edge_pos_enc = PosEncoding(args.weight_embed_dim, args.device, args.pos_base)
#                 self.leaf_LSTM = MultiLSTMCell(4 * args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#                 self.leaf_embed = Parameter(torch.Tensor(1, args.weight_embed_dim))
#                 self.empty_embed = Parameter(torch.Tensor(1, args.weight_embed_dim))
#             if self.method == "Test11":
#                 self.leaf_LSTM = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
# #                 self.test2_h0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
# #                 self.test2_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             
#             #self.update_wt = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#         
#         if self.method == "Test8":
#             self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = cmd_args.wt_drop)
#             self.update_wt = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#         
#         if self.method == "Test7":
#             self.edgelen_encoding = MLP(1, [2 * args.wt_embed_dim, args.wt_embed_dim], dropout = cmd_args.wt_drop)
#         
#         if self.method == "Test4":
#             self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim], dropout = cmd_args.wt_drop)
#             self.edgeLSTM = nn.LSTMCell(args.embed_dim, args.embed_dim)
#             self.leaf_h0_wt = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             self.leaf_c0_wt = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             self.merge_top_wt = BinaryTreeLSTMCell(args.embed_dim)
#             #self.update_wt = nn.LSTMCell(args.embed_dim, args.embed_dim)
#             self.update_wt = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#         
#         if self.method in ["Test", "Test2", "Test3"]:
#             self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = cmd_args.wt_drop)
#         
#         if self.method == "Test6":
#             self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim], dropout = cmd_args.wt_drop)
#             #self.edgeLSTM = MultiLSTMCell(args.embed_dim, args.embed_dim, args.rnn_layers)
#         
#         if self.method == "MLP-Repeat":
#             self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim], dropout = cmd_args.wt_drop)
#             
#         if self.method == "MLP-2":
#             self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])#, dropout = cmd_args.wt_drop)
#             self.leaf_h0_top = Parameter(torch.Tensor(1, args.embed_dim))
#             self.leaf_c0_top = Parameter(torch.Tensor(1, args.embed_dim))
#         
#         if self.method == "MLP-Multi":
#             self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim * args.rnn_layers], dropout = cmd_args.wt_drop)
#         
#         if self.method == "MLP-Double":
#             self.edgelen_encoding_h = MLP(1, [2 * args.embed_dim, args.embed_dim * args.rnn_layers], dropout = cmd_args.wt_drop)
#             self.edgelen_encoding_c = MLP(1, [2 * args.embed_dim, args.embed_dim * args.rnn_layers], dropout = cmd_args.wt_drop)
#         
#         if self.method == "LSTM":
#             self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = cmd_args.wt_drop)
#             self.edgeLSTM = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#             #self.edgeLSTM = nn.LSTMCell(args.weight_embed_dim, args.embed_dim)
#             
#             self.leaf_h0_wt = Parameter(torch.Tensor(1, args.embed_dim))
#             self.leaf_c0_wt = Parameter(torch.Tensor(1, args.embed_dim))
#             
#             self.edgelen_mean = MLP(2 * args.embed_dim, [3 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
#             self.edgelen_lvar = MLP(2 * args.embed_dim, [3 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
#         
#         if self.method == "LSTM2":
#             self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = cmd_args.wt_drop)
#             self.edgeLSTM = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#             #self.edgeLSTM = nn.LSTMCell(args.weight_embed_dim, args.embed_dim)
#             
#             self.leaf_h0_wt = Parameter(torch.Tensor(1, args.embed_dim))
#             self.leaf_c0_wt = Parameter(torch.Tensor(1, args.embed_dim))
#             
#             self.edgelen_mean = MLP(2 * args.embed_dim, [3 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
#             self.edgelen_lvar = MLP(2 * args.embed_dim, [3 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
#         
#         if self.method == "Test5":
#             self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = cmd_args.wt_drop)
#             self.edgeLSTM = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#             self.leaf_h0_wt = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             self.leaf_c0_wt = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             self.merge_top_wt = BinaryTreeLSTMCell(args.embed_dim)
#             self.update_wt = MultiLSTMCell(args.weight_embed_dim, args.embed_dim, args.rnn_layers)
#         
#         if self.method == "MLP-Leaf":
#             self.edgelen_encoding = MLP(1, [2 * args.weight_embed_dim, args.weight_embed_dim], dropout = cmd_args.wt_drop)
#             self.wt_h0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim // 2))
#             self.wt_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim // 2))
#             self.leaf_h0_2 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             self.leaf_c0_2 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             
#             self.edgeLSTM = MultiLSTMCell(args.weight_embed_dim, args.embed_dim // 2, args.rnn_layers)
#             
#             self.edgelen_mean = MLP(int(1.5 * args.embed_dim), [2 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
#             self.edgelen_lvar = MLP(int(1.5 * args.embed_dim), [2 * args.embed_dim, 1], dropout = cmd_args.wt_drop)
#         
#         self.embed_dim = args.embed_dim
#         self.num_layers = args.rnn_layers
#         
#         mu_wt = torch.tensor(0, dtype = float)
#         var_wt = torch.tensor(1, dtype = float)
#         n_obs = torch.tensor(0, dtype = int)
#         min_wt = torch.tensor(torch.inf, dtype = float)
#         max_wt = torch.tensor(-torch.inf, dtype = float)
#         epoch_num = torch.tensor(0, dtype = int)
#         
#         self.register_buffer("mu_wt", mu_wt)
#         self.register_buffer("var_wt", var_wt)
#         self.register_buffer("n_obs", n_obs)
#         self.register_buffer("min_wt", min_wt)
#         self.register_buffer("max_wt", max_wt)
#         self.register_buffer("epoch_num", epoch_num)
#         self.mode = args.wt_mode
#         
#         self.log_wt = False
#         self.sm_wt = False
#         self.wt_range = 1.0
#         self.wt_scale = 1.0
#         
#         glorot_uniform(self)
#         
# #         if args.sampling_method == "vae":
# #             self.embed_wt_vae = MLP(1, [2 * args.embed_dim, args.embed_dim])
# #             self.vae_mu = MLP(2 * args.embed_dim, [4 * args.embed_dim, 8])
# #             self.vae_sig = MLP(2 * args.embed_dim, [4 * args.embed_dim, 8])
# #             self.weight_out = MLP(args.embed_dim + 8, [2 * args.embed_dim + 16, 1])
# 
#     def standardize_edge_feats(self, edge_feats): 
#       if self.log_wt:
#         edge_feats = torch.log(edge_feats)
#       
#       elif self.sm_wt:
#         edge_feats = torch.log(torch.special.expm1(edge_feats))
#       
#       if self.mode == "none":
#         return edge_feats
#       
#       if self.epoch_num == 1:
#         self.update_weight_stats(edge_feats)
#       
#       if self.mode == "score":
#         edge_feats = (edge_feats - self.mu_wt) / (self.var_wt**0.5 + 1e-15)
#           
#       elif self.mode == "normalize":
#         edge_feats = -1 + 2 * (edge_feats - self.min_wt) / (self.max_wt - self.min_wt + 1e-15)
#         edge_feats = self.wt_range * edge_feats
#       
#       elif self.mode == "scale":
#         edge_feats = edge_feats * self.wt_scale
#       
#       elif self.mode == "exp":
#         edge_feats = torch.exp(-1/edge_feats)
#       
#       elif self.mode == "exp-log":
#         edge_feats = torch.exp(-1 / torch.log(1 + edge_feats))
#         
#       return edge_feats
#   
#     def update_weight_stats(self, edge_feats):
#       '''
#       Updates necessary global statistics (mean, variance, min, max) of edge_feats per batch
#       if standardizing edge_feats prior to MLP embedding. Only performed during the
#       first epoch of training.
#       
#       Args Used:
#         edge_feats: edge_feats from current iteration batch
#       '''
#       
#       ## Current training weight statistics
#       with torch.no_grad():
#         if self.mode == "score":
#           mu_n = self.mu_wt
#           var_n = self.var_wt
#           n = self.n_obs
#           
#           ## New weight statistics
#           m = len(edge_feats)
#           
#           if m > 1:
#             var_m = torch.var(edge_feats)
#           else:
#             var_m = 0.0
#             
#           mu_m = torch.mean(edge_feats)
#           tot = n + m
#           
#           if tot == 1:
#             self.mu_wt = mu_m
#             self.n_obs = tot
#          
#           else:
#             ## Update weight statistics
#             new_mu = (n * mu_n + m * mu_m) / tot
#             
#             new_var_avg = (max(n - 1, 0) * var_n + (m - 1) * var_m)/(tot - 1)
#             new_var_resid = n * m * (mu_n - mu_m)**2 / (tot * (tot - 1))
#             new_var = new_var_avg + new_var_resid
#             
#             ## Save
#             self.mu_wt = new_mu
#             self.var_wt = new_var
#             self.n_obs += m
#         
#         elif self.mode == "normalize":
#           batch_max = edge_feats.max()
#           batch_min = edge_feats.min()
#           self.min_wt = torch.min(batch_min, self.min_wt)
#           self.max_wt = torch.max(batch_max, self.max_wt)
#     
#     def embed_node_feats(self, node_feats):
#         return self.nodelen_encoding(node_feats)
# 
#     def embed_edge_feats(self, edge_feats, noise=0.0, prev_state=None, as_list=False, lr_seq=None, rc=None):
#         noise = 0.0
#         if not torch.is_tensor(edge_feats): 
#             edge_feats_normalized = []
#             for edge_feats in edge_feats:
#                 edge_feats_normalized_i = self.standardize_edge_feats(edge_feats)
#                 edge_feats_normalized.append(edge_feats_normalized_i)
#         
#         else:
#             if self.method == "Test10":
#                 edge_feats_normalized = self.standardize_edge_feats(edge_feats) + noise
#                 edge_row = rc[:, :, 0]
#                 edge_col = rc[:, :, 1]
#             
#             else:
#                 edge_feats_normalized = self.standardize_edge_feats(edge_feats) + noise
#         
#         
#         if self.method in ["Test9", "Test10", "Test11"]:
#             edge_embed = self.edgelen_encoding(edge_feats_normalized)
#             K = edge_embed.shape[0]
#             
#             if self.method == "Test10":
#                 row_pos = self.edge_pos_enc(edge_row.tolist())
#                 col_pos = self.edge_pos_enc(edge_col.tolist())
#                 #row_pos = torch.zeros(edge_embed.shape).to(edge_embed.device)
#                 #col_pos = torch.zeros(edge_embed.shape).to(edge_embed.device)
#                 
#                 edge_embed = torch.cat([edge_embed, row_pos, col_pos], dim = -1)
#             
#             if self.method == "Test9" or self.method == "Test10":
#                 x_in = torch.cat([self.leaf_embed.repeat(K, 1), edge_embed], dim = -1)
#             
#             elif self.method == "Test11": 
#                 x_in = edge_embed
#             
#             s_in = (self.test_h0.repeat(1, K, 1), self.test_c0.repeat(1, K, 1))
#             edge_embed = self.leaf_LSTM(x_in, s_in)
#             return edge_embed
#         
#         
#         if self.method == "Test8":
#             edge_embed = self.edgelen_encoding(edge_feats_normalized)
#             return edge_embed
#         
#         if self.method == "Test7":
#             edge_embed = self.edgelen_encoding(edge_feats_normalized)
#             return edge_embed
#         
#         if self.method == "Test6":
#             edge_embed = self.edgelen_encoding(edge_feats_normalized)
#             edge_embed = edge_embed.unsqueeze(0).repeat(self.num_layers, 1, 1)
#             edge_embed = (self.leaf_h0.repeat(1, edge_embed.shape[1], 1) + edge_embed, self.leaf_c0.repeat(1, edge_embed.shape[1], 1) + edge_embed)
#             return edge_embed
#         
#         if self.method == "Test5":
#             if prev_state is not None:
#                 weights_MLP = self.edgelen_encoding(edge_feats_normalized)
#                 if len(weights_MLP.shape) == 2:
#                     weights_MLP = weights_MLP.unsqueeze(1)
#                 weight_embedding = self.edgeLSTM(weights_MLP, prev_state)
#                 return weight_embedding
#             
#             weights_MLP = self.edgelen_encoding(edge_feats_normalized)
#             weight_embeddings = (self.leaf_h0_wt.repeat(1, edge_feats.shape[0], 1), self.leaf_c0_wt.repeat(1, edge_feats.shape[0], 1))
#             weight_embeddings = self.edgeLSTM(weights_MLP, weight_embeddings)
#             return weight_embeddings, weights_MLP
#         
#         if self.method == "Test4":
#             if prev_state is not None:
#                 weights_MLP = self.edgelen_encoding(edge_feats_normalized)
#                 weight_embedding = self.edgeLSTM(weights_MLP, prev_state)
#                 return weight_embedding
#                 
#             embeds = torch.cat([self.topdown_left_embed[1:], self.topdown_right_embed[1:]], dim = 0)
#             weight_embeddings = (self.leaf_h0_wt.repeat(edge_feats.shape[0], 1), self.leaf_c0_wt.repeat(edge_feats.shape[0], 1))
#             
#             for i, lr in enumerate(lr_seq):
#                 idx = (lr != -1)
#                 cur_lr = lr[idx]
#                 cur_lr = embeds[cur_lr]
#                 cur_weight_embeddings = self.edgeLSTM(cur_lr, (weight_embeddings[0][idx], weight_embeddings[1][idx]))
#                 
#                 weight_embeddings[0][idx] = cur_weight_embeddings[0]
#                 weight_embeddings[1][idx] = cur_weight_embeddings[1]
#             
#             weights_MLP = self.edgelen_encoding(edge_feats_normalized)
#             weight_embeddings = self.edgeLSTM(weights_MLP, weight_embeddings)
#             return weight_embeddings, weights_MLP
#         
#         if self.method in ["Test", "Test2", "Test3"]:
#             edge_embed = self.edgelen_encoding(edge_feats_normalized)
#             return edge_embed
#         
#         if self.method == "MLP-Repeat" or self.method == "MLP-2":
#             edge_embed = self.edgelen_encoding(edge_feats_normalized)
#             edge_embed = edge_embed.unsqueeze(0).repeat(self.num_layers, 1, 1)
#             edge_embed = (edge_embed, edge_embed)
#             return edge_embed
#         
#         if self.method == "MLP-Multi":
#             edge_embed = self.edgelen_encoding(edge_feats_normalized)
#             edge_embed = edge_embed.reshape(edge_feats.shape[0], self.num_layers, self.embed_dim).movedim(0, 1)
#             edge_embed = (edge_embed, edge_embed)
#             return edge_embed
#         
#         if self.method == "MLP-Double":
#             edge_embed_h = self.edgelen_encoding_h(edge_feats_normalized)
#             edge_embed_h = edge_embed_h.reshape(edge_feats.shape[0], self.num_layers, self.embed_dim).movedim(0, 1)
#             
#             edge_embed_c = self.edgelen_encoding_c(edge_feats_normalized)
#             edge_embed_c = edge_embed_c.reshape(edge_feats.shape[0], self.num_layers, self.embed_dim).movedim(0, 1)
#             
#             edge_embed = (edge_embed_h, edge_embed_c)
#             return edge_embed
#         
#         def LSTM_pad(edge_feats_normalized):
#             lens = [len(x) for x in edge_feats_normalized]
#             max_len = max(lens)
#             edge_feats_normalized_pad = []
#             for i, edge in enumerate(edge_feats_normalized):
#                 edge_feats_normalized_pad.append(torch.nn.functional.pad(edge, (0, 0, 0, max_len - lens[i]), value = np.inf))
#             
#             edge_feats_normalized = torch.cat(edge_feats_normalized_pad, dim = -1)
#             return edge_feats_normalized
#         
#         if self.method == "LSTM":
#             if prev_state is None:
#                 states_h = []
#                 states_c = []
#                 
#                 edge_feats_normalized = LSTM_pad(edge_feats_normalized)
#                 B = edge_feats_normalized.shape[1]
#                 cur_state = (self.leaf_h0_wt.repeat(B, 1), self.leaf_c0_wt.repeat(B, 1))
#                 prev_states_h = []
#                 prev_idx = None
#                 
#                 edge_feats_idx = torch.zeros(edge_feats_normalized.shape).to(edge_feats_normalized.device)
#                 i = 0
#                 for k in range(edge_feats_normalized.shape[1]):
#                     for idx in range(edge_feats_normalized.shape[0]):
#                         if torch.isfinite(edge_feats_normalized[:, k][idx]): 
#                             edge_feats_idx[:, k][idx] = i
#                             i = i + 1
#                         else:
#                             edge_feats_idx[:, k][idx] = np.inf
#                 
#                 L = torch.sum(torch.isfinite(edge_feats_normalized))
#                 
#                 prev_states_h = torch.zeros(L, self.embed_dim).to(edge_feats_normalized.device)
#                 states_h = torch.zeros(L, self.embed_dim).to(edge_feats_normalized.device)
#                 states_c = torch.zeros(L, self.embed_dim).to(edge_feats_normalized.device)
#                 
#                 for i, edge in enumerate(edge_feats_normalized):
#                     idx = torch.isfinite(edge)
#                     if prev_idx is None:
#                         prev_idx = idx
#                         state_idx = idx
#                     else:
#                         state_idx = idx[prev_idx]
#                         prev_idx = idx
#                     
#                     cur_idx = edge_feats_idx[i][torch.isfinite(edge_feats_idx[i])]
#                     prev_states_h[cur_idx.long()] = cur_state[0][state_idx]
#                     
#                     edge = edge[idx]
#                     edge = self.edgelen_encoding(edge.unsqueeze(-1))
#                     cur_state = self.edgeLSTM(edge, (cur_state[0][state_idx], cur_state[1][state_idx]))
#                     
#                     states_h[cur_idx.long()] = cur_state[0]
#                     states_c[cur_idx.long()] = cur_state[1]
#                 
#                 state = (states_h, states_c)
#                 prev_h = prev_states_h
#                 return state, prev_h
#                 
#             else:
#                  edge_embed = self.edgelen_encoding(edge_feats_normalized)
#                  state = self.edgeLSTM(edge_embed, prev_state)   
#                  return state
#         
#         if self.method == "LSTM2":
#             edge_feats_normalized = torch.cat(edge_feats_normalized, dim = -1)
#             edge_embed = self.edgelen_encoding(edge_feats_normalized)
#             edge_embed = self.edgeLSTM(edge_embed, (self.leaf_h0_wt.repeat(edge_embed.shape[1], 1), self.leaf_c0_wt.repeat(edge_embed.shape[1], 1)))
#             return edge_embed
#             
#         
#         if self.method == "MLP-Leaf":
#             if prev_state is None:
#                 states_h = []
#                 states_c = []
#                 
#                 edge_feats_normalized = torch.cat(edge_feats_normalized, dim = -1)
#                 
#                 edge_embed = self.edgelen_encoding(edge_feats_normalized.unsqueeze(-1))
#                 
#                 B = edge_feats_normalized.shape[1]
#                 cur_state = (self.wt_h0.repeat(1, B, 1), self.wt_c0.repeat(1, B, 1))
#                 for edge in edge_embed:
#                     cur_state = self.edgeLSTM(edge, cur_state)
#                     states_h.append(cur_state[0])
#                     states_c.append(cur_state[1])       
#                 state_h = torch.cat(states_h, 1)
#                 state_c = torch.cat(states_c, 1) 
#                 out = (state_h, state_c) 
#                 
#                 out_h = torch.cat([self.leaf_h0_2.repeat(1, state_h.shape[1], 1), out[0]], dim = -1)
#                 out_c = torch.cat([self.leaf_c0_2.repeat(1, state_h.shape[1], 1), out[1]], dim = -1)
#                 return (out_h, out_c)
#                 
#             else:
#                 edge_embed = self.edgelen_encoding(edge_feats_normalized)
#                 out = self.edgeLSTM(edge_embed, prev_state)
#                 out_h = torch.cat([self.leaf_h0_2.repeat(1, edge_feats.shape[0], 1), out[0]], dim = -1)
#                 out_c = torch.cat([self.leaf_c0_2.repeat(1, edge_feats.shape[0], 1), out[1]], dim = -1)
#                 return (out_h, out_c), out
# 
#     def compute_softminus(self, edge_feats, threshold = 20):
#       '''
#       Computes 'softminus' of weights: log(exp(w) - 1). For numerical stability,
#       reverts to linear function if w > 20.
#       
#       Args Used:
#         x_adj: adjacency vector at this iteration
#         threshold: threshold value to revert to linear function
#       
#       Returns:
#         x_sm: adjacency vector with softminus applied to weight entries
#       '''
#       x_thresh = (edge_feats <= threshold).float()
#       x_sm = torch.log(torch.special.expm1(edge_feats))
#       x_sm = torch.mul(x_sm, x_thresh)
#       x_sm = x_sm + torch.mul(edge_feats, 1 - x_thresh)
#       return x_sm
#     
#     
#     def predict_node_feats(self, state, node_feats=None):
#         """
#         Args:
#             state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
#             node_feats: N x feat_dim or None
#         Returns:
#             new_state,
#             likelihood of node_feats under current state,
#             and, if node_feats is None, then return the prediction of node_feats
#             else return the node_feats as it is
#         """
#         h, _ = state
#         pred_node_len = self.nodelen_pred(h[-1])
#         state_update = self.embed_node_feats(pred_node_len) if node_feats is None else self.embed_node_feats(node_feats)
#         new_state = self.node_state_update(state_update, state)
#         if node_feats is None:
#             ll = 0
#             node_feats = pred_node_len
#         else:
#             ll = -(node_feats - pred_node_len) ** 2
#             ll = torch.sum(ll)
#         return new_state, ll, node_feats
# 
#     def predict_edge_feats(self, state, edge_feats=None,prev_state=None,batch_idx=None,ll_batch_wt=None):
#         """
#         Args:
#             state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
#             edge_feats: N x feat_dim or None
#         Returns:
#             likelihood of edge_feats under current state,
#             and, if edge_feats is None, then return the prediction of edge_feats
#             else return the edge_feats as it is
#         """
#         #h = h[-1]
#         
#         if self.method == "MLP-2":
#             B = state[0].shape[0]
#             h, _ = self.l2r_cell(state, (self.leaf_h0_top.repeat(B, 1), self.leaf_c0_top.repeat(B, 1)))
#         
#         else:
#             h, _ = state
#         
#         if prev_state is not None:
#             h = torch.cat([h, prev_state], dim = -1)
#         
#         mus, lvars = self.edgelen_mean(h[-1]), self.edgelen_lvar(h[-1])
#         
#         if edge_feats is None:
#             ll = 0
#             ll_batch_wt = 0
#             
#             
#             if self.sampling_method == "softplus": 
#                 pred_mean = mus
#                 pred_lvar = lvars
#                 pred_sd = torch.exp(0.5 * pred_lvar)
#                 edge_feats = torch.normal(pred_mean, pred_sd)
#                 #edge_feats = edge_feats * (self.var_wt**0.5 + 1e-15) + self.mu_wt
#                 edge_feats = torch.nn.functional.softplus(edge_feats)
#             
#             elif self.sampling_method  == "lognormal":
#                 pred_mean = mus
#                 pred_lvar = lvars
#                 pred_sd = torch.exp(0.5 * pred_lvar)
#                 edge_feats = torch.normal(pred_mean, pred_sd)
#                 #edge_feats = edge_feats * (self.var_wt**0.5 + 1e-15) + self.mu_wt
#                 edge_feats = torch.exp(edge_feats)
#             
#             elif self.sampling_method  == "gamma": 
#                 loga = mus
#                 logb = lvars
#                 a = torch.exp(loga)
#                 b = torch.exp(logb)
#                 
#                 edge_feats = torch.distributions.gamma.Gamma(a, b).sample()
#             
#             return ll, ll_batch_wt, edge_feats
#             
# #             elif self.sampling_method == "vae":
# #                 z = torch.randn(1, 8).to(h.device)
# #                 edge_feats, _ = self.decode_weight(z, h)
#                 
#         else:
#             if self.sampling_method  == "softplus":
#                 ### Update log likelihood with weight prediction
#                 ### Trying with softplus parameterization...
#                 edge_feats_invsp = self.compute_softminus(edge_feats)
#                 
#                 ### Standardize
#                 #edge_feats_invsp = self.standardize_edge_feats(edge_feats_invsp)
#                 
#                 ## MEAN AND VARIANCE OF LOGNORMAL
#                 var = torch.exp(lvars) 
#                 
#                 ## diff_sq = (mu - softminusw)^2
#                 diff_sq = torch.square(torch.sub(mus, edge_feats_invsp))
#                 
#                 ## diff_sq2 = v^-1*diff_sq
#                 diff_sq2 = torch.div(diff_sq, var)
#                 
#                 ## add to ll
#                 ## For lognormal, you WOULD need to include the constant terms w/ x & sigma
#                 ll = - torch.mul(lvars, 0.5) - torch.mul(diff_sq2, 0.5) #+ edge_feats - edge_feats_invsp - 0.5 * np.log(2*np.pi)
#             
#             elif self.sampling_method  == "lognormal":
#                 ### Trying with softplus parameterization...
#                 log_edge_feats = torch.log(edge_feats)
#                 var = torch.exp(lvars) 
#                 
#                 ll = torch.sub(log_edge_feats - mus)
#                 ll = torch.square(ll)
#                 ll = torch.div(ll, var)
#                 ll = ll + np.log(2 * np.pi) + lvars
#                 ll = -0.5 * ll - log_edge_feats 
#                 #ll = torch.mean(ll)
#             
#             elif self.sampling_method  == "gamma":
#                 loga = mus
#                 logb = lvars
#                 a = torch.exp(loga)
#                 b = torch.exp(logb)
#                 log_edge_feats = torch.log(edge_feats)
#                 
#                 ll = torch.mul(a, logb)
#                 ll = ll - torch.lgamma(a)
#                 ll = ll + torch.mul(a - 1, log_edge_feats)
#                 ll = ll - torch.mul(b, edge_feats)
#             
#             if batch_idx is not None:
#                 i = 0
#                 for B in np.unique(batch_idx):
#                     ll_batch_wt[i] = ll_batch_wt[i] + torch.sum(ll[batch_idx == B])
#                     i = i + 1
#             
#             ll = torch.sum(ll)
#             
# #             elif self.sampling_method == "vae":
# #                 z, ll_kl = self.encode_weight(edge_feats, h)
# #                 _, ll = self.decode_weight(z, h, edge_feats)
# #                 ll = ll + ll_kl
#         return ll, ll_batch_wt, edge_feats
#     
# #     def encode_weight(self, edge_feats, h):
# #         edge_feats = self.standardize_edge_feats(edge_feats)
# #         vae_embed = self.embed_wt_vae(edge_feats)
# #         input_ = torch.cat([vae_embed, h[-1]], -1)
# #         mu = self.vae_mu(input_)
# #         logvar = self.vae_sig(input_)
# #         eps = torch.randn_like(mu)
# #         z = mu + torch.exp(0.5 * logvar) * eps
# #         ll_kl = 0.5 * torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar))
# #         return z, ll_kl
# #     
# #     def decode_weight(self, z, h, edge_feats=None):
# #         input_ = torch.cat([z, h[-1]], -1)
# #         w_star = self.weight_out(input_)
# #         ll = 0
# #         if edge_feats is not None:
# #             ll = -torch.square(w_star - edge_feats)
# #             ll = torch.sum(ll)
# #         return w_star, ll