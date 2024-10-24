

## Graph Convolution:

## ~~ https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv
## ~~ Good reading: https://docs.dgl.ai/en/0.8.x/guide/training-node.html

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch_geometric
from torch_geometric import nn
from torch_geometric.nn import conv
from bigg.common.pytorch_util import glorot_uniform, MLP
from bigg.common.configs import cmd_args, set_device

## Demo ##
#class SAGE(nn.Module):
#    def __init__(self, in_feats, hid_feats, out_dim):
#        super().__init__()
#        self.conv1 = dglnn.SAGEConv(
#            in_feats=in_feats, out_dim=hid_feats, aggregator_type='mean')
#        self.conv2 = dglnn.SAGEConv(
#            in_feats=hid_feats, out_dim=out_dim, aggregator_type='mean')
#
#    def forward(self, graph, inputs):
#        # inputs are features of nodes
#        h = self.conv1(graph, inputs)
#        h = F.relu(h)
#        h = self.conv2(graph, h)
#        return h

## Basic build ##

## Questions
## (1) No node features ---> use pretrained embeddings
	### Seeding with a Gaussian...
## (2) Varying graph sizes (node embeddings ...?) ---> just for tree problem right now...
## (3) Make model autoregressive ... use GRU to update embeddings?
## (4) Batching: https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html
#### GCN can pool in node info from diff parts of the graph...



class GCN(torch.nn.Module):
    def __init__(self, node_embed_dim, embed_dim, out_dim, max_num_nodes):
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.embed_dim = embed_dim
        self.node_embed_dim = node_embed_dim
        self.out_dim = out_dim
        
        self.conv1 = conv.GCNConv(in_channels = self.node_embed_dim, out_channels = self.embed_dim)
        self.conv2 = conv.GCNConv(in_channels = self.embed_dim, out_channels = self.out_dim)
        self.node_embedding = torch.nn.Embedding(self.max_num_nodes, self.node_embed_dim)
    
    def forward(self, feat_idx, edge_list):
        node_embeddings = self.node_embedding.weight[feat_idx.long()]
        h = self.conv1(node_embeddings, edge_list.long())
        h = F.relu(h)
        h = self.conv2(h, edge_list.long())
        return h


class GCN_Generate(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        ### Hyperparameters
        self.node_embed_dim = args.node_embed_dim
        self.embed_dim = args.embed_dim
        self.out_dim = args.out_dim
        self.max_num_nodes = args.max_num_nodes
        self.num_layers = args.rnn_layers
        
        ### GCN Model and GRU 
        self.GCN_mod = GCN(self.node_embed_dim, self.embed_dim, self.out_dim, self.max_num_nodes)
        #self.update_hidden = nn.GRU(node_embed_dim, out_dim)
        
        ### MLPs for mu, logvar, and weight embeddings
        self.hidden_to_mu = MLP(2 * self.embed_dim, [4 * self.embed_dim, 1])
        self.hidden_to_logvar = MLP(2 * self.embed_dim, [4 * self.embed_dim, 1])
        #self.embed_weight = MLP(1, [2 * self.node_embed_dim, self.node_embed_dim])
        
        self.softplus = torch.nn.Softplus()
        
        ### Statistics for weight standardization
        mu_wt = torch.tensor(0, dtype = float)
        var_wt = torch.tensor(1, dtype = float)
        n_obs = torch.tensor(0, dtype = int)
        min_wt = torch.tensor(np.inf, dtype = float)
        max_wt = torch.tensor(-np.inf, dtype = float)
        
        self.register_buffer("mu_wt", mu_wt)
        self.register_buffer("var_wt", var_wt)
        self.register_buffer("n_obs", n_obs)
        self.register_buffer("min_wt", min_wt)
        self.register_buffer("max_wt", max_wt)
        
        ### Test...
        self.embed_weight = MLP(1, [2 * self.out_dim, self.out_dim])#, dropout = args.wt_drop)
        self.GRU = torch.nn.GRU(input_size = self.out_dim, hidden_size = self.embed_dim, num_layers = self.num_layers, batch_first = True, bias = False)
        self.init_h0 = Parameter(torch.Tensor(self.num_layers, self.embed_dim))
        self.init_c0 = Parameter(torch.Tensor(self.num_layers, self.embed_dim))
        
        
        epoch_num = torch.tensor(0, dtype = int)
        self.register_buffer("epoch_num", epoch_num)
        
        self.mode = args.wt_mode
        
        self.log_wt = False
        self.sm_wt = False
        self.wt_range = 1.0
        self.wt_scale = 1.0
        
        glorot_uniform(self)
        
    
    ## Helper functions from LSTM model that are needed (weight loss, standardizing, ...)
    def compute_ll_w(self, mus, logvars, weights):
      '''
      Computes loss of graph weights, if graph is weighted
      
      Args Used:
        mus: model predicted means for each weight
        logvars: model predicted variances for each weight; log-scale
        weights: current weights of graph(s)
      
      Returns:
        loss_w: total loss of graph weights
      '''
      #if not self.weighted:
      #    return 0.0
      
      mus = mus.flatten()
      logvars = logvars.flatten()
      weights = weights.flatten()
    
      # Compute Loss using a "SoftPlus Normal" Distribution
      ll = torch.log(torch.exp(weights) - 1)
      ll = torch.square(torch.sub(mus, ll))
      ll = torch.mul(ll, torch.exp(-logvars))
      ll = -torch.mul(logvars, 0.5) - torch.mul(ll, 0.5)
      ll = torch.sum(ll)
      return ll
    
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
  
    def update_weight_stats(self, weights):
      '''
      Updates global mean and standard deviation of weights per batch if
      standardizing weights prior to MLP embedding. Only performed during the
      first epoch of training.
      
      Args Used:
        weights: weights from current iteration batch
      '''
      
      ## Current training weight statistics
      with torch.no_grad():
          if self.mode == "standardize":
              mu_n = self.mu_wt
              var_n = self.var_wt
              n = self.n_obs
              
              ## New weight statistics
              m = len(weights)
              
              if m > 1:
                  var_m = torch.var(weights)
              else:
                  var_m = 0.0
              
              mu_m = torch.mean(weights)
              tot = n + m
              
              
              ## Update weight statistics
              new_mu = (n * mu_n + m * mu_m) / tot
              
              new_var_avg = (max(n - 1, 0) * var_n + (m - 1) * var_m)/(tot - 1)
              new_var_resid = (m * mu_n**2 + n * mu_m**2)/(tot - 1) - (n * mu_m + m * mu_n)**2/(tot * (tot - 1))
              new_var = new_var_avg + new_var_resid
              
              ## Save
              self.mu_wt = new_mu
              self.var_wt = new_var
              self.n_obs += m
              
          
          else:
              batch_max = weights.max()
              batch_min = weights.min()
              self.min_wt = torch.min(batch_min, self.min_wt)
              self.max_wt = torch.max(batch_max, self.max_wt)
    
    def forward(self, feat_idx, edge_list, batch_weight_idx):
        h = self.GCN_mod.forward(feat_idx, edge_list[0:2, :])
        
        
        edges = batch_weight_idx[:, 0:2].long()
        weights = batch_weight_idx[:, 2:3]
        
        embedded_weights = self.embed_weight(self.standardize_edge_feats(weights))
        
        #print(embedded_weights.shape)
        
        batch_idx = edge_list[2:3, :].flatten()
        
        nodes = h[edges].flatten(1)

        
        #b_size = len(torch.unique(batch_idx))
        GRU_out = None
        for idx in torch.unique(batch_idx):
            b_weights = embedded_weights[batch_idx.flatten() == idx]
            out, _ = self.GRU(b_weights.unsqueeze(0), self.init_h0.unsqueeze(1))
            out = torch.cat([self.init_h0[-1].unsqueeze(0), out.squeeze(0)[:-1, :]])
            
            if GRU_out is None:
                GRU_out = out
            
            else:
                GRU_out = torch.cat([GRU_out, out], dim = 0)
        
        combined = torch.cat([nodes, GRU_out], dim = -1)
        
        mu_wt = self.hidden_to_mu(combined)
        logvar_wt = self.hidden_to_logvar(combined)
        
        ll_wt = self.compute_ll_w(mu_wt, logvar_wt, weights)
        
        #for (n1, n2) in edge_list:
        #    h1 = h[n1] ## Embedding for node 1
        #    h2 = h[n2] ## Embedding for node 2
            
            ## Predict Means and LogVariances
        #    mu_wt = self.hidden_to_mu(torch.cat([h1, h2], -1))
        #    logvar_wt = self.hidden_to_logvar(torch.cat([h1, h2], -1))
            
            ## Update Loss
        #    loss = loss + self.compute_loss_w(mu_wt, logvar_wt, weights)
            
            ## Embed weights
            #if self.epoch_num == 1:
            #    self.update_weight_stats(weights)
            #w_embedding = self.standardize_weights(weights, mode = self.mode, range_ = self.wt_range)
            #w_embedding = self.embed_weight(w_embedding.unsqueeze(0)).squeeze(0)
            
            ## Update Node States
            #h1 = self.update_hidden(w_embedding, h1)
            #h2 = self.update_hidden(w_embedding, h2)
            
            #h[n1] = h1
            #h[n2] = h2
        
        return ll_wt
    
    def sample(self, num_nodes, edge_list):
        feat_idx = torch.arange(num_nodes).to(edge_list.device)
        h = self.GCN_mod.forward(feat_idx, edge_list.t())
        edges = edge_list.long()
        
        num_edges = edges.shape[0]
        nodes = h[edges].flatten(1)
        
        weights = None
        for idx in range(num_edges):
            if idx == 0:
                hidden = self.init_h0.data.unsqueeze(1)
            
            cur_nodes = nodes[idx]
            #print(cur_nodes)
            print(cur_nodes.shape)
            print(hidden.shape)
            combined = torch.cat([cur_nodes, hidden[-1].squeeze(1)])
            mu_wt = self.hidden_to_mu(combined)
            logvar_wt = self.hidden_to_logvar(combined)
            std_wt = torch.exp(0.5 * logvar_wt)
            weight = torch.normal(mu_wt, std_wt)
            w = self.softplus(weight)
            
            if weights is None:
                weights = w
            
            else:
                weights = torch.cat([weights, w])
            
            embed_w = self.embed_weight(w)
            _, hidden = self.GRU(embed_w.reshape(1, 1, self.out_dim), hidden)
        
        weighted_edges = torch.cat([edge_list, weights.unsqueeze(-1)], dim = -1)
        
        return weighted_edges






































