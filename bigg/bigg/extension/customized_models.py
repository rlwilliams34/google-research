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
import torch
from bigg.common.pytorch_util import glorot_uniform, MLP
import torch.nn as nn

# pylint: skip-file


class BiggWithEdgeLen(RecurTreeGen):

    def __init__(self, args):
        super().__init__(args)
        self.edgelen_encoding = MLP(1, [args.embed_dim // 4, args.embed_dim])
        self.nodelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])
        self.nodelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, 1])
        self.edgelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, 1])
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)

    # to be customized
    def embed_node_feats(self, node_feats):
        return self.nodelen_encoding(node_feats)

    def embed_edge_feats(self, edge_feats):
        return self.edgelen_encoding(edge_feats)

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
        pred_node_len = self.nodelen_pred(h)
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
        mus, lvars = self.edgelen_mean(h), self.edgelen_lvar(h)
        
        if edge_feats is None:
            ll = 0
            pred_mean = mus
            pred_lvar = lvars
            pred_sd = torch.exp(0.5 * pred_lvar)
            edge_feats = torch.normal(pred_mean, pred_sd)
            edge_feats = torch.nn.functional.softplus(edge_feats)
            
        else:
            ### Update log likelihood with weight prediction
            
            ### Trying with softplus parameterization...
            edge_feats_invsp = torch.log(torch.special.expm1(edge_feats))
            
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
