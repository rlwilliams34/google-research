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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: skip-file

def lv_offset(num_edges):
    offset_list = []
    lv = 0
    while num_edges >= 1:
        offset_list.append(num_edges)
        num_edges = num_edges // 2
        lv += 1
    num_entries = np.sum(offset_list)
    return offset_list, num_entries

## Note number of entries per graph's set of edges will be sum of the lv offset lsit

def lv_list(k, n):
    offset, _ = lv_offset(n)
    lv = 0
    lv_list = []
    for i in range(len(bin(k)[2:])):
        if k & 2**i == 2**i:
            lv_list += [int(k // 2**i + np.sum(offset[:i])) - 1]
    return lv_list

### THIS GIVES A LIST OF INDICES FOR THE THINGIE!!!!!
def lv_list(k, offset):
    lv = 0
    lv_list = []
    for i in range(len(bin(k)[2:])):
        if k & 2**i == 2**i:
            lv_list += [int(k // 2**i + np.sum(offset[:i]))]
    return lv_list


# We have a graph with M edges...
# First, we need to compute all of the g i j 's and put them in a list
# LEVEL OFFSET helps give the indices of each state needed for the summary cell merges


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from collections import defaultdict
from torch.nn.parameter import Parameter
from bigg.common.pytorch_util import * #glorot_uniform, MLP, BinaryTreeLSTMCell, MultiLSTMCell, WeightedBinaryTreeLSTMCell
from tqdm import tqdm
from bigg.model.util import AdjNode, ColAutomata, AdjRow
from bigg.model.tree_clib.tree_lib import TreeLib
from bigg.torch_ops import multi_index_select, PosEncoding
from functools import partial


def hc_multi_select(ids_from, ids_to, h_froms, c_froms):
    h_vecs = multi_index_select(ids_from,
                                ids_to,
                                *h_froms)
    c_vecs = multi_index_select(ids_from,
                                ids_to,
                                *c_froms)
    return h_vecs, c_vecs


def tree_state_select(h_bot, c_bot, h_buf, c_buf, fn_all_ids):
    bot_froms, bot_tos, prev_froms, prev_tos = fn_all_ids()
    if h_buf is None or prev_tos is None:
        h_vecs = multi_index_select([bot_froms], [bot_tos], h_bot)
        c_vecs = multi_index_select([bot_froms], [bot_tos], c_bot)
    elif h_bot is None or bot_tos is None:
        h_vecs = multi_index_select([prev_froms], [prev_tos], h_buf)
        c_vecs = multi_index_select([prev_froms], [prev_tos], c_buf)
    else:
        h_vecs, c_vecs = hc_multi_select([bot_froms, prev_froms],
                                         [bot_tos, prev_tos],
                                         [h_bot, h_buf], [c_bot, c_buf])
    return h_vecs, c_vecs


def batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_all_ids, cell):
    h_list = []
    c_list = []
    for i in range(2):
        h_vecs, c_vecs = tree_state_select(h_bot, c_bot, h_buf, c_buf, lambda : fn_all_ids(i))
        h_list.append(h_vecs)
        c_list.append(c_vecs)
    return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))


def selective_update_hc(h, c, zero_one, feats):
    nz_idx = torch.tensor(np.nonzero(zero_one)[0]).to(h.device)
    num_layers = h.shape[0]
    embed_dim = h.shape[2]
    local_edge_feats_h = scatter(feats[0], nz_idx, dim=1, dim_size=h.shape[1])
    local_edge_feats_c = scatter(feats[1], nz_idx, dim=1, dim_size=h.shape[1])
    zero_one = torch.tensor(zero_one, dtype=torch.bool).to(h.device).unsqueeze(1)
    h = torch.where(zero_one, local_edge_feats_h, h)
    c = torch.where(zero_one, local_edge_feats_c, c)
    return h, c



def featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_all_ids, cell, t_lch=None, t_rch=None, cell_node=None):
    new_ids = [list(fn_all_ids(0)), list(fn_all_ids(1))]
    lch_isleaf, rch_isleaf = new_ids[0][0], new_ids[1][0]
    new_ids[0][0] = new_ids[1][0] = None
    is_leaf = [lch_isleaf, rch_isleaf]
    if edge_feats is not None:
        edge_feats = [(edge_feats[0][:, ~is_rch], edge_feats[1][:, ~is_rch]), (edge_feats[0][:, is_rch], edge_feats[1][:, is_rch])]
        assert np.sum(is_rch) == np.sum(rch_isleaf)
    node_feats = [t_lch, t_rch]
    h_list = []
    c_list = []
    for i in range(2):
        leaf_check = is_leaf[i]
        local_hbot, local_cbot = h_bot[:, leaf_check], c_bot[:, leaf_check]         
        if edge_feats is not None:
            local_hbot, local_cbot = selective_update_hc(local_hbot, local_cbot, leaf_check, edge_feats[i])
        if cell_node is not None:
            local_hbot, local_cbot = cell_node(node_feats[i], (local_hbot, local_cbot))
        h_vecs, c_vecs = tree_state_select(local_hbot, local_cbot, h_buf, c_buf, lambda : new_ids[i])
        h_list.append(h_vecs)
        c_list.append(c_vecs)
    return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))


def batch_tree_lstm3(h_bot, c_bot, h_buf, c_buf, h_past, c_past, fn_all_ids, cell):
    if h_past is None:
        return batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, lambda i: fn_all_ids(i)[:-2], cell)
    elif h_bot is None:
        return batch_tree_lstm2(h_buf, c_buf, h_past, c_past, lambda i: fn_all_ids(i)[2:], cell)
    elif h_buf is None:
        return batch_tree_lstm2(h_bot, c_bot, h_past, c_past, lambda i: fn_all_ids(i)[0, 1, 4, 5], cell)
    else:
        h_list = []
        c_list = []
        for i in range(2):
            bot_froms, bot_tos, prev_froms, prev_tos, past_froms, past_tos = fn_all_ids(i)
            h_vecs, c_vecs = hc_multi_select([bot_froms, prev_froms, past_froms],
                                             [bot_tos, prev_tos, past_tos],
                                             [h_bot, h_buf, h_past],
                                             [c_bot, c_buf, c_past])
            h_list.append(h_vecs)
            c_list.append(c_vecs)
        return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))


def featured_batch_tree_lstm3(feat_dict, h_bot, c_bot, h_buf, c_buf, h_past, c_past, fn_all_ids, cell, cell_node):
    edge_feats = is_rch = None
    t_lch = t_rch = None
    if 'edge' in feat_dict:
        edge_feats, is_rch = feat_dict['edge']
    if 'node' in feat_dict:
        t_lch, t_rch = feat_dict['node']
    if h_past is None:
        return featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, lambda i: fn_all_ids(i)[:-2], cell, t_lch, t_rch, cell_node)
    elif h_bot is None:
        return batch_tree_lstm2(h_buf, c_buf, h_past, c_past, lambda i: fn_all_ids(i)[2:], cell)
    elif h_buf is None:
        return featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_past, c_past, lambda i: fn_all_ids(i)[0, 1, 4, 5], cell, t_lch, t_rch, cell_node)
    else:
        raise NotImplementedError  #TODO: handle model parallelism with features


class FenwickTree(nn.Module):
    def __init__(self, args):
        super(FenwickTree, self).__init__()
        self.has_edge_feats = args.has_edge_feats
        self.has_node_feats = args.has_node_feats
        self.init_h0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
        self.init_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
        glorot_uniform(self)
        if self.has_node_feats:
            self.node_feat_update = nn.LSTMCell(args.embed_dim, args.embed_dim)
        self.merge_cell = BinaryTreeLSTMCell(args.embed_dim)
        self.summary_cell = BinaryTreeLSTMCell(args.embed_dim)
        if args.pos_enc:
            self.pos_enc = PosEncoding(args.embed_dim, args.device, args.pos_base)
        else:
            self.pos_enc = lambda x: 0

    def reset(self, list_states=[]):
        self.list_states = []
        for l in list_states:
            t = []
            for e in l:
                t.append(e)
            self.list_states.append(t)

    def append_state(self, state, level):
        if level >= len(self.list_states):
            num_aug = level - len(self.list_states) + 1
            for i in range(num_aug):
                self.list_states.append([])
        self.list_states[level].append(state)

    def forward(self, new_state=None):
        if new_state is None:
            if len(self.list_states) == 0:
                return (self.init_h0, self.init_c0)
        else:
            self.append_state(new_state, 0)
        pos = 0
        while pos < len(self.list_states):
            if len(self.list_states[pos]) >= 2:
                lch_state, rch_state = self.list_states[pos]  # assert the length is 2
                new_state = self.merge_cell(lch_state, rch_state)
                self.list_states[pos] = []
                self.append_state(new_state, pos + 1)
            pos += 1
        state = None
        for pos in range(len(self.list_states)):
            if len(self.list_states[pos]) == 0:
                continue
            cur_state = self.list_states[pos][0]
            if state is None:
                state = cur_state
            else:
                state = self.summary_cell(state, cur_state)
        return state

    def forward_train(self, h_bot, c_bot, h_buf0, c_buf0, prev_rowsum_h, prrev_rowsum_c):
        # embed row tree
        tree_agg_ids = TreeLib.PrepareRowEmbed()
        row_embeds = [(self.init_h0, self.init_c0)]
        if self.has_edge_feats or self.has_node_feats:
            feat_dict = c_bot
            if 'node' in feat_dict:
                node_feats, is_tree_trivial, t_lch, t_rch = feat_dict['node']
                sel_feat = node_feats[is_tree_trivial]
                feat_dict['node'] = (sel_feat[t_lch], sel_feat[t_rch])
            h_bot, c_bot = h_bot
        if h_bot is not None:
            row_embeds.append((h_bot, c_bot))
        if prev_rowsum_h is not None:
            row_embeds.append((prev_rowsum_h, prrev_rowsum_c))
        if h_buf0 is not None:
            row_embeds.append((h_buf0, c_buf0))
        print(row_embeds)
        for x in row_embeds:
            print(x[0].shape)

        for i, all_ids in enumerate(tree_agg_ids):
            print("i: ", i)
            print("all ids: ", all_ids)
            print("======================================================")
            fn_ids = lambda x: all_ids[x]
            lstm_func = batch_tree_lstm3
            if i == 0 and (self.has_edge_feats or self.has_node_feats):
                lstm_func = featured_batch_tree_lstm3
            lstm_func = partial(lstm_func, h_buf=row_embeds[-1][0], c_buf=row_embeds[-1][1],
                                h_past=prev_rowsum_h, c_past=prrev_rowsum_c, fn_all_ids=fn_ids, cell=self.merge_cell)
            if i == 0:
                if self.has_edge_feats or self.has_node_feats:
                    new_states = lstm_func(feat_dict, h_bot, c_bot, cell_node=None if not self.has_node_feats else self.node_feat_update)
                else:
                    new_states = lstm_func(h_bot, c_bot)
            else:
                new_states = lstm_func(None, None)
            row_embeds.append(new_states)
        h_list, c_list = zip(*row_embeds)
        joint_h = torch.cat(h_list, dim=1)
        joint_c = torch.cat(c_list, dim=1)
        print(joint_h.shape)

        # get history representation
        init_select, all_ids, last_tos, next_ids, pos_info = TreeLib.PrepareRowSummary()
        print("Init select: ", init_select)
        print("All IDs: ", all_ids)
        print("Last Tos: ", last_tos)
        print("Next Ids: ", next_ids)
        cur_state = (joint_h[:, init_select], joint_c[:, init_select])
        if self.has_node_feats:
            base_nodes, _ = TreeLib.GetFenwickBase()
            if len(base_nodes):
                needs_base_nodes = (init_select >= 1) & (init_select <= 2)
                sub_states = (cur_state[0][needs_base_nodes], cur_state[1][needs_base_nodes])
                sub_states = self.node_feat_update(node_feats[base_nodes], sub_states)
                nz_idx = torch.tensor(np.nonzero(needs_base_nodes)[0]).to(node_feats.device)
                new_cur = [scatter(x, nz_idx, dim=0, dim_size=init_select.shape[0]) for x in sub_states]
                needs_base_nodes = torch.tensor(needs_base_nodes, dtype=torch.bool).to(node_feats.device).unsqueeze(1)
                cur_state = [torch.where(needs_base_nodes, new_cur[i], cur_state[i]) for i in range(2)]
                cur_state = tuple(cur_state)
        ret_state = (joint_h[:, next_ids], joint_c[:, next_ids])
        hist_rnn_states = []
        hist_froms = []
        hist_tos = []
        for i, (done_from, done_to, proceed_from, proceed_input) in enumerate(all_ids):
            print("I: ", i)
            print("Done from", done_from)
            print("Don to", done_to)
            print("Proceed_from", proceed_from)
            print("Proceed_input", proceed_input)
            print("====================================================")
            hist_froms.append(done_from)
            hist_tos.append(done_to)
            hist_rnn_states.append(cur_state)

            next_input = joint_h[:, proceed_input], joint_c[:, proceed_input]
            sub_state = cur_state[0][:, proceed_from], cur_state[1][:, proceed_from]
            cur_state = self.summary_cell(sub_state, next_input)
        print(STOP)
        hist_rnn_states.append(cur_state)
        hist_froms.append(None)
        hist_tos.append(last_tos)
        hist_h_list, hist_c_list = zip(*hist_rnn_states)
        pos_embed = self.pos_enc(pos_info)
        row_h = multi_index_select(hist_froms, hist_tos, *hist_h_list) + pos_embed
        row_c = multi_index_select(hist_froms, hist_tos, *hist_c_list) + pos_embed
        return (row_h, row_c), ret_state



    def forward_train_EDIT(self, edge_feats_init_embed):
        # embed row tree
        tree_agg_ids = TreeLib.PrepareRowEmbed() ##### REPLACE...
        edge_embeds = [edge_feats_init_embed]
        
        for i, all_ids in enumerate(tree_agg_ids):
            fn_ids = lambda x: all_ids[x]
            lstm_func = partial(batch_tree_lstm3, h_buf=edge_embeds[-1][0], c_buf=edge_embeds[-1][1],
                                 fn_all_ids=fn_ids, cell=self.merge_cell)
            new_states = lstm_func(None, None)
            edge_embeds.append(new_states)
        h_list, c_list = zip(*edge_embeds)
        joint_h = torch.cat(h_list, dim=1)
        joint_c = torch.cat(c_list, dim=1)
        print(joint_h.shape)

        # get history representation
        init_select, all_ids, last_tos, next_ids, pos_info = TreeLib.PrepareRowSummary() #### REPLACE
        print("Init select: ", init_select)
        print("All IDs: ", all_ids)
        print("Last Tos: ", last_tos)
        print("Next Ids: ", next_ids)
        cur_state = (joint_h[:, init_select], joint_c[:, init_select])
        
        ret_state = (joint_h[:, next_ids], joint_c[:, next_ids])
        hist_rnn_states = []
        hist_froms = []
        hist_tos = []
        for i, (done_from, done_to, proceed_from, proceed_input) in enumerate(all_ids):
            print("I: ", i)
            print("Done from", done_from)
            print("Don to", done_to)
            print("Proceed_from", proceed_from)
            print("Proceed_input", proceed_input)
            print("====================================================")
            hist_froms.append(done_from)
            hist_tos.append(done_to)
            hist_rnn_states.append(cur_state)

            next_input = joint_h[:, proceed_input], joint_c[:, proceed_input]
            sub_state = cur_state[0][:, proceed_from], cur_state[1][:, proceed_from]
            cur_state = self.summary_cell(sub_state, next_input)
        print(STOP)
        hist_rnn_states.append(cur_state)
        hist_froms.append(None)
        hist_tos.append(last_tos)
        hist_h_list, hist_c_list = zip(*hist_rnn_states)
        pos_embed = self.pos_enc(pos_info)
        row_h = multi_index_select(hist_froms, hist_tos, *hist_h_list) + pos_embed
        row_c = multi_index_select(hist_froms, hist_tos, *hist_c_list) + pos_embed
        return edge_embeddings






class BitsRepNet(nn.Module):
    def __init__(self, args):
        super(BitsRepNet, self).__init__()
        self.bits_compress = args.bits_compress
        self.out_dim = args.embed_dim
        assert self.out_dim >= self.bits_compress
        self.device = args.device

    def forward(self, on_bits, n_cols):
        h = torch.zeros(1, self.out_dim).to(self.device)
        h[0, :n_cols] = -1.0
        h[0, on_bits] = 1.0

        return h, h


class RecurTreeGen(nn.Module):

    # to be customized
    def embed_node_feats(self, node_feats):
        raise NotImplementedError

    def embed_edge_feats(self, edge_feats):
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def __init__(self, args):
        super(RecurTreeGen, self).__init__()

        self.directed = args.directed
        self.sigma = args.sigma
        self.batch_size = args.batch_size
        self.self_loop = args.self_loop
        self.bits_compress = args.bits_compress
        self.has_edge_feats = args.has_edge_feats
        self.has_node_feats = args.has_node_feats
        if self.has_edge_feats:
            assert self.bits_compress == 0
        self.greedy_frac = args.greedy_frac
        self.share_param = args.share_param
        if not self.bits_compress:
            self.leaf_h0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
            self.leaf_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
            self.empty_h0 = Parameter(torch.Tensor(args.rnn_layers, 1,  args.embed_dim))
            self.empty_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))

        self.topdown_left_embed = Parameter(torch.Tensor(2, args.embed_dim))
        self.topdown_right_embed = Parameter(torch.Tensor(2, args.embed_dim))
        glorot_uniform(self)
        self.method = args.method

        if self.bits_compress > 0:
            self.bit_rep_net = BitsRepNet(args)

        if self.share_param:
            self.m_l2r_cell = BinaryTreeLSTMCell(args.embed_dim)
            self.lr2p_cell = BinaryTreeLSTMCell(args.embed_dim)
            self.pred_has_ch = MLP(args.embed_dim, [2 * args.embed_dim, 1])
            self.m_pred_has_left = MLP(args.embed_dim, [2 * args.embed_dim, 1])
            self.m_pred_has_right = MLP(args.embed_dim, [2 * args.embed_dim, 1])
            self.m_cell_topdown = MultiLSTMCell(args.embed_dim, args.embed_dim, args.rnn_layers)
            self.m_cell_topright = MultiLSTMCell(args.embed_dim, args.embed_dim, args.rnn_layers)
        else:
            fn_pred = lambda: MLP(args.embed_dim, [2 * args.embed_dim, 1])
            fn_tree_cell = lambda: BinaryTreeLSTMCell(args.embed_dim)
            fn_lstm_cell = lambda: MultiLSTMCell(args.embed_dim, args.embed_dim, args.rnn_layers)
            num_params = int(np.ceil(np.log2(args.max_num_nodes))) + 1
            self.pred_has_ch = fn_pred()

            pred_modules = [[] for _ in range(2)]
            tree_cell_modules = []
            lstm_cell_modules = [[] for _ in range(2)]
            for _ in range(num_params):
                for i in range(2):
                    pred_modules[i].append(fn_pred())
                    lstm_cell_modules[i].append(fn_lstm_cell())
                tree_cell_modules.append(fn_tree_cell())

            self.has_left_modules, self.has_right_modules = [nn.ModuleList(l) for l in pred_modules]
            self.l2r_modules= nn.ModuleList(tree_cell_modules)
            self.cell_topdown_modules, self.cell_topright_modules = [nn.ModuleList(l) for l in lstm_cell_modules]
            self.lr2p_cell = fn_tree_cell()
        self.row_tree = FenwickTree(args)

        if args.tree_pos_enc:
            self.tree_pos_enc = PosEncoding(args.embed_dim, args.device, args.pos_base, bias=np.pi / 4)
        else:
            self.tree_pos_enc = lambda x: 0

    def cell_topdown(self, x, y, lv):
        cell = self.m_cell_topdown if self.share_param else self.cell_topdown_modules[lv]
        return cell(x, y)

    def cell_topright(self, x, y, lv):
        cell = self.m_cell_topright if self.share_param else self.cell_topright_modules[lv]
        return cell(x, y)

    def l2r_cell(self, x, y, lv=-1):
        cell = self.m_l2r_cell if self.share_param else self.l2r_modules[lv]
        return cell(x, y)

    def pred_has_left(self, x, lv):
        mlp = self.m_pred_has_left if self.share_param else self.has_left_modules[lv]
        return mlp(x)

    def pred_has_right(self, x, lv):
        mlp = self.m_pred_has_right if self.share_param else self.has_right_modules[lv]
        return mlp(x)

    def get_empty_state(self):
        if self.bits_compress:
            return self.bit_rep_net([], 1)
        elif self.method == "Test9":
            return (self.test9_h0, self.test9_c0)
        else:
            return (self.empty_h0, self.empty_c0)

    def get_prob_fix(self, prob):
        p = prob * (1 - self.greedy_frac)
        if prob >= 0.5:
            p += self.greedy_frac
        return p

    def gen_row(self, ll, ll_wt, state, tree_node, col_sm, lb, ub, edge_feats=None, row=None, prev_state=None):
        assert lb <= ub
        if tree_node.is_root:
            prob_has_edge = torch.sigmoid(self.pred_has_ch(state[0][-1]))
            if col_sm.supervised:
                has_edge = len(col_sm.indices) > 0
            else:
                has_edge = np.random.rand() < self.get_prob_fix(prob_has_edge.item())
                if ub == 0:
                    has_edge = False
                if tree_node.n_cols <= 0:
                    has_edge = False
                if lb:
                    has_edge = True
            if has_edge:
                ll = ll + torch.log(prob_has_edge)
            else:
                ll = ll + torch.log(1 - prob_has_edge)
            tree_node.has_edge = has_edge
        else:
            assert ub > 0
            tree_node.has_edge = True
        
        if not tree_node.has_edge:  # an empty tree
            return ll, ll_wt, self.get_empty_state(), 0, None, prev_state

        if tree_node.is_leaf:
            tree_node.bits_rep = [0]
            col_sm.add_edge(tree_node.col_range[0])
            if self.bits_compress:
                return ll, self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols), 1, None
            else:
                if self.has_edge_feats:
                    cur_feats = edge_feats[col_sm.pos - 1].unsqueeze(0) if col_sm.supervised else None
                    rc = None
                    if self.method in ["Test10", "Test12"]:
                        col = tree_node.col_range[0]
                        rc = np.array([col, row]).reshape(1, 1, 2)
                    
                    edge_ll, _, cur_feats = self.predict_edge_feats(state, cur_feats)
                    ll_wt = ll_wt + edge_ll
                    edge_embed = self.embed_edge_feats(cur_feats, rc=rc, prev_state=prev_state)
                    if prev_state is not None:
                        prev_state = edge_embed
                    return ll, ll_wt, edge_embed, 1, cur_feats, prev_state
                    
                else:
                    return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, None, None
        else:
            tree_node.split()

            mid = (tree_node.col_range[0] + tree_node.col_range[1]) // 2
            left_prob = torch.sigmoid(self.pred_has_left(state[0][-1], tree_node.depth))

            if col_sm.supervised:
                has_left = col_sm.next_edge < mid
            else:
                has_left = np.random.rand() < self.get_prob_fix(left_prob.item())
                if ub == 0:
                    has_left = False
                if lb > tree_node.rch.n_cols:
                    has_left = True
            ll = ll + (torch.log(left_prob) if has_left else torch.log(1 - left_prob))
            left_pos = self.tree_pos_enc([tree_node.lch.n_cols])
            state = self.cell_topdown(self.topdown_left_embed[[int(has_left)]] + left_pos, state, tree_node.depth)
            pred_edge_feats = []
            if has_left:
                lub = min(tree_node.lch.n_cols, ub)
                llb = max(0, lb - tree_node.rch.n_cols)
                ll, ll_wt, left_state, num_left, left_edge_feats, prev_state = self.gen_row(ll, ll_wt, state, tree_node.lch, col_sm, llb, lub, edge_feats, row=row, prev_state=prev_state)
                pred_edge_feats.append(left_edge_feats)
            else:
                left_state = self.get_empty_state()
                num_left = 0

            right_pos = self.tree_pos_enc([tree_node.rch.n_cols])
            topdown_state = self.l2r_cell(state, (left_state[0] + right_pos, left_state[1] + right_pos), tree_node.depth)
            rlb = max(0, lb - num_left)
            rub = min(tree_node.rch.n_cols, ub - num_left)
            if not has_left:
                has_right = True
            else:
                right_prob = torch.sigmoid(self.pred_has_right(topdown_state[0][-1], tree_node.depth))
                if col_sm.supervised:
                    has_right = col_sm.has_edge(mid, tree_node.col_range[1])
                else:
                    has_right = np.random.rand() < self.get_prob_fix(right_prob.item())
                    if rub == 0:
                        has_right = False
                    if rlb:
                        has_right = True
                ll = ll + (torch.log(right_prob) if has_right else torch.log(1 - right_prob))

            topdown_state = self.cell_topright(self.topdown_right_embed[[int(has_right)]], topdown_state, tree_node.depth)

            if has_right:  # has edge in right child
                ll, ll_wt, right_state, num_right, right_edge_feats, prev_state = self.gen_row(ll, ll_wt, topdown_state, tree_node.rch, col_sm, rlb, rub, edge_feats, row=row, prev_state=prev_state)
                pred_edge_feats.append(right_edge_feats)
            else:
                right_state = self.get_empty_state()
                num_right = 0
            if tree_node.col_range[1] - tree_node.col_range[0] <= self.bits_compress:
                summary_state = self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols)
            else:
                summary_state = self.lr2p_cell(left_state, right_state)
            if self.has_edge_feats:
                edge_feats = torch.cat(pred_edge_feats, dim=0)
            return ll, ll_wt, summary_state, num_left + num_right, edge_feats, prev_state
            

    def forward(self, node_end, edge_list=None, node_feats=None, edge_feats=None, node_start=0, list_states=[], lb_list=None, ub_list=None, col_range=None, num_nodes=None, display=False):
        pos = 0
        total_ll = 0.0
        total_ll_wt = 0.0
        edges = []
        self.row_tree.reset(list_states)
        controller_state = self.row_tree()
        if num_nodes is None:
            num_nodes = node_end
        pbar = range(node_start, node_end)
        if display:
            pbar = tqdm(pbar)
        list_pred_node_feats = []
        list_pred_edge_feats = []
        
        prev_state = None
        if self.method == "Test12":
            prev_state = (self.leaf_h0, self.leaf_c0)
        
        if self.method == "Test9":
            x_in = torch.cat([self.empty_embed, torch.zeros(1, self.weight_embed_dim).to(self.empty_embed.device)], dim = -1)
            h, c = self.leaf_LSTM(x_in, (self.leaf_h0, self.leaf_c0))
            self.test9_h0 = h
            self.test9_c0 = c
        
        for i in pbar:
            if edge_list is None:
                col_sm = ColAutomata(supervised=False)
            else:
                indices = []
                while pos < len(edge_list) and i == edge_list[pos][0]:
                    indices.append(edge_list[pos][1])
                    pos += 1
                indices.sort()
                col_sm = ColAutomata(supervised=True, indices=indices)

            cur_row = AdjRow(i, self.directed, self.self_loop, col_range=col_range)
            lb = 0 if lb_list is None else lb_list[i]
            ub = cur_row.root.n_cols if ub_list is None else ub_list[i]
            cur_pos_embed = self.row_tree.pos_enc([num_nodes - i])
            controller_state = [x + cur_pos_embed for x in controller_state]
            if self.has_node_feats:
                target_node_feats = None if node_feats is None else node_feats[[i]]
                controller_state, ll_node, target_node_feats = self.predict_node_feats(controller_state, target_node_feats)
                total_ll = total_ll + ll_node
                list_pred_node_feats.append(target_node_feats)
            if self.has_edge_feats:
                target_edge_feats = None if edge_feats is None else edge_feats[len(edges) : len(edges) + len(col_sm)]
            else:
                target_edge_feats = None
            ll, ll_wt, cur_state, _, target_edge_feats, prev_state = self.gen_row(0, 0, controller_state, cur_row.root, col_sm, lb, ub, target_edge_feats, row=i, prev_state=prev_state)
            if target_edge_feats is not None and target_edge_feats.shape[0]:
                list_pred_edge_feats.append(target_edge_feats)
            if self.has_node_feats:
                target_feat_embed = self.embed_node_feats(target_node_feats)
                cur_state = self.row_tree.node_feat_update(target_feat_embed, cur_state)
            assert lb <= len(col_sm.indices) <= ub
            controller_state = self.row_tree(cur_state)
            edges += [(i, x) for x in col_sm.indices]
            total_ll = total_ll + ll
            total_ll_wt = total_ll_wt + ll_wt

        if self.has_node_feats:
            node_feats = torch.cat(list_pred_node_feats, dim=0)
        if self.has_edge_feats:
            edge_feats = torch.cat(list_pred_edge_feats, dim=0)
        return total_ll, total_ll_wt, edges, self.row_tree.list_states, node_feats, edge_feats


    def binary_ll(self, pred_logits, np_label, need_label=False, reduction='sum', batch_idx=None, ll_batch=None):
        pred_logits = pred_logits.view(-1, 1)
        label = torch.tensor(np_label, dtype=torch.float32).to(pred_logits.device).view(-1, 1)
        loss = F.binary_cross_entropy_with_logits(pred_logits, label, reduction=reduction)
        
        ind_loss = F.binary_cross_entropy_with_logits(pred_logits, label, reduction='none')
        
        if batch_idx is not None:
            i = 0
            for B in np.unique(batch_idx):
                ll_batch[i] = ll_batch[i] - torch.sum(ind_loss[batch_idx == B])
                i = i + 1
        
        if need_label:
            return -loss, label, ll_batch
        return -loss, ll_batch

    def forward_row_trees(self, graph_ids, node_feats=None, edge_feats=None, list_node_starts=None, num_nodes=-1, list_col_ranges=None):
        TreeLib.PrepareMiniBatch(graph_ids, list_node_starts, num_nodes, list_col_ranges)
        # embed trees
        all_ids = TreeLib.PrepareTreeEmbed()
        if self.has_node_feats:
            node_feats = self.embed_node_feats(node_feats)

        if not self.bits_compress:
            if self.method == "Test9":
                x_in = torch.cat([self.empty_embed, torch.zeros(1, self.weight_embed_dim).to(self.empty_embed.device)], dim = -1)
                empty_h0, empty_c0 = self.leaf_LSTM(x_in, (self.leaf_h0, self.leaf_c0))
                h_bot = torch.cat([empty_h0, self.leaf_h0], dim=1)
                c_bot = torch.cat([empty_c0, self.leaf_c0], dim=1)
                
            else:
                h_bot = torch.cat([self.empty_h0, self.leaf_h0], dim=1)
                c_bot = torch.cat([self.empty_c0, self.leaf_c0], dim=1)
            
            fn_hc_bot = lambda d: (h_bot, c_bot)
        
        else:
            binary_embeds, base_feat = TreeLib.PrepareBinary()
            fn_hc_bot = lambda d: (binary_embeds[d], binary_embeds[d]) if d < len(binary_embeds) else base_feat
        max_level = len(all_ids) - 1
        h_buf_list = [None] * (len(all_ids) + 1)
        c_buf_list = [None] * (len(all_ids) + 1)

        for d in range(len(all_ids) - 1, -1, -1):
            fn_ids = lambda i: all_ids[d][i]
            if d == max_level:
                h_buf = c_buf = None
            else:
                h_buf = h_buf_list[d + 1]
                c_buf = c_buf_list[d + 1]
            h_bot, c_bot = fn_hc_bot(d + 1)
            if self.has_edge_feats:
                edge_idx, is_rch = TreeLib.GetEdgeAndLR(d + 1)
                local_edge_feats = (edge_feats[0][:, edge_idx], edge_feats[1][:, edge_idx])
                new_h, new_c = featured_batch_tree_lstm2(local_edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell)
            else:
                new_h, new_c = batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell)
            h_buf_list[d] = new_h
            c_buf_list[d] = new_c
        hc_bot = fn_hc_bot(0)
        feat_dict = {}
        if self.has_edge_feats:
            edge_idx, is_rch = TreeLib.GetEdgeAndLR(0)
            local_edge_feats = (edge_feats[0][:, edge_idx], edge_feats[1][:, edge_idx])
            feat_dict['edge'] = (local_edge_feats, is_rch)
        if self.has_node_feats:
            is_tree_trivial = TreeLib.GetIsTreeTrivial()
            new_h, new_c = self.row_tree.node_feat_update(node_feats[~is_tree_trivial], (new_h, new_c))
            h_buf_list[0] = new_h
            c_buf_list[0] = new_c
            t_lch, t_rch = TreeLib.GetTrivialNodes()
            feat_dict['node'] = (node_feats, is_tree_trivial, t_lch, t_rch)
        if len(feat_dict):
            hc_bot = (hc_bot, feat_dict)
        return hc_bot, fn_hc_bot, h_buf_list, c_buf_list
    
    def forward_row_summaries(self, graph_ids, node_feats=None, edge_feats=None,
                             list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None):
        hc_bot, _, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, node_feats, edge_feats,
                                                                   list_node_starts, num_nodes, list_col_ranges)
        row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states)
        return row_states, next_states

    def forward_train(self, graph_ids, node_feats=None, edge_feats=None,
                      list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None, batch_idx=None):
        ll = 0.0
        ll_wt = 0.0
        noise = 0.0
        ll_batch = (None if batch_idx is None else np.zeros(len(np.unique(batch_idx))))
        ll_batch_wt = (None if batch_idx is None else np.zeros(len(np.unique(batch_idx))))
        edge_feats_embed = None
        
        if self.has_edge_feats:
            rc = None
            if self.method in ["Test10", "Test12"]:
                edge_feats, rc = edge_feats    
            edge_feats_embed = self.embed_edge_feats(edge_feats, sigma=self.sigma, rc=rc)
            if self.method == "Test12":
                edge_feats = torch.cat(edge_feats, dim = 0)
       
        hc_bot, fn_hc_bot, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, node_feats, edge_feats_embed, list_node_starts, num_nodes, list_col_ranges)
        row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states)
        if self.has_node_feats:
            row_states, ll_node_feats, _ = self.predict_node_feats(row_states, node_feats)
            ll = ll + ll_node_feats
        logit_has_edge = self.pred_has_ch(row_states[0][-1])
        has_ch, _ = TreeLib.GetChLabel(0, dtype=bool)
        ll_cur, ll_batch = self.binary_ll(logit_has_edge, has_ch, batch_idx = batch_idx, ll_batch = ll_batch)
        ll = ll + ll_cur
        cur_states = (row_states[0][:, has_ch], row_states[1][:, has_ch])
        
        if batch_idx is not None:
            batch_idx = batch_idx[has_ch]
        
        lv=0
        while True:
            is_nonleaf = TreeLib.QueryNonLeaf(lv)
            if self.has_edge_feats:
                edge_of_lv = TreeLib.GetEdgeOf(lv)
                edge_state = (cur_states[0][:, ~is_nonleaf], cur_states[1][:, ~is_nonleaf])
                cur_batch_idx = (None if batch_idx is None else batch_idx[~is_nonleaf])
                target_feats = edge_feats[edge_of_lv]
                edge_ll, ll_batch_wt, _ = self.predict_edge_feats(edge_state, target_feats, batch_idx = cur_batch_idx, ll_batch_wt = ll_batch_wt)
                ll_wt = ll_wt + edge_ll
            if is_nonleaf is None or np.sum(is_nonleaf) == 0:
                break
            cur_states = (cur_states[0][:, is_nonleaf], cur_states[1][:, is_nonleaf])
            
            if batch_idx is not None:
                batch_idx = batch_idx[is_nonleaf]
            
            left_logits = self.pred_has_left(cur_states[0][-1], lv)
            has_left, num_left = TreeLib.GetChLabel(-1, lv)
            left_update = self.topdown_left_embed[has_left] + self.tree_pos_enc(num_left)
            left_ll, float_has_left, ll_batch = self.binary_ll(left_logits, has_left, need_label=True, reduction='sum')
            ll = ll + left_ll

            cur_states = self.cell_topdown(left_update, cur_states, lv)

            left_ids = TreeLib.GetLeftRootStates(lv)
            h_bot, c_bot = fn_hc_bot(lv + 1)
            if lv + 1 < len(h_buf_list):
                h_next_buf, c_next_buf = h_buf_list[lv + 1], c_buf_list[lv + 1]
            else:
                h_next_buf = c_next_buf = None
            if self.has_edge_feats:
                edge_idx, is_rch = TreeLib.GetEdgeAndLR(lv + 1)
                left_feats = (edge_feats_embed[0][:, edge_idx[~is_rch]], edge_feats_embed[1][:, edge_idx[~is_rch]])
                h_bot, c_bot = h_bot[:, left_ids[0]], c_bot[:, left_ids[0]]
                h_bot, c_bot = selective_update_hc(h_bot, c_bot, left_ids[0], left_feats)
                left_ids = tuple([None] + list(left_ids[1:]))

            left_subtree_states = tree_state_select(h_bot, c_bot,
                                                    h_next_buf, c_next_buf,
                                                    lambda: left_ids)

            has_right, num_right = TreeLib.GetChLabel(1, lv)
            right_pos = self.tree_pos_enc(num_right)
            left_subtree_states = [x + right_pos for x in left_subtree_states]
            topdown_state = self.l2r_cell(cur_states, left_subtree_states, lv)

            right_logits = self.pred_has_right(topdown_state[0][-1], lv)
            right_update = self.topdown_right_embed[has_right]
            topdown_state = self.cell_topright(right_update, topdown_state, lv)
            right_ll, _ = self.binary_ll(right_logits, has_right, reduction='none')
            right_ll = right_ll * float_has_left
            ll = ll + torch.sum(right_ll)
            
            if batch_idx is not None:
                i = 0
                for B in np.unique(batch_idx):
                    ll_batch[i] = ll_batch[i] + torch.sum(right_ll[batch_idx == B])
            
            lr_ids = TreeLib.GetLeftRightSelect(lv, np.sum(has_left), np.sum(has_right))
            new_states = []
            for i in range(2):
                new_s = multi_index_select([lr_ids[0], lr_ids[2]], [lr_ids[1], lr_ids[3]],
                                            cur_states[i], topdown_state[i])
                new_states.append(new_s)
            cur_states = tuple(new_states)
            lv += 1

        return ll, ll_wt, ll_batch, ll_batch_wt, next_states













# 
# 
# 
# # coding=utf-8
# # Copyright 2024 The Google Research Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# 
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# # pylint: skip-file
# 
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_scatter import scatter
# from collections import defaultdict
# from torch.nn.parameter import Parameter
# from bigg.common.pytorch_util import * #glorot_uniform, MLP, BinaryTreeLSTMCell, MultiLSTMCell, WeightedBinaryTreeLSTMCell
# from tqdm import tqdm
# from bigg.model.util import AdjNode, ColAutomata, AdjRow
# from bigg.model.tree_clib.tree_lib import TreeLib
# from bigg.torch_ops import multi_index_select, PosEncoding
# from functools import partial
# 
# 
# def hc_multi_select(ids_from, ids_to, h_froms, c_froms):
#     h_vecs = multi_index_select(ids_from,
#                                 ids_to,
#                                 *h_froms)
#     c_vecs = multi_index_select(ids_from,
#                                 ids_to,
#                                 *c_froms)
#     return h_vecs, c_vecs
# 
# 
# def tree_state_select(h_bot, c_bot, h_buf, c_buf, fn_all_ids):
#     bot_froms, bot_tos, prev_froms, prev_tos = fn_all_ids()
#     if h_buf is None or prev_tos is None:
#         h_vecs = multi_index_select([bot_froms], [bot_tos], h_bot)
#         c_vecs = multi_index_select([bot_froms], [bot_tos], c_bot)
#     elif h_bot is None or bot_tos is None:
#         h_vecs = multi_index_select([prev_froms], [prev_tos], h_buf)
#         c_vecs = multi_index_select([prev_froms], [prev_tos], c_buf)
#     else:
#         h_vecs, c_vecs = hc_multi_select([bot_froms, prev_froms],
#                                          [bot_tos, prev_tos],
#                                          [h_bot, h_buf], [c_bot, c_buf])
#     return h_vecs, c_vecs
# 
# 
# def batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_all_ids, cell):
#     h_list = []
#     c_list = []
#     for i in range(2):
#         h_vecs, c_vecs = tree_state_select(h_bot, c_bot, h_buf, c_buf, lambda : fn_all_ids(i))
#         h_list.append(h_vecs)
#         c_list.append(c_vecs)
#     return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))
# 
# def selective_update_hc(h, c, zero_one, feats):
#     nz_idx = torch.tensor(np.nonzero(zero_one)[0]).to(h.device)
#     num_layers = h.shape[0]
#     embed_dim = h.shape[2]
#     local_edge_feats_h = scatter(feats[0], nz_idx, dim=1, dim_size=h.shape[1])
#     local_edge_feats_c = scatter(feats[1], nz_idx, dim=1, dim_size=h.shape[1])
#     zero_one = torch.tensor(zero_one, dtype=torch.bool).to(h.device).unsqueeze(1)
#     h = torch.where(zero_one, local_edge_feats_h, h)
#     c = torch.where(zero_one, local_edge_feats_c, c)
#     return h, c
# 
# def featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_all_ids, cell, t_lch=None, t_rch=None, cell_node=None, wt_update=None, method=None, lv=-1):
#     new_ids = [list(fn_all_ids(0)), list(fn_all_ids(1))]
#     lch_isleaf, rch_isleaf = new_ids[0][0], new_ids[1][0]
#     new_ids[0][0] = new_ids[1][0] = None
#     is_leaf = [lch_isleaf, rch_isleaf]
#     
#     if edge_feats is not None:
#         if method in ["Test", "Test2", "Test3"]:
#             edge_feats = [edge_feats[~is_rch], edge_feats[is_rch]]
#         
#         elif method == "Test8":
#             if lv == -1:
#                 edge_feats = [edge_feats[~is_rch], edge_feats[is_rch]]
#             else:
#                 edge_feats = [(edge_feats[0][:, ~is_rch], edge_feats[1][:, ~is_rch]), (edge_feats[0][:, is_rch], edge_feats[1][:, is_rch])]
#         
#         else:
#             edge_feats = [(edge_feats[0][:, ~is_rch], edge_feats[1][:, ~is_rch]), (edge_feats[0][:, is_rch], edge_feats[1][:, is_rch])]
#         assert np.sum(is_rch) == np.sum(rch_isleaf)
#     node_feats = [t_lch, t_rch]
#     h_list = []
#     c_list = []
#     list_edge_feats = [None, None]
#     
#     for i in range(2):
#         leaf_check = is_leaf[i]
#         local_hbot, local_cbot = h_bot[:, leaf_check], c_bot[:, leaf_check]
#                      
#         if edge_feats is not None and method not in ["Test", "Test2", "Test3"]:
#             if method == "Test8" and lv == 0:
#                 local_hbot, local_cbot = selective_update_hc(local_hbot, local_cbot, leaf_check, edge_feats[i])
#             elif method not in ["Test4", "Test5", "Test8"] or lv == 0:
#                 local_hbot, local_cbot = selective_update_hc(local_hbot, local_cbot, leaf_check, edge_feats[i])
#         if cell_node is not None:
#             local_hbot, local_cbot = cell_node(node_feats[i], (local_hbot, local_cbot))
#         
#         h_vecs, c_vecs = tree_state_select(local_hbot, local_cbot, h_buf, c_buf, lambda : new_ids[i])
#         
#         if method == "Test8" and lv == -1:
#             dev = local_hbot.device
#             h0 = local_hbot.shape[1]
#             h1 = h_vecs.shape[1]
#             h2 = edge_feats[i].shape[-1]
#             z = torch.zeros(h1, h2).to(dev)
#             leaf_check2 = np.array(leaf_check).astype(bool)
#             edge_ids = np.arange(h0)[leaf_check2]
#             test = new_ids[i][1]
#             edge_ids = test[edge_ids]
#             z[edge_ids] = edge_feats[i]
#             list_edge_feats[i] = z
#         
#         h_list.append(h_vecs)
#         c_list.append(c_vecs)
#     
#     summary_state = cell((h_list[0], c_list[0]), (h_list[1], c_list[1]), list_edge_feats[0], list_edge_feats[1])
#     
#     if method != "Test" or edge_feats is None:
#         return summary_state
#     
#     for i in range(2):
#         leaf_check = list(map(bool, is_leaf[i]))
#         local_idx = new_ids[i][1][leaf_check]
#         local_hbot, local_cbot = summary_state[0][:, local_idx], summary_state[1][:, local_idx]
#         cur_summary = (local_hbot, local_cbot)
#         cur_edge_feats = edge_feats[i]
#         cur_summary = wt_update(cur_edge_feats, cur_summary)
#         summary_state[0][:, local_idx] = cur_summary[0]
#         summary_state[1][:, local_idx] = cur_summary[1]
#         
#     return summary_state
# 
# 
# def batch_tree_lstm3(h_bot, c_bot, h_buf, c_buf, h_past, c_past, fn_all_ids, cell):
#     if h_past is None:
#         return batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, lambda i: fn_all_ids(i)[:-2], cell)
#     elif h_bot is None:
#         return batch_tree_lstm2(h_buf, c_buf, h_past, c_past, lambda i: fn_all_ids(i)[2:], cell)
#     elif h_buf is None:
#         return batch_tree_lstm2(h_bot, c_bot, h_past, c_past, lambda i: fn_all_ids(i)[0, 1, 4, 5], cell)
#     else:
#         h_list = []
#         c_list = []
#         for i in range(2):
#             bot_froms, bot_tos, prev_froms, prev_tos, past_froms, past_tos = fn_all_ids(i)
#             h_vecs, c_vecs = hc_multi_select([bot_froms, prev_froms, past_froms],
#                                              [bot_tos, prev_tos, past_tos],
#                                              [h_bot, h_buf, h_past],
#                                              [c_bot, c_buf, c_past])
#             h_list.append(h_vecs)
#             c_list.append(c_vecs)
#         return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))
# 
# 
# def featured_batch_tree_lstm3(feat_dict, h_bot, c_bot, h_buf, c_buf, h_past, c_past, fn_all_ids, cell, cell_node, wt_update, method):
#     edge_feats = is_rch = None
#     t_lch = t_rch = None
#     if method in ["Test4", "Test5", "Test8"]:
#         lv = 0
#     else:
#         lv = -1
#     if 'edge' in feat_dict:
#         edge_feats, is_rch = feat_dict['edge']
#     if 'node' in feat_dict:
#         t_lch, t_rch = feat_dict['node']
#     if h_past is None:
#         return featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, lambda i: fn_all_ids(i)[:-2], cell, t_lch, t_rch, cell_node, wt_update, method, lv=lv)
#     elif h_bot is None:
#         return batch_tree_lstm2(h_buf, c_buf, h_past, c_past, lambda i: fn_all_ids(i)[2:], cell)
#     elif h_buf is None:
#         return featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_past, c_past, lambda i: fn_all_ids(i)[0, 1, 4, 5], cell, t_lch, t_rch, cell_node, wt_update, method, lv=lv)
#     else:
#         raise NotImplementedError  #TODO: handle model parallelism with features
# 
# 
# class FenwickTree(nn.Module):
#     def __init__(self, args):
#         super(FenwickTree, self).__init__()
#         self.method = args.method
#         self.has_edge_feats = args.has_edge_feats
#         self.has_node_feats = args.has_node_feats
#         self.init_h0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#         self.init_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#         glorot_uniform(self)
#         if self.has_node_feats:
#             self.node_feat_update = nn.LSTMCell(args.embed_dim, args.embed_dim)
#         self.merge_cell = BinaryTreeLSTMCell(args.embed_dim)
#         self.summary_cell = BinaryTreeLSTMCell(args.embed_dim)
#         if args.pos_enc:
#             self.pos_enc = PosEncoding(args.embed_dim, args.device, args.pos_base)
#         else:
#             self.pos_enc = lambda x: 0
#         
#         if self.method == "LSTM2":
#             self.merge_top_wt = BinaryTreeLSTMCell(args.embed_dim, args.embed_dim)
# 
#     def reset(self, list_states=[]):
#         self.list_states = []
#         for l in list_states:
#             t = []
#             for e in l:
#                 t.append(e)
#             self.list_states.append(t)
# 
#     def append_state(self, state, level):
#         if level >= len(self.list_states):
#             num_aug = level - len(self.list_states) + 1
#             for i in range(num_aug):
#                 self.list_states.append([])
#         self.list_states[level].append(state)
# 
#     def forward(self, new_state=None):
#         if new_state is None:
#             if len(self.list_states) == 0:
#                 return (self.init_h0, self.init_c0)
#         else:
#             self.append_state(new_state, 0)
#         pos = 0
#         while pos < len(self.list_states):
#             if len(self.list_states[pos]) >= 2:
#                 lch_state, rch_state = self.list_states[pos]  # assert the length is 2
#                 new_state = self.merge_cell(lch_state, rch_state)
#                 self.list_states[pos] = []
#                 self.append_state(new_state, pos + 1)
#             pos += 1
#         state = None
#         for pos in range(len(self.list_states)):
#             if len(self.list_states[pos]) == 0:
#                 continue
#             cur_state = self.list_states[pos][0]
#             if state is None:
#                 state = cur_state
#             else:
#                 state = self.summary_cell(state, cur_state)
#         return state
# 
#     def forward_train(self, h_bot, c_bot, h_buf0, c_buf0, prev_rowsum_h, prrev_rowsum_c, wt_update):
#         # embed row tree
#         tree_agg_ids = TreeLib.PrepareRowEmbed()
#         row_embeds = [(self.init_h0, self.init_c0)]
#         if self.has_edge_feats or self.has_node_feats:
#             feat_dict = c_bot
#             if 'node' in feat_dict:
#                 node_feats, is_tree_trivial, t_lch, t_rch = feat_dict['node']
#                 sel_feat = node_feats[is_tree_trivial]
#                 feat_dict['node'] = (sel_feat[t_lch], sel_feat[t_rch])
#             h_bot, c_bot = h_bot
#         if h_bot is not None:
#             row_embeds.append((h_bot, c_bot))
#         if prev_rowsum_h is not None:
#             row_embeds.append((prev_rowsum_h, prrev_rowsum_c))
#         if h_buf0 is not None:
#             row_embeds.append((h_buf0, c_buf0))
#         
# 
#         for i, all_ids in enumerate(tree_agg_ids):
#             fn_ids = lambda x: all_ids[x]
#             lstm_func = batch_tree_lstm3
#             if i == 0 and (self.has_edge_feats or self.has_node_feats):
#                 lstm_func = featured_batch_tree_lstm3
#             lstm_func = partial(lstm_func, h_buf=row_embeds[-1][0], c_buf=row_embeds[-1][1],
#                                 h_past=prev_rowsum_h, c_past=prrev_rowsum_c, fn_all_ids=fn_ids, cell=self.merge_cell)
#             if i == 0:
#                 if self.has_edge_feats or self.has_node_feats:
#                     new_states = lstm_func(feat_dict, h_bot, c_bot, cell_node=None if not self.has_node_feats else self.node_feat_update, wt_update=wt_update, method=self.method)
#                 else:
#                     new_states = lstm_func(h_bot, c_bot)
#             else:
#                 new_states = lstm_func(None, None)
#             row_embeds.append(new_states)
#         
#         h_list, c_list = zip(*row_embeds)
#         joint_h = torch.cat(h_list, dim=1)
#         joint_c = torch.cat(c_list, dim=1)
#         
#         # get history representation
#         init_select, all_ids, last_tos, next_ids, pos_info = TreeLib.PrepareRowSummary()
#         cur_state = (joint_h[:, init_select], joint_c[:, init_select])
#         
#         if self.has_node_feats:
#             base_nodes, _ = TreeLib.GetFenwickBase()
#             if len(base_nodes):
#                 needs_base_nodes = (init_select >= 1) & (init_select <= 2)
#                 sub_states = (cur_state[0][needs_base_nodes], cur_state[1][needs_base_nodes])
#                 sub_states = self.node_feat_update(node_feats[base_nodes], sub_states)
#                 nz_idx = torch.tensor(np.nonzero(needs_base_nodes)[0]).to(node_feats.device)
#                 new_cur = [scatter(x, nz_idx, dim=0, dim_size=init_select.shape[0]) for x in sub_states]
#                 needs_base_nodes = torch.tensor(needs_base_nodes, dtype=torch.bool).to(node_feats.device).unsqueeze(1)
#                 cur_state = [torch.where(needs_base_nodes, new_cur[i], cur_state[i]) for i in range(2)]
#                 cur_state = tuple(cur_state)
#         ret_state = (joint_h[:, next_ids], joint_c[:, next_ids])
#         hist_rnn_states = []
#         hist_froms = []
#         hist_tos = []
#         for i, (done_from, done_to, proceed_from, proceed_input) in enumerate(all_ids):
#             hist_froms.append(done_from)
#             hist_tos.append(done_to)
#             hist_rnn_states.append(cur_state)
# 
#             next_input = joint_h[:, proceed_input], joint_c[:, proceed_input]
#             sub_state = cur_state[0][:, proceed_from], cur_state[1][:, proceed_from]
#             
#             cur_state = self.summary_cell(sub_state, next_input)
#         hist_rnn_states.append(cur_state)
#         hist_froms.append(None)
#         hist_tos.append(last_tos)
#         hist_h_list, hist_c_list = zip(*hist_rnn_states)
#         pos_embed = self.pos_enc(pos_info)
#         row_h = multi_index_select(hist_froms, hist_tos, *hist_h_list) + pos_embed
#         row_c = multi_index_select(hist_froms, hist_tos, *hist_c_list) + pos_embed
#         return (row_h, row_c), ret_state
# 
# 
# class BitsRepNet(nn.Module):
#     def __init__(self, args):
#         super(BitsRepNet, self).__init__()
#         self.bits_compress = args.bits_compress
#         self.out_dim = args.embed_dim
#         assert self.out_dim >= self.bits_compress
#         self.device = args.device
# 
#     def forward(self, on_bits, n_cols):
#         h = torch.zeros(1, self.out_dim).to(self.device)
#         h[0, :n_cols] = -1.0
#         h[0, on_bits] = 1.0
# 
#         return h, h
# 
# 
# class RecurTreeGen(nn.Module):
# 
#     # to be customized
#     def embed_node_feats(self, node_feats):
#         raise NotImplementedError
# 
#     def embed_edge_feats(self, edge_feats):
#         raise NotImplementedError
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
#         raise NotImplementedError
# 
#     def predict_edge_feats(self, state, edge_feats=None):
#         """
#         Args:
#             state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
#             edge_feats: N x feat_dim or None
#         Returns:
#             likelihood of edge_feats under current state,
#             and, if edge_feats is None, then return the prediction of edge_feats
#             else return the edge_feats as it is
#         """
#         raise NotImplementedError
# 
#     def __init__(self, args):
#         super(RecurTreeGen, self).__init__()
# 
#         self.directed = args.directed
#         self.sigma = args.sigma
#         self.batch_size = args.batch_size
#         self.self_loop = args.self_loop
#         self.bits_compress = args.bits_compress
#         self.has_edge_feats = args.has_edge_feats
#         self.has_node_feats = args.has_node_feats
#         if self.has_edge_feats:
#             assert self.bits_compress == 0
#         self.greedy_frac = args.greedy_frac
#         self.share_param = args.share_param
#         if not self.bits_compress:
#             self.leaf_h0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             self.leaf_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             
#             if self.has_edge_feats and args.method in ["Test10"]:
#                 self.empty_h0 = None
#                 self.empty_c0 = None
#             
#             else:
#                 self.empty_h0 = Parameter(torch.Tensor(args.rnn_layers, 1,  args.embed_dim))
#                 self.empty_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
# 
#         self.topdown_left_embed = Parameter(torch.Tensor(2, args.embed_dim))
#         self.topdown_right_embed = Parameter(torch.Tensor(2, args.embed_dim))
#         glorot_uniform(self)
#         self.method = args.method
# 
#         if self.bits_compress > 0:
#             self.bit_rep_net = BitsRepNet(args)
# 
#         if self.share_param:
#             self.m_l2r_cell = BinaryTreeLSTMCell(args.embed_dim)
#             if self.method == "Test8":
#                 self.lr2p_cell = WeightedBinaryTreeLSTMCell(args.embed_dim, args.weight_embed_dim)
#             
#             else:
#                 self.lr2p_cell = BinaryTreeLSTMCell(args.embed_dim)
#             self.pred_has_ch = MLP(args.embed_dim, [2 * args.embed_dim, 1])
#             self.m_pred_has_left = MLP(args.embed_dim, [2 * args.embed_dim, 1])
#             self.m_pred_has_right = MLP(args.embed_dim, [2 * args.embed_dim, 1])
#             self.m_cell_topdown = MultiLSTMCell(args.embed_dim, args.embed_dim, args.rnn_layers)
#             self.m_cell_topright = MultiLSTMCell(args.embed_dim, args.embed_dim, args.rnn_layers)
#         else:
#             fn_pred = lambda: MLP(args.embed_dim, [2 * args.embed_dim, 1])
#             fn_tree_cell = lambda: BinaryTreeLSTMCell(args.embed_dim)
#             fn_lstm_cell = lambda: MultiLSTMCell(args.embed_dim, args.embed_dim, args.rnn_layers)
#             num_params = int(np.ceil(np.log2(args.max_num_nodes))) + 1
#             self.pred_has_ch = fn_pred()
# 
#             pred_modules = [[] for _ in range(2)]
#             tree_cell_modules = []
#             lstm_cell_modules = [[] for _ in range(2)]
#             for _ in range(num_params):
#                 for i in range(2):
#                     pred_modules[i].append(fn_pred())
#                     lstm_cell_modules[i].append(fn_lstm_cell())
#                 tree_cell_modules.append(fn_tree_cell())
# 
#             self.has_left_modules, self.has_right_modules = [nn.ModuleList(l) for l in pred_modules]
#             self.l2r_modules= nn.ModuleList(tree_cell_modules)
#             self.cell_topdown_modules, self.cell_topright_modules = [nn.ModuleList(l) for l in lstm_cell_modules]
#             self.lr2p_cell = fn_tree_cell()
#         self.row_tree = FenwickTree(args)
# 
#         if args.tree_pos_enc:
#             self.tree_pos_enc = PosEncoding(args.embed_dim, args.device, args.pos_base, bias=np.pi / 4)
#         else:
#             self.tree_pos_enc = lambda x: 0
# 
#     def cell_topdown(self, x, y, lv):
#         cell = self.m_cell_topdown if self.share_param else self.cell_topdown_modules[lv]
#         return cell(x, y)
# 
#     def cell_topright(self, x, y, lv):
#         cell = self.m_cell_topright if self.share_param else self.cell_topright_modules[lv]
#         return cell(x, y)
# 
#     def l2r_cell(self, x, y, lv=-1):
#         cell = self.m_l2r_cell if self.share_param else self.l2r_modules[lv]
#         return cell(x, y)
# 
#     def pred_has_left(self, x, lv):
#         mlp = self.m_pred_has_left if self.share_param else self.has_left_modules[lv]
#         return mlp(x)
# 
#     def pred_has_right(self, x, lv):
#         mlp = self.m_pred_has_right if self.share_param else self.has_right_modules[lv]
#         return mlp(x)
# 
#     def get_empty_state(self):
#         if self.bits_compress:
#             return self.bit_rep_net([], 1)
#         else:
# #             if self.method in ["Test9", "Test10", "Test11"]:
# #                if self.method != "Test10":
# #                    x_in = torch.cat([self.empty_embed, torch.zeros(1, self.weight_embed_dim).to(self.empty_embed.device)], dim = -1)
# #                    if self.method == "Test11":
# #                        self.leaf_LSTM(x_in, (self.test2_h0, self.test2_c0))
# #                
# #                else:
# #                    x_in = torch.cat([self.empty_embed, torch.zeros(1, 3 * self.empty_embed.shape[-1]).to(self.empty_embed.device)], dim = -1)
# #                
# #                return self.leaf_LSTM(x_in, (self.test_h0, self.test_c0))
# #                #self.empty_h0, self.empty_c0 = self.leaf_LSTM(x_in, (self.test_h0, self.test_c0))
#             return (self.empty_h0, self.empty_c0)
# 
#     def get_prob_fix(self, prob):
#         p = prob * (1 - self.greedy_frac)
#         if prob >= 0.5:
#             p += self.greedy_frac
#         return p
# 
#     def gen_row(self, ll, ll_wt, state, tree_node, col_sm, lb, ub, edge_feats=None, prev_wt_state=None, row=None):
#         assert lb <= ub
#         if tree_node.is_root:
#             prob_has_edge = torch.sigmoid(self.pred_has_ch(state[0][-1]))
# 
#             if col_sm.supervised:
#                 has_edge = len(col_sm.indices) > 0
#             else:
#                 has_edge = np.random.rand() < self.get_prob_fix(prob_has_edge.item())
#                 if ub == 0:
#                     has_edge = False
#                 if tree_node.n_cols <= 0:
#                     has_edge = False
#                 if lb:
#                     has_edge = True
#             if has_edge:
#                 ll = ll + torch.log(prob_has_edge)
#             else:
#                 ll = ll + torch.log(1 - prob_has_edge)
#             tree_node.has_edge = has_edge
#         else:
#             assert ub > 0
#             tree_node.has_edge = True
# 
#         if not tree_node.has_edge:  # an empty tree
#             return ll, ll_wt, self.get_empty_state(), 0, None, prev_wt_state
# 
#         if tree_node.is_leaf:
#             tree_node.bits_rep = [0]
#             col_sm.add_edge(tree_node.col_range[0])
#             if self.bits_compress:
#                 return ll, self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols), 1, None
#             else:
#                 if self.has_edge_feats:
#                     cur_feats = edge_feats[col_sm.pos - 1].unsqueeze(0) if col_sm.supervised else None
#                     #print(cur_feats)
#                     #print(prev_wt_state)
#                     rc = None
#                     if self.method == "Test10":
#                         col = tree_node.col_range[0]
#                         rc = np.array([col, row]).reshape(1, 1, 2)
#                     
#                     if self.method != "LSTM":
#                         edge_ll, _, cur_feats = self.predict_edge_feats(state, cur_feats)
#                     else:
#                         edge_ll, _, cur_feats = self.predict_edge_feats(state, cur_feats, prev_wt_state[0])
#                     
#                     ll_wt = ll_wt + edge_ll
#                     
#                     if self.method in ["Test", "Test2", "Test3"]:
#                         return ll, ll_wt,  (self.leaf_h0, self.leaf_c0), 1, cur_feats, prev_wt_state
#                     
#                     elif self.method in ["Test4", "Test5"]:
#                         prev_wt_state = self.embed_edge_feats(cur_feats, prev_state=prev_wt_state)
#                         return ll, ll_wt,  (self.leaf_h0, self.leaf_c0), 1, cur_feats, prev_wt_state
#                     
#                     elif self.method == "LSTM":
#                         edge_embed = self.embed_edge_feats(cur_feats, prev_state=prev_wt_state)
#                         prev_wt_state = edge_embed
#                         return ll, ll_wt, edge_embed, 1, cur_feats, prev_wt_state
#                     
#                     elif self.method == "MLP-2":
#                         #edge_embed = self.embed_edge_feats(cur_feats)
#                         return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, cur_feats, None
#                     
#                     elif self.method == "Test7":
#                         return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, cur_feats, None
#                     
#                     elif self.method == "Test8":
#                         return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, cur_feats, None
#                     
# #                     elif self.method == "Test6":
# #                         edge_embed = self.embed_edge_feats(cur_feats)
# #                         return ll, ll_wt, (self.leaf_h0 + edge_embed, self.leaf_c0 + edge_embed), 1, cur_feats, None
#                     
#                     edge_embed = self.embed_edge_feats(cur_feats, rc=rc)
#                     return ll, ll_wt, edge_embed, 1, cur_feats, None
#                     
#                 else:
#                     return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, None, None
#         else:
#             tree_node.split()
# 
#             mid = (tree_node.col_range[0] + tree_node.col_range[1]) // 2
#             left_prob = torch.sigmoid(self.pred_has_left(state[0][-1], tree_node.depth))
# 
#             if col_sm.supervised:
#                 has_left = col_sm.next_edge < mid
#             else:
#                 has_left = np.random.rand() < self.get_prob_fix(left_prob.item())
#                 if ub == 0:
#                     has_left = False
#                 if lb > tree_node.rch.n_cols:
#                     has_left = True
#             ll = ll + (torch.log(left_prob) if has_left else torch.log(1 - left_prob))
#             left_pos = self.tree_pos_enc([tree_node.lch.n_cols])
#             state = self.cell_topdown(self.topdown_left_embed[[int(has_left)]] + left_pos, state, tree_node.depth)
#             pred_edge_feats = []
#             if has_left:
#                 if self.has_edge_feats and self.method == "Test4":
#                     prev_wt_state = self.edgeLSTM(self.topdown_left_embed[[int(has_left)]], prev_wt_state)
#                 lub = min(tree_node.lch.n_cols, ub)
#                 llb = max(0, lb - tree_node.rch.n_cols)
#                 ll, ll_wt, left_state, num_left, left_edge_feats, prev_wt_state = self.gen_row(ll, ll_wt, state, tree_node.lch, col_sm, llb, lub, edge_feats, prev_wt_state, row=row)
#                 pred_edge_feats.append(left_edge_feats)
#             else:
#                 left_state = self.get_empty_state()
#                 num_left = 0
# 
#             right_pos = self.tree_pos_enc([tree_node.rch.n_cols])
#             topdown_state = self.l2r_cell(state, (left_state[0] + right_pos, left_state[1] + right_pos), tree_node.depth)
#             
#             if self.has_edge_feats and self.method in ["Test", "LSTM2", "Test2", "Test4", "Test5", "Test7", "Test8"] and tree_node.lch.is_leaf and has_left:
#                 if self.method in ["Test4", "Test5", "Test7", "Test8"]:
#                     left_edge_embed = self.standardize_edge_feats(left_edge_feats)
#                     left_edge_embed = self.edgelen_encoding(left_edge_feats)
#                 
# #                 elif self.method == "Test6":
# #                     ### Here we need to update the topdown state with an LSTM
# #                     continue
#                 
#                 else:
#                     left_edge_embed = self.embed_edge_feats(left_edge_feats, prev_state=prev_wt_state)
#                 
#                 if self.update_left:
#                     topdown_state = self.topdown_update_wt(left_edge_embed, topdown_state)
#                     #left_state = self.update_wt(left_edge_embed, left_state)
#                     
#                 else:
#                     topdown_state = self.update_wt(left_edge_embed, topdown_state)
#             
#             rlb = max(0, lb - num_left)
#             rub = min(tree_node.rch.n_cols, ub - num_left)
#             if not has_left:
#                 has_right = True
#             else:
#                 right_prob = torch.sigmoid(self.pred_has_right(topdown_state[0][-1], tree_node.depth))
#                 if col_sm.supervised:
#                     has_right = col_sm.has_edge(mid, tree_node.col_range[1])
#                 else:
#                     has_right = np.random.rand() < self.get_prob_fix(right_prob.item())
#                     if rub == 0:
#                         has_right = False
#                     if rlb:
#                         has_right = True
#                 ll = ll + (torch.log(right_prob) if has_right else torch.log(1 - right_prob))
# 
#             topdown_state = self.cell_topright(self.topdown_right_embed[[int(has_right)]], topdown_state, tree_node.depth)
# 
#             if has_right:  # has edge in right child
#                 if self.has_edge_feats and self.method == "Test4":
#                     prev_wt_state = self.edgeLSTM(self.topdown_right_embed[[int(has_right)]], prev_wt_state)
#                 ll, ll_wt, right_state, num_right, right_edge_feats, prev_wt_state = self.gen_row(ll, ll_wt, topdown_state, tree_node.rch, col_sm, rlb, rub, edge_feats, prev_wt_state, row=row)
#                 pred_edge_feats.append(right_edge_feats)
#             else:
#                 right_state = self.get_empty_state()
#                 num_right = 0
#             if tree_node.col_range[1] - tree_node.col_range[0] <= self.bits_compress:
#                 summary_state = self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols)
#             else:
#                 if self.method == "Test7":   
#                     left_edge_embed = None
#                     right_edge_embed = None  
#                                    
#                     if has_left and tree_node.lch.is_leaf:
#                         left_edge_embed = self.embed_edge_feats(left_edge_feats)
#                     
#                     if has_right and tree_node.rch.is_leaf:
#                         right_edge_embed = self.embed_edge_feats(right_edge_feats)
#                      
#                     summary_state = self.lr2p_cell(left_state, right_state, left_edge_embed, right_edge_embed)
#                 
#                 elif self.method == "Test8":
#                     left_edge_embed = None
#                     right_edge_embed = None
#                     
#                     if has_left and tree_node.lch.is_leaf:
#                         left_edge_embed = self.embed_edge_feats(left_edge_feats)
#                         
#                     if has_right and tree_node.rch.is_leaf:
#                         right_edge_embed = self.embed_edge_feats(right_edge_feats)
#                     
#                     summary_state = self.lr2p_cell(left_state, right_state, left_edge_embed, right_edge_embed)
#                 
#                 else:
#                     summary_state = self.lr2p_cell(left_state, right_state)
#                 
#             if self.has_edge_feats:
#                 edge_feats = torch.cat(pred_edge_feats, dim=0)
#                 if self.method == "Test":
#                     if has_left and tree_node.lch.is_leaf:
#                         left_edge_embed = self.embed_edge_feats(left_edge_feats, prev_state=prev_wt_state)
#                         summary_state = self.update_wt(left_edge_embed, summary_state)
#                     
#                     if has_right and tree_node.rch.is_leaf:
#                         right_edge_embed = self.embed_edge_feats(right_edge_feats, prev_state=prev_wt_state)
#                         summary_state = self.update_wt(right_edge_embed, summary_state)
#             
#             return ll, ll_wt, summary_state, num_left + num_right, edge_feats, prev_wt_state
# 
#     def forward(self, node_end, edge_list=None, node_feats=None, edge_feats=None, node_start=0, list_states=[], lb_list=None, ub_list=None, col_range=None, num_nodes=None, display=False):
#         pos = 0
#         total_ll = 0.0
#         total_ll_wt = 0.0
#         edges = []
#         self.row_tree.reset(list_states)
#         controller_state = self.row_tree()
#         if num_nodes is None:
#             num_nodes = node_end
#         pbar = range(node_start, node_end)
#         if display:
#             pbar = tqdm(pbar)
#         list_pred_node_feats = []
#         list_pred_edge_feats = []
#         
#         prev_wt_state = None
#         if self.has_edge_feats and self.method in ["LSTM", "Test4", "Test5"]:
#             prev_wt_state = (self.leaf_h0_wt, self.leaf_c0_wt)
#         
#         for i in pbar:
#             if edge_list is None:
#                 col_sm = ColAutomata(supervised=False)
#             else:
#                 indices = []
#                 while pos < len(edge_list) and i == edge_list[pos][0]:
#                     indices.append(edge_list[pos][1])
#                     pos += 1
#                 indices.sort()
#                 col_sm = ColAutomata(supervised=True, indices=indices)
# 
#             cur_row = AdjRow(i, self.directed, self.self_loop, col_range=col_range)
#             lb = 0 if lb_list is None else lb_list[i]
#             ub = cur_row.root.n_cols if ub_list is None else ub_list[i]
#             cur_pos_embed = self.row_tree.pos_enc([num_nodes - i])
#             controller_state = [x + cur_pos_embed for x in controller_state]
#             if self.has_node_feats:
#                 target_node_feats = None if node_feats is None else node_feats[[i]]
#                 controller_state, ll_node, target_node_feats = self.predict_node_feats(controller_state, target_node_feats)
#                 total_ll = total_ll + ll_node
#                 list_pred_node_feats.append(target_node_feats)
#             if self.has_edge_feats:
#                 target_edge_feats = None if edge_feats is None else edge_feats[len(edges) : len(edges) + len(col_sm)]
#             else:
#                 target_edge_feats = None
#             #print(prev_wt_state)
#             
#             if self.has_edge_feats and self.method in ["Test4", "Test5"]:
#                 prev_wt_state = (self.leaf_h0_wt, self.leaf_c0_wt)
#             
#             ll, ll_wt, cur_state, _, target_edge_feats, prev_wt_state = self.gen_row(0, 0, controller_state, cur_row.root, col_sm, lb, ub, target_edge_feats, prev_wt_state, row=i)
#             
#             if target_edge_feats is not None and target_edge_feats.shape[0]:
#                 list_pred_edge_feats.append(target_edge_feats)
#             if self.has_node_feats:
#                 target_feat_embed = self.embed_node_feats(target_node_feats)
#                 cur_state = self.row_tree.node_feat_update(target_feat_embed, cur_state)
#             assert lb <= len(col_sm.indices) <= ub
#             
#             if self.has_edge_feats and self.method == "LSTM2":
#                 cur_state = self.merge_top_wt(cur_state, prev_wt_state)
#             
#             if self.has_edge_feats and self.method == "Test8" and i == 1 and target_edge_feats is not None and target_edge_feats.shape[0]:
#                 left_edge_embed = self.embed_edge_feats(target_edge_feats)
#                 cur_state = self.update_wt(left_edge_embed, cur_state)
#             
#             controller_state = self.row_tree(cur_state)
#             
#             if self.has_edge_feats and self.method == "Test" and cur_row.root.is_leaf and target_edge_feats is not None:
#                 edge_embed = self.embed_edge_feats(target_edge_feats, prev_state=prev_wt_state)
#                 controller_state = self.update_wt(edge_embed, controller_state)
#                 self.row_tree.list_states[1] = [controller_state]
#             
#             edges += [(i, x) for x in col_sm.indices]
#             total_ll = total_ll + ll
#             total_ll_wt = total_ll_wt + ll_wt
# 
#         if self.has_node_feats:
#             node_feats = torch.cat(list_pred_node_feats, dim=0)
#         if self.has_edge_feats:
#             edge_feats = torch.cat(list_pred_edge_feats, dim=0)
#         return total_ll, total_ll_wt, edges, self.row_tree.list_states, node_feats, edge_feats
# 
#     def binary_ll(self, pred_logits, np_label, need_label=False, reduction='sum', batch_idx=None, ll_batch=None):
#         pred_logits = pred_logits.view(-1, 1)
#         label = torch.tensor(np_label, dtype=torch.float32).to(pred_logits.device).view(-1, 1)
#         loss = F.binary_cross_entropy_with_logits(pred_logits, label, reduction=reduction)
#         
#         ind_loss = F.binary_cross_entropy_with_logits(pred_logits, label, reduction='none')
#         
#         if batch_idx is not None:
#             i = 0
#             for B in np.unique(batch_idx):
#                 ll_batch[i] = ll_batch[i] - torch.sum(ind_loss[batch_idx == B])
#                 i = i + 1
#         
#         if need_label:
#             return -loss, label, ll_batch
#         return -loss, ll_batch
# 
#     def forward_row_trees(self, graph_ids, node_feats=None, edge_feats=None, list_node_starts=None, num_nodes=-1, list_col_ranges=None):
#         TreeLib.PrepareMiniBatch(graph_ids, list_node_starts, num_nodes, list_col_ranges)
#         # embed trees
#         all_ids = TreeLib.PrepareTreeEmbed()
#         if self.has_node_feats:
#             node_feats = self.embed_node_feats(node_feats)
# 
#         if not self.bits_compress:
#             empty_h0, empty_c0 = self.get_empty_state()
#             h_bot = torch.cat([empty_h0, self.leaf_h0], dim=1)
#             c_bot = torch.cat([empty_c0, self.leaf_c0], dim=1)
#             
#             fn_hc_bot = lambda d: (h_bot, c_bot)
#         else:
#             binary_embeds, base_feat = TreeLib.PrepareBinary()
#             fn_hc_bot = lambda d: (binary_embeds[d], binary_embeds[d]) if d < len(binary_embeds) else base_feat
#         max_level = len(all_ids) - 1
#         h_buf_list = [None] * (len(all_ids) + 1)
#         c_buf_list = [None] * (len(all_ids) + 1)
#         
#         for d in range(len(all_ids) - 1, -1, -1):
#             fn_ids = lambda i: all_ids[d][i]
#             if d == max_level:
#                 h_buf = c_buf = None
#             else:
#                 h_buf = h_buf_list[d + 1]
#                 c_buf = c_buf_list[d + 1]
#             h_bot, c_bot = fn_hc_bot(d + 1)
#             if self.has_edge_feats:
#                 edge_idx, is_rch = TreeLib.GetEdgeAndLR(d + 1)
#                 if self.method in ["Test", "Test2", "Test3", "Test8"]:
#                     local_edge_feats = edge_feats[edge_idx]
#                 else:
#                     local_edge_feats = (edge_feats[0][:, edge_idx], edge_feats[1][:, edge_idx])
#                 
#                 new_h, new_c = featured_batch_tree_lstm2(local_edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell, wt_update = self.update_wt, method = self.method)
#             else:
#                 new_h, new_c = batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell)
#             if self.method in ["Test4", "Test5"] and d == 0:
#                 b = self.batch_size
#                 m = edge_feats[0].shape[1] // b
#                 idx = ([False] + [True]*(m-1))*b
#                 idx = np.array(idx)
#                 edge_embed_cur = (edge_feats[0][:, idx], edge_feats[1][:, idx])
#                 new_h, new_c = self.merge_top_wt((new_h, new_c), edge_embed_cur)
#             
#             #elif self.method == "Test5" and d == 0:
#             #    edge_embed_cur = (edge_feats[0][idx], edge_feats[1][idx])
#             #    new_h, new_c = self.merge_top_wt((new_h, new_c), edge_embed_cur)
#             
#             h_buf_list[d] = new_h
#             c_buf_list[d] = new_c
#         hc_bot = fn_hc_bot(0)
#         feat_dict = {}
#         if self.has_edge_feats:
#             edge_idx, is_rch = TreeLib.GetEdgeAndLR(0)
#             if self.method in ["Test", "Test2", "Test3", "Test8"]:
#                 local_edge_feats = edge_feats[edge_idx]
#                 K = local_edge_feats.shape[0]
#                 local_edge_feats = self.update_wt(local_edge_feats, (self.leaf_h0.repeat(1, K, 1), self.leaf_c0.repeat(1, K, 1)))
#             elif self.method in ["Test4", "Test5"]:
#                 local_edge_feats = (edge_feats[0][:, edge_idx], edge_feats[1][:, edge_idx])
#                 init_state = (self.leaf_h0.repeat(1, len(edge_idx), 1), self.leaf_c0.repeat(1, len(edge_idx), 1))
#                 local_edge_feats = self.merge_top_wt(init_state, local_edge_feats)
#             else:
#                 local_edge_feats = (edge_feats[0][:, edge_idx], edge_feats[1][:, edge_idx])
#             feat_dict['edge'] = (local_edge_feats, is_rch)
#         if self.has_node_feats:
#             is_tree_trivial = TreeLib.GetIsTreeTrivial()
#             new_h, new_c = self.row_tree.node_feat_update(node_feats[~is_tree_trivial], (new_h, new_c))
#             h_buf_list[0] = new_h
#             c_buf_list[0] = new_c
#             t_lch, t_rch = TreeLib.GetTrivialNodes()
#             feat_dict['node'] = (node_feats, is_tree_trivial, t_lch, t_rch)
#         if len(feat_dict):
#             hc_bot = (hc_bot, feat_dict)
#         
#         return hc_bot, fn_hc_bot, h_buf_list, c_buf_list
# 
#     def forward_row_summaries(self, graph_ids, node_feats=None, edge_feats=None,
#                              list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None):
#         hc_bot, _, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, node_feats, edge_feats,
#                                                                    list_node_starts, num_nodes, list_col_ranges)
#         row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states)
#         return row_states, next_states
# 
#     def forward_train(self, graph_ids, node_feats=None, edge_feats=None,
#                       list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None, batch_idx=None):
#         ll = 0.0
#         ll_wt = 0.0
#         noise = 0.0
#         ll_batch = (None if batch_idx is None else np.zeros(len(np.unique(batch_idx))))
#         ll_batch_wt = (None if batch_idx is None else np.zeros(len(np.unique(batch_idx))))
#         edge_feats_embed = None
#         
#         if self.has_edge_feats:
#             rc = None
#             if self.method == "Test10":
#                 edge_feats, rc = edge_feats
#             if self.method == "LSTM":
#                 edge_feats_embed, state_h_prior = self.embed_edge_feats(edge_feats, noise)
#                 edge_feats = torch.cat(edge_feats, dim = 0)
#             
#             elif self.method == "Test4":
#                 edge_feats, lr = edge_feats
#                 edge_feats_embed, weights_MLP = self.embed_edge_feats(edge_feats, lr_seq=lr)
#             
#             elif self.method == "Test5":
#                 edge_feats_embed, weights_MLP = self.embed_edge_feats(edge_feats)
#             
#             else:
#                 edge_feats_embed = self.embed_edge_feats(edge_feats, rc=rc)
#         
#         hc_bot, fn_hc_bot, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, node_feats, edge_feats_embed,
#                                                                            list_node_starts, num_nodes, list_col_ranges)
#         row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states, wt_update=self.update_wt)
#         if self.has_node_feats:
#             row_states, ll_node_feats, _ = self.predict_node_feats(row_states, node_feats)
#             ll = ll + ll_node_feats
#         logit_has_edge = self.pred_has_ch(row_states[0][-1])
#         has_ch, _ = TreeLib.GetChLabel(0, dtype=bool)
#         ll_cur, ll_batch = self.binary_ll(logit_has_edge, has_ch, batch_idx = batch_idx, ll_batch = ll_batch)
#         ll = ll + ll_cur
#         
#         cur_states = (row_states[0][:, has_ch], row_states[1][:, has_ch])
#         
#         if batch_idx is not None:
#             batch_idx = batch_idx[has_ch]
# 
#         lv = 0
#         while True:
#             is_nonleaf = TreeLib.QueryNonLeaf(lv)
#             if self.has_edge_feats:
#                 edge_of_lv = TreeLib.GetEdgeOf(lv)
#                 edge_state = (cur_states[0][:, ~is_nonleaf], cur_states[1][:, ~is_nonleaf])
#                 cur_batch_idx = (None if batch_idx is None else batch_idx[~is_nonleaf])
#                 
#                 target_feats = edge_feats[edge_of_lv]
#                 prior_h_target = None
#                 if self.method == "LSTM": 
#                     prior_h_target = state_h_prior[edge_of_lv]
#                 edge_ll, ll_batch_wt, _ = self.predict_edge_feats(edge_state, target_feats, prior_h_target, batch_idx = cur_batch_idx, ll_batch_wt = ll_batch_wt)
#                 ll_wt = ll_wt + edge_ll
#             if is_nonleaf is None or np.sum(is_nonleaf) == 0:
#                 break
#             cur_states = (cur_states[0][:, is_nonleaf], cur_states[1][:, is_nonleaf])
#             
#             if batch_idx is not None:
#                 batch_idx = batch_idx[is_nonleaf]
#             
#             left_logits = self.pred_has_left(cur_states[0][-1], lv)
#             has_left, num_left = TreeLib.GetChLabel(-1, lv)
#             left_update = self.topdown_left_embed[has_left] + self.tree_pos_enc(num_left)
#             left_ll, float_has_left, ll_batch = self.binary_ll(left_logits, has_left, need_label=True, reduction='sum', batch_idx = batch_idx, ll_batch = ll_batch)
#             ll = ll + left_ll
# 
#             cur_states = self.cell_topdown(left_update, cur_states, lv)
# 
#             left_ids = TreeLib.GetLeftRootStates(lv)
#             h_bot, c_bot = fn_hc_bot(lv + 1)
#             if lv + 1 < len(h_buf_list):
#                 h_next_buf, c_next_buf = h_buf_list[lv + 1], c_buf_list[lv + 1]
#             else:
#                 h_next_buf = c_next_buf = None
#             if self.has_edge_feats:
#                 edge_idx, is_rch = TreeLib.GetEdgeAndLR(lv + 1)
#                 if self.method in ["Test", "Test2", "Test3", "Test8"]: 
#                     left_feats = edge_feats_embed[edge_idx[~is_rch]]
#                 
#                 elif self.method in ["Test4", "Test5"]:
#                     left_feats = weights_MLP[edge_idx[~is_rch]]
#                     
#                 else:
#                     left_feats = (edge_feats_embed[0][:, edge_idx[~is_rch]], edge_feats_embed[1][:, edge_idx[~is_rch]])
#                 
#                 h_bot, c_bot = h_bot[:, left_ids[0]], c_bot[:, left_ids[0]]
#                 
#                 if self.method not in ["Test", "Test2", "Test3", "Test4", "Test5", "Test8"]:
#                     h_bot, c_bot = selective_update_hc(h_bot, c_bot, left_ids[0], left_feats)
#                 left_wt_ids = left_ids[1][list(map(bool, left_ids[0]))]
#                 left_ids = tuple([None] + list(left_ids[1:]))
# 
#             left_subtree_states = tree_state_select(h_bot, c_bot,
#                                                     h_next_buf, c_next_buf,
#                                                     lambda: left_ids)
# 
#             has_right, num_right = TreeLib.GetChLabel(1, lv)
#             right_pos = self.tree_pos_enc(num_right)
#             left_subtree_states = [x + right_pos for x in left_subtree_states]
#             topdown_state = self.l2r_cell(cur_states, left_subtree_states, lv)
#             
#             if self.has_edge_feats and self.method in ["Test", "LSTM2", "Test2", "Test4", "Test5", "Test8"] and len(left_wt_ids) > 0:
#                 leaf_topdown_states = (topdown_state[0][:, left_wt_ids], topdown_state[1][:, left_wt_ids])
#                 
#                 if self.update_left:
#                     leaf_topdown_states = self.topdown_update_wt(left_feats, leaf_topdown_states)
#                 
#                 else:
#                     leaf_topdown_states = self.update_wt(left_feats, leaf_topdown_states)
#                 topdown_state[0][:, left_wt_ids] = leaf_topdown_states[0]
#                 topdown_state[1][:, left_wt_ids] = leaf_topdown_states[1]
#             
#             right_logits = self.pred_has_right(topdown_state[0][-1], lv)
#             right_update = self.topdown_right_embed[has_right]
#             topdown_state = self.cell_topright(right_update, topdown_state, lv)
#             right_ll, _ = self.binary_ll(right_logits, has_right, reduction='none')
#             right_ll = right_ll * float_has_left
#             ll = ll + torch.sum(right_ll)
#             
#             if batch_idx is not None:
#                 i = 0
#                 for B in np.unique(batch_idx):
#                     ll_batch[i] = ll_batch[i] + torch.sum(right_ll[batch_idx == B])
#             
#             lr_ids = TreeLib.GetLeftRightSelect(lv, np.sum(has_left), np.sum(has_right))
#             new_states = []
#             for i in range(2):
#                 new_s = multi_index_select([lr_ids[0], lr_ids[2]], [lr_ids[1], lr_ids[3]],
#                                             cur_states[i], topdown_state[i])
#                 new_states.append(new_s)
#             cur_states = tuple(new_states)
#             lv += 1
#         return ll, ll_wt, ll_batch, ll_batch_wt, next_states

# 
# # coding=utf-8
# # Copyright 2024 The Google Research Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# 
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# # pylint: skip-file
# 
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_scatter import scatter
# from collections import defaultdict
# from torch.nn.parameter import Parameter
# from bigg.common.pytorch_util import glorot_uniform, MLP, BinaryTreeLSTMCell
# from tqdm import tqdm
# from bigg.model.util import AdjNode, ColAutomata, AdjRow
# from bigg.model.tree_clib.tree_lib import TreeLib
# from bigg.torch_ops import multi_index_select, PosEncoding
# from functools import partial
# 
# # def f(idx_list, y):    
# #     if len(idx_list) == 2:
# #         if y == idx_list[0]:
# #             return 'L'
# #         else:
# #             return 'R'
# #     
# #     else:
# #         midpoint = len(idx_list) // 2
# #         left_idx_list = idx_list[:midpoint]
# #                 
# #         if y in left_idx_list:
# #             if len(left_idx_list) == 1:
# #                 return 'L'
# #             return 'L' + f(left_idx_list, y)
# #         
# #         else:
# #             right_idx_list = idx_list[midpoint:]
# #             return 'R' + f(right_idx_list, y)
# # 
# # def get_lr_seq(row, col):
# #     assert col < row
# #     idx_list = list(range(row))
# #     return f(idx_list, col)
# 
# 
# 
# 
# 
# 
# def hc_multi_select(ids_from, ids_to, h_froms, c_froms):
#     h_vecs = multi_index_select(ids_from,
#                                 ids_to,
#                                 *h_froms)
#     c_vecs = multi_index_select(ids_from,
#                                 ids_to,
#                                 *c_froms)
#     return h_vecs, c_vecs
# 
# 
# def tree_state_select(h_bot, c_bot, h_buf, c_buf, fn_all_ids):
#     bot_froms, bot_tos, prev_froms, prev_tos = fn_all_ids()
#     if h_buf is None or prev_tos is None:
#         h_vecs = multi_index_select([bot_froms], [bot_tos], h_bot)
#         c_vecs = multi_index_select([bot_froms], [bot_tos], c_bot)
#     elif h_bot is None or bot_tos is None:
#         h_vecs = multi_index_select([prev_froms], [prev_tos], h_buf)
#         c_vecs = multi_index_select([prev_froms], [prev_tos], c_buf)
#     else:
#         h_vecs, c_vecs = hc_multi_select([bot_froms, prev_froms],
#                                          [bot_tos, prev_tos],
#                                          [h_bot, h_buf], [c_bot, c_buf])
#     return h_vecs, c_vecs
# 
# 
# def batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_all_ids, cell):
#     h_list = []
#     c_list = []
#     for i in range(2):
#         h_vecs, c_vecs = tree_state_select(h_bot, c_bot, h_buf, c_buf, lambda : fn_all_ids(i))
#         h_list.append(h_vecs)
#         c_list.append(c_vecs)
#     return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))
# 
# def selective_update_hc(h, c, zero_one, feats):
#     nz_idx = torch.tensor(np.nonzero(zero_one)[0]).to(h.device)
#     local_edge_feats_h = scatter(feats[0], nz_idx, dim=0, dim_size=h.shape[0])
#     local_edge_feats_c = scatter(feats[1], nz_idx, dim=0, dim_size=h.shape[0])
#     zero_one = torch.tensor(zero_one, dtype=torch.bool).to(h.device).unsqueeze(1)
#     h = torch.where(zero_one, local_edge_feats_h, h)
#     c = torch.where(zero_one, local_edge_feats_c, c)
#     return h, c
# 
# def featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_all_ids, cell, t_lch=None, t_rch=None, cell_node=None, wt_update=None, method=None, lv=-1):
#     new_ids = [list(fn_all_ids(0)), list(fn_all_ids(1))]
#     lch_isleaf, rch_isleaf = new_ids[0][0], new_ids[1][0]
#     new_ids[0][0] = new_ids[1][0] = None
#     is_leaf = [lch_isleaf, rch_isleaf]
#     
#     if edge_feats is not None:
#         if method in ["Test", "Test2", "Test3"]:
#             edge_feats = [edge_feats[~is_rch], edge_feats[is_rch]]
#         
#         else:
#             edge_feats = [(edge_feats[0][~is_rch], edge_feats[1][~is_rch]), (edge_feats[0][is_rch], edge_feats[1][is_rch])]
#         assert np.sum(is_rch) == np.sum(rch_isleaf)
#     node_feats = [t_lch, t_rch]
#     h_list = []
#     c_list = []
#     
#     for i in range(2):
#         leaf_check = is_leaf[i]
#         local_hbot, local_cbot = h_bot[leaf_check], c_bot[leaf_check]
#         if edge_feats is not None and method not in ["Test", "Test2", "Test3"]:
#             if method != "Test4" or lv == 0:
#                 local_hbot, local_cbot = selective_update_hc(local_hbot, local_cbot, leaf_check, edge_feats[i])
#         if cell_node is not None:
#             local_hbot, local_cbot = cell_node(node_feats[i], (local_hbot, local_cbot))
#         
#         h_vecs, c_vecs = tree_state_select(local_hbot, local_cbot, h_buf, c_buf, lambda : new_ids[i])
#         h_list.append(h_vecs)
#         c_list.append(c_vecs)
#     
#     summary_state = cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))
#     
#     if method != "Test" or edge_feats is None:
#         return summary_state
#     
#     for i in range(2):
#         leaf_check = list(map(bool, is_leaf[i]))
#         local_idx = new_ids[i][1][leaf_check]
#         local_hbot, local_cbot = summary_state[0][local_idx], summary_state[1][local_idx]
#         cur_summary = (local_hbot, local_cbot)
#         cur_edge_feats = edge_feats[i]
#         cur_summary = wt_update(cur_edge_feats, cur_summary)
#         summary_state[0][local_idx] = cur_summary[0]
#         summary_state[1][local_idx] = cur_summary[1]
#         
#     return summary_state
# 
# 
# 
# # def featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_all_ids, cell, t_lch=None, t_rch=None, cell_node=None):
# #     new_ids = [list(fn_all_ids(0)), list(fn_all_ids(1))]
# #     lch_isleaf, rch_isleaf = new_ids[0][0], new_ids[1][0]
# #     new_ids[0][0] = new_ids[1][0] = None
# #     is_leaf = [lch_isleaf, rch_isleaf]
# #     if edge_feats is not None:
# #         #edge_feats = [edge_feats[~is_rch], edge_feats[is_rch]]
# #         edge_feats = [(edge_feats[0][~is_rch], edge_feats[1][~is_rch]), (edge_feats[0][is_rch], edge_feats[1][is_rch])]
# #         assert np.sum(is_rch) == np.sum(rch_isleaf)
# #     node_feats = [t_lch, t_rch]
# #     h_list = []
# #     c_list = []
# #     for i in range(2):
# #         leaf_check = is_leaf[i]
# #         local_hbot, local_cbot = h_bot[leaf_check], c_bot[leaf_check]
# #         if edge_feats is not None:
# #             local_hbot, local_cbot = selective_update_hc(local_hbot, local_cbot, leaf_check, edge_feats[i])
# #         if cell_node is not None:
# #             local_hbot, local_cbot = cell_node(node_feats[i], (local_hbot, local_cbot))
# #         h_vecs, c_vecs = tree_state_select(local_hbot, local_cbot, h_buf, c_buf, lambda : new_ids[i])
# #         h_list.append(h_vecs)
# #         c_list.append(c_vecs)
# #     return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))
# 
# 
# def batch_tree_lstm3(h_bot, c_bot, h_buf, c_buf, h_past, c_past, fn_all_ids, cell):
#     if h_past is None:
#         return batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, lambda i: fn_all_ids(i)[:-2], cell)
#     elif h_bot is None:
#         return batch_tree_lstm2(h_buf, c_buf, h_past, c_past, lambda i: fn_all_ids(i)[2:], cell)
#     elif h_buf is None:
#         return batch_tree_lstm2(h_bot, c_bot, h_past, c_past, lambda i: fn_all_ids(i)[0, 1, 4, 5], cell)
#     else:
#         h_list = []
#         c_list = []
#         for i in range(2):
#             bot_froms, bot_tos, prev_froms, prev_tos, past_froms, past_tos = fn_all_ids(i)
#             h_vecs, c_vecs = hc_multi_select([bot_froms, prev_froms, past_froms],
#                                              [bot_tos, prev_tos, past_tos],
#                                              [h_bot, h_buf, h_past],
#                                              [c_bot, c_buf, c_past])
#             h_list.append(h_vecs)
#             c_list.append(c_vecs)
#         return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))
# 
# 
# def featured_batch_tree_lstm3(feat_dict, h_bot, c_bot, h_buf, c_buf, h_past, c_past, fn_all_ids, cell, cell_node, wt_update, method):
#     edge_feats = is_rch = None
#     t_lch = t_rch = None
#     if method == "Test4":
#         lv = 0
#     else:
#         lv = -1
#     if 'edge' in feat_dict:
#         edge_feats, is_rch = feat_dict['edge']
#     if 'node' in feat_dict:
#         t_lch, t_rch = feat_dict['node']
#     if h_past is None:
#         return featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, lambda i: fn_all_ids(i)[:-2], cell, t_lch, t_rch, cell_node, wt_update, method, lv=lv)
#     elif h_bot is None:
#         return batch_tree_lstm2(h_buf, c_buf, h_past, c_past, lambda i: fn_all_ids(i)[2:], cell)
#     elif h_buf is None:
#         return featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_past, c_past, lambda i: fn_all_ids(i)[0, 1, 4, 5], cell, t_lch, t_rch, cell_node, wt_update, method, lv=lv)
#     else:
#         raise NotImplementedError  #TODO: handle model parallelism with features
# 
# 
# class FenwickTree(nn.Module):
#     def __init__(self, args):
#         super(FenwickTree, self).__init__()
#         self.method = args.method
#         self.has_edge_feats = args.has_edge_feats
#         self.has_node_feats = args.has_node_feats
#         self.init_h0 = Parameter(torch.Tensor(1, args.embed_dim))
#         self.init_c0 = Parameter(torch.Tensor(1, args.embed_dim))
#         glorot_uniform(self)
#         if self.has_node_feats:
#             self.node_feat_update = nn.LSTMCell(args.embed_dim, args.embed_dim)
#         self.merge_cell = BinaryTreeLSTMCell(args.embed_dim)
#         self.summary_cell = BinaryTreeLSTMCell(args.embed_dim)
#         if args.pos_enc:
#             self.pos_enc = PosEncoding(args.embed_dim, args.device, args.pos_base)
#         else:
#             self.pos_enc = lambda x: 0
#         
#         if self.method == "LSTM2":
#             self.merge_top_wt = BinaryTreeLSTMCell(args.embed_dim, args.embed_dim)
# 
#     def reset(self, list_states=[]):
#         self.list_states = []
#         for l in list_states:
#             t = []
#             for e in l:
#                 t.append(e)
#             self.list_states.append(t)
# 
#     def append_state(self, state, level):
#         if level >= len(self.list_states):
#             num_aug = level - len(self.list_states) + 1
#             for i in range(num_aug):
#                 self.list_states.append([])
#         self.list_states[level].append(state)
# 
#     def forward(self, new_state=None):
#         if new_state is None:
#             if len(self.list_states) == 0:
#                 return (self.init_h0, self.init_c0)
#         else:
#             self.append_state(new_state, 0)
#         pos = 0
#         while pos < len(self.list_states):
#             if len(self.list_states[pos]) >= 2:
#                 lch_state, rch_state = self.list_states[pos]  # assert the length is 2
#                 new_state = self.merge_cell(lch_state, rch_state)
#                 self.list_states[pos] = []
#                 self.append_state(new_state, pos + 1)
#             pos += 1
#         state = None
#         for pos in range(len(self.list_states)):
#             if len(self.list_states[pos]) == 0:
#                 continue
#             cur_state = self.list_states[pos][0]
#             if state is None:
#                 state = cur_state
#             else:
#                 state = self.summary_cell(state, cur_state)
#         return state
# 
#     def forward_train(self, h_bot, c_bot, h_buf0, c_buf0, prev_rowsum_h, prrev_rowsum_c, wt_update):
#         # embed row tree
#         tree_agg_ids = TreeLib.PrepareRowEmbed()
#         row_embeds = [(self.init_h0, self.init_c0)]
# #         print(h_bot)
# #         print(c_bot)
# #         print(h_buf0)
#         if self.has_edge_feats or self.has_node_feats:
#             feat_dict = c_bot
#             if 'node' in feat_dict:
#                 node_feats, is_tree_trivial, t_lch, t_rch = feat_dict['node']
#                 sel_feat = node_feats[is_tree_trivial]
#                 feat_dict['node'] = (sel_feat[t_lch], sel_feat[t_rch])
#             h_bot, c_bot = h_bot
#         if h_bot is not None:
#             row_embeds.append((h_bot, c_bot))
#         if prev_rowsum_h is not None:
#             row_embeds.append((prev_rowsum_h, prrev_rowsum_c))
#         if h_buf0 is not None:
#             row_embeds.append((h_buf0, c_buf0))
#         
# 
#         for i, all_ids in enumerate(tree_agg_ids):
#             fn_ids = lambda x: all_ids[x]
#             lstm_func = batch_tree_lstm3
#             if i == 0 and (self.has_edge_feats or self.has_node_feats):
#                 lstm_func = featured_batch_tree_lstm3
#             lstm_func = partial(lstm_func, h_buf=row_embeds[-1][0], c_buf=row_embeds[-1][1],
#                                 h_past=prev_rowsum_h, c_past=prrev_rowsum_c, fn_all_ids=fn_ids, cell=self.merge_cell)
#             if i == 0:
#                 if self.has_edge_feats or self.has_node_feats:
#                     new_states = lstm_func(feat_dict, h_bot, c_bot, cell_node=None if not self.has_node_feats else self.node_feat_update, wt_update=wt_update, method=self.method)
#                 else:
#                     new_states = lstm_func(h_bot, c_bot)
#             else:
#                 new_states = lstm_func(None, None)
#             row_embeds.append(new_states)
# #             print("========================")
# #             print("i: ", i)
# #             print("New States: ", new_states)
# #             print("========================")
# #         
# #         for r in row_embeds:
# #             print("+++++++++++++++++++++")
# #             print("i: ", i)
# #             print(r)
# #             print("+++++++++++++++++++++")
#         
#         
#         h_list, c_list = zip(*row_embeds)
#         joint_h = torch.cat(h_list, dim=0)
#         joint_c = torch.cat(c_list, dim=0)
# #         print(joint_h)
# #         print(joint_c)
# #         if self.method == "Test4":
# #             num_nodes = None
# #             joint_h = joint_h[1:n+1, :]
# #             joint_c = joint_c[1:n+1, :]
#         # get history representation
#         init_select, all_ids, last_tos, next_ids, pos_info = TreeLib.PrepareRowSummary()
#         #print("init select: ", init_select)
#         #print("all ids: ", all_ids)
#         #print("last tos: ", last_tos)
#         #print("next ids: ", next_ids)
#         #print("pos_info: ", pos_info)
#         cur_state = (joint_h[init_select], joint_c[init_select])
#         #print(cur_state)
#         if self.has_node_feats:
#             base_nodes, _ = TreeLib.GetFenwickBase()
#             if len(base_nodes):
#                 needs_base_nodes = (init_select >= 1) & (init_select <= 2)
#                 sub_states = (cur_state[0][needs_base_nodes], cur_state[1][needs_base_nodes])
#                 sub_states = self.node_feat_update(node_feats[base_nodes], sub_states)
#                 nz_idx = torch.tensor(np.nonzero(needs_base_nodes)[0]).to(node_feats.device)
#                 new_cur = [scatter(x, nz_idx, dim=0, dim_size=init_select.shape[0]) for x in sub_states]
#                 needs_base_nodes = torch.tensor(needs_base_nodes, dtype=torch.bool).to(node_feats.device).unsqueeze(1)
#                 cur_state = [torch.where(needs_base_nodes, new_cur[i], cur_state[i]) for i in range(2)]
#                 cur_state = tuple(cur_state)
#         ret_state = (joint_h[next_ids], joint_c[next_ids])
#         hist_rnn_states = []
#         hist_froms = []
#         hist_tos = []
#         for i, (done_from, done_to, proceed_from, proceed_input) in enumerate(all_ids):
#             hist_froms.append(done_from)
#             hist_tos.append(done_to)
#             hist_rnn_states.append(cur_state)
#             
#             #print(done_from)
#             #print(done_to)
#             #print(proceed_from)
#             #print(proceed_input)
#             
#             
# #             ### Put topology updates here
# #             if self.method == "LSTM2":
# #                 ### UPDATE ROW STATE W EDGE EMBED
# 
#             next_input = joint_h[proceed_input], joint_c[proceed_input]
#             sub_state = cur_state[0][proceed_from], cur_state[1][proceed_from]
#             
#             #print(next_input)
#             #print(sub_state)
#             
#             cur_state = self.summary_cell(sub_state, next_input)
#         hist_rnn_states.append(cur_state)
#         hist_froms.append(None)
#         hist_tos.append(last_tos)
#         hist_h_list, hist_c_list = zip(*hist_rnn_states)
#         pos_embed = self.pos_enc(pos_info)
#         row_h = multi_index_select(hist_froms, hist_tos, *hist_h_list) + pos_embed
#         row_c = multi_index_select(hist_froms, hist_tos, *hist_c_list) + pos_embed
#         return (row_h, row_c), ret_state
# 
# 
# class BitsRepNet(nn.Module):
#     def __init__(self, args):
#         super(BitsRepNet, self).__init__()
#         self.bits_compress = args.bits_compress
#         self.out_dim = args.embed_dim
#         assert self.out_dim >= self.bits_compress
#         self.device = args.device
# 
#     def forward(self, on_bits, n_cols):
#         h = torch.zeros(1, self.out_dim).to(self.device)
#         h[0, :n_cols] = -1.0
#         h[0, on_bits] = 1.0
# 
#         return h, h
# 
# 
# class RecurTreeGen(nn.Module):
# 
#     # to be customized
#     def embed_node_feats(self, node_feats):
#         raise NotImplementedError
# 
#     def embed_edge_feats(self, edge_feats):
#         raise NotImplementedError
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
#         raise NotImplementedError
# 
#     def predict_edge_feats(self, state, edge_feats=None):
#         """
#         Args:
#             state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
#             edge_feats: N x feat_dim or None
#         Returns:
#             likelihood of edge_feats under current state,
#             and, if edge_feats is None, then return the prediction of edge_feats
#             else return the edge_feats as it is
#         """
#         raise NotImplementedError
# 
#     def __init__(self, args):
#         super(RecurTreeGen, self).__init__()
# 
#         self.directed = args.directed
#         self.batch_size = args.batch_size
#         self.self_loop = args.self_loop
#         self.bits_compress = args.bits_compress
#         self.has_edge_feats = args.has_edge_feats
#         self.has_node_feats = args.has_node_feats
#         if self.has_edge_feats:
#             assert self.bits_compress == 0
#         self.greedy_frac = args.greedy_frac
#         self.share_param = args.share_param
#         if not self.bits_compress:
#             self.leaf_h0 = Parameter(torch.Tensor(1, args.embed_dim))
#             self.leaf_c0 = Parameter(torch.Tensor(1, args.embed_dim))
#             self.empty_h0 = Parameter(torch.Tensor(1, args.embed_dim))
#             self.empty_c0 = Parameter(torch.Tensor(1, args.embed_dim))
# 
#         self.topdown_left_embed = Parameter(torch.Tensor(2, args.embed_dim))
#         self.topdown_right_embed = Parameter(torch.Tensor(2, args.embed_dim))
#         glorot_uniform(self)
# 
#         if self.bits_compress > 0:
#             self.bit_rep_net = BitsRepNet(args)
# 
#         if self.share_param:
#             self.m_l2r_cell = BinaryTreeLSTMCell(args.embed_dim)
#             self.lr2p_cell = BinaryTreeLSTMCell(args.embed_dim)
#             self.pred_has_ch = MLP(args.embed_dim, [2 * args.embed_dim, 1])
#             self.m_pred_has_left = MLP(args.embed_dim, [2 * args.embed_dim, 1])
#             self.m_pred_has_right = MLP(args.embed_dim, [2 * args.embed_dim, 1])
#             self.m_cell_topdown = nn.LSTMCell(args.embed_dim, args.embed_dim)
#             self.m_cell_topright = nn.LSTMCell(args.embed_dim, args.embed_dim)
#         else:
#             fn_pred = lambda: MLP(args.embed_dim, [2 * args.embed_dim, 1])
#             fn_tree_cell = lambda: BinaryTreeLSTMCell(args.embed_dim)
#             fn_lstm_cell = lambda: nn.LSTMCell(args.embed_dim, args.embed_dim)
#             num_params = int(np.ceil(np.log2(args.max_num_nodes))) + 1
#             self.pred_has_ch = fn_pred()
# 
#             pred_modules = [[] for _ in range(2)]
#             tree_cell_modules = []
#             lstm_cell_modules = [[] for _ in range(2)]
#             for _ in range(num_params):
#                 for i in range(2):
#                     pred_modules[i].append(fn_pred())
#                     lstm_cell_modules[i].append(fn_lstm_cell())
#                 tree_cell_modules.append(fn_tree_cell())
# 
#             self.has_left_modules, self.has_right_modules = [nn.ModuleList(l) for l in pred_modules]
#             self.l2r_modules= nn.ModuleList(tree_cell_modules)
#             self.cell_topdown_modules, self.cell_topright_modules = [nn.ModuleList(l) for l in lstm_cell_modules]
#             self.lr2p_cell = fn_tree_cell()
#         self.row_tree = FenwickTree(args)
# 
#         if args.tree_pos_enc:
#             self.tree_pos_enc = PosEncoding(args.embed_dim, args.device, args.pos_base, bias=np.pi / 4)
#         else:
#             self.tree_pos_enc = lambda x: 0
# 
#     def cell_topdown(self, x, y, lv):
#         cell = self.m_cell_topdown if self.share_param else self.cell_topdown_modules[lv]
#         return cell(x, y)
# 
#     def cell_topright(self, x, y, lv):
#         cell = self.m_cell_topright if self.share_param else self.cell_topright_modules[lv]
#         return cell(x, y)
# 
#     def l2r_cell(self, x, y, lv=-1):
#         cell = self.m_l2r_cell if self.share_param else self.l2r_modules[lv]
#         return cell(x, y)
# 
#     def pred_has_left(self, x, lv):
#         mlp = self.m_pred_has_left if self.share_param else self.has_left_modules[lv]
#         return mlp(x)
# 
#     def pred_has_right(self, x, lv):
#         mlp = self.m_pred_has_right if self.share_param else self.has_right_modules[lv]
#         return mlp(x)
# 
#     def get_empty_state(self):
#         if self.bits_compress:
#             return self.bit_rep_net([], 1)
#         else:
#             return (self.empty_h0, self.empty_c0)
# 
#     def get_prob_fix(self, prob):
#         p = prob * (1 - self.greedy_frac)
#         if prob >= 0.5:
#             p += self.greedy_frac
#         return p
# 
#     def gen_row(self, ll, ll_wt, state, tree_node, col_sm, lb, ub, edge_feats=None, prev_wt_state=None):
#         assert lb <= ub
#         if tree_node.is_root:
#             prob_has_edge = torch.sigmoid(self.pred_has_ch(state[0]))
# 
#             if col_sm.supervised:
#                 has_edge = len(col_sm.indices) > 0
#             else:
#                 has_edge = np.random.rand() < self.get_prob_fix(prob_has_edge.item())
#                 if ub == 0:
#                     has_edge = False
#                 if tree_node.n_cols <= 0:
#                     has_edge = False
#                 if lb:
#                     has_edge = True
#             if has_edge:
#                 ll = ll + torch.log(prob_has_edge)
#             else:
#                 ll = ll + torch.log(1 - prob_has_edge)
#             tree_node.has_edge = has_edge
#         else:
#             assert ub > 0
#             tree_node.has_edge = True
# 
#         if not tree_node.has_edge:  # an empty tree
#             return ll, ll_wt, self.get_empty_state(), 0, None, prev_wt_state
# 
#         if tree_node.is_leaf:
#             tree_node.bits_rep = [0]
#             col_sm.add_edge(tree_node.col_range[0])
#             if self.bits_compress:
#                 return ll, self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols), 1, None
#             else:
#                 if self.has_edge_feats:
#                     cur_feats = edge_feats[col_sm.pos - 1].unsqueeze(0) if col_sm.supervised else None
#                     #print(cur_feats)
#                     #print(prev_wt_state)
#                     if self.method != "LSTM":
#                         edge_ll, cur_feats = self.predict_edge_feats(state, cur_feats)
#                     else:
#                         edge_ll, cur_feats = self.predict_edge_feats(state, cur_feats, prev_wt_state[0])
#                     
#                     ll_wt = ll_wt + edge_ll
#                     
#                     if self.method in ["Test", "Test2", "Test3"]:
#                         return ll, ll_wt,  (self.leaf_h0, self.leaf_c0), 1, cur_feats, prev_wt_state
#                     
#                     elif self.method == "Test4":
#                         prev_wt_state = self.embed_edge_feats(cur_feats, prev_state=prev_wt_state)
#                         return ll, ll_wt,  (self.leaf_h0, self.leaf_c0), 1, cur_feats, prev_wt_state
#                     
#                     elif self.method == "LSTM":
#                         edge_embed = self.embed_edge_feats(cur_feats, prev_state=prev_wt_state)
#                         prev_wt_state = edge_embed
#                         return ll, ll_wt, edge_embed, 1, cur_feats, prev_wt_state
#                     
#                     elif self.method == "MLP-2":
#                         #edge_embed = self.embed_edge_feats(cur_feats)
#                         return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, cur_feats, None
#                     
#                     edge_embed = self.embed_edge_feats(cur_feats)
#                     return ll, ll_wt, edge_embed, 1, cur_feats, None
#                     
#                 else:
#                     return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, None, None
#         else:
#             tree_node.split()
# 
#             mid = (tree_node.col_range[0] + tree_node.col_range[1]) // 2
#             left_prob = torch.sigmoid(self.pred_has_left(state[0], tree_node.depth))
# 
#             if col_sm.supervised:
#                 has_left = col_sm.next_edge < mid
#             else:
#                 has_left = np.random.rand() < self.get_prob_fix(left_prob.item())
#                 if ub == 0:
#                     has_left = False
#                 if lb > tree_node.rch.n_cols:
#                     has_left = True
#             ll = ll + (torch.log(left_prob) if has_left else torch.log(1 - left_prob))
#             left_pos = self.tree_pos_enc([tree_node.lch.n_cols])
#             state = self.cell_topdown(self.topdown_left_embed[[int(has_left)]] + left_pos, state, tree_node.depth)
#             pred_edge_feats = []
#             if has_left:
#                 if self.method == "Test4":
#                     prev_wt_state = self.edgeLSTM(self.topdown_left_embed[[int(has_left)]], prev_wt_state)
#                 lub = min(tree_node.lch.n_cols, ub)
#                 llb = max(0, lb - tree_node.rch.n_cols)
#                 ll, ll_wt, left_state, num_left, left_edge_feats, prev_wt_state = self.gen_row(ll, ll_wt, state, tree_node.lch, col_sm, llb, lub, edge_feats, prev_wt_state)
#                 pred_edge_feats.append(left_edge_feats)
#             else:
#                 left_state = self.get_empty_state()
#                 num_left = 0
# 
#             right_pos = self.tree_pos_enc([tree_node.rch.n_cols])
#             topdown_state = self.l2r_cell(state, (left_state[0] + right_pos, left_state[1] + right_pos), tree_node.depth)
#             
#             if self.has_edge_feats and self.method in ["Test", "LSTM2", "Test2", "Test4"] and tree_node.lch.is_leaf and has_left:
#                 if self.method == "Test4":
#                     left_edge_embed = self.standardize_edge_feats(left_edge_feats)
#                     left_edge_embed = self.edgelen_encoding(left_edge_feats)
#                 
#                 else:
#                     left_edge_embed = self.embed_edge_feats(left_edge_feats, prev_state=prev_wt_state)
#                 if self.update_left:
#                     topdown_state = self.topdown_update_wt(left_edge_embed, topdown_state)
#                     #left_state = self.update_wt(left_edge_embed, left_state)
#                     
#                 else:
#                     topdown_state = self.update_wt(left_edge_embed, topdown_state)
#             
#             rlb = max(0, lb - num_left)
#             rub = min(tree_node.rch.n_cols, ub - num_left)
#             if not has_left:
#                 has_right = True
#             else:
#                 right_prob = torch.sigmoid(self.pred_has_right(topdown_state[0], tree_node.depth))
#                 if col_sm.supervised:
#                     has_right = col_sm.has_edge(mid, tree_node.col_range[1])
#                 else:
#                     has_right = np.random.rand() < self.get_prob_fix(right_prob.item())
#                     if rub == 0:
#                         has_right = False
#                     if rlb:
#                         has_right = True
#                 ll = ll + (torch.log(right_prob) if has_right else torch.log(1 - right_prob))
# 
#             topdown_state = self.cell_topright(self.topdown_right_embed[[int(has_right)]], topdown_state, tree_node.depth)
# 
#             if has_right:  # has edge in right child
#                 if self.method == "Test4":
#                     prev_wt_state = self.edgeLSTM(self.topdown_right_embed[[int(has_right)]], prev_wt_state)
#                 ll, ll_wt, right_state, num_right, right_edge_feats, prev_wt_state = self.gen_row(ll, ll_wt, topdown_state, tree_node.rch, col_sm, rlb, rub, edge_feats, prev_wt_state)
#                 pred_edge_feats.append(right_edge_feats)
#             else:
#                 right_state = self.get_empty_state()
#                 num_right = 0
#             if tree_node.col_range[1] - tree_node.col_range[0] <= self.bits_compress:
#                 summary_state = self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols)
#             else:
#                 summary_state = self.lr2p_cell(left_state, right_state)
#             if self.has_edge_feats:
#                 edge_feats = torch.cat(pred_edge_feats, dim=0)
#                 if self.method == "Test":
#                     if has_left and tree_node.lch.is_leaf:
#                         left_edge_embed = self.embed_edge_feats(left_edge_feats, prev_state=prev_wt_state)
#                         summary_state = self.update_wt(left_edge_embed, summary_state)
#                     
#                     if has_right and tree_node.rch.is_leaf:
#                         right_edge_embed = self.embed_edge_feats(right_edge_feats, prev_state=prev_wt_state)
#                         summary_state = self.update_wt(right_edge_embed, summary_state)
#             
#             return ll, ll_wt, summary_state, num_left + num_right, edge_feats, prev_wt_state
# 
#     def forward(self, node_end, edge_list=None, node_feats=None, edge_feats=None, node_start=0, list_states=[], lb_list=None, ub_list=None, col_range=None, num_nodes=None, display=False):
#         pos = 0
#         total_ll = 0.0
#         total_ll_wt = 0.0
#         edges = []
#         self.row_tree.reset(list_states)
#         controller_state = self.row_tree()
#         if num_nodes is None:
#             num_nodes = node_end
#         pbar = range(node_start, node_end)
#         if display:
#             pbar = tqdm(pbar)
#         list_pred_node_feats = []
#         list_pred_edge_feats = []
#         
#         prev_wt_state = None
#         if self.has_edge_feats and self.method in ["LSTM", "Test4"]:
#             prev_wt_state = (self.leaf_h0_wt, self.leaf_c0_wt)
#         
#         for i in pbar:
#             if edge_list is None:
#                 col_sm = ColAutomata(supervised=False)
#             else:
#                 indices = []
#                 while pos < len(edge_list) and i == edge_list[pos][0]:
#                     indices.append(edge_list[pos][1])
#                     pos += 1
#                 indices.sort()
#                 col_sm = ColAutomata(supervised=True, indices=indices)
# 
#             cur_row = AdjRow(i, self.directed, self.self_loop, col_range=col_range)
#             lb = 0 if lb_list is None else lb_list[i]
#             ub = cur_row.root.n_cols if ub_list is None else ub_list[i]
#             cur_pos_embed = self.row_tree.pos_enc([num_nodes - i])
#             controller_state = [x + cur_pos_embed for x in controller_state]
#             if self.has_node_feats:
#                 target_node_feats = None if node_feats is None else node_feats[[i]]
#                 controller_state, ll_node, target_node_feats = self.predict_node_feats(controller_state, target_node_feats)
#                 total_ll = total_ll + ll_node
#                 list_pred_node_feats.append(target_node_feats)
#             if self.has_edge_feats:
#                 target_edge_feats = None if edge_feats is None else edge_feats[len(edges) : len(edges) + len(col_sm)]
#             else:
#                 target_edge_feats = None
#             #print(prev_wt_state)
#             
#             if self.method == "Test4":
#                 prev_wt_state = (self.leaf_h0_wt, self.leaf_c0_wt)
#             
#             ll, ll_wt, cur_state, _, target_edge_feats, prev_wt_state = self.gen_row(0, 0, controller_state, cur_row.root, col_sm, lb, ub, target_edge_feats, prev_wt_state)
#             if target_edge_feats is not None and target_edge_feats.shape[0]:
#                 list_pred_edge_feats.append(target_edge_feats)
#             if self.has_node_feats:
#                 target_feat_embed = self.embed_node_feats(target_node_feats)
#                 cur_state = self.row_tree.node_feat_update(target_feat_embed, cur_state)
#             assert lb <= len(col_sm.indices) <= ub
#             
#             if self.method == "LSTM2":
#                 cur_state = self.merge_top_wt(cur_state, prev_wt_state)
#             
#             if self.method == "Test4" and i > 0:
#                 #print("=================================================================")
#                 #print("i:", i)
#                 #print("top state: ", cur_state)
#                 #print("wt state: ", prev_wt_state)
#                 #print("=================================================================")# 
#                 cur_state = self.merge_top_wt(cur_state, prev_wt_state)
#                 #print("Updated Staet: ", cur_state)
#                 #print("=================================================================")
#             
#             controller_state = self.row_tree(cur_state)
#             
#             if self.method == "Test" and cur_row.root.is_leaf and target_edge_feats is not None:
#                 edge_embed = self.embed_edge_feats(target_edge_feats, prev_state=prev_wt_state)
#                 controller_state = self.update_wt(edge_embed, controller_state)
#                 self.row_tree.list_states[1] = [controller_state]
#             
#             edges += [(i, x) for x in col_sm.indices]
#             total_ll = total_ll + ll
#             total_ll_wt = total_ll_wt + ll_wt
# 
#         if self.has_node_feats:
#             node_feats = torch.cat(list_pred_node_feats, dim=0)
#         if self.has_edge_feats:
#             edge_feats = torch.cat(list_pred_edge_feats, dim=0)
#         return total_ll, total_ll_wt, edges, self.row_tree.list_states, node_feats, edge_feats
# 
#     def binary_ll(self, pred_logits, np_label, need_label=False, reduction='sum'):
#         pred_logits = pred_logits.view(-1, 1)
#         label = torch.tensor(np_label, dtype=torch.float32).to(pred_logits.device).view(-1, 1)
#         loss = F.binary_cross_entropy_with_logits(pred_logits, label, reduction=reduction)
#         if need_label:
#             return -loss, label
#         return -loss
# 
#     def forward_row_trees(self, graph_ids, node_feats=None, edge_feats=None, list_node_starts=None, num_nodes=-1, list_col_ranges=None):
#         TreeLib.PrepareMiniBatch(graph_ids, list_node_starts, num_nodes, list_col_ranges)
#         # embed trees
#         all_ids = TreeLib.PrepareTreeEmbed()
#         if self.has_node_feats:
#             node_feats = self.embed_node_feats(node_feats)
#         #if self.has_edge_feats:
#         #    edge_feats = self.embed_edge_feats(edge_feats)
# 
#         if not self.bits_compress:
#             h_bot = torch.cat([self.empty_h0, self.leaf_h0], dim=0)
#             c_bot = torch.cat([self.empty_c0, self.leaf_c0], dim=0)
#             fn_hc_bot = lambda d: (h_bot, c_bot)
#         else:
#             binary_embeds, base_feat = TreeLib.PrepareBinary()
#             fn_hc_bot = lambda d: (binary_embeds[d], binary_embeds[d]) if d < len(binary_embeds) else base_feat
#         max_level = len(all_ids) - 1
#         h_buf_list = [None] * (len(all_ids) + 1)
#         c_buf_list = [None] * (len(all_ids) + 1)
#         
#         for d in range(len(all_ids) - 1, -1, -1):
#             fn_ids = lambda i: all_ids[d][i]
#             if d == max_level:
#                 h_buf = c_buf = None
#             else:
#                 h_buf = h_buf_list[d + 1]
#                 c_buf = c_buf_list[d + 1]
#             h_bot, c_bot = fn_hc_bot(d + 1)
#             if self.has_edge_feats:
#                 edge_idx, is_rch = TreeLib.GetEdgeAndLR(d + 1)
#                 if self.method in ["Test", "Test2", "Test3"]:
#                     local_edge_feats = edge_feats[edge_idx]
#                 else:
#                     local_edge_feats = (edge_feats[0][edge_idx], edge_feats[1][edge_idx])
#                 
#                 new_h, new_c = featured_batch_tree_lstm2(local_edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell, wt_update =self.update_wt, method = self.method)
#             else:
#                 new_h, new_c = batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell)
#             if self.method == "Test4" and d == 0:
#                 b = self.batch_size
#                 m = edge_feats[0].shape[0] // b 
#                 idx = ([False] + [True]*(m-1))*b
#                 idx = np.array(idx)
#                 edge_embed_cur = (edge_feats[0][idx], edge_feats[1][idx])
# #                 print("+++++++++++++++++++++++++++++++++++++++++++++++")
# #                 print("Top Before: ", new_h)
# #                 print("Weight: ", edge_embed_cur[0])
#                 new_h, new_c = self.merge_top_wt((new_h, new_c), edge_embed_cur)
# #                 print("Top After: ", new_h)
# #                 print("+++++++++++++++++++++++++++++++++++++++++++++++")
#                 #print("new_h: ", new_h)
#             
#             h_buf_list[d] = new_h
#             c_buf_list[d] = new_c
#         hc_bot = fn_hc_bot(0)
#         feat_dict = {}
#         if self.has_edge_feats:
#             edge_idx, is_rch = TreeLib.GetEdgeAndLR(0)
#             if self.method in ["Test", "Test2", "Test3"]:
#                 local_edge_feats = edge_feats[edge_idx]
#             elif self.method == "Test4":
#                 local_edge_feats = (edge_feats[0][edge_idx], edge_feats[1][edge_idx])
# #                 print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# #                 print(edge_idx)
# #                 print("wt state: ", local_edge_feats)
#                 init_state = (self.leaf_h0.repeat(len(edge_idx), 1), self.leaf_c0.repeat(len(edge_idx), 1))
# #                 print("top state: ", init_state)
#                 local_edge_feats = self.merge_top_wt(init_state, local_edge_feats)
# #                 print("updated state: ", local_edge_feats)
# #                 print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#             else:
#                 local_edge_feats = (edge_feats[0][edge_idx], edge_feats[1][edge_idx])
#             feat_dict['edge'] = (local_edge_feats, is_rch)
#         if self.has_node_feats:
#             is_tree_trivial = TreeLib.GetIsTreeTrivial()
#             new_h, new_c = self.row_tree.node_feat_update(node_feats[~is_tree_trivial], (new_h, new_c))
#             h_buf_list[0] = new_h
#             c_buf_list[0] = new_c
#             t_lch, t_rch = TreeLib.GetTrivialNodes()
#             feat_dict['node'] = (node_feats, is_tree_trivial, t_lch, t_rch)
#         if len(feat_dict):
#             hc_bot = (hc_bot, feat_dict)
#         
# #         print(h_buf_list)
# #         if self.method == "Test4":
# #             cur_edge_embed_h = torch.cat([self.leaf_h0_wt, edge_feats[0][0:1]], dim = 0)
# #             cur_edge_embed_c = torch.cat([self.leaf_c0_wt, edge_feats[1][0:1]], dim = 0)
# #             top = hc_bot[0]
# #             #print(cur_edge_embed_h)
# #             #print(h_bot)
# #             top = self.merge_top_wt(top, (cur_edge_embed_h, cur_edge_embed_c))
# #             hc_bot = (top, hc_bot[1])
#         return hc_bot, fn_hc_bot, h_buf_list, c_buf_list
# 
#     def forward_row_summaries(self, graph_ids, node_feats=None, edge_feats=None,
#                              list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None):
#         hc_bot, _, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, node_feats, edge_feats,
#                                                                    list_node_starts, num_nodes, list_col_ranges)
#         row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states)
#         return row_states, next_states
# 
#     def forward_train(self, graph_ids, node_feats=None, edge_feats=None,
#                       list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None):
#         ll = 0.0
#         ll_wt = 0.0
#         noise = 0.0
#         edge_feats_embed = None
#         if self.has_edge_feats:
#             if self.method == "LSTM":
#                 edge_feats_embed, state_h_prior = self.embed_edge_feats(edge_feats, noise)
#                 #print(edge_feats)
#                 edge_feats = torch.cat(edge_feats, dim = 0)
#                 #print(edge_feats)
#                 #print(STOP)
#             
#             elif self.method == "Test4":
#                 edge_feats, lr = edge_feats
#                 edge_feats_embed, weights_MLP = self.embed_edge_feats(edge_feats, lr_seq=lr)
#                 #print(STOP)
#             
#             else:
#                 edge_feats_embed = self.embed_edge_feats(edge_feats)
#         
#         hc_bot, fn_hc_bot, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, node_feats, edge_feats_embed,
#                                                                            list_node_starts, num_nodes, list_col_ranges)
#         row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states, wt_update=self.update_wt)
#         if self.has_node_feats:
#             row_states, ll_node_feats, _ = self.predict_node_feats(row_states, node_feats)
#             ll = ll + ll_node_feats
#         logit_has_edge = self.pred_has_ch(row_states[0])
#         has_ch, _ = TreeLib.GetChLabel(0, dtype=bool)
#         ll = ll + self.binary_ll(logit_has_edge, has_ch)
#         cur_states = (row_states[0][has_ch], row_states[1][has_ch])
# 
#         lv = 0
#         while True:
#             is_nonleaf = TreeLib.QueryNonLeaf(lv)
#             if self.has_edge_feats:
#                 edge_of_lv = TreeLib.GetEdgeOf(lv)
#                 edge_state = (cur_states[0][~is_nonleaf], cur_states[1][~is_nonleaf])
#                 target_feats = edge_feats[edge_of_lv]
#                 prior_h_target = None
#                 if self.method == "LSTM": 
#                     prior_h_target = state_h_prior[edge_of_lv]
#                 edge_ll, _ = self.predict_edge_feats(edge_state, target_feats, prior_h_target)
#                 ll_wt = ll_wt + edge_ll
#             if is_nonleaf is None or np.sum(is_nonleaf) == 0:
#                 break
#             cur_states = (cur_states[0][is_nonleaf], cur_states[1][is_nonleaf])
#             left_logits = self.pred_has_left(cur_states[0], lv)
#             has_left, num_left = TreeLib.GetChLabel(-1, lv)
#             left_update = self.topdown_left_embed[has_left] + self.tree_pos_enc(num_left)
#             left_ll, float_has_left = self.binary_ll(left_logits, has_left, need_label=True, reduction='sum')
#             ll = ll + left_ll
# 
#             cur_states = self.cell_topdown(left_update, cur_states, lv)
# 
#             left_ids = TreeLib.GetLeftRootStates(lv)
#             h_bot, c_bot = fn_hc_bot(lv + 1)
#             if lv + 1 < len(h_buf_list):
#                 h_next_buf, c_next_buf = h_buf_list[lv + 1], c_buf_list[lv + 1]
#             else:
#                 h_next_buf = c_next_buf = None
#             if self.has_edge_feats:
#                 edge_idx, is_rch = TreeLib.GetEdgeAndLR(lv + 1)
#                 if self.method in ["Test", "Test2", "Test3"]: 
#                     left_feats = edge_feats_embed[edge_idx[~is_rch]]
#                 
#                 elif self.method == "Test4":
#                     left_feats = weights_MLP[edge_idx[~is_rch]]
#                     
#                 else:
#                     left_feats = (edge_feats_embed[0][edge_idx[~is_rch]], edge_feats_embed[1][edge_idx[~is_rch]])
#                 
#                 h_bot, c_bot = h_bot[left_ids[0]], c_bot[left_ids[0]]
#                 
#                 if self.method not in ["Test", "Test2", "Test3", "Test4"]:
#                     h_bot, c_bot = selective_update_hc(h_bot, c_bot, left_ids[0], left_feats)
#                 left_wt_ids = left_ids[1][list(map(bool, left_ids[0]))]
#                 left_ids = tuple([None] + list(left_ids[1:]))
# 
#             left_subtree_states = tree_state_select(h_bot, c_bot,
#                                                     h_next_buf, c_next_buf,
#                                                     lambda: left_ids)
# 
#             has_right, num_right = TreeLib.GetChLabel(1, lv)
#             right_pos = self.tree_pos_enc(num_right)
#             left_subtree_states = [x + right_pos for x in left_subtree_states]
#             topdown_state = self.l2r_cell(cur_states, left_subtree_states, lv)
#             
#             if self.has_edge_feats and self.method in ["Test", "LSTM2", "Test2", "Test4"] and len(left_wt_ids) > 0:
#                 leaf_topdown_states = (topdown_state[0][left_wt_ids], topdown_state[1][left_wt_ids])
#                 
#                 if self.update_left:
#                     leaf_topdown_states = self.topdown_update_wt(left_feats, leaf_topdown_states)
#                 
#                 else:
#                     leaf_topdown_states = self.update_wt(left_feats, leaf_topdown_states)
#                 topdown_state[0][left_wt_ids] = leaf_topdown_states[0]
#                 topdown_state[1][left_wt_ids] = leaf_topdown_states[1]
#             
#             right_logits = self.pred_has_right(topdown_state[0], lv)
#             right_update = self.topdown_right_embed[has_right]
#             topdown_state = self.cell_topright(right_update, topdown_state, lv)
#             right_ll = self.binary_ll(right_logits, has_right, reduction='none') * float_has_left
#             ll = ll + torch.sum(right_ll)
#             lr_ids = TreeLib.GetLeftRightSelect(lv, np.sum(has_left), np.sum(has_right))
#             new_states = []
#             for i in range(2):
#                 new_s = multi_index_select([lr_ids[0], lr_ids[2]], [lr_ids[1], lr_ids[3]],
#                                             cur_states[i], topdown_state[i])
#                 new_states.append(new_s)
#             cur_states = tuple(new_states)
#             lv += 1
#         return ll, ll_wt, next_states
































































# # coding=utf-8
# # Copyright 2024 The Google Research Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# 
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# # pylint: skip-file
# 
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_scatter import scatter
# from collections import defaultdict
# from torch.nn.parameter import Parameter
# from bigg.common.pytorch_util import glorot_uniform, MLP, BinaryTreeLSTMCell, MultiLSTMCell
# from tqdm import tqdm
# from bigg.model.util import AdjNode, ColAutomata, AdjRow
# from bigg.model.tree_clib.tree_lib import TreeLib
# from bigg.torch_ops import multi_index_select, PosEncoding
# from functools import partial
# 
# 
# def hc_multi_select(ids_from, ids_to, h_froms, c_froms):
#     h_vecs = multi_index_select(ids_from,
#                                 ids_to,
#                                 *h_froms)
#     c_vecs = multi_index_select(ids_from,
#                                 ids_to,
#                                 *c_froms)
#     return h_vecs, c_vecs
# 
# 
# def tree_state_select(h_bot, c_bot, h_buf, c_buf, fn_all_ids):
#     bot_froms, bot_tos, prev_froms, prev_tos = fn_all_ids()
#     if h_buf is None or prev_tos is None:
#         h_vecs = multi_index_select([bot_froms], [bot_tos], h_bot)
#         c_vecs = multi_index_select([bot_froms], [bot_tos], c_bot)
#     elif h_bot is None or bot_tos is None:
#         h_vecs = multi_index_select([prev_froms], [prev_tos], h_buf)
#         c_vecs = multi_index_select([prev_froms], [prev_tos], c_buf)
#     else:
#         h_vecs, c_vecs = hc_multi_select([bot_froms, prev_froms],
#                                          [bot_tos, prev_tos],
#                                          [h_bot, h_buf], [c_bot, c_buf])
#     return h_vecs, c_vecs
# 
# 
# def batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_all_ids, cell):
#     h_list = []
#     c_list = []
#     for i in range(2):
#         h_vecs, c_vecs = tree_state_select(h_bot, c_bot, h_buf, c_buf, lambda : fn_all_ids(i))
#         h_list.append(h_vecs)
#         c_list.append(c_vecs)
#     return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))
# 
# 
# def selective_update_hc(h, c, zero_one, feats):
#     nz_idx = torch.tensor(np.nonzero(zero_one)[0]).to(h.device)
#     num_layers = h.shape[0]
#     embed_dim = h.shape[2]
#     #feats = feats.reshape(feats.shape[0], num_layers, embed_dim).movedim(0, 1)
#     local_edge_feats_h = scatter(feats[0], nz_idx, dim=1, dim_size=h.shape[1])
#     local_edge_feats_c = scatter(feats[1], nz_idx, dim=1, dim_size=h.shape[1])
#     zero_one = torch.tensor(zero_one, dtype=torch.bool).to(h.device).unsqueeze(1)
#     h = torch.where(zero_one, local_edge_feats_h, h)
#     c = torch.where(zero_one, local_edge_feats_c, c)
#     return h, c
# 
# def featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_all_ids, cell, t_lch=None, t_rch=None, cell_node=None, wt_update=None, method=None):
#     ### NEED TO PASS IN "METHOD...."
#     new_ids = [list(fn_all_ids(0)), list(fn_all_ids(1))]
#     lch_isleaf, rch_isleaf = new_ids[0][0], new_ids[1][0]
#     new_ids[0][0] = new_ids[1][0] = None
#     is_leaf = [lch_isleaf, rch_isleaf]
#     if edge_feats is not None:
#         if method == "Test":
#             edge_feats = [edge_feats[~is_rch], edge_feats[is_rch]]
#         
#         else:
#             edge_feats = [(edge_feats[0][:, ~is_rch], edge_feats[1][:, ~is_rch]), (edge_feats[0][:, is_rch], edge_feats[1][:, is_rch])]
#         assert np.sum(is_rch) == np.sum(rch_isleaf)
#     node_feats = [t_lch, t_rch]
#     h_list = []
#     c_list = []
#     
#     for i in range(2):
#         leaf_check = is_leaf[i]
#         local_hbot, local_cbot = h_bot[:, leaf_check], c_bot[:, leaf_check]
#         if edge_feats is not None and method != "Test":
#             local_hbot, local_cbot = selective_update_hc(local_hbot, local_cbot, leaf_check, edge_feats[i])
#         if cell_node is not None:
#             local_hbot, local_cbot = cell_node(node_feats[i], (local_hbot, local_cbot))
#         
#         h_vecs, c_vecs = tree_state_select(local_hbot, local_cbot, h_buf, c_buf, lambda : new_ids[i])
#         h_list.append(h_vecs)
#         c_list.append(c_vecs)
#     
#     summary_state = cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))
#     
#     if method != "Test":
#         return summary_state
#     
#     for i in range(2):
#         leaf_check = list(map(bool, is_leaf[i]))
#         local_idx = new_ids[i][1][leaf_check]
#         local_hbot, local_cbot = summary_state[0][:, local_idx], summary_state[1][:, local_idx]
#         cur_summary = (local_hbot, local_cbot)
#         cur_edge_feats = edge_feats[i]
#         cur_summary = wt_update(cur_edge_feats, cur_summary)
#         summary_state[0][:, local_idx] = cur_summary[0]
#         summary_state[1][:, local_idx] = cur_summary[1]
#         
#     return summary_state
# 
# 
# def batch_tree_lstm3(h_bot, c_bot, h_buf, c_buf, h_past, c_past, fn_all_ids, cell):
#     if h_past is None:
#         return batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, lambda i: fn_all_ids(i)[:-2], cell)
#     elif h_bot is None:
#         return batch_tree_lstm2(h_buf, c_buf, h_past, c_past, lambda i: fn_all_ids(i)[2:], cell)
#     elif h_buf is None:
#         return batch_tree_lstm2(h_bot, c_bot, h_past, c_past, lambda i: fn_all_ids(i)[0, 1, 4, 5], cell)
#     else:
#         h_list = []
#         c_list = []
#         for i in range(2):
#             bot_froms, bot_tos, prev_froms, prev_tos, past_froms, past_tos = fn_all_ids(i)
#             h_vecs, c_vecs = hc_multi_select([bot_froms, prev_froms, past_froms],
#                                              [bot_tos, prev_tos, past_tos],
#                                              [h_bot, h_buf, h_past],
#                                              [c_bot, c_buf, c_past])
#             h_list.append(h_vecs)
#             c_list.append(c_vecs)
#         return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))
# 
# 
# def featured_batch_tree_lstm3(feat_dict, h_bot, c_bot, h_buf, c_buf, h_past, c_past, fn_all_ids, cell, cell_node, wt_update, method):
#     edge_feats = is_rch = None
#     t_lch = t_rch = None
#     if 'edge' in feat_dict:
#         edge_feats, is_rch = feat_dict['edge']
#     if 'node' in feat_dict:
#         t_lch, t_rch = feat_dict['node']
#     if h_past is None:
#         #print("Hello 1")
#         ## This one is for updating state 0...
#         return featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, lambda i: fn_all_ids(i)[:-2], cell, t_lch, t_rch, cell_node, wt_update, method)
#     elif h_bot is None:
#         return batch_tree_lstm2(h_buf, c_buf, h_past, c_past, lambda i: fn_all_ids(i)[2:], cell)
#     elif h_buf is None:
#         #print("Hello 2")
#         return featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_past, c_past, lambda i: fn_all_ids(i)[0, 1, 4, 5], cell, t_lch, t_rch, cell_node, wt_update, method)
#     else:
#         raise NotImplementedError  #TODO: handle model parallelism with features
# 
# 
# class FenwickTree(nn.Module):
#     def __init__(self, args):
#         super(FenwickTree, self).__init__()
#         self.has_edge_feats = args.has_edge_feats
#         self.has_node_feats = args.has_node_feats
#         self.method = args.method
#         self.embed_dim = args.embed_dim
#         self.rnn_layers = args.rnn_layers
#         
#         multiplier = 1.0
#         if args.method == "MLP-Leaf":
#             multiplier = 1.5
#         
#         #if args.method == "MLP-Leaf":
#         #    self.init_h0 = torch.cat([Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim)), torch.zeros(args.rnn_layers, 1, args.embed_dim // 2)], dim = -1).to(args.device)
#         #    self.init_c0 = torch.cat([Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim)), torch.zeros(args.rnn_layers, 1, args.embed_dim // 2)], dim = -1).to(args.device)
#         
#         #else:
#         self.init_h0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#         self.init_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#         
#         glorot_uniform(self)
#         if self.has_node_feats:
#             self.node_feat_update = nn.LSTMCell(args.embed_dim, args.embed_dim)
#         self.merge_cell = BinaryTreeLSTMCell(int(multiplier * args.embed_dim))
#         self.summary_cell = BinaryTreeLSTMCell(int(multiplier * args.embed_dim))
#         if args.pos_enc:
#             self.pos_enc = PosEncoding(int(multiplier * args.embed_dim), args.device, args.pos_base)
#         else:
#             self.pos_enc = lambda x: 0
# 
#     def reset(self, list_states=[]):
#         self.list_states = []
#         for l in list_states:
#             t = []
#             for e in l:
#                 t.append(e)
#             self.list_states.append(t)
# 
#     def append_state(self, state, level):
#         if level >= len(self.list_states):
#             num_aug = level - len(self.list_states) + 1
#             for i in range(num_aug):
#                 self.list_states.append([])
#         self.list_states[level].append(state)
# 
#     def forward(self, new_state=None):
#         if new_state is None:
#             if len(self.list_states) == 0:
#                 if self.method == "MLP-Leaf":
#                     dev = self.init_h0.device
#                     init_h0 = torch.cat([self.init_h0, torch.zeros(self.rnn_layers, 1, self.embed_dim // 2, device = dev)], dim = -1)
#                     init_c0 = torch.cat([self.init_h0, torch.zeros(self.rnn_layers, 1, self.embed_dim // 2, device = dev)], dim = -1)
#                     return (init_h0, init_c0)
#                 
#                 else:
#                     return (self.init_h0, self.init_c0)
#         else:
#             self.append_state(new_state, 0)
#         pos = 0
#         
#         while pos < len(self.list_states):
#             if len(self.list_states[pos]) >= 2:
#                 lch_state, rch_state = self.list_states[pos]  # assert the length is 2
#                 new_state = self.merge_cell(lch_state, rch_state)
#                 self.list_states[pos] = []
#                 self.append_state(new_state, pos + 1)
#             pos += 1
#         state = None
#         for pos in range(len(self.list_states)):
#             #print("pos: ", pos)
#             if len(self.list_states[pos]) == 0:
#                 continue
#             cur_state = self.list_states[pos][0]
#             if state is None:
#                 state = cur_state
#             else:
#                 state = self.summary_cell(state, cur_state)
#         return state
# 
#     def forward_train(self, h_bot, c_bot, h_buf0, c_buf0, prev_rowsum_h, prrev_rowsum_c, wt_update, method):
#         # embed row tree
#         tree_agg_ids = TreeLib.PrepareRowEmbed()
#         
#         if self.method == "MLP-Leaf":
#             dev = self.init_h0.device
#             init_h0 = torch.cat([self.init_h0, torch.zeros(self.rnn_layers, 1, self.embed_dim // 2, device = dev)], dim = -1)
#             init_c0 = torch.cat([self.init_c0, torch.zeros(self.rnn_layers, 1, self.embed_dim // 2, device = dev)], dim = -1)
#             row_embeds = [(init_h0, init_c0)]
#         
#         else:
#             row_embeds = [(self.init_h0, self.init_c0)]
#         
#         if self.has_edge_feats or self.has_node_feats:
#             feat_dict = c_bot
#             if 'node' in feat_dict:
#                 node_feats, is_tree_trivial, t_lch, t_rch = feat_dict['node']
#                 sel_feat = node_feats[is_tree_trivial]
#                 feat_dict['node'] = (sel_feat[t_lch], sel_feat[t_rch])
#             h_bot, c_bot = h_bot
#         if h_bot is not None:
#             row_embeds.append((h_bot, c_bot))
#         if prev_rowsum_h is not None:
#             row_embeds.append((prev_rowsum_h, prrev_rowsum_c))
#         if h_buf0 is not None:
#             row_embeds.append((h_buf0, c_buf0))
# 
#         for i, all_ids in enumerate(tree_agg_ids):
#             fn_ids = lambda x: all_ids[x]
#             lstm_func = batch_tree_lstm3
#             if i == 0 and (self.has_edge_feats or self.has_node_feats):
#                 lstm_func = featured_batch_tree_lstm3
#             
#             lstm_func = partial(lstm_func, h_buf=row_embeds[-1][0], c_buf=row_embeds[-1][1],
#                                 h_past=prev_rowsum_h, c_past=prrev_rowsum_c, fn_all_ids=fn_ids, cell=self.merge_cell)
#             if i == 0:
#                 if self.has_edge_feats or self.has_node_feats:
#                     new_states = lstm_func(feat_dict, h_bot, c_bot, cell_node=None if not self.has_node_feats else self.node_feat_update, wt_update=wt_update, method=method)
#                 else:
#                     new_states = lstm_func(h_bot, c_bot)
#             else:
#                 new_states = lstm_func(None, None)
#             
#             row_embeds.append(new_states)
#         
#         h_list, c_list = zip(*row_embeds)
#         
#         joint_h = torch.cat(h_list, dim=1)
#         joint_c = torch.cat(c_list, dim=1)
# 
#         # get history representation
#         init_select, all_ids, last_tos, next_ids, pos_info = TreeLib.PrepareRowSummary()
#         cur_state = (joint_h[:, init_select], joint_c[:, init_select])
#         if self.has_node_feats:
#             base_nodes, _ = TreeLib.GetFenwickBase()
#             if len(base_nodes):
#                 needs_base_nodes = (init_select >= 1) & (init_select <= 2)
#                 sub_states = (cur_state[0][needs_base_nodes], cur_state[1][needs_base_nodes])
#                 sub_states = self.node_feat_update(node_feats[base_nodes], sub_states)
#                 nz_idx = torch.tensor(np.nonzero(needs_base_nodes)[0]).to(node_feats.device)
#                 new_cur = [scatter(x, nz_idx, dim=0, dim_size=init_select.shape[0]) for x in sub_states]
#                 needs_base_nodes = torch.tensor(needs_base_nodes, dtype=torch.bool).to(node_feats.device).unsqueeze(1)
#                 cur_state = [torch.where(needs_base_nodes, new_cur[i], cur_state[i]) for i in range(2)]
#                 cur_state = tuple(cur_state)
#         ret_state = (joint_h[:, next_ids], joint_c[:, next_ids])
#         hist_rnn_states = []
#         hist_froms = []
#         hist_tos = []
#         for i, (done_from, done_to, proceed_from, proceed_input) in enumerate(all_ids):
#             hist_froms.append(done_from)
#             hist_tos.append(done_to)
#             hist_rnn_states.append(cur_state)
# 
#             next_input = joint_h[:, proceed_input], joint_c[:, proceed_input]
#             sub_state = cur_state[0][:, proceed_from], cur_state[1][:, proceed_from]
#             cur_state = self.summary_cell(sub_state, next_input)
#         hist_rnn_states.append(cur_state)
#         hist_froms.append(None)
#         hist_tos.append(last_tos)
#         hist_h_list, hist_c_list = zip(*hist_rnn_states)
#         pos_embed = self.pos_enc(pos_info)
#         row_h = multi_index_select(hist_froms, hist_tos, *hist_h_list) + pos_embed
#         row_c = multi_index_select(hist_froms, hist_tos, *hist_c_list) + pos_embed
#         return (row_h, row_c), ret_state
# 
# 
# class BitsRepNet(nn.Module):
#     def __init__(self, args):
#         super(BitsRepNet, self).__init__()
#         self.bits_compress = args.bits_compress
#         self.out_dim = args.embed_dim
#         assert self.out_dim >= self.bits_compress
#         self.device = args.device
# 
#     def forward(self, on_bits, n_cols):
#         h = torch.zeros(1, self.out_dim).to(self.device)
#         h[0, :n_cols] = -1.0
#         h[0, on_bits] = 1.0
# 
#         return h, h
# 
# 
# class RecurTreeGen(nn.Module):
# 
#     # to be customized
#     def embed_node_feats(self, node_feats):
#         raise NotImplementedError
# 
#     def embed_edge_feats(self, edge_feats):
#         raise NotImplementedError
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
#         raise NotImplementedError
# 
#     def predict_edge_feats(self, state, edge_feats=None):
#         """
#         Args:
#             state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
#             edge_feats: N x feat_dim or None
#         Returns:
#             likelihood of edge_feats under current state,
#             and, if edge_feats is None, then return the prediction of edge_feats
#             else return the edge_feats as it is
#         """
#         raise NotImplementedError
# 
#     def __init__(self, args):
#         super(RecurTreeGen, self).__init__()
# 
#         self.directed = args.directed
#         self.eps = args.eps
#         multiplier = 1.0
#         self.method = args.method
#         
#         if args.method == "MLP-Leaf":
#             multiplier = 1.5
#         
#         self.self_loop = args.self_loop
#         self.bits_compress = args.bits_compress
#         self.has_edge_feats = args.has_edge_feats
#         self.has_node_feats = args.has_node_feats
#         self.rnn_layers = args.rnn_layers
#         if self.has_edge_feats:
#             assert self.bits_compress == 0
#         self.greedy_frac = args.greedy_frac
#         self.share_param = args.share_param
#         if not self.bits_compress:
#             self.leaf_h0 = Parameter(torch.Tensor(args.rnn_layers, 1, int(multiplier * args.embed_dim)))
#             self.leaf_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, int(multiplier * args.embed_dim)))
#             self.empty_h0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
#             self.empty_c0 = Parameter(torch.Tensor(args.rnn_layers, 1, args.embed_dim))
# 
#         self.topdown_left_embed = Parameter(torch.Tensor(2,  args.embed_dim))
#         self.topdown_right_embed = Parameter(torch.Tensor(2, args.embed_dim))
#         glorot_uniform(self)
#         self.num_layers = args.rnn_layers
#         self.embed_dim = args.embed_dim
# 
#         if self.bits_compress > 0:
#             self.bit_rep_net = BitsRepNet(args)
# 
#         if self.share_param:
#             self.m_l2r_cell = BinaryTreeLSTMCell(int(multiplier * args.embed_dim))
#             self.lr2p_cell = BinaryTreeLSTMCell(int(multiplier * args.embed_dim))
#             self.pred_has_ch = MLP(int(multiplier * args.embed_dim), [2 * args.embed_dim, 1])
#             self.m_pred_has_left = MLP(int(multiplier * args.embed_dim), [2 * args.embed_dim, 1])
#             self.m_pred_has_right = MLP(int(multiplier * args.embed_dim), [2 * args.embed_dim, 1])
#             #self.m_cell_topdown = nn.LSTMCell(args.embed_dim, args.embed_dim)
#             #self.m_cell_topright = nn.LSTMCell(args.embed_dim, args.embed_dim)
#             ## CHANGED HERE
#             self.m_cell_topdown = MultiLSTMCell(args.embed_dim, int(multiplier * args.embed_dim), args.rnn_layers)
#             self.m_cell_topright = MultiLSTMCell(args.embed_dim, int(multiplier * args.embed_dim), args.rnn_layers)
#         else:
#             fn_pred = lambda: MLP(args.embed_dim, [2 * args.embed_dim, 1])
#             fn_tree_cell = lambda: BinaryTreeLSTMCell(args.embed_dim)
#             fn_lstm_cell = lambda: nn.LSTMCell(args.embed_dim, args.embed_dim)
#             num_params = int(np.ceil(np.log2(args.max_num_nodes))) + 1
#             self.pred_has_ch = fn_pred()
# 
#             pred_modules = [[] for _ in range(2)]
#             tree_cell_modules = []
#             lstm_cell_modules = [[] for _ in range(2)]
#             for _ in range(num_params):
#                 for i in range(2):
#                     pred_modules[i].append(fn_pred())
#                     lstm_cell_modules[i].append(fn_lstm_cell())
#                 tree_cell_modules.append(fn_tree_cell())
# 
#             self.has_left_modules, self.has_right_modules = [nn.ModuleList(l) for l in pred_modules]
#             self.l2r_modules= nn.ModuleList(tree_cell_modules)
#             self.cell_topdown_modules, self.cell_topright_modules = [nn.ModuleList(l) for l in lstm_cell_modules]
#             self.lr2p_cell = fn_tree_cell()
#         self.row_tree = FenwickTree(args)
# 
#         if args.tree_pos_enc:
#             self.tree_pos_enc = PosEncoding(args.embed_dim, args.device, args.pos_base, bias=np.pi / 4)
#         else:
#             self.tree_pos_enc = lambda x: 0
# 
#     def cell_topdown(self, x, y, lv):
#         cell = self.m_cell_topdown if self.share_param else self.cell_topdown_modules[lv]
#         return cell(x, y)
# 
#     def cell_topright(self, x, y, lv):
#         cell = self.m_cell_topright if self.share_param else self.cell_topright_modules[lv]
#         return cell(x, y)
# 
#     def l2r_cell(self, x, y, lv):
#         cell = self.m_l2r_cell if self.share_param else self.l2r_modules[lv]
#         return cell(x, y)
# 
#     def pred_has_left(self, x, lv):
#         mlp = self.m_pred_has_left if self.share_param else self.has_left_modules[lv]
#         return mlp(x)
# 
#     def pred_has_right(self, x, lv):
#         mlp = self.m_pred_has_right if self.share_param else self.has_right_modules[lv]
#         return mlp(x)
# 
#     def get_empty_state(self):
#         if self.bits_compress:
#             return self.bit_rep_net([], 1)
#         else:
#             if self.method == "MLP-Leaf":
#                     dev = self.empty_h0.device
#                     #mask = torch.cat([torch.ones(1, self.embed_dim, device = dev), torch.zeros(1, int(self.embed_dim // 2), device = dev)], dim = -1)
#                     empty_h0 = torch.cat([self.empty_h0, torch.zeros(self.rnn_layers, 1, self.embed_dim // 2, device = dev)], dim = -1)
#                     empty_c0 = torch.cat([self.empty_h0, torch.zeros(self.rnn_layers, 1, self.embed_dim // 2, device = dev)], dim = -1)
#                     return (empty_h0, empty_c0)
#             return (self.empty_h0, self.empty_c0)
# 
#     def get_prob_fix(self, prob):
#         p = prob * (1 - self.greedy_frac)
#         if prob >= 0.5:
#             p += self.greedy_frac
#         return p
# 
#     def gen_row(self, ll, ll_wt, state, tree_node, col_sm, lb, ub, edge_feats=None, prev_wt_state=None):
#         assert lb <= ub
#         if tree_node.is_root:
#             prob_has_edge = torch.sigmoid(self.pred_has_ch(state[0][-1]))
# 
#             if col_sm.supervised:
#                 has_edge = len(col_sm.indices) > 0
#             else:
#                 has_edge = np.random.rand() < self.get_prob_fix(prob_has_edge.item())
#                 if ub == 0:
#                     has_edge = False
#                 if tree_node.n_cols <= 0:
#                     has_edge = False
#                 if lb:
#                     has_edge = True
#             if has_edge:
#                 ll = ll + torch.log(prob_has_edge)
#             else:
#                 ll = ll + torch.log(1 - prob_has_edge)
#             tree_node.has_edge = has_edge
#         else:
#             assert ub > 0
#             tree_node.has_edge = True
# 
#         if not tree_node.has_edge:  # an empty tree
#             return ll, ll_wt, self.get_empty_state(), 0, None, prev_wt_state
# 
#         if tree_node.is_leaf:
#             tree_node.bits_rep = [0]
#             col_sm.add_edge(tree_node.col_range[0])
#             if self.bits_compress:
#                 return ll, ll_wt, self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols), 1, None
#             else:
#                 if self.has_edge_feats:
#                     cur_feats = edge_feats[col_sm.pos - 1].unsqueeze(0) if col_sm.supervised else None
#                     if self.method != "LSTM":
#                         edge_ll, cur_feats = self.predict_edge_feats(state, cur_feats)
#                     else:
#                         edge_ll, cur_feats = self.predict_edge_feats(state, cur_feats, prev_wt_state[0][-1])
#                     ll_wt = ll_wt + edge_ll
#                     
#                     if self.method == "Test":
#                         return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, cur_feats, prev_wt_state
#                     
#                     elif self.method == "MLP-Leaf":
#                         edge_embed, prev_wt_state = self.embed_edge_feats(cur_feats, prev_state=prev_wt_state)
#                         return ll, ll_wt, edge_embed, 1, cur_feats, prev_wt_state
#                     
#                     elif self.method == "LSTM":
#                         edge_embed = self.embed_edge_feats(cur_feats, prev_state=prev_wt_state)
#                         prev_wt_state = edge_embed
#                         return ll, ll_wt, edge_embed, 1, cur_feats, prev_wt_state
#                     
#                     edge_embed = self.embed_edge_feats(cur_feats)
#                     return ll, ll_wt, edge_embed, 1, cur_feats, None
#                 else:
#                     return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, None, None
#         else:
#             tree_node.split()
# 
#             mid = (tree_node.col_range[0] + tree_node.col_range[1]) // 2
#             left_prob = torch.sigmoid(self.pred_has_left(state[0][-1], tree_node.depth))
# 
#             if col_sm.supervised:
#                 has_left = col_sm.next_edge < mid
#             else:
#                 has_left = np.random.rand() < self.get_prob_fix(left_prob.item())
#                 if ub == 0:
#                     has_left = False
#                 if lb > tree_node.rch.n_cols:
#                     has_left = True
#             ll = ll + (torch.log(left_prob) if has_left else torch.log(1 - left_prob))
#             left_pos = self.tree_pos_enc([tree_node.lch.n_cols])
#             state = self.cell_topdown(self.topdown_left_embed[[int(has_left)]] + left_pos, state, tree_node.depth)
#             pred_edge_feats = []
#             if has_left:
#                 lub = min(tree_node.lch.n_cols, ub)
#                 llb = max(0, lb - tree_node.rch.n_cols)
#                 ll, ll_wt, left_state, num_left, left_edge_feats, prev_wt_state = self.gen_row(ll, ll_wt, state, tree_node.lch, col_sm, llb, lub, edge_feats, prev_wt_state)
#                 pred_edge_feats.append(left_edge_feats)
#             else:
#                 left_state = self.get_empty_state()
#                 num_left = 0
# 
#             right_pos = self.tree_pos_enc([tree_node.rch.n_cols])
#             topdown_state = self.l2r_cell(state, (left_state[0] + right_pos, left_state[1] + right_pos), tree_node.depth)
#             
#             if self.has_edge_feats and self.method == "Test" and tree_node.lch.is_leaf and has_left:
#                 left_edge_embed = self.embed_edge_feats(left_edge_feats, prev_state=prev_wt_state)
#                 if self.update_left:
#                     topdown_state = self.topdown_update_wt(left_edge_embed, topdown_state)
#                     #left_state = self.update_wt(left_edge_embed, left_state)
#                     
#                 else:
#                     topdown_state = self.update_wt(left_edge_embed, topdown_state)
#             
#             rlb = max(0, lb - num_left)
#             rub = min(tree_node.rch.n_cols, ub - num_left)
#             if not has_left:
#                 has_right = True
#             else:
#                 right_prob = torch.sigmoid(self.pred_has_right(topdown_state[0][-1], tree_node.depth))
#                 if col_sm.supervised:
#                     has_right = col_sm.has_edge(mid, tree_node.col_range[1])
#                 else:
#                     has_right = np.random.rand() < self.get_prob_fix(right_prob.item())
#                     if rub == 0:
#                         has_right = False
#                     if rlb:
#                         has_right = True
#                 ll = ll + (torch.log(right_prob) if has_right else torch.log(1 - right_prob))
# 
#             topdown_state = self.cell_topright(self.topdown_right_embed[[int(has_right)]], topdown_state, tree_node.depth)
# 
#             if has_right:  # has edge in right child
#                 ll, ll_wt, right_state, num_right, right_edge_feats, prev_wt_state = self.gen_row(ll, ll_wt, topdown_state, tree_node.rch, col_sm, rlb, rub, edge_feats, prev_wt_state)
#                 pred_edge_feats.append(right_edge_feats)
#                 
# #                 if self.has_edge_feats and tree_node.rch.is_leaf:
# #                     right_edge_embed = self.embed_edge_feats(right_edge_feats, prev_state=prev_wt_state)
# #                     right_state = self.update_wt(right_edge_embed, right_state)
#                 
#             else:
#                 right_state = self.get_empty_state()
#                 num_right = 0
#             
#             if tree_node.col_range[1] - tree_node.col_range[0] <= self.bits_compress:
#                 summary_state = self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols)
#             else:
#                 summary_state = self.lr2p_cell(left_state, right_state)
#             if self.has_edge_feats and self.method == "Test":
#                 edge_feats = torch.cat(pred_edge_feats, dim=0)
#                 if has_left and tree_node.lch.is_leaf:
#                     left_edge_embed = self.embed_edge_feats(left_edge_feats, prev_state=prev_wt_state)
#                     summary_state = self.update_wt(left_edge_embed, summary_state)
#                 
#                 if has_right and tree_node.rch.is_leaf:
#                     right_edge_embed = self.embed_edge_feats(right_edge_feats, prev_state=prev_wt_state)
#                     summary_state = self.update_wt(right_edge_embed, summary_state)
#             
#             elif self.has_edge_feats:
#                 edge_feats = torch.cat(pred_edge_feats, dim=0)
#             return ll, ll_wt, summary_state, num_left + num_right, edge_feats, prev_wt_state
# 
#     def forward(self, node_end, edge_list=None, node_feats=None, edge_feats=None, node_start=0, list_states=[], lb_list=None, ub_list=None, col_range=None, num_nodes=None, display=False):
#         pos = 0
#         total_ll = 0.0
#         total_ll_wt = 0.0
#         edges = []
#         self.row_tree.reset(list_states)
#         controller_state = self.row_tree()
#         if num_nodes is None:
#             num_nodes = node_end
#         pbar = range(node_start, node_end)
#         if display:
#             pbar = tqdm(pbar)
#         list_pred_node_feats = []
#         list_pred_edge_feats = []
#         
#         prev_wt_state = None
#         if self.has_edge_feats and self.method == "LSTM":
#             prev_wt_state = (self.leaf_h0_wt, self.leaf_c0_wt)
#         
#         if self.has_edge_feats and self.method == "MLP-Leaf":
#             prev_wt_state = (self.wt_h0, self.wt_c0)
#         for i in pbar:
#             if edge_list is None:
#                 col_sm = ColAutomata(supervised=False)
#             else:
#                 indices = []
#                 while pos < len(edge_list) and i == edge_list[pos][0]:
#                     indices.append(edge_list[pos][1])
#                     pos += 1
#                 indices.sort()
#                 col_sm = ColAutomata(supervised=True, indices=indices)
# 
#             cur_row = AdjRow(i, self.directed, self.self_loop, col_range=col_range)
#             lb = 0 if lb_list is None else lb_list[i]
#             ub = cur_row.root.n_cols if ub_list is None else ub_list[i]
#             cur_pos_embed = self.row_tree.pos_enc([num_nodes - i])
#             controller_state = [x + cur_pos_embed for x in controller_state]
#             
#             if self.has_node_feats:
#                 target_node_feats = None if node_feats is None else node_feats[[i]]
#                 controller_state, ll_node, target_node_feats = self.predict_node_feats(controller_state, target_node_feats)
#                 total_ll = total_ll + ll_node
#                 list_pred_node_feats.append(target_node_feats)
#             if self.has_edge_feats:
#                 target_edge_feats = None if edge_feats is None else edge_feats[len(edges) : len(edges) + len(col_sm)]
#             else:
#                 target_edge_feats = None
#             ll, ll_wt, cur_state, _, target_edge_feats, prev_wt_state = self.gen_row(0, 0, controller_state, cur_row.root, col_sm, lb, ub, target_edge_feats, prev_wt_state)
#             
#             if target_edge_feats is not None and target_edge_feats.shape[0]:
#                 list_pred_edge_feats.append(target_edge_feats)
#             if self.has_node_feats:
#                 target_feat_embed = self.embed_node_feats(target_node_feats)
#                 cur_state = self.row_tree.node_feat_update(target_feat_embed, cur_state)
#             
#             assert lb <= len(col_sm.indices) <= ub
#             controller_state = self.row_tree(cur_state)
#             
#             if self.method == "Test" and cur_row.root.is_leaf and target_edge_feats is not None:
#                 edge_embed = self.embed_edge_feats(target_edge_feats, prev_state=prev_wt_state)
#                 controller_state = self.update_wt(edge_embed, controller_state)
#                 self.row_tree.list_states[1] = [controller_state]
#             
#             edges += [(i, x) for x in col_sm.indices]
#             total_ll = total_ll + ll
#             total_ll_wt = total_ll_wt + ll_wt
# 
#         if self.has_node_feats:
#             node_feats = torch.cat(list_pred_node_feats, dim=0)
#         if self.has_edge_feats:
#             edge_feats = torch.cat(list_pred_edge_feats, dim=0)
#         return total_ll, total_ll_wt, edges, self.row_tree.list_states, node_feats, edge_feats
# 
#     def binary_ll(self, pred_logits, np_label, need_label=False, reduction='sum'):
#         pred_logits = pred_logits.view(-1, 1)
#         label = torch.tensor(np_label, dtype=torch.float32).to(pred_logits.device).view(-1, 1)
#         loss = F.binary_cross_entropy_with_logits(pred_logits, label, reduction=reduction)
#         if need_label:
#             return -loss, label
#         return -loss
# 
#     def forward_row_trees(self, graph_ids, node_feats=None, edge_feats=None, list_node_starts=None, num_nodes=-1, list_col_ranges=None, noise=0.0, edge_feats_embed=None):
#         TreeLib.PrepareMiniBatch(graph_ids, list_node_starts, num_nodes, list_col_ranges)
#         # embed trees
#         all_ids = TreeLib.PrepareTreeEmbed()
#         if self.has_node_feats:
#             node_feats = self.embed_node_feats(node_feats)
#         
#         #if self.has_edge_feats:
#         #    if self.method == "LSTM":
#         #        #edge_feats, _ = self.embed_edge_feats(edge_feats, noise)
#         #        edge_feats = edge_feats_embed
#         #    
#         #    else:
#         #        edge_feats = self.embed_edge_feats(edge_feats, noise)
#         #
#         #else:
#         #    edge_feats = edge_feats_embed
#         
#         if not self.bits_compress:
#             ### CHANGED HERE
#             
#             if self.method == "MLP-Leaf":
#                 dev = self.empty_h0.device
#                 empty_h0 = torch.cat([self.empty_h0, torch.zeros(self.rnn_layers, 1, self.embed_dim // 2, device = dev)], dim = -1)
#                 empty_c0 = torch.cat([self.empty_c0, torch.zeros(self.rnn_layers, 1, self.embed_dim // 2, device = dev)], dim = -1)
#                 h_bot = torch.cat([empty_h0, self.leaf_h0], dim=1)
#                 c_bot = torch.cat([empty_c0, self.leaf_c0], dim=1)
#                 
#             
#             else: 
#                 h_bot = torch.cat([self.empty_h0, self.leaf_h0], dim=1)
#                 c_bot = torch.cat([self.empty_c0, self.leaf_c0], dim=1)
#             
#             fn_hc_bot = lambda d: (h_bot, c_bot)
#         else:
#             binary_embeds, base_feat = TreeLib.PrepareBinary()
#             fn_hc_bot = lambda d: (binary_embeds[d], binary_embeds[d]) if d < len(binary_embeds) else base_feat
#         max_level = len(all_ids) - 1
#         h_buf_list = [None] * (len(all_ids) + 1)
#         c_buf_list = [None] * (len(all_ids) + 1)
# 
#         for d in range(len(all_ids) - 1, -1, -1):
#             fn_ids = lambda i: all_ids[d][i]
#             if d == max_level:
#                 h_buf = c_buf = None
#             else:
#                 h_buf = h_buf_list[d + 1]
#                 c_buf = c_buf_list[d + 1]
#             h_bot, c_bot = fn_hc_bot(d + 1)
#             if self.has_edge_feats:
#                 edge_idx, is_rch = TreeLib.GetEdgeAndLR(d + 1)
#                 if self.method == "Test":
#                     local_edge_feats = edge_feats[edge_idx]
#                 else:
#                     local_edge_feats = (edge_feats[0][:, edge_idx], edge_feats[1][:, edge_idx])
#                 
#                 new_h, new_c = featured_batch_tree_lstm2(local_edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell, wt_update =self.update_wt, method = self.method)
#             else:
#                 new_h, new_c = batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell)
#             h_buf_list[d] = new_h
#             c_buf_list[d] = new_c
#         hc_bot = fn_hc_bot(0)
#         feat_dict = {}
#         if self.has_edge_feats:
#             edge_idx, is_rch = TreeLib.GetEdgeAndLR(0)
#             if self.method == "Test":
#                 local_edge_feats = edge_feats[edge_idx]
#             else:
#                 local_edge_feats = (edge_feats[0][:, edge_idx], edge_feats[1][:, edge_idx])
#             feat_dict['edge'] = (local_edge_feats, is_rch)
#         if self.has_node_feats:
#             is_tree_trivial = TreeLib.GetIsTreeTrivial()
#             new_h, new_c = self.row_tree.node_feat_update(node_feats[~is_tree_trivial], (new_h, new_c))
#             h_buf_list[0] = new_h
#             c_buf_list[0] = new_c
#             t_lch, t_rch = TreeLib.GetTrivialNodes()
#             feat_dict['node'] = (node_feats, is_tree_trivial, t_lch, t_rch)
#         if len(feat_dict):
#             hc_bot = (hc_bot, feat_dict)
#         return hc_bot, fn_hc_bot, h_buf_list, c_buf_list
# 
#     def forward_row_summaries(self, graph_ids, node_feats=None, edge_feats=None,
#                              list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None, noise=0.0):
#         hc_bot, _, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, node_feats, edge_feats,
#                                                                    list_node_starts, num_nodes, list_col_ranges, noise)
#         row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states)
#         return row_states, next_states
# 
#     def forward_train(self, graph_ids, node_feats=None, edge_feats=None,
#                       list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None, edge_feats_embed=None):
#         ll = 0.0
#         ll_wt = 0.0        
#         noise = 0.0
#         if self.has_edge_feats:
#             noise = 0.0#self.eps * torch.randn_like(edge_feats).to(edge_feats.device)
#         
#         if self.has_edge_feats:
#             if self.method == "LSTM":
#                 edge_feats_embed, state_h_prior = self.embed_edge_feats(edge_feats, noise)
#                 edge_feats = torch.cat(edge_feats, dim = 0)
#             
#             elif self.method == "MLP-Leaf":
#                 edge_feats_embed = self.embed_edge_feats(edge_feats, noise)
#                 edge_feats = torch.cat(edge_feats, dim = 0)
#             
#             else:
#                 edge_feats_embed = self.embed_edge_feats(edge_feats, noise)
#         
#         hc_bot, fn_hc_bot, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, node_feats, edge_feats_embed,
#                                                                            list_node_starts, num_nodes, list_col_ranges, noise)
#         row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states,  wt_update=self.update_wt, method=self.method)
#         
#         #print(row_states)
#         #print("Row States: ", row_states)
#         if self.has_node_feats:
#             row_states, ll_node_feats, _ = self.predict_node_feats(row_states, node_feats)
#             ll = ll + ll_node_feats
#         
#         logit_has_edge = self.pred_has_ch(row_states[0][-1])
#         has_ch, _ = TreeLib.GetChLabel(0, dtype=bool)
#         ll = ll + self.binary_ll(logit_has_edge, has_ch)
#         cur_states = (row_states[0][:, has_ch], row_states[1][:, has_ch])
# 
#         lv = 0
#         scale = 1
#         while True:
#             is_nonleaf = TreeLib.QueryNonLeaf(lv)
#             if self.has_edge_feats:
#                 edge_of_lv = TreeLib.GetEdgeOf(lv)
#                 edge_state = (cur_states[0][:, ~is_nonleaf], cur_states[1][:, ~is_nonleaf])
#                 target_feats = edge_feats[edge_of_lv]
#                 prior_h_target = None
#                 if self.method == "LSTM": 
#                     prior_h_target = state_h_prior[edge_of_lv, :]
#                 edge_ll, _ = self.predict_edge_feats(edge_state, target_feats, prior_h_target)
#                 ll_wt = ll_wt + edge_ll #/ len(edge_feats.flatten())
#             if is_nonleaf is None or np.sum(is_nonleaf) == 0:
#                 break
#             cur_states = (cur_states[0][:, is_nonleaf], cur_states[1][:, is_nonleaf])
#             left_logits = self.pred_has_left(cur_states[0][-1], lv)
#             has_left, num_left = TreeLib.GetChLabel(-1, lv)
#             left_update = self.topdown_left_embed[has_left] + self.tree_pos_enc(num_left)
#             left_ll, float_has_left = self.binary_ll(left_logits, has_left, need_label=True, reduction='sum')
#             ll = ll + scale * left_ll
# 
#             cur_states = self.cell_topdown(left_update, cur_states, lv)
# 
#             left_ids = TreeLib.GetLeftRootStates(lv)
#             h_bot, c_bot = fn_hc_bot(lv + 1)
#             if lv + 1 < len(h_buf_list):
#                 h_next_buf, c_next_buf = h_buf_list[lv + 1], c_buf_list[lv + 1]
#             else:
#                 h_next_buf = c_next_buf = None
#             if self.has_edge_feats:
#                 edge_idx, is_rch = TreeLib.GetEdgeAndLR(lv + 1)
#                 if self.method == "Test": 
#                     left_feats = edge_feats_embed[edge_idx[~is_rch]]
#                 else:
#                     left_feats = (edge_feats_embed[0][:, edge_idx[~is_rch]], edge_feats_embed[1][:, edge_idx[~is_rch]])
#                 h_bot, c_bot = h_bot[:, left_ids[0]], c_bot[:, left_ids[0]]
#                 if self.method != "Test":
#                     h_bot, c_bot = selective_update_hc(h_bot, c_bot, left_ids[0], left_feats)
#                 left_wt_ids = left_ids[1][list(map(bool, left_ids[0]))]
#                 left_ids = tuple([None] + list(left_ids[1:]))
# 
#             left_subtree_states = tree_state_select(h_bot, c_bot,
#                                           h_next_buf, c_next_buf,
#                                                     lambda: left_ids)
#             
#             has_right, num_right = TreeLib.GetChLabel(1, lv)
#             right_pos = self.tree_pos_enc(num_right)
#             left_subtree_states = [x + right_pos for x in left_subtree_states]
#             topdown_state = self.l2r_cell(cur_states, left_subtree_states, lv)
#             
#             if self.has_edge_feats and self.method == "Test" and len(left_wt_ids) > 0:
#                 leaf_topdown_states = (topdown_state[0][:, left_wt_ids], topdown_state[1][:, left_wt_ids])
#                 
#                 if self.update_left:
#                     leaf_topdown_states = self.topdown_update_wt(left_feats, leaf_topdown_states)
#                 
#                 else:
#                     leaf_topdown_states = self.update_wt(left_feats, leaf_topdown_states)
#                 topdown_state[0][:, left_wt_ids] = leaf_topdown_states[0]
#                 topdown_state[1][:, left_wt_ids] = leaf_topdown_states[1]
#                         
# 
#             right_logits = self.pred_has_right(topdown_state[0][-1], lv)
#             right_update = self.topdown_right_embed[has_right]
#             topdown_state = self.cell_topright(right_update, topdown_state, lv)
#             right_ll = self.binary_ll(right_logits, has_right, reduction='none') * float_has_left
#             ll = ll + scale * torch.sum(right_ll)
#             lr_ids = TreeLib.GetLeftRightSelect(lv, np.sum(has_left), np.sum(has_right))
#             new_states = []
#             for i in range(2):
#                 new_s = multi_index_select([lr_ids[0], lr_ids[2]], [lr_ids[1], lr_ids[3]],
#                                             cur_states[i], topdown_state[i])
#                 new_states.append(new_s)
#             cur_states = tuple(new_states)
#             
#             lv += 1
# 
#         return ll, ll_wt, next_states
# 
