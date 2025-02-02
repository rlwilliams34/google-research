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
import numpy as np
torch.set_printoptions(threshold=10_000)
np.set_printoptions(threshold=10_000)



## RUN LSTM THROUGH EACH OF THESE SEQUENCES
## THEN WILL NEED SOME INDEXING THAT GRABS EACH GET FROM THE CORRECT LIST


## HELPER FUNCTIONS FOR FENWICK TREE WEIGHTS
def get_list_edge(cur_nedge_list):
    offset = 0
    list_edge = []
    for nedge in cur_nedge_list:
        nedge2 = nedge - nedge % 2
        list_edge += list(range(offset, nedge2 + offset))
        offset += nedge
    return list_edge

def get_list_indices(nedge_list):
    '''Retries list of indices for states of batched graphs'''
    max_lv = int(np.log(max(nedge_list)) / np.log(2))
    list_indices = []
    list_edge = get_list_edge(nedge_list)
    cur_nedge_list = nedge_list
    empty = np.array([], dtype=np.int32)
    for lv in range(max_lv):
        left = list_edge[0::2]
        right = list_edge[1::2]
        cur_nedge_list = [x // 2 for x in cur_nedge_list]
        list_edge = get_list_edge(cur_nedge_list)
        list_indices.append([(empty, empty, np.array(left, dtype=np.int32), np.array(range(len(left)), dtype = np.int32), empty, empty), (empty, empty, np.array(right, dtype=np.int32), np.array(range(len(right)), dtype=np.int32), empty, empty)])
    return list_indices

####
def lv_offset(num_edges, max_lv = -1):
    offset_list = []
    lv = 0
    while num_edges >= 1:
        offset_list.append(num_edges)
        num_edges = num_edges // 2
        lv += 1
    
    if max_lv > 0:
        offset_list = np.pad(offset_list, (0, max_lv - len(offset_list)), 'constant', constant_values=0)
    num_entries = np.sum(offset_list)
    return offset_list, num_entries

## Note number of entries per graph's set of edges will be sum of the lv offset list

# def lv_list(k, n):
#     offset, _ = lv_offset(n)
#     lv = 0
#     lv_list = []
#     for i in range(len(bin(k)[2:])):
#         if k & 2**i == 2**i:
#             lv_list += [int(k // 2**i + np.sum(offset[:i])) - 1]
#     return lv_list

# def lv_list(k, offset):
#     lv = 0
#     lv_list = []
#     for i in range(len(bin(k)[2:])):
#         if k & 2**i == 2**i:
#             lv_list += [int(k // 2**i + np.sum(offset[:i])) - 1]
#     return lv_list

def lv_list(k, list_offset, batch_id):
    offset = list_offset[batch_id]
    lv_list = []
    for i in range(len(bin(k)[2:])):
        if k & 2**i == 2**i:
            offset_tot = np.sum([np.sum(l[:i]) for l in list_offset])
            val = int(k // 2**i + offset_tot - 1)
            offset_batch = np.sum([l[i] for l in list_offset[:batch_id] if len(l) >= i])
            val += offset_batch
            lv_list += [int(val)]
    return lv_list

def batch_lv_list(k, list_offset):
    lv_list = []
    for i in range(len(bin(k)[2:])):
        if k & 2**i == 2**i:
            offset_tot = np.sum(list_offset[:, :i])
            val = int(k // 2**i + offset_tot - 1)
            offset_batch = np.cumsum([0] + [l[i] for l in list_offset[:-1]])
            offset_batch = offset_batch[list_offset[:,0] >= k]
            val = val + offset_batch
            lv_list.append(val)
    lv_list = np.stack(lv_list, axis = 1)
    return lv_list


def get_batch_lv_list_fast(list_num_edges): 
    list_offset = []
    max_lv = int(np.max([np.log(e)/np.log(2) for e in list_num_edges]) + 1)
    list_offset = np.array([lv_offset(num_edges, max_lv)[0] for num_edges in list_num_edges])
    
    max_edge = np.max(list_num_edges)
    batch_size = len(list_num_edges)
    out = np.empty((batch_size,), object)
    out.fill([])
    
    for k in range(1, max_edge+1):
        cur = (k <= np.array(list_num_edges))
        cur_lvs = batch_lv_list(k, list_offset)
        i = 0
        for batch, cur_it in enumerate(cur):
            if cur_it:
                out[batch] = out[batch] + [cur_lvs[i].tolist()]
                i += 1
    return out.tolist()

# def get_batch_lv_list(list_num_edges): ### SLOWDOWN CULPRIT!!!!!
#     batch_id = 0
#     list_offset = []
#     for num_edges in list_num_edges:
#         offset, _ = lv_offset(num_edges)
#         list_offset += [offset]
#     ## THIS SECTION NEEDS TO BE FASTER!
#     out = [] 
#     for num_edges in list_num_edges: #Get rid of this and do things by batch instead...
#         vals = []
#         for k in range(1, num_edges + 1):
#             cur = lv_list(k, list_offset, batch_id)
#             vals.append(cur)
#         out.append(vals)
#         batch_id += 1
#     return out


def flatten(xss):
    return [x for xs in xss for x in xs]

def prepare_batch(batch_lv_in):
    batch_size = len(batch_lv_in)
    list_num_edges = [len(lv_in) for lv_in in batch_lv_in]
    tot_num_edges = np.sum(list_num_edges)
    flat_lv_in = flatten(batch_lv_in)
    list_lvs = [[len(l) for l in lv_in] for lv_in in batch_lv_in]
    flat_list_lvs = flatten(list_lvs)
    max_len = np.max([np.max(l) for l in list_lvs])
    all_ids = []
    init_select = flatten([[x[0] for x in batch_lv_in[i]] for i in range(batch_size)])
    last_tos = [j for j in range(len(flat_lv_in)) if flat_list_lvs[j] == max_len]
    lv = 1
    while True:
        done_from = [j for j in range(len(flat_lv_in)) if len(flat_lv_in[j]) == 1]
        done_to = [j for j in range(tot_num_edges) if flat_list_lvs[j] == lv]
        proceed_from = [j for j in range(len(flat_lv_in)) if len(flat_lv_in[j]) > 1]
        proceed_input = [l[1] for l in flat_lv_in if len(l) > 1]
        all_ids.append((done_from, done_to, proceed_from, proceed_input))
        flat_lv_in = [l[1:] for l in flat_lv_in if len(l) > 1]
        lv += 1
        if max([len(l) for l in flat_lv_in]) <= 1:
            break
    return init_select, all_ids, last_tos

# def prepare(lv_in):
#     num_edges = len(lv_in)
#     lvs = [len(l) for l in lv_in]
#     max_len = np.max(lvs)
#     all_ids = []
#     init_select = [x[0] for x in lv_in]
#     last_tos = [j for j in range(num_edges) if len(lv_in[j]) == max_len]
#     lv = 1
#     while True:
#         if lv_in == []:
#             break
#         done_from = [j for j in range(len(lv_in)) if len(lv_in[j]) == 1]
#         done_to = [j for j in range(num_edges) if lvs[j] == lv]
#         proceed_from = [j for j in range(len(lv_in)) if len(lv_in[j]) > 1]
#         proceed_input = [l[1] for l in lv_in if len(l) > 1]
#         all_ids.append((done_from, done_to, proceed_from, proceed_input))
#         lv_in= [l[1:] for l in lv_in if len(l) > 1]
#         lv += 1
#         if max([len(l) for l in lv_in]) <= 1:
#             break
#     return init_select, all_ids, last_tos
####

# cur_lv_offsets = [l[lv] if len(l) >= lv+1 else 0 for l in all_lv]
# 
# def prepare(lv_in, batch_lvs):
#     num_edges = len(lv_in)
#     lvs = [len(l) for l in lv_in]
#     max_len = np.max(lvs)
#     all_ids = []
#     init_select = [x[0] for x in lv_in]
#     last_tos = [j for j in range(num_edges) if len(lv_in[j]) == max_len]
#     lv = 1
#     while True:
#         done_from = [j for j in range(len(lv_in)) if len(lv_in[j]) == 1]
#         done_to = [j for j in range(num_edges) if lvs[j] == lv]
#         proceed_from = [j for j in range(len(lv_in)) if len(lv_in[j]) > 1]
#         proceed_input = [l[1] for l in lv_in if len(l) > 1]
#         all_ids.append((done_from, done_to, proceed_from, proceed_input))
#         lv_in= [l[1:] for l in lv_in if len(l) > 1]
#         lv += 1
#         if max([len(l) for l in lv_in]) <= 1:
#             break
#     return init_select, all_ids, last_tos
# 



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



def featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_all_ids, cell, t_lch=None, t_rch=None, cell_node=None, method=None, func=None, weight_state=None, edge_embed_idx=None):
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
        if edge_feats is not None and method not in ["Test75", "Test85", "Special"]:
            local_hbot, local_cbot = selective_update_hc(local_hbot, local_cbot, leaf_check, edge_feats[i])
        if cell_node is not None:
            local_hbot, local_cbot = cell_node(node_feats[i], (local_hbot, local_cbot))
        h_vecs, c_vecs = tree_state_select(local_hbot, local_cbot, h_buf, c_buf, lambda : new_ids[i])
        
        if method == "Special" and np.sum(leaf_check) > 0:
            new_local_hbot, new_local_cbot = func((h_bot[:, 1:2].repeat(1, np.sum(leaf_check), 1), c_bot[:, 1:2].repeat(1, np.sum(leaf_check), 1)), weight_state)
            h_vecs[:, new_ids[i][1][leaf_check == 1]] = new_local_hbot
            c_vecs[:, new_ids[i][1][leaf_check == 1]] = new_local_cbot
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


def featured_batch_tree_lstm3(feat_dict, h_bot, c_bot, h_buf, c_buf, h_past, c_past, fn_all_ids, cell, cell_node, method, func=None, weight_state=None):
    edge_feats = is_rch = None
    t_lch = t_rch = None
    if 'edge' in feat_dict:
        edge_feats, is_rch = feat_dict['edge']
    if 'node' in feat_dict:
        t_lch, t_rch = feat_dict['node']
    if h_past is None:
        return featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, lambda i: fn_all_ids(i)[:-2], cell, t_lch, t_rch, cell_node, method, func, weight_state)
    elif h_bot is None:
        return batch_tree_lstm2(h_buf, c_buf, h_past, c_past, lambda i: fn_all_ids(i)[2:], cell)
    elif h_buf is None:
        return featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_past, c_past, lambda i: fn_all_ids(i)[0, 1, 4, 5], cell, t_lch, t_rch, cell_node, method, func, weight_state)
    else:
        raise NotImplementedError  #TODO: handle model parallelism with features


class FenwickTree(nn.Module):
    def __init__(self, args):
        super(FenwickTree, self).__init__()
        self.method = args.method
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

    def forward(self, new_state=None, print_it=False):
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

    def forward_train(self, h_bot, c_bot, h_buf0, c_buf0, prev_rowsum_h, prrev_rowsum_c, weight_state=None, func=None):
        # embed row tree
        tree_agg_ids = TreeLib.PrepareRowEmbed()
        row_embeds = [(self.init_h0, self.init_c0)]
        if self.method not in ["Test75", "Test85"] and (self.has_edge_feats or self.has_node_feats):
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
        
        for i,(x,y) in enumerate(row_embeds):
            if len(x.shape) == 2:
                new_embed = (x.unsqueeze(0), y.unsqueeze(0))
                row_embeds[i] = new_embed
            print(row_embeds[i][0].shape)
        
        
        for i, all_ids in enumerate(tree_agg_ids):
            fn_ids = lambda x: all_ids[x]
            lstm_func = batch_tree_lstm3
            if self.method == "Test75" or self.method == "Test85" or not self.has_edge_feats:
                has_edge_feats = False
            
            if i == 0 and self.method not in ["Test75", "Test85"] and (has_edge_feats or self.has_node_feats):
                lstm_func = featured_batch_tree_lstm3
            lstm_func = partial(lstm_func, h_buf=row_embeds[-1][0], c_buf=row_embeds[-1][1],
                                h_past=prev_rowsum_h, c_past=prrev_rowsum_c, fn_all_ids=fn_ids, cell=self.merge_cell)
            if i == 0:
                if has_edge_feats or self.has_node_feats:
                    new_states = lstm_func(feat_dict, h_bot, c_bot, cell_node=None if not self.has_node_feats else self.node_feat_update, method=self.method, func=func, weight_state=weight_state)
                else:
                    new_states = lstm_func(h_bot, c_bot)
            else:
                new_states = lstm_func(None, None)
            row_embeds.append(new_states)
        
        h_list, c_list = zip(*row_embeds)
        joint_h = torch.cat(h_list, dim=1)
        joint_c = torch.cat(c_list, dim=1)

        # get history representation
        init_select, all_ids, last_tos, next_ids, pos_info = TreeLib.PrepareRowSummary()
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
            hist_froms.append(done_from)
            hist_tos.append(done_to)
            hist_rnn_states.append(cur_state)

            next_input = joint_h[:, proceed_input], joint_c[:, proceed_input]
            sub_state = cur_state[0][:, proceed_from], cur_state[1][:, proceed_from]
            cur_state = self.summary_cell(sub_state, next_input)
        
        hist_rnn_states.append(cur_state)
        hist_froms.append(None)
        hist_tos.append(last_tos)
        hist_h_list, hist_c_list = zip(*hist_rnn_states)
        pos_embed = self.pos_enc(pos_info)
        row_h = multi_index_select(hist_froms, hist_tos, *hist_h_list) + pos_embed
        row_c = multi_index_select(hist_froms, hist_tos, *hist_c_list) + pos_embed
        return (row_h, row_c), ret_state

    def forward_train_weights(self, edge_feats_init_embed, list_num_edges, db_info):
        # embed row tree
        if db_info is None:
            list_indices = get_list_indices(list_num_edges)
        
        else:
            list_indices = db_info[0]
        edge_embeds = [edge_feats_init_embed]
        
        for i, all_ids in enumerate(list_indices):
            fn_ids = lambda x: all_ids[x]
            new_states = batch_tree_lstm3(None, None, h_buf=edge_embeds[-1][0], c_buf=edge_embeds[-1][1], h_past=None, c_past=None, fn_all_ids=fn_ids, cell=self.merge_cell)
            edge_embeds.append(new_states)
        h_list, c_list = zip(*edge_embeds)
        joint_h = torch.cat(h_list, dim=1)
        joint_c = torch.cat(c_list, dim=1)

        # get history representation
        if db_info is None:
            batch_lv_list = get_batch_lv_list_fast(list_num_edges)
            init_select, all_ids, last_tos = prepare_batch(batch_lv_list)
        
        else:
            init_select, all_ids, last_tos = db_info[1] 
        
        cur_state = (joint_h[:, init_select], joint_c[:, init_select])
        
        hist_rnn_states = []
        hist_froms = []
        hist_tos = []
        for i, (done_from, done_to, proceed_from, proceed_input) in enumerate(all_ids):
            hist_froms.append(done_from)
            hist_tos.append(done_to)
            hist_rnn_states.append(cur_state)

            next_input = joint_h[:, proceed_input], joint_c[:, proceed_input]
            sub_state = cur_state[0][:, proceed_from], cur_state[1][:, proceed_from]
            cur_state = self.summary_cell(sub_state, next_input)
        hist_rnn_states.append(cur_state)
        hist_froms.append(None)
        hist_tos.append(last_tos)
        hist_h_list, hist_c_list = zip(*hist_rnn_states)
        edge_h = multi_index_select(hist_froms, hist_tos, *hist_h_list)
        edge_c = multi_index_select(hist_froms, hist_tos, *hist_c_list)
        edge_embeddings = (edge_h, edge_c)
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
        self.batch_size = args.batch_size
        self.self_loop = args.self_loop
        self.bits_compress = args.bits_compress
        self.has_edge_feats = args.has_edge_feats
        self.has_node_feats = args.has_node_feats
        self.method = args.method
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

    def get_empty_state(self, update_state=False):
        if self.bits_compress:
            return self.bit_rep_net([], 1)
        else:
            return (self.empty_h0, self.empty_c0)

    def get_prob_fix(self, prob):
        p = prob * (1 - self.greedy_frac)
        if prob >= 0.5:
            p += self.greedy_frac
        return p
    
    def get_merged_prob(self, top_state, wt_state, prob_func):
        if wt_state is None:
            prob = torch.sigmoid(prob_func(top_state[0][-1]))
            return prob
        
        if self.add_states:
            scale = torch.sigmoid(self.scale_tops)
            state_update = scale * top_state[0][-1] + (1 - scale) * wt_state[0][-1]
            prob = torch.sigmoid(prob_func(state_update))
            return prob
        
        if self.wt_one_layer:
            state_update = self.update_wt((top_state[0][-1:], top_state[1][-1:]), wt_state)
        else:
            state_update = self.update_wt(top_state, wt_state)
        
        prob = torch.sigmoid(prob_func(state_update[0][-1]))
        return prob

    def gen_row(self, ll, ll_wt, state, tree_node, col_sm, lb, ub, edge_feats=None, row=None, prev_state=None, num_nodes=None):
        assert lb <= ub
        if tree_node.is_root:
            prob_has_edge = self.get_merged_prob(state, prev_state, self.pred_has_ch)
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
                return ll, self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols), 1, None, None
            else:
                if self.has_edge_feats:
                    cur_feats = edge_feats[col_sm.pos - 1].unsqueeze(0) if col_sm.supervised else None
                    col = tree_node.col_range[0]
                    #rc = np.array([row * (row - 1) // 2 + col]).reshape(1, 1)
                    rc = np.array([row, col], dtype = np.float32).reshape(1, 2)
                    if prev_state is not None:
                        if self.add_states:
                            scale = torch.sigmoid(self.scale_wts)
                            state_update = [[scale * state[0][-1] + (1 - scale) * prev_state[0][-1]], None]
                        
                        elif self.wt_one_layer:
                            state_update = self.update_wt((state[0][-1:], state[1][-1:]), prev_state)
                        
                        else:
                            state_update = self.update_wt(state, prev_state)
                        edge_ll, _, cur_feats = self.predict_edge_feats(state_update, cur_feats)                    
                    else:
                        edge_ll, _, cur_feats = self.predict_edge_feats(state, cur_feats)
                    ll_wt = ll_wt + edge_ll
                    
                    if self.method in ["Test75", "Test85"]:
                        edge_embed = self.embed_edge_feats(cur_feats, prev_state=prev_state, rc=rc)
                        prev_state = edge_embed
                        self.num_edge += 1
                        return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, cur_feats, prev_state
                    
                    else:
                        edge_embed = self.embed_edge_feats(cur_feats, prev_state=prev_state)
                        prev_state = edge_embed
                        return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, cur_feats, prev_state
                            
                    return ll, ll_wt, edge_embed, 1, cur_feats, prev_state, None
                    
                else:
                    return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, None, None
        else:
            tree_node.split()
            mid = (tree_node.col_range[0] + tree_node.col_range[1]) // 2
            left_prob = self.get_merged_prob(state, prev_state, prob_func = partial(self.pred_has_left, lv = tree_node.depth))

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
                ll, ll_wt, left_state, num_left, left_edge_feats, prev_state = self.gen_row(ll, ll_wt, state, tree_node.lch, col_sm, llb, lub, edge_feats, row=row, prev_state=prev_state, num_nodes=num_nodes)
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
                right_prob = self.get_merged_prob(topdown_state, prev_state, prob_func = partial(self.pred_has_right, lv = tree_node.depth))
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
                ll, ll_wt, right_state, num_right, right_edge_feats, prev_state = self.gen_row(ll, ll_wt, topdown_state, tree_node.rch, col_sm, rlb, rub, edge_feats, row=row, prev_state=prev_state, num_nodes=num_nodes)
                pred_edge_feats.append(right_edge_feats)
            else:
                right_state = self.get_empty_state()
                joint_right_state = self.get_empty_state()
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
        if cmd_args.has_edge_feats and self.method in ["Test285", "Test286", "Test287", "Test288", "Test75", "Test85"]:
            self.weight_tree.reset([])
        
        if num_nodes is None:
            num_nodes = node_end
        pbar = range(node_start, node_end)
        if display:
            pbar = tqdm(pbar)
        list_pred_node_feats = []
        list_pred_edge_feats = []
        
        prev_state = None        
        if self.method in cmd_args.has_edge_feats and ["Test75", "Test85"]:
            prev_state =  None #self.weight_tree()
        
        self.num_edge = 0
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
            ll, ll_wt, cur_state, _, target_edge_feats, prev_state = self.gen_row(0, 0, controller_state, cur_row.root, col_sm, lb, ub, target_edge_feats, row=i, prev_state=prev_state, num_nodes=num_nodes)
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

#         print("Final Prev State: ", prev_state[0][-1, -1, :])
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

    def forward_row_trees(self, graph_ids, node_feats=None, edge_feats=None, list_node_starts=None, num_nodes=-1, list_col_ranges=None, batch_last_edges=None):
        TreeLib.PrepareMiniBatch(graph_ids, list_node_starts, num_nodes, list_col_ranges)
        # embed trees
        if self.method in ["Test75", "Test85"]:
            edge_feats=None
        all_ids = TreeLib.PrepareTreeEmbed()
        if self.has_node_feats:
            node_feats = self.embed_node_feats(node_feats)

        if not self.bits_compress:
            empty_h0, empty_c0 = self.get_empty_state()
            h_bot = torch.cat([empty_h0, self.leaf_h0], dim=1)
            c_bot = torch.cat([empty_c0, self.leaf_c0], dim=1)
            fn_hc_bot = lambda d: (h_bot, c_bot)
        
        else:
            binary_embeds, base_feat = TreeLib.PrepareBinary()
            fn_hc_bot = lambda d: (binary_embeds[d], binary_embeds[d]) if d < len(binary_embeds) else base_feat
        max_level = len(all_ids) - 1
        h_buf_list = [None] * (len(all_ids) + 1)
        c_buf_list = [None] * (len(all_ids) + 1)
        
        if self.method in ["Test75", "Test85"]:
            left_idx, right_idx = TreeLib.GetMostRecentWeight(len(all_ids) + 1, batch_last_edges=batch_last_edges)
            topdown_edge_index = (left_idx, right_idx)
        
        for d in range(len(all_ids) - 1, -1, -1):
            fn_ids = lambda i: all_ids[d][i]
            if d == max_level:
                h_buf = c_buf = None
            else:
                h_buf = h_buf_list[d + 1]
                c_buf = c_buf_list[d + 1]
            h_bot, c_bot = fn_hc_bot(d + 1)
            if self.has_edge_feats and self.method not in ["Test75", "Test85"]:
                edge_idx, is_rch = TreeLib.GetEdgeAndLR(d + 1)
                local_edge_feats = (edge_feats[0][:, edge_idx], edge_feats[1][:, edge_idx])
                new_h, new_c = featured_batch_tree_lstm2(local_edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell, method=self.method)
            else:
                new_h, new_c = batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell)
            h_buf_list[d] = new_h
            c_buf_list[d] = new_c
        hc_bot = fn_hc_bot(0)
        feat_dict = {}
        if self.has_edge_feats and self.method not in ["Test75", "Test85"]:
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
        
        if self.method in ["Test75", "Test85"]:
            return hc_bot, fn_hc_bot, h_buf_list, c_buf_list, topdown_edge_index
        return hc_bot, fn_hc_bot, h_buf_list, c_buf_list
    
    def merge_states(self, update_idx, top_states, edge_feats_embed, predict_top=True):
        if self.add_states:
            if predict_top:
                scale = torch.sigmoid(self.scale_tops)
                update_bool = (update_idx != -1)
                cur_edge_idx = update_idx[update_bool]
            else:
                scale = torch.sigmoid(self.scale_wts)
                update_bool = update_idx[0]
                edge_of_lv = update_idx[1]
                cur_edge_idx = edge_of_lv[update_bool] - 1
            
            cur_top_h, cur_top_c = top_states[0].clone(), top_states[1].clone()
            top_states_wt = (cur_top_h, cur_top_c)
            top_has_wt_states = (top_states_wt[0][:, update_bool], top_states_wt[1][:, update_bool])
            row_feats = (edge_feats_embed[0][:, cur_edge_idx], edge_feats_embed[1][:, cur_edge_idx])
            top_has_wt_states_h = scale * top_has_wt_states[0] + (1 - scale) * row_feats[0]
            top_states_wt[0][:, update_bool] = top_has_wt_states_h
            return top_states_wt[0], None
        
#         dev = edge_feats_embed[0].device
#         if predict_top:
#             update_bool = (update_idx != -1)
#             edge_update_idx = torch.from_numpy(update_idx[update_bool]).to(dev)
#             update_bool = torch.from_numpy(update_bool).to(dev)
#             
#         else:
#             update_bool = update_idx[0]
#             edge_update_idx = torch.tensor(update_idx[1][update_bool] - 1, dtype = torch.int64).to(dev)
#             update_bool = torch.from_numpy(update_bool).to(dev)
#         
#         update_bool = update_bool.reshape(1, update_bool.shape[0], 1)
#         edge_update_idx = edge_update_idx[..., None].expand(edge_feats_embed[0].size(0), -1, edge_feats_embed[0].size(2)).long()
#         edge_feats = [torch.gather(x, 1, edge_update_idx) for x in edge_feats_embed]
#         top_has_wt_states = [torch.masked_select(x, update_bool) for x in top_states]
#         top_has_wt_states = [x.reshape(self.num_layers, edge_update_idx.shape[1], self.embed_dim) for x in top_has_wt_states]
#         top_no_wt_h = torch.masked_select(top_states[0], ~update_bool)
#         top_has_wt_states_h, _ = self.update_wt((top_has_wt_states[0], top_has_wt_states[1]), edge_feats)
#         top_states_wt = torch.masked_scatter(torch.zeros_like(top_states[0]), update_bool, top_has_wt_states_h)
#         top_states_wt = torch.masked_scatter(top_states_wt, ~update_bool, top_no_wt_h)
#         return top_states_wt, None
        
        if predict_top:
            update_bool = (update_idx != -1)
            cur_edge_idx = update_idx[update_bool]
        else:
            update_bool = update_idx[0]
            edge_of_lv = update_idx[1]
            cur_edge_idx = edge_of_lv[update_bool] - 1
        
        if self.wt_one_layer:
            cur_top_h, cur_top_c = top_states[0][-1:].clone(), top_states[1][-1:].clone()

        else:
            cur_top_h, cur_top_c = top_states[0].clone(), top_states[1].clone()
        top_states_wt = (cur_top_h, cur_top_c)
        top_has_wt_states = (top_states_wt[0][:, update_bool], top_states_wt[1][:, update_bool])
        edge_feats = (edge_feats_embed[0][:, cur_edge_idx], edge_feats_embed[1][:, cur_edge_idx])
        top_has_wt_states_h, _ = self.update_wt(top_has_wt_states, edge_feats)
        top_states_wt[0][:, update_bool] = top_has_wt_states_h
        return top_states_wt[0], _
        
    def forward_row_summaries(self, graph_ids, node_feats=None, edge_feats=None,
                             list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None):
        hc_bot, _, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, node_feats, edge_feats,
                                                                   list_node_starts, num_nodes, list_col_ranges)
        row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states)
        return row_states, next_states

    def forward_train(self, graph_ids, node_feats=None, edge_feats=None,
                      list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None, batch_idx=None, list_num_edges=None, db_info=None, list_last_edge=None, edge_feats_lstm=None, batch_last_edges=None,rc=None):
        ll = 0.0
        ll_wt = 0.0
        noise = 0.0
        ll_batch = (None if batch_idx is None else np.zeros(len(np.unique(batch_idx))))
        ll_batch_wt = (None if batch_idx is None else np.zeros(len(np.unique(batch_idx))))
        edge_feats_embed = None
        
        if self.has_edge_feats and self.sigma > 0:
            noise = torch.randn_like(edge_feats).to(edge_feats.device)
            edge_feats = edge_feats * torch.exp(self.sigma * noise)
        
        if rc is not None:
            if len(rc.shape) == 3:
                rc = rc.reshape(rc.shape[0], rc.shape[2])
        
        if list_num_edges is not None:
            first_edge = [0]
            for i in range(len(list_num_edges) - 1):
                first_edge += [first_edge[-1] + list_num_edges[i]]
        
        if self.has_edge_feats:
            if self.row_LSTM:
                edge_feats_embed = self.embed_edge_feats(edge_feats, sigma=self.sigma, list_num_edges=list_num_edges, db_info=db_info,edge_feats_lstm=edge_feats_lstm)
            else:
                edge_feats_embed = self.embed_edge_feats(edge_feats, sigma=self.sigma, list_num_edges=list_num_edges, db_info=db_info, rc=rc)
        
#         print("Edge feats embed: ", edge_feats_embed[0][-1, -1, :])
        if self.has_edge_feats and self.method in ["Test75", "Test85"]:
            hc_bot, fn_hc_bot, h_buf_list, c_buf_list, topdown_edge_index = self.forward_row_trees(graph_ids, node_feats, edge_feats_embed, list_node_starts, num_nodes, list_col_ranges, batch_last_edges)
            row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states, None, None)
        
        else:
            hc_bot, fn_hc_bot, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, node_feats, edge_feats_embed, list_node_starts, num_nodes, list_col_ranges)
            row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states)
        if self.has_node_feats:
            row_states, ll_node_feats, _ = self.predict_node_feats(row_states, node_feats)
            ll = ll + ll_node_feats
            
        ## HERE WE NEED TO ADD AN UPDATE USING MOST. RECENT. EDGE...
        if self.has_edge_feats and self.method in ["Test75", "Test85"]:
            row_states_wt = self.merge_states(batch_last_edges, row_states, edge_feats_embed)
            logit_has_edge = self.pred_has_ch(row_states_wt[0][-1])
        
        else:
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
                has_prev = np.array([k not in first_edge for k in edge_of_lv])
#                 print("Edge of lv", edge_of_lv)
#                 print("Has prev", has_prev)
                if self.method in ["Test75", "Test85"] and np.sum(has_prev) > 0:
                    edge_state_wt = self.merge_states([has_prev, edge_of_lv], edge_state, edge_feats_embed, False)
                    edge_ll, ll_batch_wt, _ = self.predict_edge_feats(edge_state_wt, target_feats, batch_idx = cur_batch_idx, ll_batch_wt = ll_batch_wt)
                else:
                    edge_ll, ll_batch_wt, _ = self.predict_edge_feats(edge_state, target_feats, batch_idx = cur_batch_idx, ll_batch_wt = ll_batch_wt)
                
                ll_wt = ll_wt + edge_ll
            if is_nonleaf is None or np.sum(is_nonleaf) == 0:
                break
            cur_states = (cur_states[0][:, is_nonleaf], cur_states[1][:, is_nonleaf])
            
            if batch_idx is not None:
                batch_idx = batch_idx[is_nonleaf]        
            
            if self.has_edge_feats and self.method in ["Test75", "Test85"]:
                cur_left_updates = topdown_edge_index[0][lv]
                cur_states_wt = self.merge_states(cur_left_updates, cur_states, edge_feats_embed)
                left_logits = self.pred_has_left(cur_states_wt[0][-1], lv)
            else:
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
                h_bot, c_bot = h_bot[:, left_ids[0]], c_bot[:, left_ids[0]]
                if self.method not in ["Test75", "Test85"]:
                    edge_idx, is_rch = TreeLib.GetEdgeAndLR(lv + 1)
                    left_feats = (edge_feats_embed[0][:, edge_idx[~is_rch]], edge_feats_embed[1][:, edge_idx[~is_rch]])
                    h_bot, c_bot = selective_update_hc(h_bot, c_bot, left_ids[0], left_feats)
                left_ids = tuple([None] + list(left_ids[1:]))
            
            left_subtree_states = tree_state_select(h_bot, c_bot,
                                                    h_next_buf, c_next_buf,
                                                    lambda: left_ids)
            
            has_right, num_right = TreeLib.GetChLabel(1, lv)
            right_pos = self.tree_pos_enc(num_right)
            left_subtree_states = [x + right_pos for x in left_subtree_states]
            topdown_state = self.l2r_cell(cur_states, left_subtree_states, lv)
            
            if self.has_edge_feats and self.method in ["Test75", "Test85"]:
                cur_right_updates = topdown_edge_index[1][lv]
                topdown_wt_state = self.merge_states(cur_right_updates, topdown_state, edge_feats_embed)
                right_logits = self.pred_has_right(topdown_wt_state[0][-1], lv)
            
            else:
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



































