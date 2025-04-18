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
import cmd
from copyreg import pickle
# pylint: skip-file

import os
import sys
import numpy as np
import pickle as cp
import networkx as nx
import random
from tqdm import tqdm
import torch
import torch.optim as optim
from collections import OrderedDict
from bigg.common.configs import cmd_args, set_device
from bigg.extension.customized_models import BiggWithEdgeLen, BiggWithGCN
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from bigg.evaluation.graph_stats import *
from bigg.evaluation.mmd import *
from bigg.evaluation.mmd_stats import *
from bigg.experiments.train_utils import get_node_dist
from torch.nn import functional as F

import gc
torch.cuda.empty_cache()
gc.collect()

## HELPER FUNCTIONS FOR ROW LSTM
def get_max_deg(train_graphs):
    max_degrees = []
    for G in train_graphs:
        degrees = [deg for (node, deg) in G.degree()]
        max_degrees.append(np.max(degrees))
    return np.max(degrees)

def get_edge_feats_lstm(g, max_deg=-1, offset=0):
    lstm_lens = [len([k for _, k, in g.edges(i) if k < i]) for i in list(g.nodes())]
    if max_deg==-1:
        max_deg = np.max(lstm_lens)
    #print("Max Degree: ", max_deg)
    list_of_edge_feats = []
    init_edge = []
    num_edge = 0
    for i in range(len(g.nodes())):
        x = list(g.edges(i, data=True))
        x = sorted(x, key= lambda y:y[1])
        weights = [x[2]['weight'] for x in x if x[0] > x[1]]
        num_new_edge = np.sum([x[0] > x[1] for x in x])
        if num_new_edge > 0:
            weights = np.pad(np.array(weights), (0, max_deg - len(weights)), 'constant', constant_values=-1)
            list_of_edge_feats.append(weights[:, np.newaxis])
    return np.concatenate(list_of_edge_feats, -1)
### To pad: F.pad(input=g1t, pad=(0,0,0,MAX_DEG - SHAPE1),mode='constant',value=-1).shape



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

def batch_lv_list1(k, list_offset):
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
        cur_lvs = batch_lv_list1(k, list_offset)
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

def GCNN_batch_train_graphs(train_graphs, batch_indices, cmd_args):
    batch_g = nx.Graph()
    feat_idx = torch.Tensor().to(cmd_args.device)
    batch_weight_idx = []
    edge_list = []
    offset = 0
    for idx in batch_indices:
        g = train_graphs[idx]
        n = len(g)
        feat_idx = torch.cat([feat_idx, torch.arange(n).to(cmd_args.device)])
        for e1, e2, w in g.edges(data=True):
            batch_weight_idx.append((int(e1), int(e2), w['weight']))
            edge_list.append((int(e1) + offset, int(e2) + offset, idx))
        offset += n
    edge_idx = torch.Tensor(edge_list).to(cmd_args.device).t()
    batch_weight_idx = torch.Tensor(batch_weight_idx).to(cmd_args.device)
    return feat_idx, edge_idx, batch_weight_idx


def t(n1, n2):
    r = max(n1, n2)
    c = min(n1, n2)
    t = r * (r - 1) // 2 + c
    return t


def get_edge_feats(g, method=None):
    edges = sorted(g.edges(data=True), key=lambda x: t(x[0], x[1]))
    weights = [x[2]['weight'] for x in edges]
    return np.expand_dims(np.array(weights, dtype=np.float32), axis=1)


def get_rc(g):
    num_nodes = len(g.nodes())
    edges = sorted(g.edges(data=True), key=lambda x: t(x[0], x[1]))
    #rc = [[num_nodes - x[0], num_nodes - x[1]] for x in edges]
    rc = [[max([x[0], x[1]]), min([x[0], x[1]])] for x in edges]
    #rc = [t(x[0], x[1]) for x in edges]
    return np.expand_dims(np.array(rc, dtype=np.float32), axis=1)



def get_last_edge(g):
    last_edges = []
    last_edges_1 = []
    idx = -1
    idx_count = -1
    for r in sorted(g.nodes()):
        neighbors = [n for n in list(g.neighbors(r)) if n < r]
        idx_count += len(neighbors)
        if len(neighbors) > 0:
            c = max(neighbors)
            idx = idx_count
            if r == 1:
                last_edges_1.append(idx)
            elif r > 1:
                last_edges.append(idx)
    return np.array(last_edges), np.array(last_edges_1)


def get_last_edge2(g):
    last_edges = []
    last_edges_1 = []
    idx = -1
    idx_count = -1
    for r in sorted(g.nodes()):
        neighbors = [n for n in list(g.neighbors(r)) if n < r]
        idx_count += len(neighbors)
        if len(neighbors) > 0:
            c = max(neighbors)
            idx = idx_count
            if r == 1:
                last_edges_1.append(idx)
            last_edges.append(idx)
        else:
            if r == 0:
                last_edges.append(-1)
            else:
                last_edges.append(last_edges[-1])
    
    last_edges = [-1] + last_edges[:-1]
    return np.array(last_edges)



def get_last_edge_2(g):
    last_edges = []
    last_edges_1 = []
    idx = -1
    idx_count = -1
    for r in sorted(g.nodes()):
        neighbors = [n for n in list(g.neighbors(r)) if n < r]
        idx_count += len(neighbors)
        if len(neighbors) > 0:
            c = max(neighbors)
            idx = idx_count
            if r == 1:
                last_edges_1.append(idx)
            last_edges.append(idx)
        else:
            if r == 0:
                last_edges.append(-1)
            else:
                last_edges.append(last_edges[-1])
    
    last_edges = [-1] + last_edges[:-1]
    return np.array(last_edges)


def debug_model(model, graph, node_feats, edge_feats, method=None, info=None,edge_feats_lstm=None,batch_last_edges=None,rc=None):
    ll_t1 = 0
    ll_w1 = 0
    ll_t2 = 0
    ll_w2 = 0
    
    if len(graph[0]) < 25:
        print(graph[0].edges())
        print(graph[1].edges())
    
    for i in range(0, 2):
        g = graph[i]
        edge_feats_i = (edge_feats[i] if edge_feats is not None else None)
        edges = []
        for e in g.edges():
            if e[1] > e[0]:
                e = (e[1], e[0])
            edges.append(e)
        edges = sorted(edges)
        
        if edge_feats_i is not None and not torch.is_tensor(edge_feats_i):
            edge_feats_i = edge_feats_i[0]
        
        with torch.no_grad():
            ll, ll_wt, _, _, _, _ = model(len(g), edges, node_feats=node_feats, edge_feats=edge_feats_i)
        ll_t2 = ll + ll_t2
        ll_w2 = ll_wt + ll_w2
    
    
    if edge_feats_lstm is not None:
        list_num_edges = [len(edge_feats[0]), len(edge_feats[1])]
        
        if isinstance(edge_feats, list):
            edge_feats = torch.cat(edge_feats, dim = 0)
    
    else:
        list_num_edges = None
        if cmd_args.method in ["Test285", "Test286", "Test287", "Test288", "Test75", "Test85"]:
            list_num_edges = [len(edge_feats[0]), len(edge_feats[1])]
        
        if isinstance(edge_feats, list):
            edge_feats = torch.cat(edge_feats, dim = 0)
        
        elif method == "Test4":
            print("Neeed to implement")
    
    #print(info)
    
    with torch.no_grad():
        ll_t1, ll_w1, _, _, _ = model.forward_train([0, 1], node_feats=node_feats, edge_feats=edge_feats, list_num_edges=list_num_edges, list_last_edge=info, edge_feats_lstm=edge_feats_lstm, batch_last_edges=batch_last_edges,rc=rc)
    
    print("=============================")
    print("Fast Code Top+Wt Likelihoods: ")
    print(ll_t1)
    print(ll_w1)
    print("Slow Code Top+Wt Likelihoods: ")
    print(ll_t2)
    print(ll_w2)
    print("=============================")
    
    diff1 = abs(ll_t1 - ll_t2)
    diff2 = abs(ll_w1 - ll_w2)

    print("Absolute Differences: ")
    print("diff top: ", diff1)
    print("diff weight: ", diff2)
    print("=============================")
    
    rel1 = (ll_t1 - ll_t2) / (ll_t1 + 1e-15)
    rel2 = (ll_w1 - ll_w2) / (ll_w1 + 1e-15)
    
    print("Relative Differences (%): ")
    print("rel diff top: ", rel1 * 100)
    print("rel diff weight: ", rel2 * 100)
    
    import sys
    sys.exit()


# def debug_model(model, graph, node_feats, edge_feats, method=None, info=None):
#     ll_t1 = 0
#     ll_w1 = 0
#     ll_t2 = 0
#     ll_w2 = 0
#     
#     for i in range(0, 1):
#         g = graph[i]
#         edge_feats_i = (edge_feats[i] if edge_feats is not None else None)
#         edges = []
#         for e in g.edges():
#             if e[1] > e[0]:
#                 e = (e[1], e[0])
#             edges.append(e)
#         edges = sorted(edges)
#         
#         if edge_feats_i is not None and not torch.is_tensor(edge_feats_i):
#             edge_feats_i = edge_feats_i[0]
#         
#         ll, ll_wt, _, _, _, _ = model(len(g), edges, node_feats=node_feats, edge_feats=edge_feats_i)
#         ll_t2 = ll + ll_t2
#         ll_w2 = ll_wt + ll_w2
#     
#     list_num_edges = None
#     if cmd_args.method in ["Test285", "Test286", "Test287", "Test288", "Test75"]:
#         #list_num_edges = [len(edge_feats[0]), len(edge_feats[1])]
#         list_num_edges = [len(edge_feats[0])]
#     
#     if isinstance(edge_feats, list):
#         #edge_feats = torch.cat(edge_feats, dim = 0)
#         edge_feats = edge_feats[0]
#     
#     #print(info)
#     
#     ll_t1, ll_w1, _, _, _ = model.forward_train([0], node_feats=node_feats, edge_feats=edge_feats, list_num_edges=list_num_edges, list_last_edge=info)
#     
#     print("=============================")
#     print("Fast Code Top+Wt Likelihoods: ")
#     print(ll_t1)
#     print(ll_w1)
#     print("Slow Code Top+Wt Likelihoods: ")
#     print(ll_t2)
#     print(ll_w2)
#     print("=============================")
#     
#     diff1 = abs(ll_t1 - ll_t2)
#     diff2 = abs(ll_w1 - ll_w2)
# 
#     print("Absolute Differences: ")
#     print("diff top: ", diff1)
#     print("diff weight: ", diff2)
#     print("=============================")
#     
#     rel1 = (ll_t1 - ll_t2) / (ll_t1 + 1e-15)
#     rel2 = (ll_w1 - ll_w2) / (ll_w1 + 1e-15)
#     
#     print("Relative Differences (%): ")
#     print("rel diff top: ", rel1 * 100)
#     print("rel diff weight: ", rel2 * 100)
#     
#     import sys
#     sys.exit()





if __name__ == '__main__':
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)
    assert cmd_args.blksize < 0  # assume graph is not that large, otherwise model parallelism is needed
    
    if cmd_args.debug:
        cmd_args.batch_size = 2
    
    if cmd_args.g_type == "db":
        import pickle5 as cp
    db_info = None
    path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'train')
    
    with open(path, 'rb') as f:
        train_graphs = cp.load(f)
    
    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    cmd_args.max_num_nodes = max_num_nodes
    
    print(train_graphs[1].edges())
    print(train_graphs[1].edges(data=True))
    
    if cmd_args.phase == "train": 
        [TreeLib.InsertGraph(g) for g in train_graphs]
        list_node_feats = ([torch.from_numpy(get_node_feats(g)).to(cmd_args.device) for g in train_graphs] if cmd_args.has_node_feats else None)
        list_edge_feats = None
        if cmd_args.has_edge_feats:
            if cmd_args.row_LSTM:
                list_edge_feats_lstm = [torch.from_numpy(get_edge_feats_lstm(g)).to(cmd_args.device) for g in train_graphs]
            
            list_edge_feats = [torch.from_numpy(get_edge_feats(g, cmd_args.method)).to(cmd_args.device) for g in train_graphs]
            list_last_edges = [get_last_edge2(g) for g in train_graphs]
            list_rc = [get_rc(g) for g in train_graphs]
            
            ### To pad: F.pad(input=g1t, pad=(0,0,0,MAX_DEG - SHAPE1),mode='constant',value=-1).shape

            last_edge_list = None
            if cmd_args.has_edge_feats and cmd_args.method in ["Test75", "Test85"]:
                last_edge_list = [get_last_edge(g)[0] for g in train_graphs]
                last_edge_1_list = [get_last_edge(g)[1] for g in train_graphs]
            
            if cmd_args.has_edge_feats and cmd_args.g_type == "db":
                list_num_edges = [len(g.edges()) for g in train_graphs]
                db_info = []
                for num_edges in list_num_edges:
                    info1 = get_list_indices([num_edges])
                    batch_lv_list = get_batch_lv_list_fast([num_edges])
                    info2 = prepare_batch(batch_lv_list)
                    db_info += [(info1, info2)]
            
            elif cmd_args.has_edge_feats and  cmd_args.g_type == "tree" and max_num_nodes == 199:
                list_num_edges = [len(train_graphs[0].edges())] * cmd_args.batch_size
                info1 = get_list_indices(list_num_edges)
                batch_lv_list = get_batch_lv_list_fast(list_num_edges)
                info2 = prepare_batch(batch_lv_list)
                db_info = (info1, info2)
        print('# graphs', len(train_graphs), 'max # nodes', max_num_nodes)
    
    if cmd_args.model == "BiGG_GCN":
        cmd_args.has_edge_feats = False
        cmd_args.has_node_feats = False
        model = BiggWithGCN(cmd_args).to(cmd_args.device)
        cmd_args.has_edge_feats = True
    else:
        model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
    
    if cmd_args.g_type == "db" or cmd_args.g_type == "tree" or cmd_args.g_type == "er":
        optimizer = optim.AdamW(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-3)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    
    
    if cmd_args.model_dump is not None and os.path.isfile(cmd_args.model_dump):
        print('loading from', cmd_args.model_dump)
        checkpoint = torch.load(cmd_args.model_dump)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = cmd_args.learning_rate
    
    #print("LIST RC")
    #(list_rc)
    #########################################################################################################
    if cmd_args.phase == 'validate':
        # get num nodes dist
        print("Now generating sampled graphs...")
        num_node_dist = get_node_dist(train_graphs)
        path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'val')
        with open(path, 'rb') as f:
            val_graphs = cp.load(f)
        print('# val graphs', len(val_graphs))
        k=1
        gen_graphs = []
        
        with torch.no_grad():
            model.eval()
            for _ in tqdm(range(cmd_args.num_test_gen)):
                num_nodes = np.argmax(np.random.multinomial(1, num_node_dist)) 
                
                _, _, pred_edges, _, pred_node_feats, pred_edge_feats = model(node_end = num_nodes, display=cmd_args.display)
                
                if cmd_args.model == "BiGG_GCN":
                    fix_edges = []
                    for e1, e2 in pred_edges:
                        if e1 > e2:
                            fix_edges.append((e2, e1))
                        else:
                            fix_edges.append((e1, e2))
                    pred_edge_tensor = torch.tensor(fix_edges).to(cmd_args.device)
                    pred_weighted_tensor = model.gcn_mod.sample(num_nodes, pred_edge_tensor)
                    pred_weighted_tensor = pred_weighted_tensor.cpu().detach().numpy()
                    
                    weighted_edges = []
                    for e1, e2, w in pred_weighted_tensor:
                        weighted_edges.append((int(e1), int(e2), np.round(w.item(), 4)))
                    
                    pred_g = nx.Graph()
                    pred_g.add_weighted_edges_from(weighted_edges)
                    gen_graphs.append(pred_g)
                
                elif cmd_args.has_edge_feats:
                    weighted_edges = []
                    for e, w in zip(pred_edges, pred_edge_feats):
                        assert e[0] > e[1]
                        weighted_edges.append((e[1], e[0], np.round(w.item(), 4)))
                    pred_g = nx.Graph()
                    pred_g.add_weighted_edges_from(weighted_edges)
                    gen_graphs.append(pred_g)
                
                else:
                    pred_g = nx.Graph()
                    fixed_edges = []
                    for e in pred_edges:
                        w = 1.0
                        if e[0] < e[1]:
                            edge = (e[0], e[1], w)
                        else:
                            edge = (e[1], e[0], w)
                        fixed_edges.append(edge)
                    pred_g.add_weighted_edges_from(fixed_edges)
                    gen_graphs.append(pred_g)
        
        print("Generating Graph Validation Stats")
        get_graph_stats(gen_graphs, val_graphs, cmd_args.g_type)
        sys.exit()
        
    
    elif cmd_args.phase == 'test':
        # get num nodes dist
        print("Now generating sampled graphs...")
        num_node_dist = get_node_dist(train_graphs)
        path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'test')
        with open(path, 'rb') as f:
            gt_graphs = cp.load(f)
        print('# gt graphs', len(gt_graphs))
        
        ##
        g = train_graphs[0]
        degree_list = [0] * cmd_args.num_leaves + [2] * (cmd_args.num_leaves - 1)
        ##
#         
#         print("Training graphs MMD Check")
#         get_graph_stats(train_graphs, gt_graphs, cmd_args.g_type)
#         print("Test graphs MMD Check")
#         gt_graphs2 = [gt_graphs[8 - i] for i in range(9)]
#         get_graph_stats(gt_graphs2, gt_graphs, cmd_args.g_type)
        
        gen_graphs = []
        with torch.no_grad():
            model.eval()
            k = 1
            for _ in tqdm(range(cmd_args.num_test_gen)):
                if k == 1 and cmd_args.add_states:
                    print("Info from Scale Tops: ", torch.sigmoid(model.scale_tops))
                    print("Info from Scale Wts: ", torch.sigmoid(model.scale_wts))
                    k += 1
                num_nodes = np.argmax(np.random.multinomial(1, num_node_dist)) 
                _, _, pred_edges, _, pred_node_feats, pred_edge_feats = model(node_end = num_nodes, display=cmd_args.display)
                #_, _, pred_edges, _, pred_node_feats, pred_edge_feats = model(node_end = num_nodes, lb_list = degree_list, ub_list = degree_list, display=cmd_args.display)
                for e in pred_edges:
                    assert e[0] > e[1]
                
                if cmd_args.model == "BiGG_GCN":
                    fix_edges = []
                    for e1, e2 in pred_edges:
                        if e1 > e2:
                            fix_edges.append((e2, e1))
                        else:
                            fix_edges.append((e1, e2))
                    pred_edge_tensor = torch.tensor(fix_edges).to(cmd_args.device)
                    pred_weighted_tensor = model.gcn_mod.sample(num_nodes, pred_edge_tensor)
                    pred_weighted_tensor = pred_weighted_tensor.cpu().detach().numpy()
                    
                    weighted_edges = []
                    for e1, e2, w in pred_weighted_tensor:
                        weighted_edges.append((int(e1), int(e2), np.round(w.item(), 4)))
                    
                    pred_g = nx.Graph()
                    pred_g.add_weighted_edges_from(weighted_edges)
                    gen_graphs.append(pred_g)
                
                elif cmd_args.has_edge_feats:
                    weighted_edges = []
                    for e, w in zip(pred_edges, pred_edge_feats):
                        assert e[0] > e[1]
                        weighted_edges.append((e[1], e[0], np.round(w.item(), 4)))
                    pred_g = nx.Graph()
                    pred_g.add_weighted_edges_from(weighted_edges)
                    gen_graphs.append(pred_g)
                
                else:
                    pred_g = nx.Graph()
                    fixed_edges = []
                    for e in pred_edges:
                        w = 1.0
                        if e[0] < e[1]:
                            edge = (e[0], e[1], w)
                        else:
                            edge = (e[1], e[0], w)
                        fixed_edges.append(edge)
                    pred_g.add_weighted_edges_from(fixed_edges)
                    gen_graphs.append(pred_g)
        
        if cmd_args.max_num_nodes > -1:
            for idx in range(min(2, cmd_args.num_test_gen)):
                print("edges:")
                print(gen_graphs[idx].edges(data=True))
        
        print(cmd_args.g_type)
        print("Generating Graph Stats")
        get_graph_stats(gen_graphs, gt_graphs, cmd_args.g_type)
        
        save_graphs = False
        if save_graphs:
            print('saving graphs')
            with open(cmd_args.model_dump + '.graphs-%s' % str(cmd_args.greedy_frac), 'wb') as f:
                cp.dump(gen_graphs, f, cp.HIGHEST_PROTOCOL)
            print('graph generation complete')
            sys.exit()
    #########################################################################################################
    
    top_losses = []
    wt_losses = []
    times = []
    loss_times = []
    epoch_list = []
    lr_scheduler = {'lobster': 100, 'tree': 200 , 'db': 2000, 'er': 250, 'span': 500, 'franken': 200}
    epoch_lr_decrease = lr_scheduler[cmd_args.g_type]
    db_info_it = None
    
    if cmd_args.epoch_plateu > -1:
        epoch_lr_decrease = cmd_args.epoch_plateu
    
    if cmd_args.g_type == 'db':
        offset_val = 500
    
    else:
        offset_val = 100
    
    batch_loss = 0.0
    sigma_t = 1.0
    sigma_w = 1.0
    
    N = len(train_graphs)
    B = cmd_args.batch_size
    indices = list(range(N))
    num_iter = int(N / B)
    
    num_node_dist = get_node_dist(train_graphs)
    grad_accum_counter = 0
    optimizer.zero_grad()
    prev = datetime.now()
    
    if cmd_args.epoch_load is None:
        cmd_args.epoch_load = 0
    
    if cmd_args.schedule:
        pass
    elif cmd_args.g_type == 'db':
        cmd_args.scale_loss = 1
    else:
        cmd_args.scale_loss = 1
    
    if cmd_args.epoch_load >= epoch_lr_decrease:
        cmd_args.learning_rate = 1e-4
        cmd_args.scale_loss = 1 * cmd_args.scale_loss
        for param_group in optimizer.param_groups:
                param_group['lr'] = cmd_args.learning_rate
        if cmd_args.epoch_load >= epoch_lr_decrease + offset_val:
            cmd_args.learning_rate = 1e-5
            cmd_args.scale_loss = 1 * cmd_args.scale_loss
            for param_group in optimizer.param_groups:
                param_group['lr'] = cmd_args.learning_rate
    
    print("Current Learning Rate is: ", cmd_args.learning_rate)
    print("Dividing Weight Loss by: ", cmd_args.scale_loss)
    
    model.train()
    
    ### DEBUG
    if cmd_args.debug:
        if cmd_args.has_edge_feats and cmd_args.method == "LSTM":
            debug_model(model, [train_graphs[0], train_graphs[1]], None, [list_edge_feats[i] for i in [0,1]], False)
        
        elif cmd_args.method in ["Test75", "Test85"]:
            if cmd_args.row_LSTM:
                batch_indices = [0, 1]
                edge_feats_lstm = [list_edge_feats_lstm[i] for i in batch_indices]
                max_len = np.max([x.shape[0] for x in edge_feats_lstm])
                edge_feats_lstm = [F.pad(input=x, pad = (0, 0, 0, max_len - x.shape[0]), mode='constant',value=-1) for x in edge_feats_lstm]
                edge_feats_lstm = torch.cat(edge_feats_lstm, dim = -1)
                edge_feats = [get_edge_feats(train_graphs[0]), get_edge_feats(train_graphs[1])]
                edge_feats = [torch.from_numpy(x).to(cmd_args.device) for x in edge_feats]
                
                batch_indices = [0, 1]
                list_last_edge = [last_edge_list[i] for i in batch_indices]
                list_last_edge_1 = [last_edge_1_list[i] for i in batch_indices]
                list_offsets = [len(list_edge_feats[i]) for i in batch_indices]
                offset = 0
                for k in range(len(list_last_edge)):
                    list_last_edge_k = list_last_edge[k]
                    list_last_edge_1_k = list_last_edge_1[k]
                    offset_list_last_edge_k = [k + offset if k > -1 else 0 for k in list_last_edge_k]
                    offset_list_last_edge_1_k = [k + offset if k > -1 else 0 for k in list_last_edge_1_k]
                    offset += list_offsets[k]
                    list_last_edge[k] = np.array(offset_list_last_edge_k)
                    list_last_edge_1[k] = np.array(offset_list_last_edge_1_k)
                
                list_last_edge = np.concatenate(list_last_edge, axis = 0)
                list_last_edge_1 = np.concatenate(list_last_edge_1, axis = 0)
                
                
                
                last_edge_1_idx = []
                id_ = 1
                for b in batch_indices:
                    last_edge_1_idx.append(id_)
                    id_ += len(train_graphs[b])
                list_last_edge_1 = [list_last_edge_1, np.array(last_edge_1_idx)]
                list_last_edge = (list_last_edge, list_last_edge_1)    
                rc = np.concatenate([list_rc[i] for i in batch_indices], axis=0)
                
                debug_model(model, [train_graphs[0], train_graphs[1]], None,edge_feats, True, info=list_last_edge, edge_feats_lstm=edge_feats_lstm, rc=rc)
                
            else:
                batch_indices = [0, 1]
                list_last_edge = [last_edge_list[i] for i in batch_indices]
                list_last_edge_1 = [last_edge_1_list[i] for i in batch_indices]
                list_offsets = [len(list_edge_feats[i]) for i in batch_indices]
                offset = 0
                for k in range(len(list_last_edge)):
                    list_last_edge_k = list_last_edge[k]
                    list_last_edge_1_k = list_last_edge_1[k]
                    offset_list_last_edge_k = [k + offset if k > -1 else 0 for k in list_last_edge_k]
                    offset_list_last_edge_1_k = [k + offset if k > -1 else 0 for k in list_last_edge_1_k]
                    offset += list_offsets[k]
                    list_last_edge[k] = np.array(offset_list_last_edge_k)
                    list_last_edge_1[k] = np.array(offset_list_last_edge_1_k)
                
                list_last_edge = np.concatenate(list_last_edge, axis = 0)
                list_last_edge_1 = np.concatenate(list_last_edge_1, axis = 0)
                
                
                
                last_edge_1_idx = []
                id_ = 1
                for b in batch_indices:
                    last_edge_1_idx.append(id_)
                    id_ += len(train_graphs[b])
                list_last_edge_1 = [list_last_edge_1, np.array(last_edge_1_idx)]
                list_last_edge = (list_last_edge, list_last_edge_1)    
                
                edge_feats = ([list_edge_feats[i] for i in [0,1]] if cmd_args.has_edge_feats else None)
                
                batch_last_edges = [list_last_edges[i] for i in batch_indices]
                offset = 0
                for b in range(len(batch_last_edges)):
                    batch_last_edges[b] = np.array([x + offset if x != -1 else x for x in batch_last_edges[b]])
                    offset += len(train_graphs[batch_indices[b]].edges())
                batch_last_edges = np.concatenate(batch_last_edges)
                
                rc = np.concatenate([list_rc[i] for i in batch_indices], axis=0)
                debug_model(model, [train_graphs[0], train_graphs[1]], None, edge_feats, True, info=list_last_edge, batch_last_edges=batch_last_edges, rc=rc)
                
        
        else:
            edge_feats = ([list_edge_feats[i] for i in [0,1]] if cmd_args.has_edge_feats else None)
            debug_model(model, [train_graphs[0], train_graphs[1]], None, edge_feats, True)
        
    loss_tops_list = np.zeros(cmd_args.num_epochs - cmd_args.epoch_load)
    loss_wts_list = np.zeros(cmd_args.num_epochs - cmd_args.epoch_load)
    list_last_edge = None
    
    for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
        tot_loss = 0.0
        pbar = tqdm(range(num_iter))
        random.shuffle(indices)
        epoch_losses_t = []
        epoch_losses_w = []
        batch_idx = None
        list_num_edges = None
        edge_feats_lstm = None
        
        if epoch == 0 and cmd_args.has_edge_feats and cmd_args.model == "BiGG_E":
            for i in range(len(list_edge_feats)):
                edge_feats = list_edge_feats[i]
                model.update_weight_stats(edge_feats)
        
        if cmd_args.model == "BiGG_GCN":
            model.gcn_mod.epoch_num += 1
        
        else:
            model.epoch_num += 1
        
        if epoch >= epoch_lr_decrease and cmd_args.learning_rate == 1e-3:
            cmd_args.learning_rate = cmd_args.learning_rate / 10
            cmd_args.scale_loss = cmd_args.scale_loss * 1
            print("Lowering Larning Rate to: ", cmd_args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cmd_args.learning_rate
        
        elif epoch >= epoch_lr_decrease + offset_val and cmd_args.learning_rate == 1e-4:
            cmd_args.learning_rate = cmd_args.learning_rate / 10
            cmd_args.scale_loss = cmd_args.scale_loss * 1
            print("Lowering Larning Rate to: ", cmd_args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cmd_args.learning_rate
        edge_feats_embed = None
        epoch_loss_top = 0.0
        epoch_loss_wt = 0.0        
        
        for idx in pbar:
            start = B * idx
            stop = B * (idx + 1)
            batch_indices = indices[start:stop]
            batch_indices = np.sort(batch_indices)
            
            num_nodes = sum([len(train_graphs[i]) for i in batch_indices])
            
            node_feats = (torch.cat([list_node_feats[i] for i in batch_indices], dim=0) if list_node_feats is not None else None)
            
            if cmd_args.method in ["Test12", "MLP-Leaf"] and cmd_args.has_edge_feats:
                edge_feats = [list_edge_feats[i] for i in batch_indices]
            
            else:
                edge_feats = (torch.cat([list_edge_feats[i] for i in batch_indices], dim=0) if list_edge_feats is not None else None)
                rc = (np.concatenate([list_rc[i] for i in batch_indices], axis=0) if cmd_args.has_edge_feats else None)
                batch_last_edges = None
                list_last_edge = None
                
                if cmd_args.has_edge_feats and cmd_args.method in ["Test285", "Test286", "Test287", "Test288", "Test75", "Test85"]:
                    list_num_edges = [len(train_graphs[i].edges()) for i in batch_indices]
                    if db_info is not None:
                        if cmd_args.g_type == "db":
                            i = batch_indices[0]
                            db_info_it = db_info[i]
                        elif cmd_args.g_type == "tree":
                            db_info_it = db_info
                    if cmd_args.method in ["Test75", "Test85"]:
                        if cmd_args.row_LSTM:
                            edge_feats_lstm = [list_edge_feats_lstm[i] for i in batch_indices]
                            max_len = np.max([x.shape[0] for x in edge_feats_lstm])
                            edge_feats_lstm = [F.pad(input=x, pad = (0, 0, 0, max_len - x.shape[0]), mode='constant',value=-1) for x in edge_feats_lstm]
                            edge_feats_lstm = torch.cat(edge_feats_lstm, dim = -1)
                        
                        list_last_edge = [last_edge_list[i] for i in batch_indices]
                        list_last_edge_1 = [last_edge_1_list[i] for i in batch_indices]
                        list_offsets = [len(list_edge_feats[i]) for i in batch_indices]
                        offset = 0
                        for k in range(len(list_last_edge)):
                            list_last_edge_k = list_last_edge[k]
                            list_last_edge_1_k = list_last_edge_1[k]
                            offset_list_last_edge_k = [k + offset if k > -1 else 0 for k in list_last_edge_k]
                            offset_list_last_edge_1_k = [k + offset if k > -1 else 0 for k in list_last_edge_1_k]
                            offset += list_offsets[k]
                            list_last_edge[k] = np.array(offset_list_last_edge_k)
                            list_last_edge_1[k] = np.array(offset_list_last_edge_1_k)
                        
                        list_last_edge = np.concatenate(list_last_edge, axis = 0)
                        list_last_edge_1 = np.concatenate(list_last_edge_1, axis = 0)
                        
                    
                        last_edge_1_idx = []
                        id_ = 1
                        for b in batch_indices:
                            last_edge_1_idx.append(id_)
                            id_ += len(train_graphs[b])
                        list_last_edge_1 = [list_last_edge_1, np.array(last_edge_1_idx)]
                        list_last_edge = (list_last_edge, list_last_edge_1)
                        
                        
                        batch_last_edges = [list_last_edges[i] for i in batch_indices]
                        offset = 0
                        for b in range(len(batch_last_edges)):
                            batch_last_edges[b] = np.array([x + offset if x != -1 else x for x in batch_last_edges[b]])
                            offset += len(train_graphs[batch_indices[b]].edges())
                        batch_last_edges = np.concatenate(batch_last_edges)
            
            
            if cmd_args.has_edge_feats and cmd_args.sigma:
                batch_idx = np.concatenate([np.repeat(i, len(train_graphs[i])) for i in batch_indices])
            
            if cmd_args.has_edge_feats and len(batch_indices) == 1 and cmd_args.g_type == "er":
                print("BATCH: ", batch_indices)
                print(train_graphs[batch_indices[0]].edges())
            
            if cmd_args.model == "BiGG_GCN":
                feat_idx, edge_list, batch_weight_idx = GCNN_batch_train_graphs(train_graphs, batch_indices, cmd_args)
                ll, ll_wt = model.forward_train2(batch_indices, feat_idx, edge_list, batch_weight_idx)
                
            else:
                ll, ll_wt, ll_batch, ll_batch_wt, _ = model.forward_train(batch_indices, node_feats = node_feats, edge_feats = edge_feats, batch_idx = batch_idx, list_num_edges = list_num_edges, db_info = db_info_it, list_last_edge=list_last_edge, edge_feats_lstm=edge_feats_lstm, batch_last_edges=batch_last_edges,rc=rc)
                
            
            loss_top = -ll / num_nodes
            loss_wt = -ll_wt / num_nodes
            true_loss = -(ll + ll_wt) / num_nodes
            epoch_loss_top = epoch_loss_top + loss_top.item()  / num_iter
            if cmd_args.has_edge_feats:
                epoch_loss_wt = epoch_loss_wt + loss_wt.item()  / num_iter
            
            loss = -(ll + ll_wt / cmd_args.scale_loss) / (num_nodes * cmd_args.accum_grad)
            
            loss.backward()
            grad_accum_counter += 1
            
            if grad_accum_counter == cmd_args.accum_grad:                
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                grad_accum_counter = 0
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / num_iter, true_loss))
        
        print('epoch complete')
        print("Epoch Loss (Topology): ", epoch_loss_top)
        print("Epoch Loss (Weights): ", epoch_loss_wt)
        loss_tops_list[epoch - cmd_args.epoch_load] = epoch_loss_top
        loss_wts_list[epoch - cmd_args.epoch_load] = epoch_loss_wt 
        
        cur = epoch + 1
        
        if cur % cmd_args.epoch_save == 0 or cur == cmd_args.num_epochs:
            print('saving epoch')
            checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch + 1)))
    
    print("Training Complete")
    print("TOPOLOGY LOSSES")
    print(loss_tops_list)
    if cmd_args.has_edge_feats:
        print("WEIGHT LOSSES")
        print(loss_wts_list)
        
        
        

        
        
        
        
        
        