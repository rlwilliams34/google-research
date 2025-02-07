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
import pickle as cp
import networkx as nx
import numpy as np
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
from bigg.experiments.train_utils import sqrtn_forward_backward, get_node_dist
#from bigg.data_process.data_util import create_graphs, get_graph_data

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
    #edge_idx_weighted = list(batch_g.edges(data=True))
    batch_weight_idx = torch.Tensor(batch_weight_idx).to(cmd_args.device)
    
    return feat_idx, edge_idx, batch_weight_idx

def get_node_map(nodelist, shift=0):
    node_map = {}
    for i, x in enumerate(nodelist):
        node_map[x + shift] = i + shift
    return node_map


def apply_order(G, nodelist, order_only):
    if order_only:
        return nodelist
    node_map = get_node_map(nodelist)
    g = nx.relabel_nodes(G, node_map)
    return g


def get_graph_data(G, node_order, leaves_last = False, order_only=False):
    G = G.to_undirected()
    out_list = []
    orig_node_labels = sorted(list(G.nodes()))
    orig_map = {}
    for i, x in enumerate(orig_node_labels):
        orig_map[x] = i
    G = nx.relabel_nodes(G, orig_map)
    
    if node_order == 'default':
        out_list.append(apply_order(G, list(range(len(G))), order_only))
    
    elif node_order == 'DFS' or node_order == 'BFS':
            ### BFS & DFS from largest-degree node
            CGs = [G.subgraph(c) for c in nx.connected_components(G)]
            
            # rank connected componets from large to small size
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            
            node_list_bfs = []
            node_list_dfs = []
            
            for ii in range(len(CGs)):
                node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                degree_sequence = sorted(
                    node_degree_list, key=lambda tt: tt[1], reverse=True)
                bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                node_list_bfs += list(bfs_tree.nodes())
                dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])
                node_list_dfs += list(dfs_tree.nodes())
            
            if node_order == 'BFS':
                node_list_bfs[0], node_list_bfs[1] = node_list_bfs[1], node_list_bfs[0]
                out_list.append(apply_order(G, node_list_bfs, order_only))
            if node_order == 'DFS':
                node_list_dfs[0], node_list_dfs[1] = node_list_dfs[1], node_list_dfs[0]
                out_list.append(apply_order(G, node_list_dfs, order_only))
    
    else: 
        if node_order == "time":
            out_list.append(order_tree(G, leaves_last))
    
    if len(out_list) == 0:
        out_list = [apply_order(G, list(range(len(G))), order_only)]
    
    return out_list

def get_node_feats(g):
    length = []
    for i, (idx, feat) in enumerate(g.nodes(data=True)):
        assert i == idx
        length.append(feat['length'])
    return np.expand_dims(np.array(length, dtype=np.float32), axis=1)


def get_edge_feats(g, blksize = -1):
    edges = sorted(g.edges(data=True), key=lambda x: x[0] * len(g) + x[1])
    weights = [x[2]['weight'] for x in edges]
    return np.expand_dims(np.array(weights, dtype=np.float32), axis=1)

def get_edge_idx(g):
    edges = sorted(g.edges(data=True), key=lambda x: x[0] * len(g) + x[1])
    edge_idx = [x[0] for x in edges]
    return np.array(edge_idx)

# def get_rand_er(num_nodes, num_graphs, low_p = 1.0, high_p = 1.0, args = cmd_args, p = 0.01):
#     npr = np.random.RandomState(args.seed)
#     graphs = []
#     
#     min_er_nodes = int(low_p * num_nodes)
#     max_er_nodes = int(high_p * num_nodes)
#     
#     for i in range(num_graphs):
#         if i % 10 == 0:
#             g_num_nodes = num_nodes
#         
#         else:
#             g_num_nodes = np.random.randint(min_er_nodes, max_er_nodes + 1)
#         
#         g = nx.fast_gnp_random_graph(g_num_nodes, p)
#         for n1, n2 in g.edges():
#             z = scipy.stats.norm.rvs(size = 1).item()
#             w = np.log(np.exp(z) + 1)
#             g[n1][n2]['weight'] = w
#         graphs += [g]
#     return graphs

def tree_generator(n):
    '''
    Generates a random bifurcating tree w/ n nodes
    Args:
        n: number of leaves
    '''
    g = nx.Graph()
    for j in range(n - 1):
        if j == 0:
            g.add_edges_from([(0, 1), (0, 2)])
        else:
            sample_set = [k for k in g.nodes() if g.degree(k) == 1]
            selected_node = random.sample(sample_set, 1).pop()
            g.add_edges_from([(selected_node, 2*j+1), (selected_node, 2*j+2)])
    return g


def graph_generator(num_leaves, num_graphs = 100, seed = 34):
    '''
    Generates requested number of bifurcating trees
    Args:
    	n: number of leaves
    	num_graphs: number of requested graphs
    	constant_topology: if True, all graphs are topologically identical
    	constant_weights: if True, all weights across all graphs are identical
    	mu_weight: mean weight 
    	scale: SD of weights
    '''
    npr = np.random.RandomState(seed)
    graphs = []
    
    for k in range(num_graphs):
        if num_graphs > 1:
            print(k)
        g = tree_generator(num_leaves)
        mu = np.random.uniform(7, 13)
        weights = np.random.gamma(mu*mu, 1/mu, 2 * num_leaves + 1)
        
        weighted_edge_list = []
        for (n1,n2),w in zip(g.edges(), weights):
            weighted_edge_list.append((n1, n2, w))
        
        g = nx.Graph()
        g.add_weighted_edges_from(weighted_edge_list)
        
        graphs.append(g)
    return graphs


if __name__ == '__main__':
#     if cmd_args.num_leaves <= 5000 and cmd_args.model != "BiGG_GCN":
#         cmd_args.scale_loss = 10
#     
#     else:
#         cmd_args.scale_loss = 1
    
    cmd_args.wt_drop = 0.0
    cmd_args.wt_mode = "score"
    cmd_args.has_edge_feats = True
    cmd_args.has_node_feats = False
    cmd_args.bits_compress = 0
    cmd_args.gpu = 0
    cmd_args.rnn_layers = 1
    cmd_args.max_num_nodes = 2 * cmd_args.num_leaves - 1
    
    ### NEWLY ADDED
    #cmd_args.method == "Test75"
    cmd_args.noise = 0.0
    
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)
    #assert cmd_args.blksize < 0  # assume graph is not that large, otherwise model parallelism is needed
    
    #model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
    if cmd_args.phase != "train": 
        num_leaves_list = [50, 250, 500, 1000, 2500, 5000]
        times_bigg_e = []
        times_bigg_gcn = []
        path = os.getcwd()
        
        for num_leaves in num_leaves_list:
            torch.cuda.mem_get_info()
            cmd_args.num_leaves = num_leaves
            cmd_args.max_num_nodes = 2 * cmd_args.num_leaves - 1
            num_nodes = 2 * cmd_args.num_leaves - 1
            
            model_bigg = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
            model_path = os.path.join(path, 'bigg-temp', 'temp%d.ckpt' % cmd_args.num_leaves)
            if os.path.isfile(model_path):
                print('Loading BiGG-E Model')
                checkpoint_bigg = torch.load(model_path)
                model_bigg.load_state_dict(checkpoint_bigg['model'])
                model_bigg.eval()
                init = datetime.now()
                with torch.no_grad():
                    _, pred_edges, _, _, pred_edge_feats = model_bigg(node_end = num_nodes, display=cmd_args.display)
                cur = datetime.now() - init
                
                times_bigg_e.append(cur.total_seconds())
                
                print("Num nodes: ", num_nodes)
                print("Num edges: ", len(pred_edges))
                print("Time: ", cur.total_seconds())
                
                del model_bigg
                pred_edges = None
                pred_edge_feats = None
            
            else:
                print('MISSING BIGG-E MODEL FOR ', num_leaves, 'LEAVES')
                times_bigg_e.append(-1)
            
            cmd_args.has_edge_feats = False
            cmd_args.has_node_feats = False
            model_gcn = BiggWithGCN(cmd_args).to(cmd_args.device)
            cmd_args.has_edge_feats = True
            
            model_path = os.path.join(path, 'gcn-temp', 'temp%d.ckpt' % cmd_args.num_leaves)
            if os.path.isfile(model_path):
                print('Loading Model')
                checkpoint_gcn = torch.load(model_path)
                model_gcn.load_state_dict(checkpoint_gcn['model'])
                model_gcn.eval()
                init = datetime.now()
                with torch.no_grad():
                    pred_edges, pred_weighted_tensor = model_gcn.sample2(num_nodes = num_nodes, display = cmd_args.display)
                cur = datetime.now() - init
                times_bigg_gcn.append(cur.total_seconds())
                
                print("Num nodes: ", num_nodes)
                print("Num edges: ", len(pred_edges))
                print("Time: ", cur.total_seconds())
                
                del model_gcn
                pred_edges = None
                pred_weighted_tensor = None
            
            else:
                print('MISSING BIGG-GCN MODEL FOR ', num_leaves, 'LEAVES')
                times_bigg_gcn.append(-1)
            
            torch.cuda.empty_cache()
        print("Num leaves: ", num_leaves_list)
        print("BiGG-E times: ", times_bigg_e)
        print("BiGG_GCN times: ", times_bigg_gcn)
        sys.exit()
    
    if cmd_args.training_time:
        print("Getting training times")
        #num_leaves_list = [cmd_args.num_nodes]
        num_leaves_list = [5, 10, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 500, 1e3, 2e3, 3e3, 4e3, 5e3]
        #num_leaves_list = 
        #num_leaves_list = [50, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 7.5e3]
        bigg_times = []
        gcn_times = []
        train_graphs = []
        i = 0
        for num_leaves in num_leaves_list:
            num_leaves = int(num_leaves)
            num_nodes = 2 * int(num_leaves) - 1
            
            ### DATA
            g = graph_generator(num_leaves, 1, cmd_args.seed) #get_rand_er(int(num_nodes), 1)[0]
            g = get_graph_data(g[0], 'DFS')
            train_graphs += g
            [TreeLib.InsertGraph(train_graphs[i])]
            edge_feats = torch.from_numpy(get_edge_feats(train_graphs[i])).to(cmd_args.device)
#             
#             feat_idx, edge_list, batch_weight_idx = GCNN_batch_train_graphs(train_graphs, [i], cmd_args)
#             edge_feats = torch.from_numpy(get_edge_feats(train_graphs[i])).to(cmd_args.device)
#             
#             ### FIRST BIGG-GCN
#             cmd_args.max_num_nodes = num_nodes
#             cmd_args.has_edge_feats = False
#             cmd_args.has_node_feats = False
#             model = BiggWithGCN(cmd_args).to(cmd_args.device)
#             cmd_args.has_edge_feats = True
#             optimizer = optim.AdamW(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
#                 
#             init = datetime.now()
#             ll, ll_wt = model.forward_train2([i], feat_idx, edge_list, batch_weight_idx)
#             loss = -(ll + ll_wt) / num_nodes
#             loss.backward()    
#             optimizer.step()
#             optimizer.zero_grad()
#             cur = datetime.now() - init
#             
#             del model
#             del optimizer
#             
#             if i >= 2:
#                 gcn_times.append(cur.total_seconds())
#                 print("BIGG-GCN")
#                 print(num_leaves)
#                 print(cur.total_seconds())
            
            ### BIGG-E
            
            ## DB Info
            m = len(g[0].edges())
            info1 = get_list_indices([m])
            batch_lv_list = get_batch_lv_list_fast([m])
            info2 = prepare_batch(batch_lv_list)
            db_info = (info1, info2)
            
            
            ## List num edges
            list_num_edges = [m]
            
            ## list_last_edge
            batch_last_edges = np.array(get_last_edge2(g[0]))
            model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
            optimizer = optim.AdamW(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
            model.update_weight_stats(edge_feats)
            model.epoch_num += 1
        
            
            
            init = datetime.now()
            #ll, ll_wt, _ = model.forward_train([i], node_feats = None, edge_feats = edge_feats)
            ll, ll_wt, ll_batch, ll_batch_wt, _ = model.forward_train([i], edge_feats = edge_feats, list_num_edges = list_num_edges, db_info = db_info, batch_last_edges=batch_last_edges)
            loss = -(ll + ll_wt) / num_nodes
            loss.backward()    
            optimizer.step()
            optimizer.zero_grad()
            cur = datetime.now() - init
            
            del model
            del optimizer
            
            if i >= 2:
                bigg_times.append(cur.total_seconds())
                print("BIGG-E")
                print(num_leaves)
                print(cur.total_seconds())
            
            i+=1
            
        print(num_leaves_list)
        print(gcn_times)
        print(bigg_times)
            
        sys.exit()
    
    
    ### Set Model
    if cmd_args.model == "BiGG_GCN":
        cmd_args.has_edge_feats = False
        cmd_args.has_node_feats = False
        model = BiggWithGCN(cmd_args).to(cmd_args.device)
        cmd_args.has_edge_feats = True
    
    else:
        model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    
    ## CREATE TRAINING GRAPHS HERE 
    path = os.path.join(os.getcwd(), 'temp_graphs')
    
    if cmd_args.num_leaves <= 7500 and os.path.isfile(path):
        with open(path, 'rb') as f:
            ordered_graphs = cp.load(f) ## List of nx val graphs
    
    else:   
        graphs = graph_generator(cmd_args.num_leaves, 100, cmd_args.seed)
        
        num_graphs = len(graphs)
        num_train = 80
        num_test_gt = num_graphs - num_train
        
        # npr = np.random.RandomState(cmd_args.seed)
        # npr.shuffle(graphs)
        ordered_graphs = []
        
        for g in graphs:
            cano_g = get_graph_data(g, 'DFS')
            ordered_graphs += cano_g
        
        if cmd_args.num_leaves > 1000:
            with open(path, 'wb') as f:
                cp.dump(ordered_graphs, f, protocol=cp.HIGHEST_PROTOCOL)
    
    num_graphs = len(ordered_graphs)
    num_train = 80
    num_test_gt = num_graphs - num_train
    
    train_graphs = ordered_graphs[:num_train]
    test_graphs = ordered_graphs[num_train:]
    graphs = None
    ordered_graphs = None
    
    if len(train_graphs[0]) < 5000:
        print(train_graphs[0].edges(data=True))
    
    
    [TreeLib.InsertGraph(g) for g in train_graphs]
    
    list_node_feats = None
    list_edge_feats = None
    if cmd_args.model == "BiGG_E":
        list_edge_feats = [torch.from_numpy(get_edge_feats(g)).to(cmd_args.device) for g in train_graphs]
    
    
    if cmd_args.blksize > 0:
        list_edge_idx = [get_edge_idx(g) for g in train_graphs]
    
    print('# graphs', len(train_graphs), '# nodes', cmd_args.max_num_nodes)
    print("Begin training")
    print("Train graph size:", len(train_graphs[0]))
    
    N = len(train_graphs)
    B = cmd_args.batch_size
    indices = list(range(N))
    num_iter = int(N / B)
    
    grad_accum_counter = 0
    optimizer.zero_grad()
    
    model.train()
    plateus = []
    prev_loss = np.inf
    
    
    path = os.path.join(os.getcwd(), 'temp%d.ckpt' % cmd_args.num_leaves)
    epoch_load = (cmd_args.epoch_load if cmd_args.epoch_load is not None else 0)
    
    if os.path.isfile(path):
        print('Loading Model')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_load = checkpoint['epoch'] + 1
    
    print(cmd_args.epoch_plateu)
    offset_val = 100
    epoch_lr_decrease = cmd_args.epoch_plateu
    cmd_args.epoch_load = (cmd_args.epoch_load if cmd_args.epoch_load is not None else 0)
    if cmd_args.epoch_load >= epoch_lr_decrease:
        cmd_args.learning_rate = 1e-4
        for param_group in optimizer.param_groups:
            param_group['lr'] = cmd_args.learning_rate
        if cmd_args.epoch_load >= epoch_lr_decrease + offset_val:
            cmd_args.learning_rate = 1e-5
            for param_group in optimizer.param_groups:
                param_group['lr'] = cmd_args.learning_rate
    
    num_epochs = cmd_args.num_epochs
    epoch_plateu = cmd_args.epoch_plateu
    
    #num_epochs = epoch_load
    model.train()
    #num_epochs = epoch_load
    
    list_num_edges = None
    db_info = None
    last_edge_list = None
    last_edge_1_list = None
    list_last_edges = None
    if cmd_args.method == "Test75":
        ## DB info
        list_last_edges = [get_last_edge2(g) for g in train_graphs]
        list_num_edges = [len(train_graphs[0].edges())] * cmd_args.batch_size
        info1 = get_list_indices(list_num_edges)
        batch_lv_list = get_batch_lv_list_fast(list_num_edges)
        info2 = prepare_batch(batch_lv_list)
        db_info = (info1, info2)
            
        ## Last Edge info
        last_edge_list = [get_last_edge(g)[0] for g in train_graphs]
        last_edge_1_list = [get_last_edge(g)[1] for g in train_graphs]
    
    
    for epoch in range(epoch_load, num_epochs):
        pbar = tqdm(range(num_iter))
        random.shuffle(indices)
        
        if epoch == 0 and cmd_args.has_edge_feats and cmd_args.model == "BiGG_E":
            for i in range(len(list_edge_feats)):
                edge_feats = list_edge_feats[i]
                model.update_weight_stats(edge_feats)
        
        if cmd_args.model == "BiGG_GCN":
            model.gcn_mod.epoch_num += 1
        
        else:
            model.epoch_num += 1
        
        epoch_loss = 0.0
        epoch_loss_top = 0.0
        epoch_loss_wt = 0.0
        if epoch >= epoch_lr_decrease and cmd_args.learning_rate == 1e-3:
            cmd_args.learning_rate = cmd_args.learning_rate / 10
            print("Lowering Larning Rate to: ", cmd_args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cmd_args.learning_rate
        
        elif epoch >= epoch_lr_decrease + offset_val and cmd_args.learning_rate == 1e-4:
            cmd_args.learning_rate = cmd_args.learning_rate / 10
            print("Lowering Larning Rate to: ", cmd_args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cmd_args.learning_rate
        
        for idx in pbar:
            start = B * idx
            stop = B * (idx + 1)
            batch_indices = indices[start:stop]
            batch_indices = np.sort(batch_indices)
            
            num_nodes = sum([len(train_graphs[i]) for i in batch_indices])
            
            node_feats = None
            edge_feats = (torch.cat([list_edge_feats[i] for i in batch_indices], dim=0) if list_edge_feats is not None else None)
            
            edge_feats = (torch.cat([list_edge_feats[i] for i in batch_indices], dim=0) if list_edge_feats is not None else None)
            batch_last_edges = None
            list_last_edge = None
            
            if cmd_args.method ==  "Test75":
                list_num_edges = [len(train_graphs[i].edges()) for i in batch_indices]
                if db_info is not None:
                    db_info_it = db_info
                
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
                if cmd_args.batch_size > 1:
                    for b in range(len(batch_last_edges)):
                        batch_last_edges[b] = np.array([x + offset if x != -1 else x for x in batch_last_edges[b]])
                        offset += len(train_graphs[batch_indices[b]].edges())
                else:
                    batch_last_edges = [np.array(batch_last_edges[0])]
                
                batch_last_edges = np.concatenate(batch_last_edges)
            
            
            if cmd_args.blksize < 0 or num_nodes <= cmd_args.blksize:
                if cmd_args.model == "BiGG_GCN":
                    feat_idx, edge_list, batch_weight_idx = GCNN_batch_train_graphs(train_graphs, batch_indices, cmd_args)
                    ll, ll_wt = model.forward_train2(batch_indices, feat_idx, edge_list, batch_weight_idx)
                
                else:
                    ll, ll_wt, ll_batch, ll_batch_wt, _ = model.forward_train(batch_indices, edge_feats = edge_feats, list_num_edges = list_num_edges, db_info = db_info_it, batch_last_edges=batch_last_edges)
                    #ll, ll_wt, _ = model.forward_train(batch_indices, node_feats = node_feats, edge_feats = edge_feats)
                
                true_loss = -(ll  + ll_wt) / num_nodes
                loss = -(ll + ll_wt / cmd_args.scale_loss) / (num_nodes * cmd_args.accum_grad)
                loss.backward()
                
                loss_top = -ll / num_nodes
                loss_wt = -ll_wt / num_nodes
                epoch_loss_top = epoch_loss_top + loss_top.item()  / num_iter
                epoch_loss_wt = epoch_loss_wt + loss_wt.item()  / num_iter
                
                
                epoch_loss += loss.item() / num_iter
            
            else:
                ll = 0.0
                for i in batch_indices:
                    n = len(train_graphs[i])
                    cur_ll, _ = sqrtn_forward_backward(model, graph_ids=[i], list_node_starts=[0],
                                                    num_nodes=n, blksize=cmd_args.blksize, loss_scale=1.0/n, edge_feats = list_edge_feats[i], edge_idx = list_edge_idx[i])
                    ll += cur_ll
                loss = -ll / num_nodes
                epoch_loss += loss / num_iter
            
            if (idx + 1) % cmd_args.accum_grad == 0:
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            
            grad_accum_counter += 1
            
            if grad_accum_counter == cmd_args.accum_grad:
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                grad_accum_counter = 0
            
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / num_iter, true_loss))
        
        
        if (epoch+1) % 20 == 0 or epoch == 0:
            print('Saving Model')
            
            if os.path.isfile(os.path.join(os.getcwd(), 'temp%d.ckpt' % cmd_args.num_leaves)):
                os.remove(os.path.join(os.getcwd(), 'temp%d.ckpt' % cmd_args.num_leaves))
            
            checkpoint = {'epoch': epoch+1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            path = os.path.join(os.getcwd(), 'temp%d.ckpt' % cmd_args.num_leaves)
            torch.save(checkpoint, path)
        
        print('epoch complete')
        print("Epoch Loss (Topology): ", epoch_loss_top)
        print("Epoch Loss (Weights): ", epoch_loss_wt)
        
    print("Evaluation...")
    num_node_dist = get_node_dist(train_graphs)
    gen_graphs = []
      
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(20)):
            num_nodes = 2 * cmd_args.num_leaves - 1
            
            init = datetime.now()
            
            if cmd_args.model == "BiGG_GCN": 
                pred_edges, pred_weighted_tensor = model.sample2(num_nodes = num_nodes, display = cmd_args.display)
            
            else:
                _, _, pred_edges, _, pred_node_feats, pred_edge_feats = model(node_end = num_nodes, display=cmd_args.display)
            
            if i % 5 == 0:
                cur = datetime.now() - init
                print("Num nodes: ", num_nodes)
                print("Times: ", cur.total_seconds())
            
            weighted_edges = []
            if cmd_args.model == "BiGG_GCN":
                    pred_weighted_tensor = pred_weighted_tensor.cpu().detach().numpy()
                    
                    weighted_edges = []
                    for e1, e2, w in pred_weighted_tensor:
                        weighted_edges.append((int(e1), int(e2), np.round(w.item(), 4)))
                    
                    pred_g = nx.Graph()
                    pred_g.add_weighted_edges_from(weighted_edges)
            
            else:
                for e, w in zip(pred_edges, pred_edge_feats):
                    assert e[0] > e[1]
                    weighted_edges.append((e[1], e[0], np.round(w.item(), 4)))
                    pred_g = nx.Graph()
                    pred_g.add_weighted_edges_from(weighted_edges)
                
            gen_graphs.append(pred_g)
            if i == 0:
                print(pred_g.edges(data=True))
            
    
    print(gen_graphs[0].edges(data=True))
    
    get_graph_stats(gen_graphs, test_graphs, 'scale_test')

    

        