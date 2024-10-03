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
#from bigg.data_process.data_util import create_graphs, get_graph_data

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

def order_tree(G, leaves_last = False): 
    n = len(G)
    leaves = sorted([x for x in G.nodes() if G.degree(x)==1])
    nodes = sorted([x for x in G.nodes() if x not in leaves])
    
    #npl = [node for node in nx.single_source_dijkstra(G, 0)[0]]
    #npl = [node for node in nx.single_source_shortest_path_length(G, 0)[0]]
    
    npl_dict = nx.single_source_shortest_path_length(G, 0)
    npl_list = [k for k in npl_dict.keys()]
    
    if leaves_last:
        npl_n = [node for node in npl if node in nodes]
        npl_l = [node for node in npl if node in leaves]
        npl = npl_n + npl_l
    
    reorder = {}
    for k in range(n):
        reorder[npl_list[k]] = k
    new_G = nx.relabel_nodes(G, mapping = reorder)
    return new_G

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


def get_edge_feats(g):
    edges = sorted(g.edges(data=True), key=lambda x: x[0] * len(g) + x[1])
    weights = [x[2]['weight'] for x in edges]
    return np.expand_dims(np.array(weights, dtype=np.float32), axis=1)

def get_rand_er(num_nodes, num_graphs, low_p = 1.0, high_p = 1.0, args = cmd_args, p = 0.01):
    npr = np.random.RandomState(args.seed)
    graphs = []
    
    min_er_nodes = int(low_p * num_nodes)
    max_er_nodes = int(high_p * num_nodes)
    
    for i in range(num_graphs):
        if i % 10 == 0:
            g_num_nodes = num_nodes
        
        else:
            g_num_nodes = np.random.randint(min_er_nodes, max_er_nodes + 1)
        
        g = nx.fast_gnp_random_graph(g_num_nodes, p)
        for n1, n2 in g.edges():
            z = scipy.stats.norm.rvs(size = 1).item()
            w = np.log(np.exp(z) + 1)
            g[n1][n2]['weight'] = w
        graphs += [g]
    return graphs


if __name__ == '__main__':
    cmd_args.scale = 20
    cmd_args.wt_drop = 0.5
    cmd_args.wt_mode = "score"
    cmd_args.has_edge_feats = True
    cmd_args.has_node_feats = False
    cmd_args.bits_compress = 0
    cmd_args.gpu = 0
    
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)
    assert cmd_args.blksize < 0  # assume graph is not that large, otherwise model parallelism is needed
    
    if cmd_args.training_time:
        print("Getting training times")
        num_nodes_list = [50, 100, 200, 500, 1e3, 5e3, 1e4, 5e4, 1e5]
        times = []
        
        for num_nodes in num_nodes_list:
            print(num_nodes)
            g = get_rand_er(int(num_nodes), 1)[0]
            g = get_graph_data(g, 'DFS')[0]
            [TreeLib.InsertGraph(g)]
            
            model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
            optimizer = optim.AdamW(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
            
            init = datetime.now()
            edge_feats = torch.from_numpy(get_edge_feats(g)).to(cmd_args.device)
            
            ll, ll_wt, _ = model.forward_train([0], node_feats = None, edge_feats = edge_feats)
            loss = -(ll + ll_wt) / num_nodes
            loss.backward()
            
            if cmd_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            cur = datetime.now() - init
            times.append(cur.total_seconds())
            print(times)
            
        print(num_nodes_list)
        print(times)
            
        sys.exit()
    
    ## CREATE TRAINING GRAPHS HERE    
    graphs = get_rand_er(cmd_args.num_nodes, 100, low_p = 0.5, high_p = 1.5)
    
    num_graphs = len(graphs)
    num_train = 80
    num_test_gt = num_graphs - num_train

    # npr = np.random.RandomState(cmd_args.seed)
    # npr.shuffle(graphs)
    ordered_graphs = []
    
    for g in graphs:
        cano_g = get_graph_data(g, 'DFS')
        ordered_graphs += cano_g
    
    train_graphs = ordered_graphs[:num_train]
    print(train_graphs[0].edges(data=True))
    test_graphs = ordered_graphs[num_train:]
    
    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    cmd_args.max_num_nodes = max_num_nodes
    
    [TreeLib.InsertGraph(g) for g in train_graphs]
    
    model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
    optimizer = optim.AdamW(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    
    list_node_feats = None
    list_edge_feats = [torch.from_numpy(get_edge_feats(g)).to(cmd_args.device) for g in train_graphs]
    
    print('# graphs', len(train_graphs), '# nodes', max_num_nodes)
    print("Begin training")
    
    N = len(train_graphs)
    B = cmd_args.batch_size
    indices = list(range(N))
    num_iter = int(N / B)
    
    grad_accum_counter = 0
    optimizer.zero_grad()
    
    model.train()
    plateus = []
    prev_loss = np.inf
    
    for epoch in range(2000):
        pbar = tqdm(range(num_iter))
        random.shuffle(indices)
        
        if epoch == 0:
            for i in range(len(list_edge_feats)):
                edge_feats = list_edge_feats[i]
                model.update_weight_stats(edge_feats)
        
        epoch_loss = 0.0
        
        for idx in pbar:
            start = B * idx
            stop = B * (idx + 1)
            batch_indices = indices[start:stop]
            
            num_nodes = sum([len(train_graphs[i]) for i in batch_indices])
            
            node_feats = None
            edge_feats = torch.cat([list_edge_feats[i] for i in batch_indices], dim=0)
            
            ll, ll_wt, _ = model.forward_train(batch_indices, node_feats = node_feats, edge_feats = edge_feats)
            
            loss = -(ll * cmd_args.scale_loss + ll_wt) / num_nodes
            loss.backward()
            grad_accum_counter += 1
            
            epoch_loss += loss.item() / num_iter
            
            if grad_accum_counter == cmd_args.accum_grad:
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                grad_accum_counter = 0
            
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / num_iter, loss))
        
        if cmd_args.learning_rate != 1e-5:
            plateu = int(epoch_loss > prev_loss)
            prev_loss = epoch_loss
            
            if len(plateus) == 10:
                plateus = plateus[1:] + [plateu]
            
            else:
                plateus.append(plateu)
            
            if sum(plateus) > 5:
                cmd_args.learning_rate = 1e-5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5
    
    print("Evaluation...")
    num_node_dist = get_node_dist(train_graphs)
    
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(20)):
            if i == 0:
                num_nodes = cmd_args.num_nodes
            
            else:
                num_nodes = np.argmax(np.random.multinomial(1, num_node_dist))
            
            if i == 0:
                init = datetime.now()
            
            _, pred_edges, _, _, pred_edge_feats = model(node_end = num_nodes, display=cmd_args.display)
            
            weighted_edges = []
            gen_graphs = []
            
            for e, w in zip(pred_edges, pred_edge_feats):
                assert e[0] > e[1]
                weighted_edges.append((e[1], e[0], np.round(w.item(), 4)))
                pred_g = nx.Graph()
                pred_g.add_weighted_edges_from(weighted_edges)
                gen_graphs.append(pred_g)
            
            if i == 0:
                cur = datetime.now() - init
                print("Num nodes: ", num_nodes)
                print("Times: ", cur)
    
    get_graph_stats(gen_graphs, test_graphs, 'scale_test')
    
    

        