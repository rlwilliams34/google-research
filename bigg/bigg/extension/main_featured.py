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
from bigg.extension.customized_models import BiggWithEdgeLen#, BiggWithGCN
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from bigg.evaluation.graph_stats import *
from bigg.evaluation.mmd import *
from bigg.evaluation.mmd_stats import *
from bigg.experiments.train_utils import get_node_dist

import gc
torch.cuda.empty_cache()
gc.collect()



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

# 
# def GCNN_batch_train_graphs(train_graphs, batch_indices, device = "cpu"):
#     batch_g = nx.Graph()
#     feat_idx = torch.Tensor().to(device)
#     batch_weight_idx = []
#     edge_list = []
#     offset = 0
#     
#     for idx in batch_indices:
#         g = train_graphs[idx]
#         n = len(g)
#         feat_idx = torch.cat([feat_idx, torch.arange(n).to(device)])
#         
#         for e1, e2, w in g.edges(data=True):
#             batch_weight_idx.append((int(e1), int(e2), w['weight']))
#             edge_list.append((int(e1) + offset, int(e2) + offset, idx))
#         
#         offset += n
#     
#     edge_idx = torch.Tensor(edge_list).to(device).t()
#     #edge_idx_weighted = list(batch_g.edges(data=True))
#     batch_weight_idx = torch.Tensor(batch_weight_idx).to(device)
#     
#     return feat_idx, edge_idx, batch_weight_idx


# def get_node_feats(g):
#     length = []
#     for i, (idx, feat) in enumerate(g.nodes(data=True)):
#         assert i == idx
#         length.append(feat['length'])
#     return np.expand_dims(np.array(length, dtype=np.float32), axis=1)

def t(n1, n2):
    r = max(n1, n2)
    c = min(n1, n2)
    t = r * (r - 1) / 2 + c
    return int(t)


def get_edge_feats(g):
    #edges = sorted(g.edges(data=True), key=lambda x: x[1]) #x[0] * len(g) + x[1])
    edges = sorted(g.edges(data=True), key=lambda x: t(x[0], x[1]))
    weights = [x[2]['weight'] for x in edges]
    
    return np.expand_dims(np.array(weights, dtype=np.float32), axis=1)


#mydict = {'L': 0, 'R': 1}
#TO GET: list(map(mydict.get, mykeys))



def lr_gen(idx_list, y):    
    if len(idx_list) == 1:
        return ''
    
    if len(idx_list) == 2:
        if y == idx_list[0]:
            return 'L'
        else:
            return 'R'
    
    else:
        midpoint = len(idx_list) // 2
        left_idx_list = idx_list[:midpoint]
                
        if y in left_idx_list:
            if len(left_idx_list) == 1:
                return 'L'
            return 'L' + lr_gen(left_idx_list, y)
        
        else:
            right_idx_list = idx_list[midpoint:]
            return 'R' + lr_gen(right_idx_list, y)

def get_lr_seq(row_, col_):
    mydict = {'L': 0, 'R': 1}
    row = max(row_, col_)
    col = min(row_, col_)
    idx_list = list(range(row))
    lr_seq = lr_gen(idx_list, col)
    bin_lr_seq = list(map(mydict.get, lr_seq))
    return bin_lr_seq #f(idx_list, col)



def get_edge_feats_2(g, device):
    #edges = sorted(g.edges(data=True), key=lambda x: x[1]) #x[0] * len(g) + x[1])
    edges = sorted(g.edges(data=True), key=lambda x: t(x[0], x[1]))
    weights = [x[2]['weight'] for x in edges]
    weights = np.expand_dims(np.array(weights, dtype=np.float32), axis=1)
    weights = torch.from_numpy(weights).to(device)
    
    lr_seq = [get_lr_seq(x[0], x[1]) for x in edges]
    max_len = max(len(x) for x in lr_seq)
    lr_seq = [x + [-1] * (max_len - len(x)) for x in lr_seq]
    return weights, np.transpose(np.array(lr_seq))



def debug_model(model, graph, node_feats, edge_feats, two_graphs=False, cat=False, method=None):
    #model.epoch_num += 2
    if two_graphs:
        ll_t1 = 0
        ll_w1 = 0
        ll_t2 = 0
        ll_w2 = 0
        i = 0
        
        for i in range(0, 2):
            #ll, ll_wt, _ = model.forward_train([i], node_feats=node_feats, edge_feats=edge_feats)
            #ll_t1 = ll + ll_t1
            #ll_w1 = ll_wt + ll_w1
            #print(g)
            #print(e)
            print(i)
            g = graph[i]
            edge_feats_i = edge_feats[0][i]
#             print(edge_feats)
#             print(edge_feats_i)
            edges = []
            for e in g.edges():
                if e[1] > e[0]:
                    e = (e[1], e[0])
                edges.append(e)
            edges = sorted(edges)
            
            if not torch.is_tensor(edge_feats_i):
                edge_feats_i = edge_feats_i[0]
            
            ll, ll_wt, _, _, _, _ = model(len(g), edges, node_feats=node_feats, edge_feats=edge_feats_i)
            ll_t2 = ll + ll_t2
            ll_w2 = ll_wt + ll_w2
            
        
        if cat:
            #edge_feats = torch.cat(edge_feats, dim = 0)
            #edge_feats, lr = zip(*edge_feats)
            #print(edge_feats)
            #print(lr)
            edge_feats, lr = edge_feats
            edge_feats = (torch.cat(edge_feats, dim = 0), np.concatenate(lr, axis = 1))
        #print(edge_feats)
        ll_t1, ll_w1, _ = model.forward_train([0, 1], node_feats=node_feats, edge_feats=edge_feats)
        
        print(ll_t1)
        print(ll_w1)
        print(ll_t2)
        print(ll_w2)
        
        diff1 = abs(ll_t1 - ll_t2)
        diff2 = abs(ll_w1 - ll_w2)
        print("diff top: ", diff1)
        print("diff weight: ", diff2)
        import sys
        sys.exit()
    
    if method=="Test4":
        ll, ll_wt, _ = model.forward_train([0], node_feats=node_feats, edge_feats=edge_feats)
        print(ll)
        ll_t1 = ll.item()
        ll_w1 = ll_wt.item()
        print(ll_wt)
        
        edges = []
        for e in graph.edges():
            if e[1] > e[0]:
                e = (e[1], e[0])
            edges.append(e)
        edges = sorted(edges)
        
        if not torch.is_tensor(edge_feats) and edge_feats is not None:
            edge_feats = edge_feats[0]
        ll, ll_wt, _, _, _, _ = model(len(graph), edges, node_feats=node_feats, edge_feats=edge_feats)
        print(ll)
        print(ll_wt)
        ll_t2 = ll.item()
        ll_w2 = ll_wt.item()
        diff_t = abs(ll_t1 - ll_t2)
        diff_w = abs(ll_w1 - ll_w2)
        print("diff top: ", diff_t)
        print("diff weight: ", diff_w)
        import sys
        sys.exit()
    
    
    ll, ll_wt, _ = model.forward_train([0], node_feats=node_feats, edge_feats=edge_feats)
    #ll, _ = model.forward_train([0], node_feats=node_feats, edge_feats=edge_feats)
    print(ll)
    ll_t1 = ll.item()
    ll_w1 = ll_wt.item()
    print(ll_wt)

    edges = []
    for e in graph.edges():
        if e[1] > e[0]:
            e = (e[1], e[0])
        edges.append(e)
    edges = sorted(edges)
    
    if not torch.is_tensor(edge_feats) and edge_feats is not None:
        edge_feats = edge_feats[0]
    ll, ll_wt, _, _, _, _ = model(len(graph), edges, node_feats=node_feats, edge_feats=edge_feats)
    #ll, _, _, _, _ = model(len(graph), edges, node_feats=node_feats, edge_feats=edge_feats)
    print(ll)
    print(ll_wt)
    ll_t2 = ll.item()
    ll_w2 = ll_wt.item()
    diff_t = abs(ll_t1 - ll_t2)
    diff_w = abs(ll_w1 - ll_w2)
    print("diff top: ", diff_t)
    print("diff weight: ", diff_w)
    import sys
    sys.exit()

# def fix_tree_weights(graphs):
#    for g in graphs:
#        for (n1, n2, w) in g.edges(data=True):
#            g[n1][n2]['weight'] = w['weight'] / 10
#    return graphs


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)
    assert cmd_args.blksize < 0  # assume graph is not that large, otherwise model parallelism is needed
    
    if cmd_args.g_type == "db":
        import pickle5 as cp
    path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'train')
    
    with open(path, 'rb') as f:
        train_graphs = cp.load(f)
    
    print(train_graphs[0].edges(data=True))
    
    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    cmd_args.max_num_nodes = max_num_nodes
    
    if cmd_args.phase == "train": 
        [TreeLib.InsertGraph(g) for g in train_graphs]
    
        if cmd_args.has_node_feats:
            list_node_feats = [torch.from_numpy(get_node_feats(g)).to(cmd_args.device) for g in train_graphs]
        
        else:
           list_node_feats = None
        
        if cmd_args.has_edge_feats:
            if cmd_args.method == "Test4":
                list_edge_feats = [get_edge_feats_2(g, cmd_args.device) for g in train_graphs]
            
            else:
                list_edge_feats = [torch.from_numpy(get_edge_feats(g)).to(cmd_args.device) for g in train_graphs]
        
        else:
            list_edge_feats = None
        
        print('# graphs', len(train_graphs), 'max # nodes', max_num_nodes)
        print(train_graphs[0].edges())
    
    
    #print(train_graphs[0].edges(data=True))
    if cmd_args.model == "BiGG_GCN":
        cmd_args.has_edge_feats = False
        cmd_args.has_node_feats = False
        model = BiggWithGCN(cmd_args).to(cmd_args.device)
        cmd_args.has_edge_feats = True
    
    else:
        model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    
    if cmd_args.model_dump is not None and os.path.isfile(cmd_args.model_dump):
        print('loading from', cmd_args.model_dump)
        checkpoint = torch.load(cmd_args.model_dump)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = cmd_args.learning_rate
    
    #########################################################################################################
    if cmd_args.phase == 'validate':
        # get num nodes dist
        print("Now generating sampled graphs...")
        num_node_dist = get_node_dist(train_graphs)
        
        path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'val')
        
        with open(path, 'rb') as f:
            val_graphs = cp.load(f)
        
        #if cmd_args.g_type == 'tree':
        #    gt_graphs = fix_tree_weights(gt_graphs)
        print('# val graphs', len(val_graphs))
        
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
        
        #if cmd_args.g_type == 'tree':
        #    gt_graphs = fix_tree_weights(gt_graphs)
        print('# gt graphs', len(gt_graphs))
        
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
        
        if cmd_args.max_num_nodes > -1:
            for idx in range(min(2, cmd_args.num_test_gen)):
                print("edges:")
                print(gen_graphs[idx].edges(data=True))
        
        print(cmd_args.g_type)
        print("Generating Graph Stats")
        get_graph_stats(gen_graphs, gt_graphs, cmd_args.g_type)
        
        print('saving graphs')
        with open(cmd_args.model_dump + '.graphs-%s' % str(cmd_args.greedy_frac), 'wb') as f:
            cp.dump(gen_graphs, f, cp.HIGHEST_PROTOCOL)
        print('graph generation complete')
        
        sys.exit()
    #########################################################################################################
    #val_size_dict = {'lobster': 50, 'tree': 50, 'db': 0, 'er': 0}
    top_losses = []
    wt_losses = []
    #best_prop = 0
    #best_prop_epoch = 0
    times = []
    loss_times = []
    epoch_list = []
    lr_scheduler = {'lobster': 100, 'tree': 300 , 'db': 1000, 'er': 250, 'span': 500}
    epoch_lr_decrease = lr_scheduler[cmd_args.g_type]
    batch_loss = 0.0
    
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
        cmd_args.scale_loss = 20
    
    if cmd_args.model == "BiGG_GCN":
        cmd_args.scale_loss = 20
    
    model.train()
    
    ### DEBUG
    if cmd_args.debug:
#         print(train_graphs[0].edges(data=True))
#         if cmd_args.has_edge_feats:
#             print(list_edge_feats[0])
#         if cmd_args.method in ["LSTM", "MLP-Leaf"]:
#             if cmd_args.has_edge_feats:
#                 debug_model(model, train_graphs[0], None, [list_edge_feats[0]])
#             debug_model(model, train_graphs[0], None, None)
#         
#         else:
#             if cmd_args.has_edge_feats:
#                 debug_model(model, train_graphs[0], None, list_edge_feats[0])
#             debug_model(model, train_graphs[0], None, None)
        
        print(train_graphs[0].edges(data=True))
        print(train_graphs[1].edges(data=True))
        
        if cmd_args.method == "LSTM":
            debug_model(model, [train_graphs[0], train_graphs[1]], None, [list_edge_feats[i] for i in [0,1]], True)
        
        elif cmd_args.method == "Test4":
            edge_feats = [list_edge_feats[i][0] for i in [0,1]]
            lr = [list_edge_feats[i][1] for i in [0, 1]]
            edge_feats = (edge_feats, lr)
            debug_model(model, [train_graphs[0], train_graphs[1]], None, edge_feats, True, True)
        
        else:
            debug_model(model, [train_graphs[0], train_graphs[1]], None, [list_edge_feats[i] for i in [0,1]], True, True)
    
#     ####
    
    
    for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
        tot_loss = 0.0
        pbar = tqdm(range(num_iter))
        random.shuffle(indices)
        
        if False: #epoch == 9:
            if cmd_args.method in ["LSTM", "MLP-Leaf"]:
                if cmd_args.has_edge_feats:
                    debug_model(model, train_graphs[0], None, [list_edge_feats[0]])
                debug_model(model, train_graphs[0], None, None)
        
            else:
                if cmd_args.has_edge_feats:
                    debug_model(model, train_graphs[0], None, list_edge_feats[0])
                debug_model(model, train_graphs[0], None, None)
        
        
        if epoch == 0 and cmd_args.has_edge_feats and cmd_args.model == "BiGG_E":
            for i in range(len(list_edge_feats)):
                if cmd_args.method == "Test4":
                    edge_feats = list_edge_feats[i][0]
                else:
                    edge_feats = list_edge_feats[i]
                model.update_weight_stats(edge_feats)
        
        if cmd_args.model == "BiGG_GCN":
            model.gcn_mod.epoch_num += 1
        
        else:
            model.epoch_num += 1
        
        if cmd_args.schedule:
            if epoch >= epoch_lr_decrease and cmd_args.learning_rate != 1e-5:
                cmd_args.learning_rate = 1e-5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5
        
        #if cmd_args.method == "LSTM" and cmd_args.has_edge_feats:
        #    list_edge_feats_embed = model.embed_edge_feats(list_edge_feats, as_list = True)
        
        #else:
        edge_feats_embed = None
        
        for idx in pbar:
            start = B * idx
            stop = B * (idx + 1)
            batch_indices = indices[start:stop]
            batch_indices = np.sort(batch_indices)
            
            num_nodes = sum([len(train_graphs[i]) for i in batch_indices])
            
            node_feats = (torch.cat([list_node_feats[i] for i in batch_indices], dim=0) if list_node_feats is not None else None)
            
            if cmd_args.method in ["LSTM", "MLP-Leaf"] and cmd_args.has_edge_feats:
                #edge_feats_embed_h = (torch.cat([list_edge_feats_embed[0][i] for i in batch_indices], dim=1)) #[list_edge_feats[i] for i in batch_indices]
                #edge_feats_embed_c = (torch.cat([list_edge_feats_embed[1][i] for i in batch_indices], dim=1)) #[list_edge_feats[i] for i in batch_indices]
                #edge_feats_embed = (edge_feats_embed_h, edge_feats_embed_c)
                edge_feats = [list_edge_feats[i] for i in batch_indices]
                
            elif cmd_args.method == "Test4":
                edge_feats = [list_edge_feats[i] for i in batch_indices]
                edge_feats, lr = zip(*edge_feats)
                edge_feats = (torch.cat(edge_feats, dim = 0), np.concatenate(lr, axis = 1))
            
            else:
                edge_feats = (torch.cat([list_edge_feats[i] for i in batch_indices], dim=0) if list_edge_feats is not None else None)
            
            if cmd_args.model == "BiGG_GCN":
                feat_idx, edge_list, batch_weight_idx = GCNN_batch_train_graphs(train_graphs, batch_indices, cmd_args)
                ll, ll_wt = model.forward_train2(batch_indices, feat_idx, edge_list, batch_weight_idx)
                
            else:
                ll, ll_wt, _ = model.forward_train(batch_indices, node_feats = node_feats, edge_feats = edge_feats)#, edge_feats_embed = edge_feats_embed)
                #ll, _ = model.forward_train(batch_indices, node_feats = node_feats, edge_feats = edge_feats)#, edge_feats_embed = edge_feats_embed)
                #ll_wt = 0.0
            
            
            loss_top = -ll / num_nodes
            loss_wt = -ll_wt / num_nodes
            #top_losses.append(loss_top.item())
            
            #if cmd_args.has_edge_feats or cmd_args.model == "BiGG_GCN":
            #    wt_losses.append(loss_wt.item())
            
            true_loss = -(ll + ll_wt) / num_nodes
            #batch_loss = true_loss.item() / cmd_args.accum_grad + batch_loss
            
            
            #ll_wt = 0
            loss = -(ll + ll_wt / cmd_args.scale_loss) / (num_nodes)#* cmd_args.accum_grad)
            loss.backward()
            grad_accum_counter += 1
            
            
            #loss_times.append(loss)
            #epoch_list.append(epoch)
            
#             if true_loss < best_loss:
#                 best_loss = true_loss
#                 torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'best-model'))

            if grad_accum_counter == cmd_args.accum_grad:
                #cur = datetime.now() - prev
                #times.append(cur.total_seconds())
                #loss_times.append(batch_loss)
                #epoch_list.append(epoch)
                #batch_loss = 0.0
                
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                grad_accum_counter = 0
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / num_iter, true_loss))
            
        #time_data = {'times': times, 'loss_times': loss_times, 'epoch_list': epoch_list}
    
        #with open('%s-' % cmd_args.model + '%s-time-data.pkl' % cmd_args.g_type, 'wb') as f:
        #    cp.dump(time_data, f, protocol=cp.HIGHEST_PROTOCOL)
        
        print('epoch complete')
        cur = epoch + 1
        
        print("CURRENT LOSSES")
        print("Top Loss: ", loss_top)
        print("Wt Loss: ", loss_wt)
        #if cur % cmd_args.epoch_save == 0 or cur == cmd_args.num_epochs: #save every 10th / last epoch
        
        if cur % cmd_args.epoch_save == 0:
            print('saving epoch')
            checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch + 1)))
        
#         if cur % cmd_args.epoch_save == 0:
#             print('validating')
#             model.eval()
#             
#             gen_graphs = []
#             with torch.no_grad():
#                 val_size = val_size_dict[cmd_args.g_type]
#                 for _ in range(val_size):
#                     num_nodes = np.argmax(np.random.multinomial(1, num_node_dist)) 
#                     _, pred_edges, _, pred_node_feats, pred_edge_feats = model(node_end = num_nodes)
#                     
#                     if cmd_args.model == "BiGG_GCN":
#                         fix_edges = []
#                         for e1, e2 in pred_edges:
#                             if e1 > e2:
#                                 fix_edges.append((e2, e1))
#                             else:
#                                 fix_edges.append((e1, e2))
#                         pred_edge_tensor = torch.tensor(fix_edges).to(cmd_args.device)
#                         pred_weighted_tensor = model.gcn_mod.sample(num_nodes, pred_edge_tensor)
#                         pred_weighted_tensor = pred_weighted_tensor.cpu().detach().numpy()
#                         
#                         weighted_edges = []
#                         for e1, e2, w in pred_weighted_tensor:
#                             weighted_edges.append((int(e1), int(e2), np.round(w.item(), 4)))
#                         
#                         pred_g = nx.Graph()
#                         pred_g.add_weighted_edges_from(weighted_edges)
#                         gen_graphs.append(pred_g)
#                     
#                     elif cmd_args.has_edge_feats:
#                         weighted_edges = []
#                         for e, w in zip(pred_edges, pred_edge_feats):
#                             weighted_edges.append((e[1], e[0], np.round(w.item(), 4)))
#                     
#                         pred_g = nx.Graph()
#                         pred_g.add_weighted_edges_from(weighted_edges)
#                         gen_graphs.append(pred_g)
#                     
#                     else:
#                         pred_g = nx.Graph()
#                         fixed_edges = []
#                         for e in pred_edges:
#                             w = 1.0
#                             if e[0] < e[1]:
#                                 edge = (e[0], e[1], w)
#                             else:
#                                 edge = (e[1], e[0], w)
#                             fixed_edges.append(edge)
#                         pred_g.add_weighted_edges_from(fixed_edges)
#                         gen_graphs.append(pred_g)
#             
# #             print("NUMBER GRAPHS:", len(gen_graphs))
# #             for g in gen_graphs:
# #                 print(g.edges(data=True))
#             model.train()
#             if val_size > 0:
#                 print("Generating Graph Stats")
#                 prop = get_graph_stats(gen_graphs, None, cmd_args.g_type)
#                 
#                 if cmd_args.g_type == "tree":
#                     cutoff = 0.70
#                 
#                 else:
#                     cutoff = 0.80
#                 
#                 if prop > cutoff or prop > best_prop:
#                     if prop >= best_prop:
#                         best_prop = prop
#                     best_prop_epoch = epoch + 1
#                     print('Saving best prop model')
#                     checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
#                     torch.save(checkpoint, os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch + 1)))
#     print('best prop: ', best_prop)
#     print('best prop epoch: ', best_prop_epoch)
#     print('training complete.')
    
    ###################################################################################
#     indices = list(range(len(train_graphs)))
#     
#     if cmd_args.epoch_load is None:
#         cmd_args.epoch_load = 0
#     
#     prev = datetime.now()
#     N = len(train_graphs)
#     B = cmd_args.batch_size
#     num_iter = N // B
#     
#     if num_iter != N / B:
#         num_iter += 1
#     
#     best_loss = np.inf
#     improvements = []
#     thresh = 10
#     patience = 0
#     prior_loss = np.inf
#     losses = []
#     top_losses = []
#     wt_losses = []
#     
#     for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
#         tot_loss = 0.0
#         pbar = tqdm(range(num_iter))
#         random.shuffle(indices)
# 
#         optimizer.zero_grad()
#         if cmd_args.test_gcn:
#             model.gcn_mod.epoch_num += 1
#         
#         else:
#             model.epoch_num += 1
#         
#         start = 0
#         for idx in pbar:
#             if idx >= cmd_args.accum_grad * int(num_iter / cmd_args.accum_grad):
#               print("Skipping iteration -- not enough sub-batches remaining for grad accumulation.")
#               continue
#             
#             start = idx * B
#             stop = (idx + 1) * B
#             
#             if stop >= N:
#                 batch_indices = indices[start:]
#             
#             else:
#                 batch_indices = indices[start:stop]
#             
#             num_nodes = sum([len(train_graphs[i]) for i in batch_indices])
#             node_feats = (torch.cat([list_node_feats[i] for i in batch_indices], dim=0) if cmd_args.has_node_feats else None)
# 
#             edge_feats = (torch.cat([list_edge_feats[i] for i in batch_indices], dim=0) if cmd_args.has_edge_feats else None)
#             
#             if cmd_args.test_gcn:
#                 feat_idx, edge_list, batch_weight_idx = GCNN_batch_train_graphs(train_graphs, batch_indices, cmd_args)
#                 ll, ll_wt = model.forward_train2(batch_indices, feat_idx, edge_list, batch_weight_idx)
#                 
#             else:
#                 ll, ll_wt, _ = model.forward_train(batch_indices, node_feats = node_feats, edge_feats = edge_feats)
#             
#             
#             loss_top = -ll / num_nodes
#             loss_wt = -ll_wt / num_nodes
#             top_losses.append(loss_top.item())
#             if cmd_args.has_edge_feats or cmd_args.test_gcn:
#                 wt_losses.append(loss_wt.item())
#             loss = -(ll + ll_wt) / num_nodes
#             loss.backward()
#             loss = loss.item()
#             tot_loss = tot_loss + loss / cmd_args.accum_grad
#             
#             if loss < best_loss:
#                 best_loss = loss
#                 torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'best-model'))
# 
#             if (idx + 1) % cmd_args.accum_grad == 0:
#                 if len(losses) > 0:
#                     avg_loss = np.mean(losses)
#                     if loss - avg_loss < 0:
#                         improvements.append(True)
#                         patience = 0
#             
#                     else:
#                         improvements.append(False)
#                         patience += 1
#               
#                 losses.append(tot_loss)
#                 if len(losses) > 10:
#                     losses = losses[1:]
#                 tot_loss = 0.0
#                 
#                 if patience > thresh:
#                     patience = 0
#                     print("Reducing Learning Rate")
#                     for param_group in optimizer.param_groups:
#                         param_group['lr'] =  1e-5
#                         print("Current Learning Rate: ", param_group['lr'])
#                 
#                 if cmd_args.accum_grad > 1:
#                     with torch.no_grad():
#                         for p in model.parameters():
#                             if p.grad is not None:
#                                 p.grad.div_(cmd_args.accum_grad)
#                 
#                 if cmd_args.grad_clip > 0:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
#                 optimizer.step()
#                 optimizer.zero_grad()
#             pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / num_iter, loss))
#         
#         print('epoch complete')
#         cur = epoch + 1
#         
#         print("CURRENT LOSSES")
#         print("Top Loss: ", loss_top)
#         print("Wt Loss: ", loss_wt)
#         if cur % cmd_args.epoch_save == 0 or cur == cmd_args.num_epochs: #save every 10th / last epoch
#             print('saving epoch')
#             print("Top Losses: ")
#             print(top_losses)
#             print("Weight Losses: ")
#             print(wt_losses)
#             checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
#             torch.save(checkpoint, os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch + 1)))
#             
#     
#     elapsed = datetime.now() - prev
#     print("Time elapsed during training: ", elapsed)
#     print("Model training complete.")
#     
#     for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
#         pbar = tqdm(range(cmd_args.epoch_save))
#     
#         optimizer.zero_grad()
#         for idx in pbar:
#             random.shuffle(indices)
#             batch_indices = indices[:cmd_args.batch_size]
#             num_nodes = sum([len(train_graphs[i]) for i in batch_indices])
#     
#             node_feats = torch.cat([list_node_feats[i] for i in batch_indices], dim=0)
#             edge_feats = torch.cat([list_edge_feats[i] for i in batch_indices], dim=0)
#     
#             ll, _ = model.forward_train(batch_indices, node_feats=node_feats, edge_feats=edge_feats)
#             loss = -ll / num_nodes
#             loss.backward()
#             loss = loss.item()
#     
#             if (idx + 1) % cmd_args.accum_grad == 0:
#                 if cmd_args.grad_clip > 0:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
#                 optimizer.step()
#                 optimizer.zero_grad()
#             pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / cmd_args.epoch_save, loss))
#         _, pred_edges, _, pred_node_feats, pred_edge_feats = model(len(train_graphs[0]))
#         print(pred_edges)
#         print(pred_node_feats)
#         print(pred_edge_feats)
        

# path = os.path.join(os.getcwd(), 'tree-time-data.pkl')
# with open(path, 'rb') as f:
#     stats = cp.load(f)
# 
# df = pd.DataFrame(stats)
# df.to_csv('tree_time_results.csv')
        
        
        
        
        
        
        
        
        
        
        
        
        
        