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
    batch_weight_idx = torch.Tensor(batch_weight_idx).to(cmd_args.device)
    return feat_idx, edge_idx, batch_weight_idx


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
    return bin_lr_seq


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


def debug_model(model, graph, node_feats, edge_feats, method=None):
    ll_t1 = 0
    ll_w1 = 0
    ll_t2 = 0
    ll_w2 = 0
    
    
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
        
        ll, ll_wt, _, _, _, _ = model(len(g), edges, node_feats=node_feats, edge_feats=edge_feats_i)
        ll_t2 = ll + ll_t2
        ll_w2 = ll_wt + ll_w2
    
    if isinstance(edge_feats, list) and method != "Test4":
        edge_feats = torch.cat(edge_feats, dim = 0)
    
    elif method == "Test4":
        print("Neeed to implement")
                
    ll_t1, ll_w1, _, _, _ = model.forward_train([0, 1], node_feats=node_feats, edge_feats=edge_feats)
    #ll_t1, ll_w1, _ = model.forward_train([0, 1]) #, node_feats=node_feats, edge_feats=edge_feats)
    
    print("=============================")
    print("Slow Code Top+Wt Likelihoods: ")
    print(ll_t1)
    print(ll_w1)
    print("Fast Code Top+Wt Likelihoods: ")
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
    
    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    cmd_args.max_num_nodes = max_num_nodes
    
    print(train_graphs[1].edges())
    
    if cmd_args.phase == "train": 
        [TreeLib.InsertGraph(g) for g in train_graphs]
        list_node_feats = ([torch.from_numpy(get_node_feats(g)).to(cmd_args.device) for g in train_graphs] if cmd_args.has_node_feats else None)
        list_edge_feats = None
        if cmd_args.has_edge_feats:
            if cmd_args.method == "Test4":
                list_edge_feats = [get_edge_feats_2(g, cmd_args.device) for g in train_graphs]
            else:
                list_edge_feats = [torch.from_numpy(get_edge_feats(g)).to(cmd_args.device) for g in train_graphs]
        print('# graphs', len(train_graphs), 'max # nodes', max_num_nodes)
    
    if cmd_args.model == "BiGG_GCN":
        cmd_args.has_edge_feats = False
        cmd_args.has_node_feats = False
        model = BiggWithGCN(cmd_args).to(cmd_args.device)
        cmd_args.has_edge_feats = True
    else:
        model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
    
    #optimizer = optim.AdamW(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    
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
        print('# gt graphs', len(gt_graphs))
#         
#         print("Training graphs MMD Check")
#         get_graph_stats(train_graphs, gt_graphs, cmd_args.g_type)
#         print("Test graphs MMD Check")
#         gt_graphs2 = [gt_graphs[8 - i] for i in range(9)]
#         get_graph_stats(gt_graphs2, gt_graphs, cmd_args.g_type)
        
        gen_graphs = []
        with torch.no_grad():
            model.eval()
            for _ in tqdm(range(cmd_args.num_test_gen)):
                num_nodes = np.argmax(np.random.multinomial(1, num_node_dist)) 
                _, _, pred_edges, _, pred_node_feats, pred_edge_feats = model(node_end = num_nodes, display=cmd_args.display)
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
    lr_scheduler = {'lobster': 100, 'tree': 100 , 'db': 3000, 'er': 250, 'span': 500, 'franken': 200}
    epoch_lr_decrease = lr_scheduler[cmd_args.g_type]
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
        if cmd_args.g_type == 'db':
            cmd_args.scale_loss = 50
        else:
            cmd_args.scale_loss = 20
    
    if cmd_args.model == "BiGG_GCN":
        cmd_args.scale_loss = 20
    
    model.train()
    
    ### DEBUG
    if cmd_args.debug:
        if cmd_args.has_edge_feats and cmd_args.method == "LSTM":
            debug_model(model, [train_graphs[0], train_graphs[1]], None, [list_edge_feats[i] for i in [0,1]], False)
        
        elif cmd_args.has_edge_feats and cmd_args.method == "Test4":
            edge_feats = [list_edge_feats[i][0] for i in [0,1]]
            lr = [list_edge_feats[i][1] for i in [0, 1]]
            edge_feats = (edge_feats, lr)
            debug_model(model, [train_graphs[0], train_graphs[1]], None, edge_feats, True, True)
        
        else:
            edge_feats = ([list_edge_feats[i] for i in [0,1]] if cmd_args.has_edge_feats else None)
            debug_model(model, [train_graphs[0], train_graphs[1]], None, edge_feats, True)
        
    for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
        tot_loss = 0.0
        pbar = tqdm(range(num_iter))
        random.shuffle(indices)
        epoch_losses_t = []
        epoch_losses_w = []
        batch_idx = None
        
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
        
        edge_feats_embed = None
        epoch_loss = 0.0
        
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
            
            if cmd_args.sigma:
                batch_idx = np.concatenate([np.repeat(i, len(train_graphs[i])) for i in batch_indices])
            
            if cmd_args.model == "BiGG_GCN":
                feat_idx, edge_list, batch_weight_idx = GCNN_batch_train_graphs(train_graphs, batch_indices, cmd_args)
                ll, ll_wt = model.forward_train2(batch_indices, feat_idx, edge_list, batch_weight_idx)
                
            else:
                ll, ll_wt, ll_batch, ll_batch_wt, _ = model.forward_train(batch_indices, node_feats = node_feats, edge_feats = edge_feats, batch_idx = batch_idx)
                
            
            loss_top = -ll / num_nodes
            loss_wt = -ll_wt / num_nodes
            true_loss = -(ll + ll_wt) / num_nodes
            epoch_loss = epoch_loss + true_loss  / num_iter 
            
            if cmd_args.sigma:
                epoch_losses_t.append(ll_batch)
                epoch_losses_w.append(ll_batch_wt)
                loss = -ll / sigma_t**0.5 - ll_wt / sigma_w**0.5
                loss = loss / B
            
            else:
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
        print("Epoch Loss: ", epoch_loss)
        
        if cmd_args.sigma and epoch > 0:
            sigma_t = np.var(epoch_losses_t, ddof = 1)
            sigma_w = np.var(epoch_losses_w, ddof = 1)
            print("sigma t: ", sigma_t)
            print("sigma w: ", sigma_w)
        
        cur = epoch + 1
        
        print("CURRENT LOSSES")
        print("Top Loss: ", loss_top)
        print("Wt Loss: ", loss_wt)
                
        if cur % cmd_args.epoch_save == 0 or cur == cmd_args.num_epochs:
            print('saving epoch')
            checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch + 1)))
        
        
        
        
        
        
        