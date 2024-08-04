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
from bigg.extension.customized_models import BiggWithEdgeLen
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from bigg.evaluation.graph_stats import *
from bigg.evaluation.mmd import *
from bigg.evaluation.mmd_stats import *
from bigg.experiments.train_utils import get_node_dist

def GCNN_batch_train_graphs(train_graphs, batch_indices, cmd_args):
    batch_g = nx.Graph()
    feat_idx = torch.Tensor().to(cmd_args.device)
    batch_weight_idx = []
    
    for idx in batch_indices:
        g = train_graphs[idx]
        n = len(g)
        
        for e1, e2, w in g.edges(data=True):
            batch_weight_idx.append((int(e1), int(e2), w['weight']))
        feat_idx = torch.cat([feat_idx, torch.arange(n).to(cmd_args.device)])
        
        batch_g = nx.union(batch_g, g, rename = ("A", "B"))
        batch_g = nx.convert_node_labels_to_integers(batch_g)
    
    edge_idx = torch.Tensor(list(batch_g.edges())).to(cmd_args.device).t()
    edge_idx_weighted = list(batch_g.edges(data=True))
    batch_weight_idx = torch.Tensor(batch_weight_idx).to(cmd_args.device)
    
    return feat_idx, edge_idx, batch_weight_idx


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


def debug_model(model, graph, node_feats, edge_feats):
    ll, _ = model.forward_train([0], node_feats=node_feats, edge_feats=edge_feats)
    print(ll)

    edges = []
    for e in graph.edges():
        if e[1] > e[0]:
            e = (e[1], e[0])
        edges.append(e)
    edges = sorted(edges)
    ll, _, _, _, _ = model(len(graph), edges, node_feats=node_feats, edge_feats=edge_feats)
    print(ll)
    import sys
    sys.exit()


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)
    assert cmd_args.blksize < 0  # assume graph is not that large, otherwise model parallelism is needed
    
    path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'train')
    with open(path, 'rb') as f:
        train_graphs = cp.load(f)
    
    [TreeLib.InsertGraph(g) for g in train_graphs]
    print(train_graphs[0].edges(data=True))

    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    cmd_args.max_num_nodes = max_num_nodes
    print('# graphs', len(train_graphs), 'max # nodes', max_num_nodes)
    
    if cmd_args.has_node_feats:
        list_node_feats = [torch.from_numpy(get_node_feats(g)).to(cmd_args.device) for g in train_graphs]
    
    else:
        list_node_feats = None
    
    if cmd_args.has_edge_feats:
        list_edge_feats = [torch.from_numpy(get_edge_feats(g)).to(cmd_args.device) for g in train_graphs]
    
    else:
        list_edge_feats = None
    
    if cmd_args.test_gcn:
        model = BiggWithGCN(cmd_args).to(cmd_args.device)
    
    else:
        model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    
    if cmd_args.model_dump is not None and os.path.isfile(cmd_args.model_dump):
        print('loading from', cmd_args.model_dump)
        checkpoint = torch.load(cmd_args.model_dump)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = cmd_args.learning_rate
    
    #########################################################################################################
    if cmd_args.phase != 'train':
        # get num nodes dist
        print("Now generating sampled graphs...")
        num_node_dist = get_node_dist(train_graphs)
        
        path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'test')
        
        with open(path, 'rb') as f:
            gt_graphs = cp.load(f)
        
        print('# gt graphs', len(gt_graphs))
        
        gen_graphs = []
        with torch.no_grad():
            for _ in tqdm(range(cmd_args.num_test_gen)):
                num_nodes = np.argmax(np.random.multinomial(1, num_node_dist)) 
                _, pred_edges, _, pred_node_feats, pred_edge_feats = model(node_end = num_nodes, display=cmd_args.display)
                
                if cmd_args.has_edge_feats:
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
                        #print(e)
                        w = 1.0
                        if e[0] < e[1]:
                            edge = (e[0], e[1], w)
                        else:
                            edge = (e[1], e[0], w)
                        
                        fixed_edges.append(edge)
                    pred_g.add_weighted_edges_from(fixed_edges)
                    gen_graphs.append(pred_g)
        
        for idx in range(min(10, cmd_args.num_test_gen)):
            print("edges: ", gen_graphs[idx].edges(data=True))
        
        print(cmd_args.g_type)
        print("Generating Graph Stats")
        get_graph_stats(gen_graphs, gt_graphs, cmd_args.g_type)
        
        print('saving graphs')
        with open(cmd_args.model_dump + '.graphs-%s' % str(cmd_args.greedy_frac), 'wb') as f:
            cp.dump(gen_graphs, f, cp.HIGHEST_PROTOCOL)
        print('graph generation complete')
        
        sys.exit()
    #########################################################################################################
    
    indices = list(range(len(train_graphs)))
    
    if cmd_args.epoch_load is None:
        cmd_args.epoch_load = 0
    
    prev = datetime.now()
    N = len(train_graphs)
    B = cmd_args.batch_size
    num_iter = N // B
    
    if num_iter != N / B:
        num_iter += 1
    
    best_loss = np.inf
    improvements = []
    thresh = 5
    patience = 0
    prior_loss = np.inf
    losses = []
    
    for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
        tot_loss = 0.0
        pbar = tqdm(range(num_iter))
        random.shuffle(indices)

        optimizer.zero_grad()
        start = 0
        #losses = []
        for idx in pbar:
            if idx >= cmd_args.accum_grad * int(num_iter / cmd_args.accum_grad):
              print("Skipping iteration -- not enough sub-batches remaining for grad accumulation.")
              continue
            
            start = idx * B
            stop = (idx + 1) * B
            
            if stop >= N:
                batch_indices = indices[start:]
            
            else:
                batch_indices = indices[start:stop]
            
            num_nodes = sum([len(train_graphs[i]) for i in batch_indices])
            node_feats = (torch.cat([list_node_feats[i] for i in batch_indices], dim=0) if cmd_args.has_node_feats else None)

            edge_feats = (torch.cat([list_edge_feats[i] for i in batch_indices], dim=0) if cmd_args.has_edge_feats else None)
            
            if cmd_args.test_gcn:
                feat_idx, edge_list, batch_weight_idx = GCNN_batch_train_graphs(train_graphs, batch_indices, cmd_args)
                ll = model.forward_train2(batch_indices, feat_idx, edge_list, batch_weight_idx)
                
            else:
                ll, _ = model.forward_train(batch_indices, node_feats = node_feats, edge_feats = edge_feats)
            
            loss = -ll / num_nodes
            loss.backward()
            loss = loss.item()
            tot_loss = tot_loss + loss / cmd_args.accum_grad
            
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'best-model'))

            if (idx + 1) % cmd_args.accum_grad == 0:
                if len(losses) > 0:
                    avg_loss = np.mean(losses)
                    if loss - avg_loss < 0:
                        improvements.append(True)
                        patience = 0
            
                    else:
                        improvements.append(False)
                        patience += 1
              
                losses.append(tot_loss)
                if len(losses) > 10:
                    losses = losses[1:]
                tot_loss = 0.0
                
                if patience > thresh:
                    patience = 0
                    print("Reducing Learning Rate")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = max(param_group['lr'] / 2, 1e-5)
                
                if cmd_args.accum_grad > 1:
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.div_(cmd_args.accum_grad)
                
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / num_iter, loss))
        
        print('epoch complete')
        cur = epoch + 1
        model.epoch_num += 1
        
        if cur % cmd_args.epoch_save == 0 or cur == cmd_args.num_epochs: #save every 10th / last epoch
            print('saving epoch')
            checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch + 1)))
            
    
    elapsed = datetime.now() - prev
    print("Time elapsed during training: ", elapsed)
    print("Model training complete.")
    
    #for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
    #    pbar = tqdm(range(cmd_args.epoch_save))
    #
    #    optimizer.zero_grad()
    #    for idx in pbar:
    #        random.shuffle(indices)
    #        batch_indices = indices[:cmd_args.batch_size]
    #        num_nodes = sum([len(train_graphs[i]) for i in batch_indices])
    #
    #        node_feats = torch.cat([list_node_feats[i] for i in batch_indices], dim=0)
    #        edge_feats = torch.cat([list_edge_feats[i] for i in batch_indices], dim=0)
    #
    #        ll, _ = model.forward_train(batch_indices, node_feats=node_feats, edge_feats=edge_feats)
    #        loss = -ll / num_nodes
    #        loss.backward()
    #        loss = loss.item()
    #
    #        if (idx + 1) % cmd_args.accum_grad == 0:
    #            if cmd_args.grad_clip > 0:
    #                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
    #            optimizer.step()
    #            optimizer.zero_grad()
    #        pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / cmd_args.epoch_save, loss))
    #    _, pred_edges, _, pred_node_feats, pred_edge_feats = model(len(train_graphs[0]))
    #    print(pred_edges)
    #    print(pred_node_feats)
    #    print(pred_edge_feats)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        