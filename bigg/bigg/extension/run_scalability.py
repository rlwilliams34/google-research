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
from bigg.data_process.data_util import create_graphs, get_graph_data



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
    
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)
    assert cmd_args.blksize < 0  # assume graph is not that large, otherwise model parallelism is needed
    
    model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
    optimizer = optim.AdamW(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    
    if cmd_args.training_time:
        print("Getting training times")
        num_nodes_list = [50, 100, 200, 500, 1e3, 5e3, 1e4, 5e4, 1e5]
        times = []
        
        for num_nodes in num_nodes_list:
            g = get_rand_er(num_nodes, 1)[0]
            [TreeLib.InsertGraph(g)]
            
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
            
            print(num_nodes_list)
            print(times)
            
            sys.exit()
    
    ## CREATE TRAINING GRAPHS HERE    
    train_graphs = get_rand_er(cmd_args.num_nodes, 80, low_p = 0.5, high_p = 1.5)
    print(train_graphs)
    val_graphs = train_graphs[:19]
    test_graphs = get_rand_er(cmd_args.num_nodes, 20, low_p = 0.5, high_p = 1.5)
    
    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    cmd_args.max_num_nodes = max_num_nodes
    
    [TreeLib.InsertGraph(g) for g in train_graphs]
    
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
            edge_feats = (torch.cat([list_edge_feats[i] for i in batch_indices], dim=0) if cmd_args.has_edge_feats else None)
            
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
            
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / num_iter, true_loss))
        
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
    
    

        