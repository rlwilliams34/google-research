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
    cmd_args.scale_loss = 20
    cmd_args.wt_drop = 0.5
    cmd_args.wt_mode = "score"
    cmd_args.has_edge_feats = True
    cmd_args.has_node_feats = False
    cmd_args.bits_compress = 0
    cmd_args.gpu = 0
    cmd_args.rnn_layers = 1
    
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)
    #assert cmd_args.blksize < 0  # assume graph is not that large, otherwise model parallelism is needed
    
    model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
    optimizer = optim.AdamW(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    
    if cmd_args.training_time:
        print("Getting training times")
        #num_leaves_list = [cmd_args.num_nodes]
        num_leaves_list = [50, 100, 200, 50, 100, 200, 500, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 7.5e3]
        #num_leaves_list = [50, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 7.5e3]
        times = []
        i = 0
        for num_leaves in num_leaves_list:
            num_leaves = int(num_leaves)
            num_nodes = 2 * int(num_leaves) - 1
            print(num_nodes)
            
            g = graph_generator(num_leaves, 1, cmd_args.seed) #get_rand_er(int(num_nodes), 1)[0]
            g = get_graph_data(g[0], 'BFS')
            g = g[0]
            
            [TreeLib.InsertGraph(g)]
            
            edge_feats = torch.from_numpy(get_edge_feats(g)).to(cmd_args.device)
            
            init = datetime.now()
            ll, ll_wt, _ = model.forward_train([i], node_feats = None, edge_feats = edge_feats)
            loss = -(ll + ll_wt) / num_nodes
            loss.backward()
            
            if cmd_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            cur = datetime.now() - init
            times.append(cur.total_seconds())
            i+=1
            print(num_leaves)
            print(cur.total_seconds())
            
        print(num_leaves_list)
        print(times)
            
        sys.exit()
    
    ## CREATE TRAINING GRAPHS HERE 
    path = os.path.join(os.getcwd(), 'temp_graphs')
    
    if cmd_args.num_leaves > 1000 and os.path.isfile(path):
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
            cano_g = get_graph_data(g, 'BFS')
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
    
    print(train_graphs[0].edges(data=True))
    
    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    cmd_args.max_num_nodes = max_num_nodes
    
    [TreeLib.InsertGraph(g) for g in train_graphs]
    
    model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
    optimizer = optim.AdamW(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    
    list_node_feats = None
    list_edge_feats = [torch.from_numpy(get_edge_feats(g)).to(cmd_args.device) for g in train_graphs]
    
    if cmd_args.blksize > 0:
        list_edge_idx = [get_edge_idx(g) for g in train_graphs]
    
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
    
    
    path = os.path.join(os.getcwd(), 'temp%d.ckpt' % cmd_args.num_leaves)
    epoch_load = 0
    
    if os.path.isfile(path):
        print('Loading Model')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_load = checkpoint['epoch']
    
    if cmd_args.num_leaves >= 5000:
        num_epochs = 500
        epoch_plateu = 400
        
        if epoch_load >= 400:
            cmd_args.learning_rate = 1e-5        
    
    else:
        num_epochs = 1500
        epoch_plateu = 800
        
        if epoch_load >= 800:
            cmd_args.learning_rate = 1e-5
    
    for epoch in range(0):#epoch_load, num_epochs):
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
            
            ###
            if cmd_args.blksize < 0 or num_nodes <= cmd_args.blksize:
                ll, ll_wt, _ = model.forward_train(batch_indices, node_feats = node_feats, edge_feats = edge_feats)
                loss = -(ll * cmd_args.scale_loss + ll_wt) / num_nodes
                loss.backward()
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
            
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / num_iter, loss))
        
        if cmd_args.learning_rate != 1e-5 and epoch > epoch_plateu:
            plateu = int(epoch_loss > prev_loss)
            prev_loss = epoch_loss
            
            if len(plateus) == 10:
                plateus = plateus[1:] + [plateu]
            
            else:
                plateus.append(plateu)
            
            if sum(plateus) > 5:
                print("Plateu LR")
                cmd_args.learning_rate = 1e-5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5
        
        if (epoch+1) % 50 == 0 or epoch == 0:
            print('Saving Model')
            
            #if os.isfile(os.path.join(os.getcwd(), 'temp%d.ckpt' % cmd_args.num_leaves)):
            #    os.remove(os.path.join(os.getcwd(), 'temp%d.ckpt' % cmd_args.num_leaves))
            
            checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            path = os.path.join(os.getcwd(), 'temp%d.ckpt' % cmd_args.num_leaves)
            torch.save(checkpoint, path)
        
    print("Evaluation...")
    num_node_dist = get_node_dist(train_graphs)
    gen_graphs = []
      
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(20)):
            if i == 0:
                num_nodes = 2 * cmd_args.num_leaves - 1
            
            else:
                num_nodes = np.argmax(np.random.multinomial(1, num_node_dist))
            
            init = datetime.now()
            
            _, pred_edges, _, _, pred_edge_feats = model(node_end = num_nodes, display=cmd_args.display)
            
            
            if i % 5 == 0:
                cur = datetime.now() - init
                print("Num nodes: ", num_nodes)
                print("Times: ", cur.total_seconds())
            
            weighted_edges = []
            
            for e, w in zip(pred_edges, pred_edge_feats):
                assert e[0] > e[1]
                weighted_edges.append((e[1], e[0], np.round(w.item(), 4)))
                pred_g = nx.Graph()
                pred_g.add_weighted_edges_from(weighted_edges)
            
            gen_graphs.append(pred_g)
            
    
    print(gen_graphs[0].edges(data=True))
    
    get_graph_stats(gen_graphs, test_graphs, 'scale_test')

    

        