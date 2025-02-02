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

import ctypes
import numpy as np
import random
import os
import sys
import networkx as nx
from tqdm import tqdm
# pylint: skip-file

try:
    import torch
except:
    print('no torch loaded')


class CtypeGraph(object):
    def __init__(self, g):
        self.num_nodes = len(g)
        self.num_edges = len(g.edges())

        self.edge_pairs = np.zeros((self.num_edges * 2, ), dtype=np.int32)
        for i, (x, y) in enumerate(g.edges()):
            self.edge_pairs[i * 2] = x
            self.edge_pairs[i * 2 + 1] = y


class _tree_lib(object):

    def __init__(self):
        pass

    def setup(self, config):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL('%s/build/dll/libtree.so' % dir_path)

        self.lib.Init.restype = ctypes.c_int
        self.lib.PrepareTrain.restype = ctypes.c_int
        self.lib.AddGraph.restype = ctypes.c_int
        self.lib.TotalTreeNodes.restype = ctypes.c_int
        self.lib.MaxTreeDepth.restype = ctypes.c_int
        self.lib.NumEdgesAtLevel.restype = ctypes.c_int
        self.lib.NumTrivialNodes.restype = ctypes.c_int
        self.lib.NumBaseNodes.restype = ctypes.c_int
        self.lib.NumBaseEdges.restype = ctypes.c_int
        self.lib.NumPrevDep.restype = ctypes.c_int
        self.lib.NumBottomDep.restype = ctypes.c_int
        self.lib.NumRowBottomDep.restype = ctypes.c_int
        self.lib.NumRowPastDep.restype = ctypes.c_int
        self.lib.NumRowTopDep.restype = ctypes.c_int
        self.lib.RowSumSteps.restype = ctypes.c_int
        self.lib.RowMergeSteps.restype = ctypes.c_int
        self.lib.NumRowSumOut.restype = ctypes.c_int
        self.lib.NumRowSumNext.restype = ctypes.c_int
        self.lib.NumCurNodes.restype = ctypes.c_int
        self.lib.NumInternalNodes.restype = ctypes.c_int
        self.lib.NumLeftBot.restype = ctypes.c_int
        self.lib.GetNumNextStates.restype = ctypes.c_int

        args = 'this -bits_compress %d -embed_dim %d -gpu %d -bfs_permute %d -seed %d' % (config.bits_compress, config.embed_dim, config.gpu, config.bfs_permute, config.seed)
        args = args.split()
        if sys.version_info[0] > 2:
            args = [arg.encode() for arg in args]  # str -> bytes for each element in args

        arr = (ctypes.c_char_p * len(args))()
        arr[:] = args
        self.lib.Init(len(args), arr)
        self.embed_dim = config.embed_dim
        self.device = config.device
        self.num_graphs = 0
        self.graph_stats = []

    def TotalTreeNodes(self):
        return self.lib.TotalTreeNodes()

    def InsertGraph(self, nx_g, bipart_stats=None):
        gid = self.num_graphs
        self.num_graphs += 1
        if isinstance(nx_g, CtypeGraph):
            ctype_g = nx_g
        else:
            ctype_g = CtypeGraph(nx_g)
        self.graph_stats.append((ctype_g.num_nodes, ctype_g.num_edges))
        if bipart_stats is None:
            n, m = -1, -1
        else:
            n, m = bipart_stats
        self.lib.AddGraph(gid, ctype_g.num_nodes, ctype_g.num_edges,
                          ctypes.c_void_p(ctype_g.edge_pairs.ctypes.data), n, m)
        return gid

    def PrepareMiniBatch(self, list_gids, list_node_start=None, num_nodes=-1, list_col_ranges=None, new_batch=True):
        n_graphs = len(list_gids)
        list_gids = np.array(list_gids, dtype=np.int32)
        if list_node_start is None:
            list_node_start = np.zeros((n_graphs,), dtype=np.int32)
        else:
            list_node_start = np.array(list_node_start, dtype=np.int32)
        if list_col_ranges is None:
            list_col_start = np.zeros((n_graphs,), dtype=np.int32) - 1
            list_col_end = np.zeros((n_graphs,), dtype=np.int32) - 1
        else:
            list_col_start, list_col_end = zip(*list_col_ranges)
            list_col_start = np.array(list_col_start, dtype=np.int32)
            list_col_end = np.array(list_col_end, dtype=np.int32)

        self.lib.PrepareTrain(n_graphs,
                              ctypes.c_void_p(list_gids.ctypes.data),
                              ctypes.c_void_p(list_node_start.ctypes.data),
                              ctypes.c_void_p(list_col_start.ctypes.data),
                              ctypes.c_void_p(list_col_end.ctypes.data),
                              num_nodes,
                              int(new_batch))
        list_nnodes = []
        for i, gid in enumerate(list_gids):
            tot_nodes = self.graph_stats[gid][0]
            if num_nodes <= 0:
                cur_num = tot_nodes - list_node_start[i]
            else:
                cur_num = min(num_nodes, tot_nodes - list_node_start[i])
            list_nnodes.append(cur_num)
        self.list_nnodes = list_nnodes
        return list_nnodes

    def PrepareTreeEmbed(self):
        max_d = self.lib.MaxTreeDepth()

        all_ids = []
        for d in range(max_d + 1):
            ids_d = []
            for i in range(2):
                num_prev = self.lib.NumPrevDep(d, i)
                num_bot = self.lib.NumBottomDep(d, i)

                bot_froms = np.empty((num_bot,), dtype=np.int32)
                bot_tos = np.empty((num_bot,), dtype=np.int32)

                prev_froms = np.empty((num_prev,), dtype=np.int32)
                prev_tos = np.empty((num_prev,), dtype=np.int32)
                self.lib.SetTreeEmbedIds(d,
                                         i,
                                         ctypes.c_void_p(bot_froms.ctypes.data),
                                         ctypes.c_void_p(bot_tos.ctypes.data),
                                         ctypes.c_void_p(prev_froms.ctypes.data),
                                         ctypes.c_void_p(prev_tos.ctypes.data))
                ids_d.append((bot_froms, bot_tos, prev_froms, prev_tos))
            all_ids.append(ids_d)
        return all_ids

    def PrepareBinary(self):
        max_d = self.lib.MaxBinFeatDepth()
        all_bin_feats = []
        base_feat = torch.zeros(2, self.embed_dim)
        base_feat[0, 0] = -1
        base_feat[1, 0] = 1
        base_feat = base_feat.to(self.device)
        for d in range(max_d):
            num_nodes = self.lib.NumBinNodes(d)
            if num_nodes == 0:
                all_bin_feats.append(base_feat)
            else:
                if self.device == torch.device('cpu'):
                    feat = torch.zeros(num_nodes + 2, self.embed_dim)
                    dev = 0
                else:
                    feat = torch.cuda.FloatTensor(num_nodes + 2, self.embed_dim).fill_(0)
                    dev = 1
                self.lib.SetBinaryFeat(d, ctypes.c_void_p(feat.data_ptr()), dev)
                all_bin_feats.append(feat)
        return all_bin_feats, (base_feat, base_feat)

    def PrepareRowEmbed(self):
        tot_levels = self.lib.RowMergeSteps()
        lv = 0
        all_ids = []
        for lv in range(tot_levels):
            ids_d = []
            for i in range(2):
                num_prev = self.lib.NumRowTopDep(lv, i)
                num_bot = self.lib.NumRowBottomDep(i) if lv == 0 else 0
                num_past = self.lib.NumRowPastDep(lv, i)
                bot_froms = np.empty((num_bot,), dtype=np.int32)
                bot_tos = np.empty((num_bot,), dtype=np.int32)
                prev_froms = np.empty((num_prev,), dtype=np.int32)
                prev_tos = np.empty((num_prev,), dtype=np.int32)
                past_froms = np.empty((num_past,), dtype=np.int32)
                past_tos = np.empty((num_past,), dtype=np.int32)
                self.lib.SetRowEmbedIds(i,
                                        lv,
                                        ctypes.c_void_p(bot_froms.ctypes.data),
                                        ctypes.c_void_p(bot_tos.ctypes.data),
                                        ctypes.c_void_p(prev_froms.ctypes.data),
                                        ctypes.c_void_p(prev_tos.ctypes.data),
                                        ctypes.c_void_p(past_froms.ctypes.data),
                                        ctypes.c_void_p(past_tos.ctypes.data))
                ids_d.append((bot_froms, bot_tos, prev_froms, prev_tos, past_froms, past_tos))
            all_ids.append(ids_d)

        return all_ids

    def PrepareRowSummary(self):
        total_steps = self.lib.RowSumSteps()
        all_ids = []
        total_nodes = np.sum(self.list_nnodes)
        init_ids = np.empty((total_nodes,), dtype=np.int32)
        self.lib.SetRowSumInit(ctypes.c_void_p(init_ids.ctypes.data))
        for i in range(total_steps):
            num_done = self.lib.NumRowSumOut(i)
            num_next = self.lib.NumRowSumNext(i)
            step_from = np.empty((num_done,), dtype=np.int32)
            step_to = np.empty((num_done,), dtype=np.int32)

            step_next = np.empty((num_next,), dtype=np.int32)
            step_input = np.empty((num_next,), dtype=np.int32)
            self.lib.SetRowSumIds(i,
                                  ctypes.c_void_p(step_from.ctypes.data),
                                  ctypes.c_void_p(step_to.ctypes.data),
                                  ctypes.c_void_p(step_input.ctypes.data),
                                  ctypes.c_void_p(step_next.ctypes.data))
            all_ids.append((step_from, step_to, step_next, step_input))
            total_nodes -= num_done
        last_ids = np.empty((total_nodes,), dtype=np.int32)
        self.lib.SetRowSumLast(ctypes.c_void_p(last_ids.ctypes.data))

        num_next = self.lib.GetNumNextStates()
        next_ids = np.empty((num_next,), dtype=np.int32)
        self.lib.GetNextStates(ctypes.c_void_p(next_ids.ctypes.data))

        np_pos = np.empty((np.sum(self.list_nnodes),), dtype=np.int32)
        self.lib.GetCurPos(ctypes.c_void_p(np_pos.ctypes.data))
        return init_ids, all_ids, last_ids, next_ids, torch.tensor(np_pos, dtype=torch.float32).to(self.device)

    def GetChLabel(self, lr, depth=-1, dtype=None):
        if lr == 0:
            total_nodes = np.sum(self.list_nnodes)
            has_ch = np.empty((total_nodes,), dtype=np.int32)
            self.lib.HasChild(ctypes.c_void_p(has_ch.ctypes.data))
            num_ch = None
        else:
            n = self.lib.NumInternalNodes(depth)
            has_ch = np.empty((n,), dtype=np.int32)
            self.lib.GetChMask(lr, depth,
                               ctypes.c_void_p(has_ch.ctypes.data))
            num_ch = np.empty((n,), dtype=np.int32)
            self.lib.GetNumCh(lr, depth,
                              ctypes.c_void_p(num_ch.ctypes.data))
            num_ch = torch.tensor(num_ch, dtype=torch.float32).to(self.device)
        if dtype is not None:
            has_ch = has_ch.astype(dtype)
        return has_ch, num_ch
    
    
#     def GetTopdownEdgeIdx(self, max_depth, dtype=None):
#         edge_idx = [None] * max_depth
#         lch = None
#         rch = None
#         for d in range(max_depth - 1, -1, -1):
#             is_nonleaf = self.QueryNonLeaf(d)
#             num_internal = np.sum(is_nonleaf)
#             num_leaves = np.sum(~is_nonleaf)
#             
#             edge_idx_it = np.zeros((num_internal, ), dtype=np.int32)
#             
#             if lch is None:
#                 assert num_internal == 0
#                 cur_edge_idx, _ = self.GetEdgeAndLR(d)
#                 is_nonleaf = self.QueryNonLeaf(d - 1)
#                 num_internal_parents = np.sum(is_nonleaf)
#                 lch = np.array([-1] * num_internal_parents)
#                 rch = np.array([-1] * num_internal_parents)
#                 
#                 is_left, _ = self.GetChLabel(-1, d - 1)
#                 is_right, _ = self.GetChLabel(1, d - 1)
#                 
#                 is_left = lch * (1 - is_left) + is_left
#                 is_right = rch * (1 - is_right) + is_right
#                 
#                 lr = np.concatenate([np.array([x, y]) for x,y in zip(is_left, is_right)])
#                 lr = lr.astype(np.int32)
#                 lr[lr == 1] = cur_edge_idx
#                 lr = lr.reshape(len(is_left), 2)
#                 lch, rch = lr[:, 0], lr[:, 1]
#                 
#                 edge_idx[d] = edge_idx_it
#             
#             else:
#                 mrs = [(lch[i] if lch[i] > -1 else rch[i]) for i in range(len(lch))]
#                 edge_idx_it = np.array(mrs, dtype=np.int32)
#                 mrs = [(rch[i] if rch[i] > -1 else lch[i]) for i in range(len(rch))]
#                 edge_idx[d] = edge_idx_it
#                 if d == 0:
#                     return edge_idx
#                 
#                 is_nonleaf = self.QueryNonLeaf(d)
#                 cur_weights = np.zeros((len(is_nonleaf), ), dtype=np.int32)
#                 cur_edge_idx, _ = self.GetEdgeAndLR(d)
#                 cur_weights[is_nonleaf] = mrs
#                 cur_weights[~is_nonleaf] = cur_edge_idx
#                 is_nonleaf = self.QueryNonLeaf(d - 1)
#                 num_internal_parents = np.sum(is_nonleaf)
#                 lch = np.array([-1] * num_internal_parents)
#                 rch = np.array([-1] * num_internal_parents)
#                 
#                 is_left, _ = self.GetChLabel(-1, d - 1)
#                 is_right, _ = self.GetChLabel(1, d - 1)
#                 is_left = lch * (1 - is_left) + is_left
#                 is_right = rch * (1 - is_right) + is_right
#                 
#                 lr = np.concatenate([np.array([x, y]) for x,y in zip(is_left, is_right)])
#                 lr = lr.astype(np.int32)
#                 lr[lr == 1] = cur_weights
#                 lr = lr.reshape(len(is_left), 2)
#                 lch, rch = lr[:, 0], lr[:, 1]
#         return edge_idx
    

    
    def GetMostRecentWeight(self, max_depth, batch_last_edges=None):
        print("cur max depth: ", max_depth)
        max_d_bin = self.lib.MaxBinFeatDepth()
        print("Max bin: ", max_d_bin)
        max_d_tree = self.lib.MaxTreeDepth()
        print("max tree: ", max_d_tree)
        #max_depth = max_d_bin + max_d_tree + 1 + int(max_d_tree != 0)
        max_depth = max_d_bin + max_depth - (max_d_tree + 1)
        print("guessed max depth: ", max_depth)
        print("ACTUAL max_depth: ", 8)
        max_depth = max_d_bin + max_depth - (max_d_tree + 1)
        
        most_recent_edge_list = [None] * max_depth
        parent_indices = [None] * max_depth
        is_lch_list = [None] * max_depth
        
        for d in range(max_depth - 1, -1, -1):
            cur_lv_nonleaf = self.QueryNonLeaf(d)
            cur_lv_edge, _ = self.GetEdgeAndLR(d)
            
            if d == max_depth - 1:
                cur_weights = cur_lv_edge
            
            else:
                cur_weights = np.zeros(len(cur_lv_nonleaf))
                cur_weights[~cur_lv_nonleaf] = cur_lv_edge
                cur_weights[cur_lv_nonleaf] = mre
            
            if d != max_depth - 1:
                cur_is_left, _ =  self.GetChLabel(-1, d)
                cur_is_right, _ =  self.GetChLabel(1, d)
            
            else:
                cur_is_left = None
                cur_is_right = None
            
            if d == 0:
                left_idx = [None] * max_depth
                right_idx = [None] * max_depth
                
                for lv in range(0, max_depth):
                    cur_left_states = None
                    cur_right_states = None
                    cur_par_idx = parent_indices[lv]
                    cur_edge = most_recent_edge_list[lv]
                    cur_is_lch = is_lch_list[lv]
                    
                    if lv == 0:
                        if batch_last_edges is None:
                            cur_left_states = np.array([-1] * len(cur_edge))
                        
                        else:
                            has_ch, _ = self.GetChLabel(0, dtype=bool)
                            cur_lv_nonleaf = self.QueryNonLeaf(lv)
                            cur_left_states = batch_last_edges[has_ch][cur_lv_nonleaf]
                            assert len(cur_left_states) == len(cur_edge)
                            
                        cur_right_states = np.array([x[0] for x in cur_edge])
                        left_idx[d] = cur_left_states
                        left_idx[d] = cur_right_states
                        par_left_edge = np.array([x[0] for x in cur_edge])
                        par_left_states = cur_left_states
                        par_right_states = cur_right_states
                    
                    elif cur_edge is not None:
                        cur_left_states = np.array([-1] * len(cur_edge))
                        idx = np.array([~x and (y != -1) for x, y in zip(cur_is_lch, par_left_edge[cur_par_idx])])
                        cur_left_states[~idx] = par_left_states[cur_par_idx[~idx]]
                        cur_left_states[idx] = par_left_edge[cur_par_idx[idx]]
                        cur_right_states = np.array([x[0] for x in cur_edge])
                        par_left_edge = np.array([x[0] for x in cur_edge])
                        par_left_states = cur_left_states
                    
                    left_idx[lv] = cur_left_states
                    right_idx[lv] = cur_right_states
                return left_idx, right_idx
            
            up_lv_nonleaf = self.QueryNonLeaf(d - 1)
            up_is_left, _ = self.GetChLabel(-1, d - 1)
            up_is_right, _ = self.GetChLabel(1, d - 1)
            
            num_internal_parents = np.sum(up_lv_nonleaf)
            lch = np.array([-1] * num_internal_parents)
            rch = np.array([-1] * num_internal_parents)
            
            up_is_left = lch * (1 - up_is_left) + up_is_left
            up_is_right = rch * (1 - up_is_right) + up_is_right

            lr = np.concatenate([np.array([x, y]) for x,y in zip(up_is_left, up_is_right)])

            is_lch = np.array([True, False]*len(up_is_left))
            is_lch = is_lch[lr != -1]
            is_lch = is_lch[cur_lv_nonleaf]
            is_lch_list[d] = is_lch
            
            lr = lr.astype(np.int32)
            lr[lr == 1] = cur_weights
            lr = lr.reshape(len(up_is_left), 2)
            lch, rch = lr[:, 0], lr[:, 1]
            
            up_level_lr = np.array([[l, r] for l, r in zip(lch, rch)])
            mre = np.array([x[1] if x[1] != -1 else x[0] for x in up_level_lr])
            most_recent_edge_list[d - 1] = up_level_lr
            
            lch_b = (lch > -1)
            rch_b = (rch > -1)
            num_chil = lch_b.astype(int) + rch_b.astype(int)
            idx_list = list(range(len(num_chil)))
            par_idx = np.array([x for i, x in zip(num_chil, idx_list) for _ in range(i)])
            par_idx = par_idx[cur_lv_nonleaf]
            parent_indices[d] = par_idx
        
#     def GetTopdownEdgeIdx(self, max_depth, dtype=None):
#         edge_idx = [None] * max_depth
#         lch = None
#         rch = None
#         is_nonleaf = self.QueryNonLeaf(max_depth - 1)
#         test_case = [None] * max_depth
#         
#         for d in range(max_depth - 1, -1, -1):
#             num_internal = np.sum(is_nonleaf)
#             num_leaves = np.sum(~is_nonleaf)
#             cur_edge_idx, _ = self.GetEdgeAndLR(d)
#             edge_idx_it = np.zeros((num_internal, ), dtype=np.int32)
#             
#             if lch is not None:
#                 cur_weights = np.zeros((len(is_nonleaf), ), dtype=np.int32)
#                 mrs = [(lch[i] if lch[i] > -1 else rch[i]) for i in range(len(lch))]
#                 edge_idx_it = np.array(mrs, dtype=np.int32)
#                 mrs = [(rch[i] if rch[i] > -1 else lch[i]) for i in range(len(rch))]
#                 cur_weights[is_nonleaf] = mrs
#                 if cur_edge_idx is not None:
#                     cur_weights[~is_nonleaf] = cur_edge_idx
#                 
#             else:
#                 assert num_internal == 0 #At the very deepest level, only leaves should exist
#                 cur_weights = cur_edge_idx
#             
#             edge_idx[d] = edge_idx_it
#             if d == 0:
#                 return edge_idx, _
#             
#             is_nonleaf = self.QueryNonLeaf(d - 1)
#             num_internal_parents = np.sum(is_nonleaf)
#             lch = np.array([-1] * num_internal_parents)
#             rch = np.array([-1] * num_internal_parents)
#             
#             is_left, num_left = self.GetChLabel(-1, d - 1)
#             is_right, num_right = self.GetChLabel(1, d - 1)
#             
#             is_left = lch * (1 - is_left) + is_left
#             is_right = rch * (1 - is_right) + is_right
#             
#             lr = np.concatenate([np.array([x, y]) for x,y in zip(is_left, is_right)])
#             is_lch = np.array([True, False]*len(is_left))
#             is_lch = is_lch[lr != -1]
#             is_nonleaf2 = self.QueryNonLeaf(d)
#             is_lch = is_lch[is_nonleaf2]
#             
#             lr = lr.astype(np.int32)
#             lr[lr == 1] = cur_weights
#             lr = lr.reshape(len(is_left), 2)
#             lch, rch = lr[:, 0], lr[:, 1]
#         return edge_idx, _




    def QueryNonLeaf(self, depth):
        n = self.lib.NumCurNodes(depth)
        if n == 0:
            return None
        is_internal = np.empty((n,), dtype=np.int32)
        self.lib.GetInternalMask(depth, ctypes.c_void_p(is_internal.ctypes.data))
        return is_internal.astype(bool)

    def GetEdgeOf(self, lv):
        n = self.lib.NumEdgesAtLevel(lv)
        if n == 0:
            return None
        edge_idx = np.empty((n,), dtype=np.int32)
        self.lib.GetEdgesOfLevel(lv, ctypes.c_void_p(edge_idx.ctypes.data))
        return edge_idx

    def GetEdgeAndLR(self, lv):
        n = self.lib.NumEdgesAtLevel(lv)
        lr = np.empty((n,), dtype=np.int32)
        self.lib.GetIsEdgeRch(lv, ctypes.c_void_p(lr.ctypes.data))
        edge_idx = self.GetEdgeOf(lv)
        return edge_idx, lr.astype(bool)

    def GetTrivialNodes(self):
        res = []
        for lr in range(2):
            n = self.lib.NumTrivialNodes(lr)
            nodes = np.empty((n,), dtype=np.int32)
            self.lib.GetTrivialNodes(lr, ctypes.c_void_p(nodes.ctypes.data))
            res.append(nodes)
        return res

    def GetLeftRootStates(self, depth):
        n = self.lib.NumInternalNodes(depth)
        left_bot = self.lib.NumLeftBot(depth)
        left_next = n - left_bot
        bot_froms = np.empty((left_bot,), dtype=np.int32)
        bot_tos = np.empty((left_bot,), dtype=np.int32)
        next_froms = np.empty((left_next,), dtype=np.int32)
        next_tos = np.empty((left_next,), dtype=np.int32)
        self.lib.SetLeftState(depth,
                              ctypes.c_void_p(bot_froms.ctypes.data),
                              ctypes.c_void_p(bot_tos.ctypes.data),
                              ctypes.c_void_p(next_froms.ctypes.data),
                              ctypes.c_void_p(next_tos.ctypes.data))
        if left_bot == 0:
            bot_froms = bot_tos = None
        if left_next == 0:
            next_froms = next_tos = None
        return bot_froms, bot_tos, next_froms, next_tos

    def GetLeftRightSelect(self, depth, num_left, num_right):
        left_froms = np.empty((num_left,), dtype=np.int32)
        left_tos = np.empty((num_left,), dtype=np.int32)
        right_froms = np.empty((num_right,), dtype=np.int32)
        right_tos = np.empty((num_right,), dtype=np.int32)

        self.lib.LeftRightSelect(depth,
                                 ctypes.c_void_p(left_froms.ctypes.data),
                                 ctypes.c_void_p(left_tos.ctypes.data),
                                 ctypes.c_void_p(right_froms.ctypes.data),
                                 ctypes.c_void_p(right_tos.ctypes.data))
        return left_froms, left_tos, right_froms, right_tos

    def GetIsTreeTrivial(self):
        total_nodes = np.sum(self.list_nnodes)
        tree_trivial = np.empty((total_nodes,), dtype=np.int32)
        self.lib.TreeTrivial(ctypes.c_void_p(tree_trivial.ctypes.data))
        return tree_trivial.astype(bool)

    def GetFenwickBase(self):
        num_nodes = self.lib.NumBaseNodes()
        nodes = np.empty((num_nodes,), dtype=np.int32)
        num_edges = self.lib.NumBaseEdges()
        edges = np.empty((num_edges,), dtype=np.int32)
        assert num_edges == 0  # code not tested when this > 0
        self.lib.GetBaseNodeEdge(ctypes.c_void_p(nodes.ctypes.data),
                                 ctypes.c_void_p(edges.ctypes.data))
        return nodes, edges

TreeLib = _tree_lib()

def setup_treelib(config):
    global TreeLib
    dll_path = '%s/build/dll/libtree.so' % os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(dll_path):
        TreeLib.setup(config)