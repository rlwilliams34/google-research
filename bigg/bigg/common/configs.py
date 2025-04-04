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

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# pylint: skip-file
import argparse
import os
import pickle as cp
import torch


cmd_opt = argparse.ArgumentParser(description='Argparser for model runs', allow_abbrev=False)

cmd_opt.add_argument('-save_dir', default='.', help='result output root')
cmd_opt.add_argument('-data_dir', default='.', help='data dir')
cmd_opt.add_argument('-eval_folder', default=None, help='data eval_dir')
cmd_opt.add_argument('-train_method', default='full', help='full/stage')
cmd_opt.add_argument('-phase', default='train', help='train/test')
cmd_opt.add_argument('-g_type', default=None, help='graph type')
cmd_opt.add_argument('-model_dump', default=None, help='load model dump')
cmd_opt.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')
cmd_opt.add_argument('-num_proc', type=int, default=1, help='number of processes')
cmd_opt.add_argument('-node_order', default='default', help='default/DFS/BFS/degree_descent/degree_accent/k_core/all, or any of them concat by +')

cmd_opt.add_argument('-dist_backend', default='gloo', help='dist package backend', choices=['gloo', 'nccl'])

cmd_opt.add_argument('-embed_dim', default=256, type=int, help='embed size')
cmd_opt.add_argument('-bits_compress', default=0, type=int, help='num of bits to compress')
cmd_opt.add_argument('-param_layers', default=1, type=int, help='num of param groups')
cmd_opt.add_argument('-num_test_gen', default=-1, type=int, help='num of graphs generated for test')
cmd_opt.add_argument('-max_num_nodes', default=-1, type=int, help='max num of nodes')


cmd_opt.add_argument('-rnn_layers', default=2, type=int, help='num layers in rnn')
cmd_opt.add_argument('-seed', default=34, type=int, help='seed')
cmd_opt.add_argument('-learning_rate', default=1e-3, type=float, help='learning rate')
cmd_opt.add_argument('-grad_clip', default=3, type=float, help='gradient clip')
cmd_opt.add_argument('-train_ratio', default=0.8, type=float, help='ratio for training')
cmd_opt.add_argument('-dev_ratio', default=0.2, type=float, help='ratio for dev')
cmd_opt.add_argument('-greedy_frac', default=0, type=float, help='prob for greedy decode')

cmd_opt.add_argument('-num_epochs', default=100000, type=int, help='num epochs')
cmd_opt.add_argument('-batch_size', default=10, type=int, help='batch size')
cmd_opt.add_argument('-pos_enc', default=True, type=eval, help='pos enc?')
cmd_opt.add_argument('-pos_base', default=10000, type=int, help='base of pos enc')

cmd_opt.add_argument('-old_model', default=False, type=eval, help='old model dumps?')

cmd_opt.add_argument('-tree_pos_enc', default=False, type=eval, help='pos enc for tree?')

cmd_opt.add_argument('-blksize', default=-1, type=int, help='num blksize steps')
cmd_opt.add_argument('-accum_grad', default=1, type=int, help='accumulate grad for batching purpose')

cmd_opt.add_argument('-epoch_save', default=50, type=int, help='num epochs between save')
cmd_opt.add_argument('-epoch_load', default=None, type=int, help='epoch for loading')

cmd_opt.add_argument('-batch_exec', default=False, type=eval, help='run with dynamic batching?')

cmd_opt.add_argument('-share_param', default=True, type=eval, help='share param in each level?')
cmd_opt.add_argument('-directed', default=False, type=eval, help='is directed graph?')
cmd_opt.add_argument('-self_loop', default=False, type=eval, help='has self-loop?')
cmd_opt.add_argument('-bfs_permute', default=False, type=eval, help='random permute with bfs?')
cmd_opt.add_argument('-display', default=False, type=eval, help='display progress?')

cmd_opt.add_argument('-has_edge_feats', default=False, type=eval, help='has edge features?')
cmd_opt.add_argument('-has_node_feats', default=False, type=eval, help='has node features?')

cmd_opt.add_argument('-leaves', default=10, type=int, help='leaves in trees')
cmd_opt.add_argument('-by_time', default=False, type=bool, help='order tree by time?')

cmd_opt.add_argument('-num_lobster_nodes', default=80, type=int, help='leaves in trees')
cmd_opt.add_argument('-p1', default=0.7, type=float, help='leaves in trees')
cmd_opt.add_argument('-p2', default=0.7, type=float, help='leaves in trees')
cmd_opt.add_argument('-min_nodes', default=5, type=int, help='leaves in trees')
cmd_opt.add_argument('-max_nodes', default=100, type=int, help='leaves in trees')
cmd_opt.add_argument('-wt_mode', default='None', type=str, help='mode to standardize weights')
cmd_opt.add_argument('-save_every', default=50, type=int, help='mode to standardize weights')
cmd_opt.add_argument('-method', default='MLP-Repeat', type=str, help='mode to standardize weights')

cmd_opt.add_argument('-wt_drop', default=-1, type=float, help='dropout for weight MLPs. -1 signifies NO dropout')
cmd_opt.add_argument('-val_every', default=10, type=int, help='dropout for weight MLPs. -1 signifies NO dropout')

## GCN
cmd_opt.add_argument('-node_embed_dim', default=128, type=int, help='embed size')
cmd_opt.add_argument('-out_dim', default=128, type=int, help='embed size')
cmd_opt.add_argument('-model', default = "BiGG_E", type = str, help = "BiGG_E or BiGG_GCN?")
cmd_opt.add_argument('-scale_loss', default=1, type=float, help='Amount to scale weight loss by during training')
cmd_opt.add_argument('-schedule', default=False, type=eval, help='Amount to scale weight loss by during training')
cmd_opt.add_argument('-sigma', default=0.0, type=eval, help='Tuning parameter for noise drawn from N(0, s) added to weights during training')
cmd_opt.add_argument('-noise', default=0.0, type=eval, help='Tuning parameter for noise drawn from N(0, s) added to weights during training')


cmd_opt.add_argument('-weight_embed_dim', default=16, type=int, help='embed size for weights')

cmd_opt.add_argument('-training_time', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-num_leaves', default=100, type=int, help='for scalability test')
cmd_opt.add_argument('-sampling_method', default='softplus', type=str, help='for scalability test')
cmd_opt.add_argument('-update_left', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-eps', default=1e-3, type=float, help='noise to be added to training data')
cmd_opt.add_argument('-debug', default=False, type=eval, help='computing training times')

cmd_opt.add_argument('-epoch_plateu', default=-1, type=int, help='computing training times')
cmd_opt.add_argument('-alt_9', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-update_ll', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-row_LSTM', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-test_topdown', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-wt_mlp', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-test', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-test2', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-test3', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-test_sep', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-wt_one_layer', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-add_states', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-penalty', default=False, type=eval, help='computing training times')

cmd_opt.add_argument('-proj', default=False, type=eval, help='computing training times')
cmd_opt.add_argument('-proj_dim', default=64, type=int, help='computing training times')
cmd_opt.add_argument('-comb_states', default=False, type=eval, help='computing training times')


cmd_args, _ = cmd_opt.parse_known_args()

if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

if cmd_args.epoch_load == -1:
    cmd_args.epoch_load = cmd_args.num_epochs

if cmd_args.epoch_load is not None and cmd_args.model_dump is None:
    cmd_args.model_dump = os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % cmd_args.epoch_load)

print(cmd_args)

def set_device(gpu):
    if torch.cuda.is_available() and gpu >= 0:
        cmd_args.gpu = gpu
        cmd_args.device = torch.device('cuda:' + str(gpu))
        print('use gpu indexed: %d' % gpu)
    else:
        cmd_args.gpu = -1
        cmd_args.device = torch.device('cpu')
        print('use cpu')
