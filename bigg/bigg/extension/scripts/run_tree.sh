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

#!/bin/bash

g_type=tree
ordering=DFS
blksize=-1
bsize=50
accum_grad=2

data_dir=/u/home/r/rlwillia/ADJ-LSTM/train_graphs/$g_type

save_dir=../../../bigg-results/$g_type

if [ ! -e $save_dir ];
then
  mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python ../main_featured.py \
  -$@ \
  -data_dir $data_dir \
  -save_dir $save_dir \
  -g_type $g_type \
  -node_order $ordering \
  -num_graphs $num_g \
  -blksize $blksize \
  -epoch_save 500 \
  -bits_compress 0 \
  -batch_size $bsize \
  -num_test_gen 100 \
  -num_epochs 100 \
  -gpu 0 \
  -has_node_feats 0 \
  -has_edge_feats 1 \
  -accum_grad $accum_grad \
  $@

