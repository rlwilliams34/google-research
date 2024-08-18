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

g_type=lobster
ordering=DFS
blksize=-1
bsize=128

data_dir=../../../../data/$g_type-$ordering

save_dir=../../../../results/$g_type/$ordering-blksize-$blksize-b-$bsize

if [ ! -e $save_dir ];
then
  mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python ../batch_train.py \
  -data_dir $data_dir \
  -save_dir $save_dir \
  -g_type $g_type \
  -node_order $ordering \
  -num_graphs $num_g \
  -blksize $blksize \
  -epoch_save 500 \
  -bits_compress 256 \
  -batch_size $bsize \
  -num_test_gen 100 \
  -num_epochs 1000 \
  -gpu 0 \
  $@



### BLKSIZE = -1
### BATCHSIZE 128
### BITS COMPRESS 256
### Epoch_SAVE 500
### NUM_EPOCHS 10000

