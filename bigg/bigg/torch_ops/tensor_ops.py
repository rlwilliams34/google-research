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
# pylint: skip-file

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.autograd import Function
import numpy as np
from bigg.common.consts import t_float


class MultiIndexSelectFunc(Function):
    @staticmethod
    def forward(ctx, idx_froms, idx_tos, *mats):
        assert len(idx_tos) == len(idx_froms) == len(mats)
        cols = mats[0].shape[1]
        assert all([len(x.shape) == 2 for x in mats])
        assert all([x.shape[1] == cols for x in mats])

        num_rows = sum([len(x) for x in idx_tos])
        out = mats[0].new(num_rows, cols)

        for i, mat in enumerate(mats):
            x_from = idx_froms[i]
            x_to = idx_tos[i]
            if x_from is None:
                out[x_to] = mat.detach()
            else:
                assert len(x_from) == len(x_to)
                out[x_to] = mat[x_from].detach()

        ctx.idx_froms = idx_froms
        ctx.idx_tos = idx_tos
        ctx.shapes = [x.shape for x in mats]
        return out

    @staticmethod
    def backward(ctx, grad_output):
        idx_froms, idx_tos = ctx.idx_froms, ctx.idx_tos

        list_grad_mats = [None, None]
        for i in range(len(idx_froms)):
            x_from = idx_froms[i]
            x_to = idx_tos[i]
            if x_from is None:
                grad_mat = grad_output[x_to].detach()
            else:
                grad_mat = grad_output.new(ctx.shapes[i]).zero_()
                grad_mat[x_from] = grad_output[x_to].detach()
            list_grad_mats.append(grad_mat)

        return tuple(list_grad_mats)


class MultiIndexSelect(Module):
    def forward(self, idx_froms, idx_tos, *mats):
        return MultiIndexSelectFunc.apply(idx_froms, idx_tos, *mats)

multi_index_select = MultiIndexSelect()

def test_multi_select():
    a = Parameter(torch.randn(4, 2))
    b = Parameter(torch.randn(3, 2))
    d = Parameter(torch.randn(5, 2))

    idx_froms = [[0, 1], [1, 2], [3, 4]]
    idx_tos = [[4, 5], [0, 1], [2, 3]]
    c = multi_index_select(idx_froms, idx_tos, a, b, d)
    print('===a===')
    print(a)
    print('===b===')
    print(b)
    print('===d===')
    print(d)
    print('===c===')
    print(c)

    t = torch.sum(c)
    t.backward()
    print(a.grad)
    print(b.grad)
    print(d.grad)


class PosEncoding(Module):
    def __init__(self, dim, device, base=10000, bias=0):
        super(PosEncoding, self).__init__()

        p = []
        sft = []
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)
            else:
                sft.append(bias)
        self.device = device
        self.sft = torch.tensor(sft, dtype=t_float).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=t_float).view(1, -1).to(device)

    def forward(self, pos):
        with torch.no_grad():
            if isinstance(pos, list):
                pos = torch.tensor(pos, dtype=t_float).to(self.device)
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft
            return torch.sin(x)


if __name__ == '__main__':
    # test_multi_select()

    pos_enc = PosEncoding(128, 'cpu')
    print(pos_enc([1, 2, 3]))






















































































# 
# # coding=utf-8
# # Copyright 2024 The Google Research Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# 
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# # pylint: skip-file
# 
# import torch
# from torch.nn import Module
# from torch.nn.parameter import Parameter
# from torch.autograd import Function
# import numpy as np
# from bigg.common.consts import t_float
# 
# 
# class MultiIndexSelectFunc(Function):
#     @staticmethod
#     def forward(ctx, idx_froms, idx_tos, *mats):
#         assert len(idx_tos) == len(idx_froms) == len(mats)
#         cols = mats[0].shape[2]
#         assert all([len(x.shape) == 3 for x in mats])
#         assert all([x.shape[2] == cols for x in mats])
#         #print(mats)
#         #print("idx_froms")
#         #print(idx_froms)
#         #print("idx_tos")
#         #print(idx_tos)
#         #print("mats")
#         #print(mats)
#         #print("mats above")
#         
# 
#         num_rows = sum([len(x) for x in idx_tos])
#         num_layers = mats[0].shape[0]
#         out = mats[0].new(num_layers, num_rows, cols)
#         #print("out")
#         #print(out)
#         #print("out above")
#         #print(mats[0])
# 
#         for i, mat in enumerate(mats):
#             #print(i)
#             x_from = idx_froms[i]
#             x_to = idx_tos[i]
#             if x_from is None:
#                 #print(out)
#                 #print(x_to)
#                 #print(mat)
#                 for layer in range(mat.shape[0]):
#                     out[layer][x_to] = mat[layer].detach()
#             else:
#                 assert len(x_from) == len(x_to)
#                 
#                 #print("out")
#                 #print(out)
#                 #print(out.shape)
#                 #print("x_to")
#                 #print(x_to)
#                 #print("max")
#                 #print(mat)
#                 #print(mat.shape)
#                 #print("x_from")
#                 #print(x_from)
#                 for layer in range(mat.shape[0]):
#                     out[layer][x_to] = mat[layer][x_from].detach()
#                 #out[x_to] = mat[x_from].detach()
# 
#         ctx.idx_froms = idx_froms
#         ctx.idx_tos = idx_tos
#         ctx.shapes = [x.shape for x in mats]
#         return out
# 
#     @staticmethod
#     def backward(ctx, grad_output):
#         idx_froms, idx_tos = ctx.idx_froms, ctx.idx_tos
#         #print(grad_output)
#         #print(idx_froms)
#         #print(idx_tos)
# 
#         list_grad_mats = [None, None]
#         grad_mat = None
#         for i in range(len(idx_froms)):
#             x_from = idx_froms[i]
#             x_to = idx_tos[i]
#             #print(x_from)
#             #print(x_to)
#             if x_from is None:
#                 #print("Hello Checking X from is None!")
#                 #print("x_to")
#                 #print(x_to)
#                 #print("grad_mat")
#                 #print(grad_mat)
#                 #print("grad output")
#                 #print(grad_output)
#                 #print("grad output 0 x to")
#                 #print(grad_output[0][x_to])
#                 #print("grad mat 0")
#                 #print(grad_mat[0])
#                 if True:
#                     grad_mat_list = []
#                     for layer in range(grad_output.shape[0]):
#                         grad_mat_list.append(grad_output[layer][x_to].unsqueeze(0).detach())
#                     
#                     grad_mat = torch.cat(grad_mat_list, dim = 0)
#                 
#                 else:
#                     for layer in range(grad_output.shape[0]):
#                         grad_mat[layer] = grad_output[layer][x_to].detach()
#             
#             else:
#                 #print("Hello 2")
#                 grad_mat = grad_output.new(ctx.shapes[i]).zero_()
#                 for layer in range(grad_output.shape[0]):
#                     grad_mat[layer][x_from] = grad_output[layer][x_to].detach()
#             list_grad_mats.append(grad_mat)
# 
#         return tuple(list_grad_mats)
# 
# 
# class MultiIndexSelect(Module):
#     def forward(self, idx_froms, idx_tos, *mats):
#         return MultiIndexSelectFunc.apply(idx_froms, idx_tos, *mats)
# 
# multi_index_select = MultiIndexSelect()
# 
# def test_multi_select():
#     a = Parameter(torch.randn(4, 2))
#     b = Parameter(torch.randn(3, 2))
#     d = Parameter(torch.randn(5, 2))
# 
#     idx_froms = [[0, 1], [1, 2], [3, 4]]
#     idx_tos = [[4, 5], [0, 1], [2, 3]]
#     c = multi_index_select(idx_froms, idx_tos, a, b, d)
#     print('===a===')
#     print(a)
#     print('===b===')
#     print(b)
#     print('===d===')
#     print(d)
#     print('===c===')
#     print(c)
# 
#     t = torch.sum(c)
#     t.backward()
#     print(a.grad)
#     print(b.grad)
#     print(d.grad)
# 
# 
# class PosEncoding(Module):
#     def __init__(self, dim, device, base=10000, bias=0):
#         super(PosEncoding, self).__init__()
# 
#         p = []
#         sft = []
#         for i in range(dim):
#             b = (i - i % 2) / dim
#             p.append(base ** -b)
#             if i % 2:
#                 sft.append(np.pi / 2.0 + bias)
#             else:
#                 sft.append(bias)
#         self.device = device
#         self.sft = torch.tensor(sft, dtype=t_float).view(1, -1).to(device)
#         self.base = torch.tensor(p, dtype=t_float).view(1, -1).to(device)
# 
#     def forward(self, pos):
#         with torch.no_grad():
#             if isinstance(pos, list):
#                 pos = torch.tensor(pos, dtype=t_float).to(self.device)
#             pos = pos.view(-1, 1)
#             x = pos / self.base + self.sft
#             return torch.sin(x)
# 
# 
# if __name__ == '__main__':
#     # test_multi_select()
# 
#     pos_enc = PosEncoding(128, 'cpu')
#     print(pos_enc([1, 2, 3]))
