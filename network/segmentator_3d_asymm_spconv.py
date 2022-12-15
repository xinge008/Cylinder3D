# -*- coding:utf-8 -*-
# author: Xinge
# @file: segmentator_3d_asymm_spconv.py

import numpy as np
#import spconv
import spconv.pytorch as spconv
import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()
          
        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef2")
        # self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")

        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef4")
        # self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        reaA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))
        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef2")
        # self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef4")
        # self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))

        resA = resA.replace_feature(resA.features + shortcut.features)

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA

class Attention_Block_Test(nn.Module):
    def __init__(self, gating_features, shortcut_features, n_coefficients, kernel_size, indice_key = None, up_key = None):
        super(Attention_Block_Test, self).__init__()

        # shortcut block 
        self.shortcut_block = spconv.SparseSequential(
            spconv.SparseConv3d(shortcut_features, n_coefficients, 3, stride = 2, padding = 1, indice_key= indice_key + 'shortcut_sparse'),
            nn.BatchNorm1d(n_coefficients),
            nn.LeakyReLU(),
        )
        
        # gating_signal block
        self.gating_block = spconv.SparseSequential(
            spconv.SparseConv3d(gating_features, n_coefficients, 3, stride = 1, padding = 1, indice_key= indice_key + 'gating_sparse'),
            nn.BatchNorm1d(n_coefficients),
            nn.LeakyReLU(),
            spconv.SparseInverseConv3d(n_coefficients, n_coefficients, kernel_size= 3, indice_key = indice_key + 'shortcut_sparse'),
            nn.BatchNorm1d(n_coefficients),
            nn.LeakyReLU(),
        )

        # sum block
        self.sum_block = spconv.SparseSequential(
            conv3x3(n_coefficients, 1, stride= 1, indice_key= indice_key + 'sum_subm'),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
            spconv.SparseInverseConv3d(1, shortcut_features, kernel_size= 3, indice_key= up_key),
            nn.BatchNorm1d(shortcut_features),
            nn.Sigmoid(),
        )

    def forward(self, shortcut, gating):

        phi_g = self.shortcut_block(shortcut)

        theta_x = self.gating_block(gating)

        sum_up = phi_g.replace_features(phi_g.features + theta_x.features)

        sum_up = self.sum_block(sum_up)

        return sum_up

class Attention_Gate(nn.Module):

    def __init__(self, gating_features, shortcut_features, n_coefficients, kernel_size, indice_key = None, up_key = None):
        super(Attention_Gate, self).__init__()

        self.shortcut_conv = spconv.SparseConv3d(shortcut_features, n_coefficients, kernel_size= 3, stride = 2, padding = 1, indice_key= indice_key + 'shortcut_sparse')
        self.shortcut_bn = nn.BatchNorm1d(n_coefficients)
        self.shortcut_act = nn.LeakyReLU()

        self.gate_conv = spconv.SparseConv3d(gating_features, n_coefficients, kernel_size= 3, stride = 1, padding = 1, indice_key= indice_key + 'gate_sparse')
        self.gate_bn = nn.BatchNorm1d(n_coefficients)
        self.gate_act = nn.LeakyReLU()
        self.gate_up = spconv.SparseInverseConv3d(n_coefficients, n_coefficients, kernel_size= 3, indice_key= indice_key + 'shortcut_sparse')
        self.gate_up_bn = nn.BatchNorm1d(n_coefficients)
        self.gate_up_act = nn.LeakyReLU()

        self.sum_conv = conv3x3(n_coefficients, 1, stride= 1, indice_key= indice_key + 'sum_conv')
        self.sum_bn = nn.BatchNorm1d(1)
        self.sum_act = nn.LeakyReLU()
        self.up_sum = spconv.SparseInverseConv3d(1, shortcut_features, kernel_size= 3, indice_key= up_key)
        self.up_sum_bn = nn.BatchNorm1d(shortcut_features)
        self.up_sum_act = nn.Sigmoid()
    
    def forward(self, gate, shortcut):
        
        theta_x = self.shortcut_conv(shortcut)
        theta_x = self.replace_features(self.shortcut_bn(theta_x.features))
        theta_x = self.replace_features(self.shortcut_act(theta_x.features))

        phi_g = self.gate_conv(gate)
        phi_g = phi_g.replace_features(self.gate_bn(phi_g.features))
        phi_g = phi_g.replace_features(self.gate_act(phi_g.features))
        phi_g = self.gate_up(phi_g)
        phi_g = phi_g.replace_features(self.gate_up_bn(phi_g.features))
        phi_g = phi_g.replace_features(self.gate_up_act(phi_g.features))

        big_sum = phi_g.replace_features(phi_g.features + theta_x.features)
        big_sum = self.sum_conv(big_sum)
        big_sum = big_sum.replace_features(self.sum_bn(big_sum.features))
        big_sum = big_sum.replace_features(self.sum_act(big_sum.features))
        big_sum = self.up_sum(big_sum)
        big_sum = big_sum.replace_features(self.up_sum_bn(big_sum.features))
        big_sum = big_sum.replace_features(self.up_sum_act(big_sum.features))

        return big_sum

# class Attention(nn.Module):

#     def __init__(self, gating_features, shortcut_features, n_coefficients, kernel_size, indice_key = None, up_key = None):
#         super(Attention, self).__init__()

#         # gating signal block
#         self.gate_conv = conv3x3(gating_features, n_coefficients, stride= 1, indice_key= indice_key + 'gate_subm')
#         self.gate_act = nn.LeakyReLU()
#         self.gate_bn = nn.BatchNorm1d(n_coefficients)
#         self.up_subm = spconv.SparseInverseConv3d(n_coefficients, n_coefficients, kernel_size= 3, indice_key= up_key)

#         # shortcut block
#         self.shortcut_conv = conv3x3(shortcut_features, n_coefficients, stride= 1, indice_key= indice_key + 'short_subm')
#         self.shortcut_act = nn.LeakyReLU()
#         self.shortcut_bn = nn.BatchNorm1d(n_coefficients)

#         # psi block
#         self.sum_pre_act = nn.LeakyReLU()
#         self.sum_conv = conv3x3(n_coefficients, 1, stride= 1, indice_key= indice_key + 'sum_subm')
#         self.sum_act = nn.Sigmoid()
#         self.sum_bn = nn.BatchNorm1d(1)

#         self.sum_up_subm = spconv.SparseInverseConv3d(1, shortcut_features, kernel_size= 3, indice_key= indice_key + 'sum_subm')

#     def forward(self, gating, shortcut):
#         phi_g = self.gate_conv(gating)
#         phi_g = phi_g.replace_features(self.gate_act(phi_g.features))
#         phi_g = phi_g.replace_features(self.gate_bn(phi_g.features))
#         phi_g = self.up_subm(phi_g)

#         pass

# class Attention_Block(nn.Module):

#     def __init__(self, gating_features, shortcut_features, n_coefficients, kernel_size, indice_key = None, up_key = None):
#         super(Attention_Block, self).__init__()

#         # gating signal block
#         self.gate_conv = conv1x1(gating_features, n_coefficients, stride= 1, indice_key= 'gate_subm')
#         self.gate_bn0 = nn.BatchNorm1d(n_coefficients)
#         self.gate_act0 = nn.ReLU()
#         self.gate_pool = spconv.SparseConv3d(n_coefficients, n_coefficients, kernel_size= (1, 1, 1), stride= 1, indice_key= 'gate_pool')
#         self.gate_bn1 = nn.BatchNorm1d(n_coefficients)
#         self.gate_act1 = nn.ReLU()

#         # shortcut block
#         self.shortcut_conv = conv1x1(shortcut_features, n_coefficients, stride= 1, indice_key= 'shortcut_subm')
#         self.shortcut_bn0 = nn.BatchNorm1d(n_coefficients)
#         self.shortcut_act0 = nn.ReLU()
#         self.shortcut_pool = spconv.SparseConv3d(n_coefficients, n_coefficients, kernel_size= (3, 3, 3), stride= 2, indice_key= 'shortcut_pool')
#         self.shortcut_bn1 = nn.BatchNorm1d(n_coefficients)
#         self.shortcut_act1 = nn.ReLU()

#         # concatenate block
#         self.concat_pre_act = nn.ReLU()
#         self.concat_conv = conv1x1(n_coefficients, 1, stride= 1, indice_key= 'concat_subm')
#         self.concat_bn0 = nn.BatchNorm1d(1)
#         self.concat_act0 = nn.Sigmoid()
#         self.concat_pool = spconv.SparseConv3d(1, 1, kernel_size= 3, stride= 2, padding= 1, indice_key= 'concat_pool')
#         self.concat_bn1 = nn.BatchNorm1d(1)
#         self.concat_act1 = nn.Sigmoid()

#         #up sampling block
#         self.concat_inv_conv = spconv.SparseInverseConv3d(1, shortcut_features, kernel_size= 3, indice_key= 'concat_pool')

# class AttentionBlock(nn.Module):
#     def __init__(self, gating_features, shortcut_features, n_coefficients, kernel_size= (1, 1, 1), indice_key= None, up_key = None):
#         super(AttentionBlock, self).__init__()
#         # gating signal block
#         self.up_subm = spconv.SparseInverseConv3d(gating_features, n_coefficients, kernel_size= 3, bias= False, indice_key = up_key)
#         self.gate_conv = conv1x1(n_coefficients, n_coefficients, stride = 1, indice_key=indice_key + 'gc0')
#         # self.gate_sparse = spconv.SparseConv3d(n_coefficients, n_coefficients, kernel_size= 3, stride = 1, padding=1, indice_key = 'cc0')
#         self.gate_bn = nn.BatchNorm1d(n_coefficients)
#         self.gate_act = nn.ReLU()
#         # shortcut block
#         self.shortcut_conv = conv1x1(shortcut_features, n_coefficients, stride = 2, indice_key= indice_key + 'sc0')
#         self.shortcut_bn = nn.BatchNorm1d(n_coefficients)
#         self.shortcut_act = nn.ReLU()

#         # psi block
#         self.concat_pre_act = nn.ReLU()
#         self.concat_conv = conv1x1(n_coefficients, 1, stride = 1, indice_key = indice_key + 'cc0')
#         self.concat_bn = nn.BatchNorm1d(1)
#         self.concat_act = nn.Sigmoid()

#         # up_sample block
#         self.concat_inv_conv = spconv.SparseInverseConv3d(1, shortcut_features, kernel_size= 3, indice_key= 'cc0')

#     def forward(self, g, x):
#         phi_g = self.up_subm(phi_g)
#         # phi_g = self.gate_sparse(phi_g)
#         import pdb
#         pdb.set_trace()
#         phi_g = self.gate_conv(g)
#         phi_g = phi_g.replace_feature(self.gate_bn(phi_g.features))
#         phi_g = phi_g.replace_feature(self.gate_act(phi_g.features))

#         theta_x = self.shortcut_conv(x)
#         theta_x = theta_x.replace_feature(self.shortcut_bn(theta_x.features))
#         theta_x = theta_x.replace_feature(self.shortcut_act(theta_x.features))

#         concat = phi_g.replace_feature(phi_g.features + theta_x.features)
#         concat = concat.replace_feature(self.concat_pre_act(concat.features))

#         concat = self.concat_conv(concat)
#         concat = concat.replace_feature(self.concat_bn(concat.features))
#         concat = concat.replace_feature(self.concat_act(concat.features))

#         up_sample = self.concat_inv_conv(concat)
#         y = up_sample.replace_feature(x.features * up_sample.features)
#         return y

class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key+'up1')
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key+'up2')
        # self.conv2 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key+'up3')
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA = upA.replace_feature(self.trans_bn(upA.features))

        ## upsample
        upA = self.up_subm(upA)

        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        upE = upE.replace_feature(self.act1(upE.features))
        upE = upE.replace_feature(self.bn1(upE.features))

        upE = self.conv2(upE)
        upE = upE.replace_feature(self.act2(upE.features))
        upE = upE.replace_feature(self.bn2(upE.features))

        upE = self.conv3(upE)
        upE = upE.replace_feature(self.act3(upE.features))
        upE = upE.replace_feature(self.bn3(upE.features))

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef2")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut.replace_feature(self.bn0_3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))
        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)

        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.AG0 = Attention_Gate(16 * init_size, 8 * init_size, indice_key = "AG0", up_key = 'down4')
        self.AG1 = Attention_Gate(8 * init_size, 4 * init_size, indice_key = "AG1", up_key = 'down3')
        self.AG2 = Attention_Gate(4 * init_size, 2 * init_size, indice_key = "AG2", up_key = 'down2')


        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        up4e = self.upBlock0(down4c, down4b)
        #a4e = self.AttentionBlock1(up4e, down3b)
        #up3e = self.upBlock1(up4e, a4e)
        up3e = self.upBlock1(up4e, down3b)
        #a3e = 
        up2e = self.upBlock2(up3e, down2b)
        #a2e = 
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y