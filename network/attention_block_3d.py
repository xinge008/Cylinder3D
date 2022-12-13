import torch
import torch.nn as nn
# from .segmentator_3d_asymm_spconv import conv1x1
# import spconv
# print(spconv.__version__)
import spconv.pytorch as spconv

def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)

def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)

def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)

class AttentionBlock(nn.Module):
    def __init__(self, gating_features, shortcut_features, n_coefficients, kernel_size= (1, 1, 1), indice_key= None, up_key = None):
        super(AttentionBlock, self).__init__()
        # gating signal block
        self.up_subm = spconv.SparseInverseConv3d(n_coefficients, n_coefficients, kernel_size= 3, bias= False, indice_key = up_key)
        self.gate_conv = conv1x1(gating_features, n_coefficients, stride = 1, indice_key=indice_key + 'gc0')
        # self.gate_sparse = spconv.SparseConv3d(n_coefficients, n_coefficients, kernel_size= 3, stride = 1, padding=1, indice_key = 'cc0')
        self.gate_bn = nn.BatchNorm1d(n_coefficients)
        self.gate_act = nn.ReLU()
        # shortcut block
        self.shortcut_conv = conv1x1(shortcut_features, n_coefficients, stride = 2, indice_key= indice_key + 'sc0')
        self.shortcut_bn = nn.BatchNorm1d(n_coefficients)
        self.shortcut_act = nn.ReLU()

        # psi block
        self.concat_pre_act = nn.ReLU()
        self.concat_conv = conv1x1(n_coefficients, 1, stride = 1, indice_key = indice_key + 'cc0')
        self.concat_bn = nn.BatchNorm1d(1)
        self.concat_act = nn.Sigmoid()

        # up_sample block
        self.concat_inv_conv = spconv.SparseInverseConv3d(1, shortcut_features, kernel_size= 3, indice_key= 'cc0')

    def forward(self, g, x):
        phi_g = self.up_subm(phi_g)
        # phi_g = self.gate_sparse(phi_g)
        import pdb
        pdb.set_trace()
        phi_g = self.gate_conv(g)
        phi_g = phi_g.replace_feature(self.gate_bn(phi_g.features))
        phi_g = phi_g.replace_feature(self.gate_act(phi_g.features))

        theta_x = self.shortcut_conv(x)
        theta_x = theta_x.replace_feature(self.shortcut_bn(theta_x.features))
        theta_x = theta_x.replace_feature(self.shortcut_act(theta_x.features))

        concat = phi_g.replace_feature(phi_g.features + theta_x.features)
        concat = concat.replace_feature(self.concat_pre_act(concat.features))

        concat = self.concat_conv(concat)
        concat = concat.replace_feature(self.concat_bn(concat.features))
        concat = concat.replace_feature(self.concat_act(concat.features))

        up_sample = self.concat_inv_conv(concat)
        y = up_sample.replace_feature(x.features * up_sample.features)

# class AG_Block(nn.Module):
#     def __init__(self, gate_filters, skip_filters, out_filters, indice_key= None, up_key= None):
#         super(AG_Block, self).__init__()
#         self.gate_conv = conv1x1(gate_filters, out_filters, stride = 1)
#         self.gate_bn = nn.BatchNorm1d(out_filters)
#         self.gate_act = nn.ReLU()
#         self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size= 3, indice_key= up_key, bias= False)

#         self.skip_conv = conv1x1(skip_filters, out_filters, stride = 2)
#         self.skip_bn = nn.BatchNorm1d(out_filters)
#         self.skip_act = nn.ReLU()

#         self.cat_preact = nn.ReLU()
#         self.cat_conv = conv1x1(out_filters, 1, stride = 1)
#         self.cat_bn = nn.BatchNorm1d(1)
#         self.cat_act = nn.Sigmoid()

#     def forward(self, gate, skip):
#         phi_gate = self.gate_conv(gate)
#         phi_gate = phi_gate.replace_feature(self.gate_bn(phi_gate.features))
#         phi_gate = phi_gate.replace_feature(self.gate_act(phi_gate.features))
#         phi_gate = self.up_subm(phi_gate)

#         theta_skip = self.skip_conv(skip)
#         theta_skip = theta_skip.replace_feature(self.skip_bn(theta_skip.features))
#         theta_skip = theta_skip.replace_feature(self.skip_act(theta_skip.features))

#         cat = phi_gate.replace_feature(phi_gate.features + theta_skip.features)
#         cat = cat.replace_feature(self.cat_preact(cat.features))
#         cat = self.cat_conv(cat)
#         cat = cat.replace_feature(self.cat_bn(cat.features))
#         cat = cat.replace_feature(self.cat_act(cat.features))

#         return cat

class AttentionBlock_Seq(nn.Module):
    def __init__(self, gating_features, shortcut_features, n_coefficients, kernel_size= (1, 1, 1), indice_key= None):
        super(AttentionBlock_Seq, self).__init__()
        self.gate_block = spconv.SparseSequential(
            spconv.SubMConv3d(gating_features, n_coefficients, kernel_size= 1, stride= 1, padding=1 , bias= False),
            spconv.SparseConv3d(n_coefficients, n_coefficients, kernel_size= 3, stride= 1, padding= 1, bias= False, indice_key = 'cc0'),
            spconv.SparseInverseConv3d(n_coefficients, n_coefficients, kernel_size= 3, bias= False, indice_key= 'cc0'),
            nn.BatchNorm1d(n_coefficients),
            nn.ReLU(),
        )
        self.shortcut_block = spconv.SparseSequential(
            spconv.SubMConv3d(shortcut_features, n_coefficients, kernel_size= 1, stride= 2, padding= 1, bias= False),
            nn.BatchNorm1d(n_coefficients),
            nn.ReLU(),
        )
        self.up_block = spconv.SparseSequential(
            nn.ReLU(),
            spconv.SubMConv3d(n_coefficients, 1, kernel_size= 1, stride= 1, padding= 1, bias= False, indice_key= 'att_up'),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
            spconv.SparseInverseConv3d(1, n_coefficients, kernel_size= 3, indice_key= 'att_up')
        )

    def forward(self, g, x):
        phi_g = self.gate_block(g)
        theta_x = self.shortcut_block(x)

        concat = phi_g.replace_feature(phi_g.features + theta_x.features)

        up_out = self.up_block(concat)
        up_out = up_out.replace_feature(up_out.features * x.features)
        return up_out

# class AttentionBlock(nn.Module):
#     """Attention block with learnable parameters"""

#     def __init__(self, F_g, F_l, n_coefficients):
#         """
#         :param F_g: number of feature maps (channels) in previous layer
#         :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
#         :param n_coefficients: number of learnable multi-dimensional attention coefficients
#         """
#         super(AttentionBlock, self).__init__()

#         self.W_gate = nn.Sequential(
#             conv1x1(F_g, n_coefficients, stride=1),
#             nn.BatchNorm1d(n_coefficients)
#         )

#         self.W_x = nn.Sequential(
#             conv1x1(F_l, n_coefficients, stride=2),
#             nn.BatchNorm1d(n_coefficients)
#         )

#         self.psi = nn.Sequential(
#             conv1x1(n_coefficients, 1, stride=1),
#             nn.BatchNorm1d(1),
#             nn.Sigmoid()
#         )

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, gate, skip_connection):
#         """
#         :param gate: gating signal from previous layer
#         :param skip_connection: activation from corresponding encoder layer
#         :return: output activations
#         """
#         print(self.W_gate)
#         g1 = self.W_gate(gate)
#         x1 = self.W_x(skip_connection)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         out = skip_connection * psi

#         return out

if __name__ == "__main__":
    AG_2 = AttentionBlock(gating_features= 512, shortcut_features= 256, n_coefficients= 256)
    print(AG_2)
    print(type(AG_2))
    AG_seq = AttentionBlock_Seq(gating_features= 512, shortcut_features= 256, n_coefficients= 256)
    print(AG_seq)
    print(type(AG_seq))
    
    # AG_new = AG_Block(gate_filters= 512, skip_filters= 256, out_filters= 256)

    # features_1 = torch.rand(80, 16)
    # print(features_1.shape)
    # abc = torch.randint(0, 100, (80, 3), dtype = torch.int32)
    # xyz = torch.zeros([80, 1], dtype = torch.int32)
    # indices_1 = torch.cat((xyz, abc), 1)
    # print(indices_1.shape)
    # spatial_shape_1 = torch.tensor([240, 180, 4], dtype= torch.int32)
    # print(len(spatial_shape_1))
    # sparse_1 = spconv.SparseConvTensor(features_1, indices_1, spatial_shape_1, batch_size= 1) 
    # print(sparse_1)

    # features_2 = torch.rand(80, 16)
    # print(features_2.shape)
    # cab = torch.randint(0, 100, (80, 3), dtype = torch.int32)
    # zxy = torch.zeros([80, 1], dtype = torch.int32)
    # indices_2 = torch.cat((xyz, abc), 1)
    # print(indices_2.shape)
    # spatial_shape_2 = torch.tensor([240, 180, 4], dtype= torch.int32)
    # print(len(spatial_shape_2))
    # sparse_2 = spconv.SparseConvTensor(features_2, indices_2, spatial_shape_2, batch_size= 1) 
    # print(sparse_2)

    features_1 = torch.rand([6658, 512])
    print(features_1.shape)
    zeros_1 = torch.zeros([6658, 1], dtype= torch.int32)
    print(zeros_1.shape)
    coors_1 = torch.randint(0, 50, (6658, 3), dtype = torch.int32)
    print(coors_1.shape)
    indices_1 = torch.cat((zeros_1, coors_1), 1)
    print(indices_1.shape)
    print(indices_1)

    up4e = spconv.SparseConvTensor(features_1, indices_1, spatial_shape= [60, 45, 8], batch_size= 1)
    print(up4e)

    features_2 = torch.rand([7792, 256])
    print(features_2.shape)
    zeros_2 = torch.zeros([7792, 1], dtype= torch.int32)
    print(zeros_2.shape)
    coors_2 = torch.randint(0, 50, (7792, 3), dtype = torch.int32)
    print(coors_2.shape)
    indices_2 = torch.cat((zeros_2, coors_2), 1)
    print(indices_2.shape)
    print(indices_2)

    down3b = spconv.SparseConvTensor(features_2, indices_2, spatial_shape= [120, 90, 8], batch_size= 1)
    print(down3b)

    print(AG_2(up4e, down3b))
    # print(AG_seq(up4e, down3b))
    # print(AG_new(up4e, down3b))
