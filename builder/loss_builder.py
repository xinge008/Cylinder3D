# -*- coding:utf-8 -*-
# author: Xinge
# @file: loss_builder.py 

import torch
from utils.lovasz_losses import lovasz_softmax


def build(wce=True, lovasz=True, num_class=20, ignore_label=0):
    ## for semanticKITTI
    weights = torch.zeros(num_class, dtype=torch.float)
    weights[0] = 1.0
    weights[1] = 2.293
    weights[2] = 85.756
    weights[3] = 71.511
    weights[4] = 31.596
    weights[5] = 35.624
    weights[6] = 74.761
    weights[7] = 88.722
    weights[8] = 96.389
    weights[9] = 1.00
    weights[10] = 6.362
    weights[11] = 1.00
    weights[12] = 20.387
    weights[13] = 1.00
    weights[14] = 1.363
    weights[15] = 1.00
    weights[16] = 14.214
    weights[17] = 1.263
    weights[18] = 25.936
    weights[19] = 61.896
    loss_funs = torch.nn.CrossEntropyLoss(weight=weights.cuda(), ignore_index=ignore_label)

    if wce and lovasz:
        return loss_funs, lovasz_softmax
    elif wce and not lovasz:
        return wce
    elif not wce and lovasz:
        return lovasz_softmax
    else:
        raise NotImplementedError
