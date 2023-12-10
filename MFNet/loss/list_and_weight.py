"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import torch

def cls_num_list_and_weight(num_class, images_class):
    # Imagenet_LT class distribution
    dist = [0 for _ in range(num_class)]
    for i in images_class:
        dist[int(i)] += 1
    num = sum(dist)
    prob = [i/num for i in dist]
    prob = torch.FloatTensor(prob)
    # normalization
    max_prob = prob.max().item()
    prob = prob / max_prob
    # class reweight
    weight = - prob.log() + 1

    return dist, weight


