import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention


class Meta_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, features, v2e,m2e, embed_dim, cuda="cpu"):
        super(Meta_Aggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.v2e = v2e
        self.m2e = m2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, history_m):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            tmp_adj = history_m[i]
            num_meta = len(tmp_adj)
            #
            e_m = self.m2e.weight[list(tmp_adj)]  # fast: user embedding
            # slow: item-space user latent factor (item aggregation)
            # feature_neigbhors = self.features(torch.LongTensor(list(tmp_adj)).to(self.device))
            # e_u = torch.t(feature_neigbhors)

            v_rep = self.v2e.weight[nodes[i]]

            att_w = self.att(e_m, v_rep, num_meta)
            att_history = torch.mm(e_m.t(), att_w).t()
            embed_matrix[i] = att_history
        to_feats = embed_matrix

        return to_feats
