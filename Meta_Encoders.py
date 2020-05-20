import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Meta_Encoder(nn.Module):

    def __init__(self, features, embed_dim, history_m_lists, aggregator, base_model=None, cuda="cpu"):
        super(Meta_Encoder, self).__init__()

        self.features = features
        self.history_m_lists = history_m_lists
        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes):
        tmp_history_m = []
        for node in nodes:
            tmp_history_m.append(self.history_m_lists[int(node)])

        neigh_feats = self.aggregator.forward(nodes, tmp_history_m)  # item-type network

        self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        self_feats = self_feats.t()
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
