import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, vocab_size, emb_dim, kernel_sizes, num_kernels, dropout):
        super(Discriminator, self).__init__()

        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_k, (k_size, emb_dim)) for k_size, n_k in zip(kernel_sizes, num_kernels)
                ])
        self.highway_linear = nn.Linear(sum(num_kernels), sum(num_kernels))
        self.highway_gate = nn.Linear(sum(num_kernels), sum(num_kernels))
        self.linear = nn.Linear(sum(num_kernels), 1)
        self.dropout = nn.Dropout(dropout)

        self._init_param()

    def _init_param(self):

        nn.init.uniform_(self.embed.weight, -1.0, 1.0)

        for conv in self.convs:
            nn.init.normal_(conv.weight, std=0.1)
            nn.init.constant_(conv.bias, 0.1)

        nn.init.normal_(self.highway_linear.weight, std=0.1)
        nn.init.constant_(self.highway_linear.bias, 0.1)
        nn.init.normal_(self.highway_gate.weight, std=0.1)
        nn.init.constant_(self.highway_gate.bias, 0.1)
        nn.init.normal_(self.linear.weight, std=0.1)
        nn.init.constant_(self.linear.bias, 0.1)

    def _get_loss(self):
        return

    def forward(self, x):
        # batch_size x 1 x seq_len x emb_dim
        x = self.embed(x).unsqueeze(1)

        feats_conv = [F.relu(conv(x).squeeze(3)) for conv in self.convs]
        # batch_size x 1
        feats_pool = [F.max_pool1d(feat, feat.size(-1)).squeeze(2) for feat in feats_conv]
        # batch_size x num_kernels
        feats = torch.cat(feats_pool, dim=1)

        # highway
        h = F.relu(self.highway_linear(feats))
        t = F.sigmoid(self.highway_gate(feats))
        output = t * h + (1 - h) * feats

        output = self.dropout(output)
        output = F.sigmoid(self.linear(output))

        return output

