import torch
import torch.nn as nn

class PGLoss(nn.Module):
    def __init__(self):
        super(PGLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, preds, tgt, reward):
        """
        :param preds: a tensor with shape [batch_size x seq_len x num_word]
        :param label: a tuple ([batch_size x seq_len], [batch_size x seq_len])
        :param reward: a tensor with shape [batch_size x seq_len]
        :return:
        """
        prob = self.softmax(preds).view(-1, preds.size(2))
        batch_size = prob.size(0)
        num_class = prob.size(1)

        tgt, tgt_pos = tgt
        pad_filter_mask = tgt.view(-1).gt(0)

        # tgt_mask: tensor [(batch_size x seq_len) x 1]
        tgt_mask = torch.zeros(batch_size, num_class).cuda().scatter_(1, tgt.view(-1, 1), 1)

        # loss: tensor [(batch_size x seq_len)]
        loss = prob.mask_select(tgt_mask).view(-1)
        loss = loss * (reward.view(-1))
        loss = loss.mask_select(pad_filter_mask)
        loss = -(loss.mean())

        return loss
