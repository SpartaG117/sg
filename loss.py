import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, preds, tgt, reward):
        """
        :param preds: a tensor with shape [batch_size x seq_len x num_word]
        :param label: a tuple ([batch_size x seq_len], [batch_size x seq_len])
        :param reward: a tensor with shape [batch_size x seq_len]
        :return:
        """
        prob = self.softmax(preds).view(-1, preds.size(2))
        tgt, tgt_pos = tgt
        pad_mask = tgt.view(-1).gt(0)
        tgt_mask = torch.zeros(prob.size(0), prob.size(1)).scatter_(1, tgt.view(-1, 1), 1)
        # loss: tensor [(batch_size x seq_len) x ]
        loss = prob.mask_select(tgt_mask).view(-1).mask_select(pad_mask)
        loss = loss * reward
        loss = -(loss.sum())

        return loss
