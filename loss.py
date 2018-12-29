import torch
import torch.nn as nn

class PGLoss(nn.Module):
    def __init__(self):
        super(PGLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, preds, label, reward):
        """
        :param preds: tensor [batch_size x seq_len x num_word]
        :param tgt: tensor ([batch_size x seq_len])
        :param reward: a tensor with shape [batch_size x seq_len]
        :return:
        """

        prob = self.log_softmax(preds)

        # tgt_mask: tensor [(batch_size x seq_len) x 1]
        log_porb = torch.gather(prob, 2, label.unsqueeze(2)).view(-1)

        pad_filter_mask = label.contiguous().view(-1).gt(0)

        # loss: tensor [(batch_size x seq_len)]
        loss = log_porb * reward
        loss = loss.masked_select(pad_filter_mask)
        loss = -(loss.mean())

        return loss
