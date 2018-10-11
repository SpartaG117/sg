import torch
import torch.nn.functional as F

def get_sample(model, batch_size, seq_len, src, x=None):
    """
    Todo whether the seq_len includes the start token or end token
    Todo padding token may be sampled
    :param model:
    :param batch_size:
    :param seq_len:
    :param src:
    :param x:
    :return:
    """
    # Todo end token
    sample = []

    # sample from zero
    if x is None:

        pad_mask = torch.zeros([batch_size]).byte()
        # Todo
        pad_tokens = torch.Tensor([2 for _ in range(batch_size)])

        # Todo should be the start token
        x = torch.zeros([batch_size, 1]).long()
        x = x.cuda()
        x_pos = torch.Tensor([batch_size, 1]).long()
        x_pos = x_pos.cuda()
        for i in range(seq_len):
            # pred: batch_size x num_word
            pred = model(src, (x, x_pos)).view(batch_size, i+1, -1)[:, -1, :]
            pred_prob = F.softmax(pred, dim=1)
            # s_word: batch_size x 1
            s_word = pred_prob.multinomial(1)
            s_word = s_word.masked_fill(pad_mask, 0)
            sample.append(s_word)
            x = torch.cat([x, s_word], dim=1)
            # make the right padding position
            cur_pos = torch.Tensor([i+1 for _ in range(batch_size)]).long()
            cur_pos = cur_pos.masked_fill(pad_mask, 0)
            x_pos = torch.cat([x_pos, cur_pos], dim=1)
            # TODO end_token=2  padding_token=0 start_token=1 ?
            pad_mask = torch.le(s_word, pad_tokens)

        output = torch.cat(sample, dim=1)
        x_pos = (x_pos - 1)[:, 1:]
        pad_mask = x_pos.eq(1)
        x_pos.masked_fill(pad_mask, 0)
        return output, x_pos

    # sample from given sequence
    else:

        x, x_pos = x
        sample.append(x)
        num_given = x.size(1)
        x_pos = x_pos.cuda()

        # Todo
        pad_tokens = torch.Tensor([2 for _ in range(batch_size)])
        pad_mask = x[:, -1].le(pad_tokens)

        for i in range(num_given, seq_len):
            # pred: batch_size x num_dic
            pred = model(src, (x, x_pos)).view(batch_size, i+1, -1)[:, -1, :]
            pred_prob = F.softmax(pred, dim=1)
            # s_word: batch_size x 1
            s_word = pred_prob.multinomial(1)
            s_word = s_word.masked_fill(pad_mask, 0)
            sample.append(s_word)
            x = torch.cat([x, s_word], dim=1)
            # make the right padding position
            cur_pos = torch.Tensor([i+1 for _ in range(batch_size)]).long()
            cur_pos = cur_pos.masked_fill(pad_mask, 0)
            x_pos = torch.cat([x_pos, cur_pos], dim=1)
            # TODO end_token=2  padding_token=0 start_token=1 ?
            pad_mask = torch.le(s_word, pad_tokens)

        output = torch.cat(sample, dim=1)
        x_pos = (x_pos - 1)[:, 1:]
        pad_mask = x_pos.eq(1)
        x_pos.masked_fill(pad_mask, 0)

        return output, x_pos






