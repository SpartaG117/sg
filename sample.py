import torch
import torch.nn.functional as F

def get_sample(generator, batch_size, max_seq_len, src, tgt=None):

    # sample from zero
    if tgt is None:

        tgt_seq_pad = torch.zeros([batch_size, max_seq_len - 1]).long()
        tgt_seq = torch.Tensor([9488] * batch_size).long().unsqueeze(1)
        tgt_seq = torch.cat([tgt_seq, tgt_seq_pad], dim=1).contiguous().cuda()

        tgt_pos_pad = torch.zeros([batch_size, max_seq_len - 1]).long()
        tgt_pos = torch.ones([batch_size, 1]).long()
        tgt_pos = torch.cat([tgt_pos, tgt_pos_pad], dim=1).contiguous().cuda()

        zero_mask = torch.zeros([batch_size]).byte().cuda()

        end_tokens = torch.Tensor([9489 for _ in range(batch_size)]).long().cuda()
        pad_tokens = torch.Tensor([0 for _ in range(batch_size)]).long().cuda()
        start_tokens = torch.Tensor([9488 for _ in range(batch_size)]).long().cuda()

        for i in range(max_seq_len-1):
            # pred: batch_size x num_word
            pred = generator(src, (tgt_seq, tgt_pos)).view(batch_size, max_seq_len, -1)[:, i, :]
            pred_prob = F.softmax(pred, dim=1)
            # pred_word: batch_size x 1
            pred_word = pred_prob.multinomial(1).view(-1)
            pred_word = pred_word.masked_fill(zero_mask, 0)

            pred_pos = (tgt_pos[:, i] + 1).masked_fill(zero_mask, 0)
            tgt_pos[:, i + 1] = pred_pos

            end_mask = pred_word.eq(end_tokens)
            start_mask = pred_word.eq(start_tokens)
            pad_mask = pred_word.eq(pad_tokens)
            zero_mask = (end_mask + start_mask + pad_mask).ge(1)

            tgt_seq[:, i + 1] = pred_word

        return tgt_seq, tgt_pos

    # sample from given sequence
    else:

        tgt_seq, tgt_pos = tgt
        num_given = tgt_seq.size(1)
        tgt_pos = tgt_pos.cuda()


        end_tokens = torch.Tensor([9489 for _ in range(batch_size)]).long().cuda()
        pad_tokens = torch.Tensor([0 for _ in range(batch_size)]).long().cuda()
        start_tokens = torch.Tensor([9488 for _ in range(batch_size)]).long().cuda()

        end_mask = tgt_seq[:, -1].eq(end_tokens)
        pad_mask = tgt_seq[:, -1].eq(pad_tokens)
        zero_mask = (end_mask + pad_mask).ge(1)

        for i in range(num_given - 1, max_seq_len):

            if zero_mask.sum().item() == len(zero_mask):
                if (max_seq_len + 1 - (i + 1)) > 0:

                    tgt_seq_pad = torch.zeros([batch_size, max_seq_len + 1 - (i + 1)]).long().cuda()
                    tgt_seq = torch.cat([tgt_seq, tgt_seq_pad], dim=1)
                    # tgt_seq = torch.cat([tgt_seq, tgt_seq_pad], dim=1)
                    tgt_pos_pad = torch.zeros([batch_size, max_seq_len + 1 - (i + 1)]).long().cuda()
                    # tgt_pos = torch.cat([tgt_pos, tgt_pos_pad], dim=1)
                    tgt_pos = torch.cat([tgt_pos, tgt_pos_pad], dim=1)
                break

            # pred: batch_size x num_word
            pred = generator(src, (tgt_seq, tgt_pos)).view(batch_size, i+1, -1)[:, -1, :]
            pred_prob = F.softmax(pred, dim=1)
            # pred_word: batch_size x 1
            pred_word = pred_prob.multinomial(1).view(-1)
            pred_word = pred_word.masked_fill(zero_mask, 0)

            pred_pos = (tgt_pos[:, i] + 1).masked_fill(zero_mask, 0)
            tgt_pos = torch.cat([tgt_pos, pred_pos.unsqueeze(1)], dim=1)

            end_mask = pred_word.eq(end_tokens)
            start_mask = pred_word.eq(start_tokens)
            pad_mask = pred_word.eq(pad_tokens)

            zero_mask = (end_mask + start_mask + pad_mask).ge(1)

            tgt_seq = torch.cat([tgt_seq, pred_word.unsqueeze(1)], dim=1)

        return tgt_seq, tgt_pos






