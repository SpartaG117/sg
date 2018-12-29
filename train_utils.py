import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from cider.cider import Cider

def get_seq_position(seq):
    assert seq.dim() == 2
    num_ne_pad = seq.ne(0).sum(1).tolist()
    seq_pos = []
    for i in range(seq.size(0)):
        seq_pos.append([p+1 for p in range(num_ne_pad[i])] + [0] * (seq.size(1) - num_ne_pad[i]))

    if seq.is_cuda:
        return torch.Tensor(seq_pos).long().cuda()

    return torch.Tensor(seq_pos).long()


def get_rec_input_data(data):

    labels = data['labels']
    tgts_label = labels[:, 1:]
    tgts = labels.copy()
    for i in range(len(tgts)):
        tgts[i, tgts[i] == 9489] = 0
    tgts = tgts[:, :-1]
    return tgts, tgts_label, labels


def gen_fake_sample(src, generator, max_seq_len):
    # generate sample with length of 31: [x,x,x,</s> ...]

    src_seq, src_pos = src
    batch_size = src_seq.size(0)

    tgt_seq = torch.Tensor([9488] * batch_size).long().unsqueeze(1).cuda()

    tgt_pos = torch.ones([batch_size, 1]).long().cuda()

    zero_mask = torch.zeros([batch_size]).byte().cuda()

    end_tokens = torch.Tensor([9489 for _ in range(batch_size)]).long().cuda()
    pad_tokens = torch.Tensor([0 for _ in range(batch_size)]).long().cuda()
    start_tokens = torch.Tensor([9488 for _ in range(batch_size)]).long().cuda()

    with torch.no_grad():
        for i in range(max_seq_len):
            # pred: batch_size x num_word
            pred = generator(src, (tgt_seq, tgt_pos)).view(batch_size, i+1, -1)[:, -1, :]
            # pred_word: batch_size x 1
            pred_word = pred.max(dim=1)[1]
            pred_word = pred_word.masked_fill(zero_mask, 0)

            pred_pos = (tgt_pos[:, -1] + 1).masked_fill(zero_mask, 0)
            tgt_pos = torch.cat([tgt_pos, pred_pos.unsqueeze(1)], dim=1)

            end_mask = pred_word.eq(end_tokens)
            start_mask = pred_word.eq(start_tokens)
            pad_mask = pred_word.eq(pad_tokens)
            zero_mask = (end_mask + start_mask + pad_mask).ge(1)

            tgt_seq = torch.cat([tgt_seq, pred_word.unsqueeze(1)], dim=1)

            if zero_mask.sum().item() == len(zero_mask):
                if (max_seq_len + 1 - (i+2)) > 0:
                    tgt_seq_pad = torch.zeros([batch_size, max_seq_len + 1 - (i+2)]).long().cuda()
                    tgt_seq = torch.cat([tgt_seq, tgt_seq_pad], dim=1)
                    # tgt_seq = torch.cat([tgt_seq, tgt_seq_pad], dim=1)
                    tgt_pos_pad = torch.zeros([batch_size, max_seq_len + 1 - (i+2)]).long().cuda()
                    # tgt_pos = torch.cat([tgt_pos, tgt_pos_pad], dim=1)
                    tgt_pos = torch.cat([tgt_pos, tgt_pos_pad], dim=1)
                break

    return tgt_seq, tgt_pos


def make_disc_data(fake_seq, gts_seq):

    disc_label = [0] * fake_seq.size(0) + [1] * gts_seq.size(0)
    disc_label = torch.tensor(disc_label).unsqueeze(1).float().cuda()
    disc_seq = torch.cat([fake_seq, gts_seq], dim=0)

    return disc_seq, disc_label

def get_disc_acc(pred, target):
    preds = F.sigmoid(pred.view(-1)).gt(0.5).float()
    targets = target.view(-1)
    total = targets.size(0)

    num_acc = torch.eq(preds, targets).sum().item()
    return num_acc, total



def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def get_loss_correct(pred, target, criterion):
    loss = criterion(pred, target.contiguous().view(-1))
    pred = pred.max(1)[1]
    target = target.contiguous().view(-1)
    correct_mask = pred.eq(target)
    num_correct = correct_mask.masked_select(target.ne(0)).sum()
    return loss, num_correct


def get_criterion(num_vocab):
    """ make the pad loss weight 0 """
    weight = torch.ones(num_vocab).cuda()
    weight[0] = 0
    return nn.CrossEntropyLoss(weight, size_average=True)


def search(att_feat, fc_feat, model, max_len):
    model.eval()
    batch_size = att_feat.size(0)
    seq = torch.Tensor([[9488]] * batch_size).long().cuda()

    with torch.no_grad():
        for _ in range(max_len+1):
            preds = model(seq, att_feat, fc_feat, return_attn=False)
            preds = preds.view(batch_size, seq.size(1), -1)
            preds = preds[:, -1, :].max(1)[1].unsqueeze(1)
            seq = torch.cat([seq, preds], dim=1)
        preds = seq[:, 1:]
    assert preds.size(1) == max_len + 1
    return preds

def greedy_sample(att_feat, fc_feat, model, max_len):
    model.train()
    batch_size = att_feat.size(0)
    seq = torch.Tensor([[9488]] * batch_size).long().cuda()

    zero_mask = torch.zeros([batch_size]).byte().cuda()
    end_tokens = torch.Tensor([9489 for _ in range(batch_size)]).long().cuda()
    pad_tokens = torch.Tensor([0 for _ in range(batch_size)]).long().cuda()
    start_tokens = torch.Tensor([9488 for _ in range(batch_size)]).long().cuda()
    with torch.no_grad():
        for _ in range(max_len + 1):
            preds = model(seq, att_feat, fc_feat, return_attn=False)
            preds = preds.view(batch_size, seq.size(1), -1)
            prob = F.softmax(preds[:, -1, :], dim=1)
            samples = torch.multinomial(prob, 1).squeeze(1)
            samples = samples.masked_fill(zero_mask, 0)

            end_mask = samples.eq(end_tokens)
            start_mask = samples.eq(start_tokens)
            pad_mask = samples.eq(pad_tokens)
            zero_mask = (end_mask + start_mask + pad_mask).ge(1)

            seq = torch.cat([seq, samples.unsqueeze(1)], dim=1)

        sample_w = seq
        sample_label = copy.deepcopy(seq)
        sample_label = sample_label[:, 1:]
        end_mask = sample_w.eq(9489)
        sample_w = sample_w.masked_fill(end_mask, 0)[:, :-1]
        assert sample_w.size(1) == max_len + 1

    return sample_w, sample_label


def trans2sentence(seq, vocab, eos_flag=False):
    captions = []
    if isinstance(seq, list):
        n_seq = len(seq)
    elif isinstance(seq, torch.Tensor):
        n_seq = seq.size(0)
    for i in range(n_seq):
        words = []
        for word in seq[i]:
            if word == 9489:
                if eos_flag:
                    if isinstance(seq, list):
                        words.append(vocab[str(word)])
                    elif isinstance(seq, torch.Tensor):
                        words.append(vocab[str(word.item())])
                break
            if word == 0:
                break
            if isinstance(seq, list):
                words.append(vocab[str(word)])
            elif isinstance(seq, torch.Tensor):
                words.append(vocab[str(word.item())])
        captions.append(' '.join(words).strip())
    return captions

def prepro_gts(data, vocab):
    gts = {}
    for batch in data:
        _, _, _, infos = batch
        for i, info in enumerate(infos):
            caps = trans2sentence(info['gts'], vocab)
            gts[int(info['image_id'])] = caps
    return gts


def calc_cider(gts, res):
    cider = Cider()
    score, scores = cider.compute_score(gts, res)
    return score, scores
