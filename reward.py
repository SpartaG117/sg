import torch
import torch.nn.functional as F
import copy
from .sample import get_sample


def get_reward(ori_model, src, tgt, num, discriminator):

    tgt_seq, tgt_pos = tgt
    batch_size = tgt_seq.size(0)
    max_seq_len = tgt_seq.size(1)

    T_idx = []
    for i in range(batch_size):
        idx = len(tgt_seq[i, tgt_seq[i] != 0])
        if idx == max_seq_len:
            idx = max_seq_len - 1
        T_idx.append(idx)
    T_idx = torch.Tensor(T_idx).unsqueeze(1).long().cuda()

    pad_tokens = torch.zeros(batch_size).long().cuda()
    rewards = []
    # while t < T
    for i in range(num):
        cum_reward = []
        for l in range(2, max_seq_len + 1):
            x = (tgt_seq[:, 0:l], tgt_pos[:, 0:l])

            if tgt_seq[:, l-1].eq(pad_tokens).sum().item() == tgt_seq.size(0):
                if (max_seq_len + 1 - l) > 0:
                    reward = torch.zeros([batch_size, max_seq_len + 1 - l]).float()
                    cum_reward.append(reward)
                break

            sample, _ = get_sample(ori_model, batch_size, max_seq_len, src, x)
            # reward: batch_size x 1
            reward = F.sigmoid(discriminator(sample))
            cum_reward.append(reward.cpu())
        # cum_rewards: batch_size x 1 x (seq_len-1)
        cum_reward = torch.cat(cum_reward, dim=1).unsqueeze(1)

        rewards.append(cum_reward)

    # rewards: batch_size x n x (seq_len-1)
    rewards = torch.cat(rewards, dim=1).cuda()
    # rewards: batch_size x (seq_len-1)
    rewards = rewards.sum(1) / num
    # rewards: batch_size x seq_len
    rewards = torch.cat([rewards, torch.zeros([batch_size, 1]).cuda()], dim=1)

    # while t = T
    tgt_seq = torch.cat([tgt_seq, torch.zeros([batch_size, 1]).long().cuda()], dim=1)
    tgt_seq = tgt_seq.scatter_(1, T_idx, 9489)

    reward = F.sigmoid(discriminator(tgt_seq))

    # rewards: batch_size x seq_len

    rewards = rewards.scatter_(1, T_idx, reward)
    

    return rewards










