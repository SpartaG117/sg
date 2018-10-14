import torch
import copy
from .sample import get_sample

class Rollout(object):
    def __init__(self, model):
        self.ori_model = model
        self.cur_model = copy.deepcopy(model)

    def get_reward(self, src, tgt, num, discriminator):
        tgt_seq, tgt_pos = tgt
        batch_size = tgt_seq.size(0)
        seq_len = tgt_seq.size(1)
        rewards = []
        # while t < T
        for i in range(num):
            cum_reward = []
            for l in range(seq_len-1):
                x = tgt_seq[:, 0:l]
                x_pos = tgt_pos[:, 0:l]
                sample, sample_pos = get_sample(self.ori_model, batch_size, seq_len, src, (x, x_pos))
                # reward: batch_size x 1
                reward = discriminator(sample)
                cum_reward.append(reward.cpu())
            # cum_rewards: batch_size x 1 x (seq_len-1)
            cum_reward = torch.cat(cum_reward, dim=1).unsqueeze(1)
            rewards.append(cum_reward)
        # rewards: batch_size x n x (seq_len-1)
        rewards = torch.cat(rewards, dim=1)
        # rewards: batch_size x (seq_len-1)
        rewards = rewards.sum(dim=1) / num

        # while t = T
        x = tgt_seq
        sample = get_sample(self.ori_model, batch_size, seq_len, src, x)
        reward = discriminator(sample)

        # rewards: batch_size x seq_len
        rewards = torch.cat([rewards, reward])
        return rewards

    def update_param(self):










