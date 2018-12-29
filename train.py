import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import time
import os
from transformer.Model import Transformer
from aug_loader import CocoDataset, collate_fn
import opts
from train_utils import *
from dc_model.DModel import Discriminator
from dc_model import Constants, sample
from dc_model.loss import PGLoss


def calc_cider(gts, res):

    cider = Cider()
    score, scores = cider.compute_score(gts, res)
    return score, scores

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


def train_disc_epoch(iters, discriminator, generator, train_data, criterion, optimizer, vocab):
    """ training epoch"""

    if iters % 1000 <= 500:
        return 0, iters
    discriminator.train()
    generator.eval()
    total_loss = 0

    for idx, batch in tqdm(enumerate(train_data), mininterval=2,
            desc='  - (Training)   ', leave=False, total=len(train_data)):

        src_seq, _, label, infos = batch
        batch_size = label.size(0)

        # both Tensor of shape [(B x 5) x 17]
        src_seq = src_seq.view(src_seq.size(0)*src_seq.size(1), src_seq.size(2)).cuda()
        label = label.view(label.size(0)*label.size(1), label.size(2))
        src_pos = get_seq_position(src_seq)

        preds, _ = gen_fake_sample((src_seq, src_pos), generator, label.size(1))

        gt_sents = []
        for info in infos:
            gt_sents.append(trans2sentence(info['gts'], vocab))

        res = {}
        gts = {}
        sents = trans2sentence(preds[:, :-1], vocab)
        for i, sent in enumerate(sents):
            res[i] = [sent]
            gts[i] = gt_sents[i//5]
        score, _ = calc_cider(gts, res)

        start_token = torch.Tensor([[9488]] * label.size(0)).long().cuda()
        src_seq = torch.cat([start_token, src_seq], dim=1)
        preds = preds.masked_fill(preds.eq(9489), 0)[:, :-1]

        gts_seq_src = src_seq.unsqueeze(1).expand(-1, 5, -1).contiguous().view(batch_size*25, -1)
        gts_seq_tgt = src_seq.view(batch_size, 5, -1).unsqueeze(1).expand(-1, 5, -1, -1).contiguous().view(
                    batch_size*25, -1)

        fake_seq = torch.cat([preds, src_seq], dim=1)
        gts_seq = torch.cat([gts_seq_tgt, gts_seq_src], dim=1)
        fake_label = torch.Tensor([0]*fake_seq.size(0)).unsqueeze(1).float().cuda()
        gts_label = torch.Tensor([1]*gts_seq.size(0)).unsqueeze(1).float().cuda()


        optimizer.zero_grad()

        fake_out = discriminator(fake_seq)
        gts_out = discriminator(gts_seq)
        fake_loss = criterion(fake_out, fake_label)
        gts_loss = criterion(gts_out, gts_label)
        loss = fake_loss + gts_loss / 5

        fake_correct, n_fake_w = get_disc_acc(fake_out, fake_label)
        gts_correct, n_gts_w = get_disc_acc(gts_out, gts_label)
        correct = fake_correct + gts_correct
        n_w = n_fake_w + n_gts_w

        print('iter', idx, 'acc', correct/n_w, 'score', score)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if iters % 1000 <=500 :
            break

        iters += 1

    return total_loss/len(train_data), iters


def train_gen_epoch(iters, discriminator, generator, generator_old, train_data, criterion, optimizer, vocab):
    """ training epoch"""

    if iters % 1000 > 500:
        return 0, iters

    generator.train()
    discriminator.eval()
    total_loss = 0
    for idx, batch in tqdm(enumerate(train_data), mininterval=2,
            desc='  - (Training)   ', leave=False, total=len(train_data)):

        src_seq, _, _, infos = batch

        # both Tensor of shape [(B x 5) x 17]
        src_seq = src_seq.view(src_seq.size(0)*src_seq.size(1), src_seq.size(2)).cuda()
        src_pos = get_seq_position(src_seq)

        gt_sents = []
        for info in infos:
            gt_sents.append(trans2sentence(info['gts'], vocab))

        with torch.no_grad():
            preds, _ = sample.get_sample(generator_old, src_seq.size(0), src_seq.size(1)+2, (src_seq, src_pos))

            disc_pred_seq = preds.masked_fill(preds.eq(9489), 0)[:, :-1]
            start_token = torch.Tensor([[9488]] * preds.size(0)).long().cuda()
            disc_src_seq = torch.cat([start_token, src_seq], dim=1)

            disc_seq = torch.cat([disc_pred_seq, disc_src_seq], dim=1)
            rewards = F.sigmoid(discriminator(disc_seq))

        tgt_seq = disc_pred_seq
        tgt_pos = get_seq_position(tgt_seq)
        label = preds[:, 1:]

        res = {}
        gts = {}
        sents = trans2sentence(label, vocab)
        for i, sent in enumerate(sents):
            res[i] = [sent]
            gts[i] = gt_sents[i // 5]

        _, scores = calc_cider(gts, res)

        scores = torch.from_numpy(scores).float().unsqueeze(1).cuda()
        lambda_ = 0.3
        rewards = (1-lambda_) * scores + lambda_ * rewards

        optimizer.zero_grad()

        batch_size = src_seq.size(0)
        max_seq_len = tgt_seq.size(1)
        outputs = generator((src_seq, src_pos), (tgt_seq, tgt_pos))
        outputs = outputs.view(batch_size, max_seq_len, -1)
        rewards = rewards.expand(-1, max_seq_len).contiguous().view(-1)
        loss = criterion(outputs, label, rewards)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        iters += 1

        if iters % 1000 > 500:
            break

    return total_loss/len(train_data), iters


def train(discriminator, generator, train_data, gen_optimizer, disc_optimizer,
          current_iter, vocab, args):

    iters = current_iter + 1

    disc_criterion = discriminator.get_criterion().cuda()
    gen_criterion = PGLoss().cuda()

    while True:

        start = time.time()

        generator_old = copy.deepcopy(generator)

        train_loss, iters = train_gen_epoch(iters, discriminator, generator, generator_old,
                                            train_data, gen_criterion, gen_optimizer, vocab)

        print('  - (Training)   loss: {loss: 3.5f}, '
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss, elapse=(time.time()-start)/60))


        model_state_dict = generator.state_dict()
        gen_checkpoint = {
            'model': model_state_dict,
            'settings': args,
            'iters': iters,
            'gen_optimizer': gen_optimizer.state_dict(),
        }

        ckpt_path = args.checkpoint_dir + 'generator_iters_{iters}.ckpt'.format(iters=iters)
        torch.save(gen_checkpoint, ckpt_path)
        print(iters, 'generator saved')

        train_loss, iters = train_disc_epoch(iters, discriminator, generator, train_data,
                                      disc_criterion, disc_optimizer, vocab)

        print('  - (Training)   loss: {loss: 3.5f}, '
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss, elapse=(time.time()-start)/60))

        model_state_dict = discriminator.state_dict()
        disc_checkpoint = {
            'model': model_state_dict,
            'settings': args,
            'iters': iters,
            'disc_optimizer': disc_optimizer.state_dict(),
        }

        ckpt_path = args.checkpoint_dir + 'discriminator_iters_{iters}.ckpt'.format(iters=iters)
        torch.save(disc_checkpoint, ckpt_path)
        print(iters, 'discriminator saved')



def main(args):

    #=================== Establish model====================#

    generator = Transformer(
        num_vocab=args.num_vocab,
        n_max_seq=args.max_length,
        num_layer=args.num_layer,
        num_head=args.num_head,
        d_word=args.d_word_vec,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        proj_share_weight=True,
        embs_share_weight=True
    )

    discriminator = Discriminator(
        args.num_vocab,
        Constants.EMB_DIM,
        Constants.KERNEL_SIZES,
        Constants.NUM_KERNEL,
        Constants.DROPOUT
    )

    if args.cuda:
        discriminator.cuda()
        generator.cuda()

    #==================== Loading checkpoint ========================#

    gen_dict = torch.load('checkpoints/ad_b1/generator_iters_108501.ckpt')
    gen_state = gen_dict['model']
    generator.load_state_dict(gen_state)

    disc_dict = torch.load('checkpoints/ad_b1/discriminator_iters_109000.ckpt')
    disc_state = disc_dict['model']
    discriminator.load_state_dict(disc_state)

    if 'iters' in gen_dict.keys() and 'iters' not in disc_dict.keys():
        current_iter = gen_dict['iters']
    elif 'iters' not in gen_dict.keys() and 'iters' in disc_dict.keys():
        current_iter = disc_dict['iters']
    elif 'iters' in gen_dict.keys() and 'iters' in disc_dict.keys():
        if gen_dict['iters'] > disc_dict['iters']:
            current_iter = gen_dict['iters']
        else:
            current_iter = disc_dict['iters']
    else:
        current_iter = 0

    #===================== criterion and optimizer ===================#

    gen_optimizer = optim.Adam(generator.get_trainable_parameters(), 1e-4)
    disc_optimizer = optim.Adam(discriminator.parameters(), 5e-4)

    if 'disc_optimizer' in disc_dict:
        disc_optimizer.load_state_dict(disc_dict['disc_optimizer'])
    if 'gen_optimizer' in gen_dict:
        gen_optimizer.load_state_dict(gen_dict['gen_optimizer'])

    #================== Loading Data ======================#

    train_dataset = CocoDataset(args, 'train')

    train_data = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn)

    vocab = train_dataset.ix_to_word
    #==================== Training ========================#

    train(discriminator, generator, train_data, gen_optimizer, disc_optimizer, current_iter, vocab, args)

if __name__ == "__main__":

    opt = opts.parse_opt()
    DEVICE_ID = opt.device_id
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
    main(opt)

