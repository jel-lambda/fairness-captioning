'''
Original code from https://github.com/AaronCCWong/Show-Attend-and-Tell
'''

import torch


class AverageMeter(object):
    """Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(preds, targets, target_lens, k):
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 2, True, True)
    correct = pred.view(-1,k).eq(targets.view(-1,1).expand_as(pred.view(-1,k)))
    correct_total = correct.view(batch_size, targets.size(1), k)
    correct_sum = 0
    for correct, target_len in zip(correct_total, target_lens):
        correct_sum += correct[:target_len].float().sum()
    return correct_sum * (100.0 / target_lens.sum())


def calculate_caption_lengths(word_dict, captions):
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (word_dict['<start>'], word_dict['<eos>'], word_dict['<pad>']):
                continue
            else:
                lengths += 1
    return lengths
