'''
Original code from https://github.com/AaronCCWong/Show-Attend-and-Tell
'''

import argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

from mi_estimators import *
from dataset import ImageCaptionTestDataset
from decoder import Decoder
from encoder import Encoder
from utils import AverageMeter, accuracy, calculate_caption_lengths


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def test(encoder, decoder, data_loader, word_dict, log_interval):
    encoder.eval()
    decoder.eval()

    top1 = AverageMeter()
    top5 = AverageMeter()

    # used for calculating bleu scores
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch_idx, (images, captions, all_captions) in enumerate(data_loader):

            images = Variable(images.cuda())
            captions = Variable(captions.cuda())

            # forward
            features = encoder(images)
            preds, _, _ = decoder(features, captions)

            # calculate caption lengths
            pad_idx = word_dict['<pad>']
            target_lens = (captions.clone().detach().long()!=pad_idx).sum(dim=1)

            # calculate accuracy
            targets = captions[:, 1:].clone()
            acc1 = accuracy(preds, targets, target_lens, 1)
            acc5 = accuracy(preds, targets, target_lens, 5)

            total_caption_length = calculate_caption_lengths(word_dict, captions)

            top1.update(acc1, total_caption_length)
            top5.update(acc5, total_caption_length)

            # calculate bleu scores
            for cap_set in all_captions.tolist():
                caps = []
                for caption in cap_set:
                    cap = [word_idx for word_idx in caption
                                    if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]
                    caps.append(cap)
                references.append(caps)

            word_idxs = torch.max(preds, dim=2)[1]
            for idxs in word_idxs.tolist():
                hypotheses.append([idx for idx in idxs
                                       if idx != word_dict['<start>'] and idx != word_dict['<pad>']])

            if batch_idx % log_interval == 0:     
                print('Test Batch: [{0}/{1}]\t'
                      'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                          batch_idx, len(data_loader), top1=top1, top5=top5))

        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(references, hypotheses)

        print('Testing :\t'
                'BLEU-1 ({})\t'
                'BLEU-2 ({})\t'
                'BLEU-3 ({})\t'
                'BLEU-4 ({})\t'.format(bleu_1, bleu_2, bleu_3, bleu_4))




def main(args):
    
    word_dict = json.load(open(args.data + '/word_dict.json', 'r'))
    vocabulary_size = len(word_dict)
    
    encoder = Encoder(args.network)
    decoder = Decoder(vocabulary_size, encoder.dim, args.tf, args.use_filter)

    if args.use_filter:
        print(f'Load pretrained weights for decoder from {args.model}')
        if args.model:
            decoder.load_state_dict(torch.load(args.model))
    else:
        print(f'Load pretrained weights for decoder from {args.model}')
        decoder_dict = decoder.state_dict()
        for k, v in torch.load(args.model).items():
            decoder_dict[k] = v
        decoder.load_state_dict(decoder_dict)
        decoder.freeze_pretrained()


    encoder.cuda()
    decoder.cuda()

    test_dataset = ImageCaptionTestDataset(transform=data_transforms, coco_data_path=args.data, word_dict=word_dict, split_type='test')
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=test_dataset.collate_fn)

    print('Starting testing with {}'.format(args))
    test(encoder, decoder, test_loader, word_dict, args.log_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='batch size for training (default: 16)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='L',
                        help='number of batches to wait before logging training stats (default: 100)')
    parser.add_argument('--data', type=str, default='data/coco',
                        help='path to data images (default: data/coco)')
    parser.add_argument('--network', choices=['vgg19', 'resnet152', 'densenet161'], default='vgg19',
                        help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--model', type=str, default='data/pretrained/VGG19_decoder.pth',
                        help='path to model (default: path for vgg19)')
    parser.add_argument('--tf', action='store_true', default=False,
                        help='Use teacher forcing when training LSTM (default: False)')

    # custom arguments
    parser.add_argument('--use-filter', action='store_true', default=False,
                        help='Use the filter in model')
    parser.add_argument('--estimator-hidden-size', type=int, default=512)
    parser.add_argument('--gpu-id', type=int, default=-1)

    main(parser.parse_args())
