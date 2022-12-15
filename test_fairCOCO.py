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
from dataset import ImagePairedCaptionDataset
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
    references1 = []
    hypotheses1 = []
    references2 = []
    hypotheses2 = []
    references_male = []
    hypotheses_male = []
    references_female = []
    hypotheses_female = []
    references_girl = []
    hypotheses_girl = []
    references_boy = []
    hypotheses_boy = []

    with torch.no_grad():
        for batch_idx, ((img1, cap1, all_cap1), (img2, cap2, all_cap2), _, _) in enumerate(data_loader):

            img1 = Variable(img1.cuda())
            img2 = Variable(img2.cuda())
            cap1 = Variable(cap1.cuda())
            cap2 = Variable(cap2.cuda())

            # forward
            features1 = encoder(img1)
            features2 = encoder(img2)
            preds1, _, _ = decoder(features1, cap1)
            preds2, _, _ = decoder(features2, cap2)

            # calculate caption lengths
            pad_idx = word_dict['<pad>']
            target_lens1 = (cap1.clone().detach().long()!=pad_idx).sum(dim=1)
            target_lens2 = (cap2.clone().detach().long()!=pad_idx).sum(dim=1)

            # calculate accuracy
            targets1 = cap1[:, 1:].clone()
            targets2 = cap2[:, 1:].clone()
            acc1_1 = accuracy(preds1, targets1, target_lens1, 1)
            acc1_5 = accuracy(preds1, targets1, target_lens1, 5)
            acc2_1 = accuracy(preds2, targets2, target_lens2, 1)
            acc2_5 = accuracy(preds2, targets2, target_lens2, 5)

            total_caption_length1 = calculate_caption_lengths(word_dict, cap1)
            total_caption_length2 = calculate_caption_lengths(word_dict, cap2)

            top1.update(acc1_1, total_caption_length1)
            top1.update(acc2_1, total_caption_length2)
            top5.update(acc1_5, total_caption_length1)
            top1.update(acc2_5, total_caption_length2)

            # calculate bleu scores
            for cap_set in all_cap1.tolist():
                caps1 = []
                cap_boy = []
                cap_girl = []
                cap_male = []
                cap_female = []

                for caption in cap_set:
                    cap = [word_idx for word_idx in caption
                                    if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]
                    caps1.append(cap)
                    if word_dict['woman'] in caption:
                        cap_female.append(cap)
                    if word_dict['man'] in caption:
                        cap_male.append(cap)
                    if word_dict['girl'] in caption:
                        cap_girl.append(cap)
                    if word_dict['boy'] in caption:
                        cap_boy.append(cap)
                references1.append(caps1)
                if cap_female:
                    references_female.append(cap_female)
                if cap_male:
                    references_male.append(cap_male)
                if cap_girl:
                    references_girl.append(cap_girl)
                if cap_boy:
                    references_boy.append(cap_boy)

            for cap_set in all_cap2.tolist():
                caps2 = []
                cap_boy = []
                cap_girl = []
                cap_male = []
                cap_female = []
                for caption in cap_set:
                    cap = [word_idx for word_idx in caption
                                    if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]
                    caps2.append(cap)
                    if word_dict['woman'] in caption:
                        cap_female.append(cap)
                    if word_dict['man'] in caption:
                        cap_male.append(cap)
                    if word_dict['girl'] in caption:
                        cap_girl.append(cap)
                    if word_dict['boy'] in caption:
                        cap_boy.append(cap)
                references2.append(caps2)
                if cap_female:
                    references_female.append(cap_female)
                if cap_male:
                    references_male.append(cap_male)
                if cap_girl:
                    references_girl.append(cap_girl)
                if cap_boy:
                    references_boy.append(cap_boy)


            word_idxs1 = torch.max(preds1, dim=2)[1]
            for idxs in word_idxs1.tolist():
                hypotheses1.append([idx for idx in idxs
                                       if idx != word_dict['<start>'] and idx != word_dict['<pad>']])
                cap = [idx for idx in idxs
                                       if idx != word_dict['<start>'] and idx != word_dict['<pad>']]
                if word_dict['woman'] in idxs:
                    hypotheses_female.append(cap)
                if word_dict['man'] in idxs:
                    hypotheses_male.append(cap)
                if word_dict['girl'] in idxs:
                    hypotheses_girl.append(cap)
                if word_dict['boy'] in idxs:
                    hypotheses_boy.append(cap)
                

            word_idxs2 = torch.max(preds2, dim=2)[1]
            for idxs in word_idxs2.tolist():
                hypotheses2.append([idx for idx in idxs
                                       if idx != word_dict['<start>'] and idx != word_dict['<pad>']])
                cap = [idx for idx in idxs
                                       if idx != word_dict['<start>'] and idx != word_dict['<pad>']]
                if word_dict['woman'] in idxs:
                    hypotheses_female.append(cap)
                if word_dict['man'] in idxs:
                    hypotheses_male.append(cap)
                if word_dict['girl'] in idxs:
                    hypotheses_girl.append(cap)
                if word_dict['boy'] in idxs:
                    hypotheses_boy.append(cap)
                
            if batch_idx % log_interval == 0:     
                print('Test Batch: [{0}/{1}]\t'
                      'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                          batch_idx, len(data_loader), top1=top1, top5=top5))
            

        bleu_1_1 = corpus_bleu(references1, hypotheses1, weights=(1, 0, 0, 0))
        bleu_1_2 = corpus_bleu(references1, hypotheses1, weights=(0.5, 0.5, 0, 0))
        bleu_1_3 = corpus_bleu(references1, hypotheses1, weights=(0.33, 0.33, 0.33, 0))
        bleu_1_4 = corpus_bleu(references1, hypotheses1)

        bleu_2_1 = corpus_bleu(references2, hypotheses2, weights=(1, 0, 0, 0))
        bleu_2_2 = corpus_bleu(references2, hypotheses2, weights=(0.5, 0.5, 0, 0))
        bleu_2_3 = corpus_bleu(references2, hypotheses2, weights=(0.33, 0.33, 0.33, 0))
        bleu_2_4 = corpus_bleu(references2, hypotheses2)

        bleu_female = corpus_bleu(references_female, hypotheses_female, weights=(1, 0, 0, 0))
        bleu_male = corpus_bleu(references_male, hypotheses_male, weights=(1, 0, 0, 0))
        bleu_boy = corpus_bleu(references_boy, hypotheses_boy, weights=(1, 0, 0, 0))
        bleu_girl = corpus_bleu(references_girl, hypotheses_girl, weights=(1, 0, 0, 0))

        print('Testing :\t'
                'BLEU-1 ({})\t'
                'BLEU-2 ({})\t'
                'BLEU-3 ({})\t'
                'BLEU-4 ({})\t'.format(bleu_1_1, bleu_1_2, bleu_1_3, bleu_1_4))
        
        print('Testing :\t'  
                'BLEU-1 ({})\t'
                'BLEU-2 ({})\t'
                'BLEU-3 ({})\t'
                'BLEU-4 ({})\t'.format(bleu_2_1, bleu_2_2, bleu_2_3, bleu_2_4))

        print('Testing :\t'  
                'BLEU-male ({})\t'
                'BLEU-female ({})\t'
                'BLEU-boy ({})\t'
                'BLEU-girl ({})\t'.format(bleu_male, bleu_female, bleu_boy, bleu_girl))


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

    test_dataset = ImagePairedCaptionDataset(transform=data_transforms, images_dir=args.images_dir, annotations_dir=args.annotations_dir, word_dict=word_dict, split_type='test')
    
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
    parser.add_argument('--images-dir', type=str, default='./data/final/')
    parser.add_argument('--annotations-dir', type=str, default='./data/annotations/')
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
