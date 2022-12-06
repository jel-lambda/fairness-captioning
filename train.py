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


def main(args):
    writer = SummaryWriter()
    
    word_dict = json.load(open(args.data + '/word_dict.json', 'r'))
    vocabulary_size = len(word_dict)
    
    encoder = Encoder(args.network)
    decoder = Decoder(vocabulary_size, encoder.dim, args.tf)

    if args.model:
        decoder.load_state_dict(torch.load(args.model))

    encoder.cuda()
    decoder.cuda()

    train_dataset = ImagePairedCaptionDataset(transform=data_transforms, images_dir='./data/final/', annotations_dir='./data/annotations/', word_dict=word_dict, split_type='train')
    val_dataset = ImagePairedCaptionDataset(transform=data_transforms, images_dir='./data/final/', annotations_dir='./data/annotations/', word_dict=word_dict, split_type='val')

    optimizer = optim.Adam(decoder.filter.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size)
    cross_entropy_loss_train = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_idx).cuda()
    cross_entropy_loss_val = nn.CrossEntropyLoss(ignore_index=val_dataset.pad_idx).cuda()
    infoNCE = eval("InfoNCE")(encoder.dim,encoder.dim,args.estimator_hidden_size).cuda()
    optimizer_nce = optim.Adam(infoNCE.parameters(), lr=args.lr)
    club = eval("CLUB")(encoder.dim,encoder.dim,args.estimator_hidden_size).cuda()
    optimizer_club = optim.Adam(club.parameters(), lr=args.lr)

    
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn=train_dataset.collate_fn)
    val_loader= DataLoader(val_dataset, shuffle=False, batch_size=1, collate_fn=val_dataset.collate_fn)

    print('Starting training with {}'.format(args))
    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        train(epoch, encoder, decoder, optimizer, optimizer_nce, optimizer_club, cross_entropy_loss_train, infoNCE, club,
              train_loader, word_dict, args.log_interval, writer, args.batch_size)
        validate(epoch, encoder, decoder, cross_entropy_loss_val, val_loader,
                infoNCE, club, word_dict, args.alpha_c, args.log_interval, writer)
        model_file = 'model/model_' + args.network + '_' + str(epoch) + '.pth'
        torch.save(decoder.state_dict(), model_file)
        print('Saved model to ' + model_file)
    writer.close()



def train(epoch, encoder, decoder, optimizer, optimizer_nce, optimizer_club, cross_entropy_loss, infoNCE, club, data_loader, word_dict, log_interval, writer, batch_size):
    encoder.eval()
    decoder.train()

    infonce_losses = AverageMeter()
    club_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, ((img1, cap1), (img2, cap2), club_mask, nce_mask) in enumerate(data_loader):
        
        img1 = Variable(img1.cuda())
        cap1 = Variable(cap1.cuda())
        img2 = Variable(img2.cuda())
        cap2 = Variable(cap2.cuda())
        club_mask = Variable(club_mask.cuda())
        nce_mask = Variable(nce_mask.cuda())

        img1_features = encoder(img1)
        img2_features = encoder(img2)
        
        optimizer.zero_grad()
        preds1, _, features1 = decoder(img1_features, cap1)
        preds2, _, features2 = decoder(img2_features, cap2)
        targets1 = cap1[:, 1:].clone()
        targets2 = cap2[:, 1:].clone()
        club_mask = club_mask[:, 1:].clone()
        nce_mask = nce_mask[:, 1:].clone()

        # club_feat1 = features1 * club_mask.unsqueeze(2)
        # club_feat2 = features2 * club_mask.unsqueeze(2)
        # nce_feat1 = features1 * nce_mask.unsqueeze(2)
        # nce_feat2 = features2 * nce_mask.unsqueeze(2)

        # num_negative = cls_mask.sum(1).max().item()
   
        club.eval()
        infoNCE.eval()

        total_caption_length = calculate_caption_lengths(word_dict, cap1)
        ce_loss = cross_entropy_loss(preds1.view(-1, preds1.size(-1)), targets1.view(-1))\
            + cross_entropy_loss(preds2.view(-1, preds2.size(-1)), targets2.view(-1))
        
        with torch.no_grad():
            club_loss = club(features1, features2, club_mask)
            infonce_loss = infoNCE(features1, features2, nce_mask)

        loss = (ce_loss + infonce_loss + club_loss) / total_caption_length
        
        loss.backward(retain_graph=True)
        optimizer.step()
        torch.autograd.set_detect_anomaly(True)
        
        club.train()
        infoNCE.train()
        
        club_learning_loss = club.learning_loss(features1, features2, club_mask)
        infonce_learning_loss = infoNCE.learning_loss(features1, features2, nce_mask)
        
        optimizer_club.zero_grad()
        optimizer_nce.zero_grad()
        # infonce_learning_loss.backward()
        club_learning_loss.backward()

        breakpoint()

        optimizer_club.step()
        optimizer_nce.step()

        acc1_1 = accuracy(preds1, targets1, 1)
        acc1_5 = accuracy(preds1, targets1, 5)
        acc2_1 = accuracy(preds2, targets2, 1)
        acc2_5 = accuracy(preds2, targets2, 5)

        infonce_losses.update(infonce_loss.item(), total_caption_length)
        club_losses.update(club_loss.item(), total_caption_length)
        losses.update(loss.item() , total_caption_length)
        top1.update((acc1_1 + acc2_1)/2, total_caption_length)
        top5.update((acc1_5 + acc2_5)/2, total_caption_length)

        if batch_idx % log_interval == 0:
            print('Train Batch: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                      batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))
    writer.add_scalar('infoNCE', infonce_losses.avg, epoch)
    writer.add_scalar('club_loss', club_losses.avg, epoch)
    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1_acc', top1.avg, epoch)
    writer.add_scalar('train_top5_acc', top5.avg, epoch)

def validate(epoch, encoder, decoder, cross_entropy_loss, data_loader, infoNCE_loss, club_loss, word_dict, alpha_c, log_interval, writer):
    encoder.eval()
    decoder.eval()
    club_loss.eval()
    infoNCE_loss.eval()

    infonce_losses = AverageMeter()
    club_losses = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # used for calculating bleu scores
    references1 = []
    hypotheses1 = []
    references2 = []
    hypotheses2 = []
    with torch.no_grad():
        for batch_idx, ((img1, cap1, all_cap1), (img2, cap2, all_cap2), club_mask, nce_mask) in enumerate(data_loader):
            print(img1.shape)
            print('captions', all_cap2)

            img1 = Variable(img1.cuda())
            cap1 = Variable(cap1.cuda())
            img2 = Variable(img2.cuda())
            cap2 = Variable(cap2.cuda())
            club_mask = Variable(club_mask.cuda())
            nce_mask = Variable(nce_mask.cuda())

            img1_features = encoder(img1)
            img2_features = encoder(img2)
            
            preds1, _, features1 = decoder(img1_features, cap1)
            preds2, _, features2 = decoder(img2_features, cap2)
            targets1 = cap1[:, 1:]
            targets2 = cap2[:, 1:]

            c_loss = club_loss(features1, features2, nce_mask)
            infonce_loss = infoNCE_loss(features1, features2, nce_mask)
            ce_loss = cross_entropy_loss(preds1.view(-1, preds1.size(-1)), targets1.view(-1))\
                 + cross_entropy_loss(preds2.view(-1, preds2.size(-1)), targets2.view(-1))

            loss = (ce_loss + infonce_loss + c_loss)/total_caption_length

            total_caption_length = calculate_caption_lengths(word_dict, cap1)

            acc1_1 = accuracy(preds1, targets1, 1)
            acc1_5 = accuracy(preds1, targets1, 5)
            acc2_1 = accuracy(preds2, targets2, 1)
            acc2_5 = accuracy(preds2, targets2, 5)

            losses.update(loss.item() , total_caption_length)
            top1.update((acc1_1 + acc2_1)/2, total_caption_length)
            top5.update((acc1_5 + acc2_5)/2, total_caption_length)

            for cap_set in all_cap1.tolist():
                caps1 = []
                for caption in cap_set:
                    cap = [word_idx for word_idx in caption
                                    if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]
                    caps1.append(cap)
                references1.append(caps1)

            for cap_set in all_cap2.tolist():
                caps2 = []
                for caption in cap_set:
                    cap = [word_idx for word_idx in caption
                                    if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]
                    caps2.append(cap)
                references2.append(caps2)

            word_idxs1 = torch.max(preds1, dim=2)[1]
            for idxs in word_idxs1.tolist():
                hypotheses1.append([idx for idx in idxs
                                       if idx != word_dict['<start>'] and idx != word_dict['<pad>']])

            word_idxs2 = torch.max(preds2, dim=2)[1]
            for idxs in word_idxs2.tolist():
                hypotheses2.append([idx for idx in idxs
                                       if idx != word_dict['<start>'] and idx != word_dict['<pad>']])
            if batch_idx % log_interval == 0:
                print('Validation Batch: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                          batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))
        writer.add_scalar('val_infoNCE', infonce_losses.avg, epoch)
        writer.add_scalar('val_club_loss', club_losses.avg, epoch)
        writer.add_scalar('val_loss', losses.avg, epoch)
        writer.add_scalar('val_top1_acc', top1.avg, epoch)
        writer.add_scalar('val_top5_acc', top5.avg, epoch)

        bleu_1_1 = corpus_bleu(references1, hypotheses1, weights=(1, 0, 0, 0))
        bleu_1_2 = corpus_bleu(references1, hypotheses1, weights=(0.5, 0.5, 0, 0))
        bleu_1_3 = corpus_bleu(references1, hypotheses1, weights=(0.33, 0.33, 0.33, 0))
        bleu_1_4 = corpus_bleu(references1, hypotheses1)

        writer.add_scalar('val_bleu1_1', bleu_1_1, epoch)
        writer.add_scalar('val_bleu1_2', bleu_1_2, epoch)
        writer.add_scalar('val_bleu1_3', bleu_1_3, epoch)
        writer.add_scalar('val_bleu1_4', bleu_1_4, epoch)

        bleu_2_1 = corpus_bleu(references2, hypotheses2, weights=(1, 0, 0, 0))
        bleu_2_2 = corpus_bleu(references2, hypotheses2, weights=(0.5, 0.5, 0, 0))
        bleu_2_3 = corpus_bleu(references2, hypotheses2, weights=(0.33, 0.33, 0.33, 0))
        bleu_2_4 = corpus_bleu(references2, hypotheses2)

        writer.add_scalar('val_bleu2_1', bleu_2_1, epoch)
        writer.add_scalar('val_bleu2_2', bleu_2_2, epoch)
        writer.add_scalar('val_bleu2_3', bleu_2_3, epoch)
        writer.add_scalar('val_bleu2_4', bleu_2_4, epoch)   

        print('Validation Epoch: {}\t'
              'BLEU-1 ({})\t'
              'BLEU-2 ({})\t'
              'BLEU-3 ({})\t'
              'BLEU-4 ({})\t'.format(epoch, bleu_1_1, bleu_1_2, bleu_1_3, bleu_1_4))
            
        print('Validation Epoch: {}\t'  
                'BLEU-1 ({})\t'
                'BLEU-2 ({})\t'
                'BLEU-3 ({})\t'
                'BLEU-4 ({})\t'.format(epoch, bleu_2_1, bleu_2_2, bleu_2_3, bleu_2_4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate of the decoder (default: 1e-4)')
    parser.add_argument('--step-size', type=int, default=5,
                        help='step size for learning rate annealing (default: 5)')
    parser.add_argument('--alpha-c', type=float, default=1, metavar='A',
                        help='regularization constant (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='L',
                        help='number of batches to wait before logging training stats (default: 100)')
    parser.add_argument('--data', type=str, default='data/coco',
                        help='path to data images (default: data/coco)')
    parser.add_argument('--network', choices=['vgg19', 'resnet152', 'densenet161'], default='vgg19',
                        help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--tf', action='store_true', default=False,
                        help='Use teacher forcing when training LSTM (default: False)')

    # custom arguments
    parser.add_argument('--estimator-hidden-size', type=int, default=512)

    main(parser.parse_args())
