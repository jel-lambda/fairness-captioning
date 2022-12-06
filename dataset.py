import json, os
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset

from PIL import Image



def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImagePairedCaptionDataset(Dataset):
    def __init__(self, transform, images_dir, annotations_dir, word_dict, split_type='train'):
        super(ImagePairedCaptionDataset, self).__init__()
        self.split_type = split_type
        self.transform = transform

        self.word_Counter = Counter()

        background_template_path = os.path.join(annotations_dir, 'background_template.json')
        self.backgroud_label2template = json.load( open(background_template_path, 'r') )

        paired_annotations = json.load( open(os.path.join(annotations_dir, f'{split_type}_paired_annotations.json'), 'r') )

        self.img_path_pairs = [] # list of tuple (img1, img2)
        self.caption_pairs = [] # list of tuple (caption1, caption2)
        self.target_word_pairs = [] # list of tuple (target_word1, target_word2)
        self.target_word_idxs = [] # list of int
        self.cls_template_idxs = [] # list of list [cls_idx_start, ..., cls_idx_end]

        self.image2captions = defaultdict(set) # dict of set of str

        for paired_annot in paired_annotations:
            target_word_pair = (paired_annot['target_word1'], paired_annot['target_word2'])

            for background_label, cls_template in self.backgroud_label2template.items():
                
                for i in range(5):
                    # Iterate over 5 background images

                    image_path_pair = (os.path.join(images_dir, f'{background_label}_{i}/{paired_annot["image1_filename"]}'),
                                    os.path.join(images_dir, f'{background_label}_{i}/{paired_annot["image2_filename"]}'))    
                    
                    target_word_idx, cls_idx = paired_annot['target_word_idx'], paired_annot['cls_word_idx']
                    cls_template = self.backgroud_label2template[background_label]

                    caption1 = self.replace_cls_to_template(paired_annot['caption1'], cls_idx, cls_template)
                    caption2 = self.replace_cls_to_template(paired_annot['caption2'], cls_idx, cls_template)
                    caption_pair = (caption1, caption2)

                    if cls_idx < target_word_idx:
                        # when target word is appeared after background_cls word
                        target_word_idx = target_word_idx + len(cls_template.split()) - 1
                    cls_templete_idx = list( range( paired_annot['cls_word_idx'], paired_annot['cls_word_idx']+len(cls_template.split()) ) )

                    self.img_path_pairs.append(image_path_pair)
                    self.caption_pairs.append(caption_pair)
                    self.target_word_pairs.append(target_word_pair)
                    self.target_word_idxs.append(target_word_idx)
                    self.cls_template_idxs.append(cls_templete_idx)

                    # update word counter
                    self.word_Counter.update(caption1.split(' '))
                    self.word_Counter.update(caption2.split(' '))

                    # Add to image2captions
                    self.image2captions[image_path_pair[0]].add(caption1)
                    self.image2captions[image_path_pair[1]].add(caption2)

        self.word2idx = defaultdict(lambda: word_dict['<unk>'])
        self.word2idx.update(word_dict)

        self.pad_idx = self.word2idx['<pad>']
        

    def replace_cls_to_template(self, caption, cls_idx, cls_template):
        caption = caption.split(' ')
        cls_template = cls_template.split(' ')
        caption = caption[:cls_idx] + cls_template + caption[cls_idx+1:]
        return ' '.join(caption)

    def __getitem__(self, index):
        img_path1, img_path2 = self.img_path_pairs[index]
        img1, img2 = pil_loader(img_path1), pil_loader(img_path2)

        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)

        caption1, caption2 = self.caption_pairs[index]
        caption1 = [self.word2idx['<start>']] + [self.word2idx[word] for word in caption1.split(' ')] + [self.word2idx['<eos>']]
        caption2 = [self.word2idx['<start>']] + [self.word2idx[word] for word in caption2.split(' ')] + [self.word2idx['<eos>']]

        # Masks for CLUB and InfoNCE
        CLUB_mask = torch.zeros( len(caption1) )
        InfoNCE_mask = torch.zeros( len(caption1) )

        target_word_pair = self.target_word_pairs[index]
        if 'person' in target_word_pair:
            # if one of the target word is 'person'
            # target word andd background template words are masked as InfoNCE loss
            InfoNCE_mask[self.target_word_idxs[index]] = 1
            InfoNCE_mask[self.cls_template_idxs[index]] = 1
        else:
            # if both of the target words are not 'person'
            # target words are masked as CLUB loss, while background template words are masked as InfoNCE loss
            CLUB_mask[self.target_word_idxs[index]] = 1
            InfoNCE_mask[self.cls_template_idxs[index]] = 1

        if self.split_type == 'train':
            return (torch.FloatTensor(img1), caption1),\
                   (torch.FloatTensor(img2), caption2),\
                   CLUB_mask.tolist(), InfoNCE_mask.tolist()

        elif self.split_type == 'val':
            all_captions1 = [ [self.word2idx['<sos>']] + [self.word2idx[word] for word in caption.split(' ')] + [self.word2idx['<eos>']]
                            for caption in self.image2captions[img_path1]]
            all_captions2 = [ [self.word2idx['<sos>']] + [self.word2idx[word] for word in caption.split(' ')] + [self.word2idx['<eos>']]
                            for caption in self.image2captions[img_path2]]
            return (torch.FloatTensor(img1), caption1, all_captions1),\
                   (torch.FloatTensor(img2), caption2, all_captions2),\
                   CLUB_mask.tolist(), InfoNCE_mask.tolist()

    def __len__(self):
        return len(self.img_path_pairs)


    def collate_fn(self, data):

        if self.split_type == 'train':
            img_cap_1s, img_cap_2s, target_word_masks, cls_template_masks = zip(*data)
            img1s, caption1s = zip(*img_cap_1s)
            img2s, caption2s = zip(*img_cap_2s)
            max_caption_len = max( max([len(caption1) for caption1 in caption1s]), max([len(caption2) for caption2 in caption2s]) )

        elif self.split_type == 'val':
            img_cap_1s, img_cap_2s, target_word_masks, cls_template_masks = zip(*data)
            img1s, caption1s, all_captions1s = zip(*img_cap_1s)
            img2s, caption2s, all_captions2s = zip(*img_cap_2s)
            max_caption_len = 0
            for all_captions in all_captions1s + all_captions2s:
                max_caption_len = max( max_caption_len, max([len(caption) for caption in all_captions]) )
            
        caption1s = [caption1 + [self.pad_idx] * (max_caption_len - len(caption1)) for caption1 in caption1s]
        caption2s = [caption2 + [self.pad_idx] * (max_caption_len - len(caption2)) for caption2 in caption2s]
        target_word_masks = [mask + [0] * (max_caption_len - len(mask)) for mask in target_word_masks]
        cls_template_masks = [mask + [0] * (max_caption_len - len(mask)) for mask in cls_template_masks]

        if self.split_type == 'train':
            return ( torch.stack(img1s), torch.LongTensor(caption1s) ),\
                   ( torch.stack(img2s), torch.LongTensor(caption2s) ),\
                   torch.LongTensor(target_word_masks), torch.LongTensor(cls_template_masks)

        elif self.split_type == 'val':
            all_captions1s = [ [caption + [self.pad_idx] * (max_caption_len - len(caption)) for caption in all_captions1]
                            for all_captions1 in all_captions1s ]
            all_captions2s = [ [caption + [self.pad_idx] * (max_caption_len - len(caption)) for caption in all_captions2]
                            for all_captions2 in all_captions2s ]

            return ( torch.stack(img1s), torch.LongTensor(caption1s), torch.LongTensor(all_captions1s) ),\
                   ( torch.stack(img2s), torch.LongTensor(caption2s), torch.LongTensor(all_captions2s) ),\
                   torch.LongTensor(target_word_masks), torch.LongTensor(cls_template_masks)



class ImageCaptionDataset(Dataset):
    pass
