import os
import json
import string
from collections import Counter, defaultdict

import random
random.seed(1029)

import argparse

#####################################
########## Argument Parser ##########
#####################################

parser = argparse.ArgumentParser()

# input file arguments
parser.add_argument('--person_image_annotations_path', type=str, default='data/annotations/raw/person_image_annotation.json',
                    help='Path to the person image annotations json file as an input.')
parser.add_argument('--target_word_gender_map_path', type=str, default='data/annotations/target_word_gender_map.json',
                    help='Path to the mapping dictionary where keys are target words and values are gender tags.')
parser.add_argument('--target_word_pairs_path', type=str, default='data/annotations/target_word_pairs.json',
                    help='Path to the paired target words such as ("man", "woman") and ("boy", "girl")')

# parameter
parser.add_argument('--n_samples_per_pair', type=int, default=-1)

# output file arguments
parser.add_argument('--paired_annotations_dir', type=str, default='data/annotations/',
                    help='Directory to the paired annotations train/valid/test json file as an output.')
parser.add_argument('--no_matched_pair_images_path', type=str, default='data/annotations/no_matched_pairs_images.txt',)

args = parser.parse_args()

#####################################


def remove_punctuations(sentence):
    # punctuations without '<' and '>' because they are used for special tokens
    punctuations = string.punctuation.replace('<', '').replace('>', '')
    for p in punctuations:
        sentence = sentence.replace(p, '')
    return sentence


def pair_person_image_annotations(image_list, person_image_annotations, target_word_pairs, tgt2gender):
    pair_infos = defaultdict(list)
    '''
    pair infos: {pair: [pair_info, pair_info, ...]}
    :: pair (tutple): ex) ('man','woman'), ...
    :: pair_info (dict):
        : {'image1_filename':(str),
           'image2_filename':(str),
           'caption1':(str),
           'caption2':(str),
           'gender1':(str),
           'gender2':(str),
           'target_word_idx':(int),
           'cls_word_idx':(int)}
    '''

    for i, anchor_image in enumerate(image_list):
        # Get the list of captions for the anchor image
        anchor_captions = person_image_annotations[anchor_image]

        for anchor_caption in anchor_captions:
            anchor_caption['caption'] = remove_punctuations(anchor_caption['caption'])
            anchor_tgt_idx = anchor_caption['target_word_idx']
            anchor_cls_idx = anchor_caption['caption'].split().index('<cls>')
            anchor_tgt_word = anchor_caption['target_word']
            
            for compare_image in image_list[i+1:]:
                # Get the list of captions for the compare image
                compare_captions = person_image_annotations[compare_image]

                for compare_caption in compare_captions:
                    compare_caption['caption'] = remove_punctuations(compare_caption['caption'])
                    compare_tgt_idx = compare_caption['target_word_idx']
                    compare_cls_idx = compare_caption['caption'].split().index('<cls>')
                    compare_tgt_word = compare_caption['target_word']

                    if (anchor_tgt_idx == compare_tgt_idx) \
                        and (anchor_cls_idx == compare_cls_idx) \
                        and ({anchor_tgt_word, compare_tgt_word} in target_word_pairs):

                        pair = tuple(sorted([anchor_tgt_word, compare_tgt_word]))

                        pair_infos[pair].append(
                                    {'image1_filename': anchor_image,
                                    'image2_filename': compare_image,
                                    'caption1': anchor_caption['caption'],
                                    'caption2': compare_caption['caption'],
                                    'gender1': tgt2gender[anchor_tgt_word],
                                    'gender2': tgt2gender[compare_tgt_word],
                                    'target_word_idx': anchor_tgt_idx,
                                    'cls_word_idx': anchor_cls_idx})

    return pair_infos


def sample_pair_infos(pair_infos, n_samples_per_pair):
    # Sample n_samples_per_pair from each pair
    sampled_pair_infos = defaultdict(list)
    n_original_pairs = {}
    for pair, pair_info in pair_infos.items():
        n_original_pairs[pair] = len(pair_info)
        n_exclude = len(pair_info) - n_samples_per_pair
        # exclude the pair with the most frequent target word index
        most_freq_tgt_word_idx, cnt = Counter([info['target_word_idx'] for info in pair_info]).most_common(1)[0]
        sampled_pair_infos[pair] = [info for info in pair_info if info['target_word_idx'] != most_freq_tgt_word_idx] + \
                                    random.sample([info for info in pair_info if info['target_word_idx'] == most_freq_tgt_word_idx],
                                                    cnt - n_exclude)

    return sampled_pair_infos


def concat_pair_infos(image_list, pair_infos):
    # Concat the sampled pair infos
    final_pair_infos = []
    no_matched_pair_images = set(image_list)
    for pair, pair_info in pair_infos.items():        
        for info in pair_info:
            new_info = {'pair_id': len(final_pair_infos)}
            new_info.update(info)
            final_pair_infos.append(new_info)
            no_matched_pair_images -= {info['image1_filename'], info['image2_filename']}

    return final_pair_infos, no_matched_pair_images



def main():

    # Load the maually annotated captions
    with open(args.person_image_annotations_path, 'r') as f:
        person_image_annotation = json.load(f)    

    # Load target word gender map
    with open(args.target_word_gender_map_path, 'r') as f:
        tgt2gender = json.load(f)

    # Load target word pairs
    with open(args.target_word_pairs_path, 'r') as f:
        target_word_pairs = json.load(f)
    target_word_pairs = [set(pair) for pair in target_word_pairs]

    images = list(person_image_annotation.keys())

    '''
    ======================================
    == Split images into train/val/test ==
    ======================================
    '''

    # Add person info to the annotation
    # take the most frequent target word(person_info) among captions in the image
    person_info_splits = {'man':[], 'woman':[], 'boy':[], 'girl':[]}

    for image in images:
        target_words = []
        for annot in person_image_annotation[image]:
            target_words.append(annot['target_word'])
        person_info = Counter(target_words).most_common(1)[0][0]
        person_info_splits[person_info].append(image)
        
        new_annot = []
        for annot in person_image_annotation[image]:
            if annot['target_word'] == person_info:
                new_annot.append(annot)
        
        person_image_annotation[image] = new_annot

    # Split images to be in train, val, and test sets
    assert len(person_info_splits['man']) == 30 and len(person_info_splits['woman']) == 30
    assert len(person_info_splits['boy']) == 20 and len(person_info_splits['girl']) == 20

    n_train_adult, n_val_adult, n_test_adult = 16, 7, 7
    n_train_kid, n_val_kid, n_test_kid = 10, 5, 5
    
    train_person_images = []
    val_person_images = []
    test_person_images = []
    
    for person_info, images in person_info_splits.items():
        random.shuffle(image)
        if person_info in ['man', 'woman']:
            train_person_images += images[:n_train_adult]
            val_person_images += images[n_train_adult:n_train_adult+n_val_adult]
            test_person_images += images[n_train_adult+n_val_adult:]
        else:
            train_person_images += images[:n_train_kid]
            val_person_images += images[n_train_kid:n_train_kid+n_val_kid]
            test_person_images += images[n_train_kid+n_val_kid:]

    breakpoint()

    '''
    ======================================
    ======== Pairing and sampling ========
    ======================================
    '''

    # Generate training dataset
    train_pair_infos = pair_person_image_annotations(train_person_images, person_image_annotation, target_word_pairs, tgt2gender)
    if args.n_samples_per_pair > 0:
        train_pair_infos = sample_pair_infos(train_pair_infos, args.n_samples_per_pair)
    train_paired_annotations, no_matched_train_images = concat_pair_infos(train_person_images, train_pair_infos)

    print(f'Number of training images: {len(train_person_images)}')
    print(f'Number of training pairs: {len(train_paired_annotations)}')
    for pair in target_word_pairs:
        print(f' ==> {pair}: {len(train_pair_infos[pair])}')
    print(f'Number of training images without matched pairs: {len(no_matched_train_images)}')
    print()

    # Generate validation dataset
    val_pair_infos = pair_person_image_annotations(val_person_images, person_image_annotation, target_word_pairs, tgt2gender)
    if args.n_samples_per_pair > 0:
        val_pair_infos = sample_pair_infos(val_pair_infos, args.n_samples_per_pair)
    val_paired_annotations, no_matched_val_images = concat_pair_infos(val_person_images, val_pair_infos)

    print(f'Number of validation images: {len(val_person_images)}')
    print(f'Number of validation pairs: {len(val_paired_annotations)}')
    for pair in target_word_pairs:
        print(f' ==> {pair}: {len(val_pair_infos[pair])}')
    print(f'Number of validation images without matched pairs: {len(no_matched_val_images)}')
    print()

    # Generate test dataset
    test_pair_infos = pair_person_image_annotations(test_person_images, person_image_annotation, target_word_pairs, tgt2gender)
    if args.n_samples_per_pair > 0:
        test_pair_infos = sample_pair_infos(test_pair_infos, args.n_samples_per_pair)
    test_paired_annotations, no_matched_test_images = concat_pair_infos(test_person_images, test_pair_infos)

    print(f'Number of test images: {len(test_person_images)}')
    print(f'Number of test pairs: {len(test_paired_annotations)}')
    for pair in target_word_pairs:
        print(f' ==> {pair}: {len(test_pair_infos[pair])}')
    print(f'Number of test images without matched pairs: {len(no_matched_test_images)}')
    print()


    '''
    ======================================
    =============== Saving ===============
    ======================================
    '''
    # Save the paired annotations
    with open( os.path.join(args.paired_annotations_dir, 'train_paired_annotations.json'), 'w')  as f:
        json.dump(train_paired_annotations, f, indent=4)
    with open( os.path.join(args.paired_annotations_dir, 'val_paired_annotations.json'), 'w')  as f:
        json.dump(val_paired_annotations, f, indent=4)
    with open( os.path.join(args.paired_annotations_dir, 'test_paired_annotations.json'), 'w')  as f:
        json.dump(test_paired_annotations, f, indent=4)

    # Save the images without matched pair
    no_matched_pair_images = no_matched_train_images + no_matched_val_images + no_matched_test_images
    if no_matched_pair_images:
        with open(args.no_matched_pair_images_path, 'w') as f:
            for image in no_matched_pair_images:
                f.write(image + '\n')

if __name__ == '__main__':
    main()