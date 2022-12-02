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
parser.add_argument('--n_samples_per_pair', type=int, default=100)

# output file arguments
parser.add_argument('--paired_annotations_path', type=str, default='data/annotations/paired_annotation.json',
                    help='Path to the paired annotations json file as an output.')
parser.add_argument('--no_matched_pair_images_path', type=str, default='data/annotations/no_matched_pairs_images.txt',)

args = parser.parse_args()

#####################################


def remove_punctuations(sentence):
    # punctuations without '<' and '>' because they are used for special tokens
    punctuations = string.punctuation.replace('<', '').replace('>', '')
    for p in punctuations:
        sentence = sentence.replace(p, '')
    return sentence


def is_ignored(prob):
    return random.random() > prob


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

    pair_infos = defaultdict(list)
    '''
    pair infos: {pair: pair_info}
    :: pair (tuple):
        a sorted pair of target words   
    :: pair_info (list):
        a list of pair information (dict)
       (dict): {'image1_filename':(str),
           'image2_filename':(str),
           'caption1':(str),
           'caption2':(str),
           'gender1':(str),
           'gender2':(str),
           'target_word_idx':(int),
           'cls_word_idx':(int)}
    '''

    for i, anchor_image in enumerate(images):
        # Get the list of captions for the anchor image
        anchor_captions = person_image_annotation[anchor_image]

        for anchor_caption in anchor_captions:
            anchor_caption['caption'] = remove_punctuations(anchor_caption['caption'])
            anchor_tgt_idx = anchor_caption['target_word_idx']
            anchor_cls_idx = anchor_caption['caption'].split().index('<cls>')
            anchor_tgt_word = anchor_caption['target_word']
            
            for compare_image in images[i+1:]:
                # Get the list of captions for the compare image
                compare_captions = person_image_annotation[compare_image]

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

    # Sample n_samples_per_pair from each pair
    sampled_pair_infos = defaultdict(list)
    n_original_pairs = {}
    for pair, pair_info in pair_infos.items():
        n_original_pairs[pair] = len(pair_info)
        n_exclude = len(pair_info) - args.n_samples_per_pair
        # exclude the pair with the most frequent target word index
        most_freq_tgt_word_idx, cnt = Counter([info['target_word_idx'] for info in pair_info]).most_common(1)[0]
        sampled_pair_infos[pair] = [info for info in pair_info if info['target_word_idx'] != most_freq_tgt_word_idx] + \
                                    random.sample([info for info in pair_info if info['target_word_idx'] == most_freq_tgt_word_idx],
                                                    cnt - n_exclude)

    # Concat the sampled pair infos
    final_pair_infos = []
    no_matched_pair_images = set(person_image_annotation.keys())
    for pair, pair_info in sampled_pair_infos.items():        
        for info in pair_info:
            new_info = {'pair_id': len(final_pair_infos)}
            new_info.update(info)
            final_pair_infos.append(new_info)
            no_matched_pair_images -= {info['image1_filename'], info['image2_filename']}

    print(f'Number of pairs: {len(final_pair_infos)}')
    print(f'Number of images without matched pair: {len(no_matched_pair_images)}')
    for pair in target_word_pairs:
        pair = tuple(sorted(pair))
        print(f'selected {pair} pairs: {len(sampled_pair_infos[pair])} / {n_original_pairs[pair]}')

    # Save the paired annotations
    with open(args.paired_annotations_path, 'w') as f:
        json.dump(final_pair_infos, f, indent=4)

    # Save the images without matched pair
    if no_matched_pair_images:
        with open(args.no_matched_pair_images_path, 'w') as f:
            for image in no_matched_pair_images:
                f.write(image + '\n')

if __name__ == '__main__':
    main()