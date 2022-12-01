import json
import string

import random
random.seed(1029)

import argparse

#####################################
########## Argument Parser ##########
#####################################

parser = argparse.ArgumentParser()

# input file arguments
parser.add_argument('--person_image_annotations_path', type=str, default='data/annotations/person_image_annotation.json',
                    help='Path to the person image annotations json file as an input.')
parser.add_argument('--target_word_gender_map_path', type=str, default='data/annotations/target_word_gender_map.json',
                    help='Path to the mapping dictionary where keys are target words and values are gender tags.')
parser.add_argument('--target_word_pairs_path', type=str, default='data/annotations/target_word_pairs.json',
                    help='Path to the paired target words such as ("man", "woman") and ("boy", "girl")')

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

    pair_info = []
    '''
    pair_info (list of dictionary)
    dict: {'pair_id':(int),
           'image1_filename':(str),
           'image2_filename':(str),
           'caption1':(str),
           'caption2':(str),
           'gender1':(str),
           'gender2':(str),
           'target_word_idx':(int),
           'cls_word_idx':(int)}
    '''

    no_matched_pair_images = []

    n_boy_girl = 0
    n_man_woman = 0
    n_man_nutural = 0
    n_woman_nutural = 0

    n_ignored_pairs = 0

    for i, anchor_image in enumerate(images):
        # Get the list of captions for the anchor image
        anchor_captions = person_image_annotation[anchor_image]

        no_match = True

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

                        # ignore (man, woman) pairs with index 1 randomly
                        if ({anchor_tgt_word, compare_tgt_word} == {'man', 'woman'}) \
                            and (anchor_tgt_idx == 1) \
                            and is_ignored(0.1):   
                            n_ignored_pairs += 1
                            continue

                        pair_info.append({'pair_id': len(pair_info),
                                        'image1_filename': anchor_image,
                                        'image2_filename': compare_image,
                                        'caption1': anchor_caption['caption'],
                                        'caption2': compare_caption['caption'],
                                        'gender1': tgt2gender[anchor_tgt_word],
                                        'gender2': tgt2gender[compare_tgt_word],
                                        'target_word_idx': anchor_tgt_idx,
                                        'cls_word_idx': anchor_cls_idx})

                        no_match = False

                        if {anchor_tgt_word, compare_tgt_word} == {'boy', 'girl'}:
                            n_boy_girl += 1
                        if {anchor_tgt_word, compare_tgt_word} == {'man', 'woman'}:
                            n_man_woman += 1
                        if {anchor_tgt_word, compare_tgt_word} == {'man', 'person'}:
                            n_man_nutural += 1
                        if {anchor_tgt_word, compare_tgt_word} == {'woman', 'person'}:
                            n_woman_nutural += 1

        if no_match:
            no_matched_pair_images.append(anchor_image)

    print(f'Number of pairs: {len(pair_info)}')
    print(f'Number of images without matched pair: {len(no_matched_pair_images)}')
    print(f':: ("man", "woman") pairs = {n_man_woman}')
    print(f':: ("boy", "girl") pairs = {n_boy_girl}')
    print(f':: ("man", "person") pairs = {n_man_nutural}')
    print(f':: ("woman", "person") pairs = {n_woman_nutural}')
    print()
    print(f':: Number of ignored ("man", "woman") pairs: {n_ignored_pairs}')

    # Save the paired annotations
    with open(args.paired_annotations_path, 'w') as f:
        json.dump(pair_info, f, indent=4)

    # Save the images without matched pair
    with open(args.no_matched_pair_images_path, 'w') as f:
        for image in no_matched_pair_images:
            f.write(image + '\n')

if __name__ == '__main__':
    main()