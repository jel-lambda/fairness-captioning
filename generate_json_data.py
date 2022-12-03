'''
Original code from https://github.com/AaronCCWong/Show-Attend-and-Tell
'''

import argparse, json
from collections import Counter


def generate_json_data(split_path, data_path, max_captions_per_image, min_word_count):
    split = json.load(open(split_path, 'r'))
    word_count = Counter()

    max_length = 0
    for img in split['images']:
        caption_count = 0
        for sentence in img['sentences']:
            if caption_count < max_captions_per_image:
                caption_count += 1
            else:
                break

            max_length = max(max_length, len(sentence['tokens']))
            word_count.update(sentence['tokens'])

    words = ['<start>', '<eos>', '<unk>', '<pad>'] + [word for word in word_count.keys() if word_count[word] >= min_word_count]
    word_dict = {word: idx + 4 for idx, word in enumerate(words)}

    with open(data_path + '/word_dict.json', 'w') as f:
        json.dump(word_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate json files')
    parser.add_argument('--split-path', type=str, default='data/coco/dataset.json')
    parser.add_argument('--data-path', type=str, default='data/coco')
    parser.add_argument('--max-captions', type=int, default=5,
                        help='maximum number of captions per image')
    parser.add_argument('--min-word-count', type=int, default=5,
                        help='minimum number of occurences of a word to be included in word dictionary')
    args = parser.parse_args()

    generate_json_data(args.split_path, args.data_path, args.max_captions, args.min_word_count)
