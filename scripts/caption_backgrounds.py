import os, sys
from glob import glob
from subprocess import check_call

background_paths = glob('data/backgrounds/30_place_dataset/*/*.jpg')
output_paths = [ 'data/backgrounds/30_place_dataset_vgg_caption/' + '/'.join(path.split('/')[-2:])
                for path in background_paths]
model_ckpt_path = 'data/pretrained/VGG19_decoder.pth'

PYTHON = sys.executable

for background_path, output_path in zip(background_paths, output_paths):

    if not os.path.exists('/'.join(output_path.split('/')[:-1])):
        os.makedirs('/'.join(output_path.split('/')[:-1]))

    cmd = f"{PYTHON} generate_caption.py --img-path {background_path} --model {model_ckpt_path} --output-img-path {output_path}"
    print(cmd)
    check_call(cmd, shell=True)