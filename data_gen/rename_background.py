import os
from glob import glob
import json

bg_folder = "./data/backgrounds/30_place_dataset/" # directory to backgrounds
final_folder = "./data/final" # directory to final sythesized images
wrong2correct_json = './data_gen/wrong2correct.json'

bg_classes = os.listdir(bg_folder)

wrong2correct = {}

temp_bg_paths = []
temp_final_dirs = []

for bg_class in bg_classes:
    original_bg_paths = sorted( glob(os.path.join(bg_folder, bg_class, '*.jpg')) )
    wrong_bg_names = [ os.path.basename(bg_path).split('.')[0] for bg_path in original_bg_paths ]
    
    for i, original_bg_path in enumerate( original_bg_paths ):
       
        wrong_bg_name = os.path.basename(original_bg_path).split('.')[0]
        wrong2correct[wrong_bg_name] = f'{bg_class}_{i}'

        temp_bg_path = os.path.join( os.path.dirname(original_bg_path),f'_{bg_class}_{i}.jpg' )
        os.rename(original_bg_path, temp_bg_path)

        original_final_dir = os.path.join( final_folder, wrong_bg_name ) 
        temp_final_dir = os.path.join( final_folder, f'_{bg_class}_{i}' )
        os.rename(original_final_dir, temp_final_dir)

        temp_bg_paths.append(temp_bg_path)
        temp_final_dirs.append(temp_final_dir)

for temp_bg_path in temp_bg_paths:
    correct_bg_path = os.path.join( os.path.dirname(temp_bg_path), os.path.basename(temp_bg_path)[1:] )
    os.rename(temp_bg_path, correct_bg_path)

for temp_final_dir in temp_final_dirs:
    correct_final_dir = os.path.join( os.path.dirname(temp_final_dir), os.path.basename(temp_final_dir)[1:] )
    os.rename(temp_final_dir, correct_final_dir)

with open(wrong2correct_json, 'w') as f:
    json.dump(wrong2correct, f, indent=4)
