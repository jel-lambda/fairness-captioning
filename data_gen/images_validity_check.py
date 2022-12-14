import os
import cv2
from tqdm import tqdm
from glob import glob

bg_folder = "../../efs/final"
bg_classes = os.listdir(bg_folder)
images = glob(os.path.join(bg_folder, '*/*.png'))
i = 0
for image in tqdm(images):
    if not image.endswith('.png'):
        print("not image", image)
        continue
    bg = cv2.imread(image)
    if bg is None:
        print("none", image)
        continue
    w, h, c = bg.shape
    i += 1

print("total_num", i)