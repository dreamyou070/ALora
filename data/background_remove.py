from PIL import Image
from rembg import remove
import numpy as np
import os
import argparse

def remove_background(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)

def main(args):

    print(f'step 2. prepare images')
    base_folder = args.base_folder
    cats = os.listdir(base_folder)

    for cat in cats:
        if cat == args.trg_cat:

            cat_dir = os.path.join(base_folder, f'{cat}')
            train_good_dir = os.path.join(cat_dir, 'train/good')

            train_rgb_dir = os.path.join(train_good_dir, 'rgb')
            origin_folder = os.path.join(train_good_dir, 'rgb_origin')
            os.makedirs(origin_folder, exist_ok=True)
            sub_folder = os.path.join(train_good_dir, 'rgb_remove_background')
            os.makedirs(sub_folder, exist_ok=True)

            images = os.listdir(train_rgb_dir)
            for image in images:

                img_dir = os.path.join(train_rgb_dir, image)
                pil_img = Image.open(img_dir)

                # [1] save original image
                origin_img_dir = os.path.join(origin_folder, image)
                pil_img.save(origin_img_dir)
                # [2] remove background
                sub_dir = os.path.join(sub_folder, image)
                remove_background(img_dir, sub_dir)
                background_removed_img = Image.open(sub_dir).convert("RGB")
                background_removed_img.save(img_dir)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec')
    parser.add_argument('--trg_cat', type=str, default='transistor')
    args = parser.parse_args()
    main(args)