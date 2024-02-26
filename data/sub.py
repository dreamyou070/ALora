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

            train_dir = os.path.join(cat_dir, 'train')
            train_cropping_dir = os.path.join(cat_dir, 'train_cropping')
            os.makedirs(train_cropping_dir, exist_ok=True)

            train_good_dir = os.path.join(train_dir, 'good')
            train_good_cropping_dir = os.path.join(train_cropping_dir, 'good')
            os.makedirs(train_good_cropping_dir, exist_ok=True)

            rgb_remove_background_folder = os.path.join(train_good_dir, 'rgb_remove_background')
            images = os.listdir(rgb_remove_background_folder)

            rgb_folder = os.path.join(train_good_dir, 'rgb')
            os.makedirs(rgb_folder, exist_ok=True)


            for image in images:
                img_path = os.path.join(rgb_remove_background_folder, image)
                Image.open(img_path).convert('RGB').save(os.path.join(rgb_folder, image))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec')
    parser.add_argument('--trg_cat', type=str, default='toothbrush')
    args = parser.parse_args()
    main(args)