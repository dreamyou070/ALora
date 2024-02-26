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
            images = os.listdir(train_good_dir)


            train_rgb_dir = os.path.join(train_good_dir, 'rgb')
            os.makedirs(train_rgb_dir, exist_ok=True)

            origin_folder = os.path.join(train_good_dir, 'rgb_origin')
            os.makedirs(origin_folder, exist_ok=True)

            sub_folder = os.path.join(train_good_dir, 'rgb_remove_background')
            os.makedirs(sub_folder, exist_ok=True)

            for image in images:
                img_dir = os.path.join(train_good_dir, image)

                # [1] save original image
                Image.open(img_dir).save(os.path.join(origin_folder, image))

                # [2] remove background
                sub_dir = os.path.join(sub_folder, image)
                remove_background(img_dir, sub_dir)
                Image.open(sub_dir).convert("RGB").save(os.path.join(train_rgb_dir, image))

                # [3] remove original image
                os.remove(img_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec')
    parser.add_argument('--trg_cat', type=str, default='zipper')
    args = parser.parse_args()
    main(args)