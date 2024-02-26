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

            origin_folder = os.path.join(train_good_dir, 'rgb')
            images = os.listdir(origin_folder)

            mask_dir = os.path.join(train_good_dir, 'object_mask')
            os.makedirs(mask_dir, exist_ok=True)

            for image in images:

                img_dir = os.path.join(origin_folder, image)
                img_pil = Image.open(img_dir)

                # [1] save original image
                #Image.open(img_dir).save(os.path.join(origin_folder, image))

                # [2] remove background
                img_np = np.array(img_pil.convert('L'))
                h,w = img_np.shape
                mask = np.ones((h,w), dtype=np.uint8)
                mask_pil = Image.fromarray(mask * 255)
                mask_pil.save(os.path.join(mask_dir, image))

                # [3] remove original image
                os.remove(img_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec')
    parser.add_argument('--trg_cat', type=str, default='transistor')
    args = parser.parse_args()
    main(args)