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

            rgb_folder = os.path.join(train_good_dir, 'rgb')
            rgb_cropping_folder = os.path.join(train_good_cropping_dir, 'rgb')
            os.makedirs(rgb_cropping_folder, exist_ok=True)

            object_mask_folder = os.path.join(train_good_dir, 'object_mask')
            object_mask_cropping_folder = os.path.join(train_good_cropping_dir, 'object_mask')
            os.makedirs(object_mask_cropping_folder, exist_ok=True)

            images = os.listdir(rgb_folder)

            for image in images:

                img_path = os.path.join(rgb_folder, image)
                object_mask_dir = os.path.join(object_mask_folder, image)

                object_mask_pil = Image.open(object_mask_dir).convert('L')
                object_mask_np = np.array(object_mask_pil)
                h, w = object_mask_np.shape
                h_indexs, w_indexs = [], []
                for h_i in range(h):
                    for w_i in range(w):
                        if object_mask_np[h_i, w_i] > 0:
                            h_indexs.append(h_i)
                            w_indexs.append(w_i)
                h_start, h_end = min(h_indexs), max(h_indexs)
                w_start, w_end = min(w_indexs), max(w_indexs)
                h_pad = 0.02 * h
                w_pad = 0.02 * w
                h_start = h_start - h_pad if h_start - h_pad > 0 else 0
                h_end = h_end + h_pad if h_end + h_pad < h else h
                w_start = w_start - w_pad if w_start - w_pad > 0 else 0
                w_end = w_end + w_pad if w_end + w_pad < w else w

                object_mask_pil = object_mask_pil.crop((w_start, h_start, w_end, h_end)).convert('L')
                cropped_img = Image.open(img_path).convert('RGB').crop((w_start, h_start, w_end, h_end))

                object_mask_pil.save(os.path.join(object_mask_cropping_folder, image))
                cropped_img.save(os.path.join(rgb_cropping_folder, image))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec')
    parser.add_argument('--trg_cat', type=str, default='capsule')
    args = parser.parse_args()
    main(args)