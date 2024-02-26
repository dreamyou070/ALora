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
            test_dir = os.path.join(cat_dir, 'test')
            ground_truth_dir = os.path.join(cat_dir, 'ground_truth')

            defetcs = os.listdir(test_dir)

            for defect in defetcs:

                defect_dir = os.path.join(test_dir, f'{defect}')
                gt_defect_dir = os.path.join(ground_truth_dir, f'{defect}')

                imgs = os.listdir(defect_dir)

                for img in imgs:
                    name, ext = os.path.splitext(img)
                    gt_name = f'{name}_mask{ext}'

                    # [1] new files
                    rgb_dir = os.path.join(defect_dir, 'rgb_origin')
                    os.makedirs(rgb_dir, exist_ok=True)
                    gt_folder = os.path.join(defect_dir, 'gt')
                    os.makedirs(gt_folder, exist_ok=True)

                    # [1] save original image
                    origin_rgb_dir = os.path.join(defect_dir, img)
                    Image.open(origin_rgb_dir).save(os.path.join(rgb_dir, img))

                    # [3] copy to rgb_origin files
                    if 'good' not in defect :
                        Image.open(os.path.join(gt_defect_dir, gt_name)).convert("L").save(os.path.join(gt_folder, img))
                    else :
                        pseudo_gt = np.array(Image.open(origin_rgb_dir))*0
                        Image.fromarray(pseudo_gt.astype(np.uint8)).convert("L").save(os.path.join(gt_folder, img))

                    os.remove(origin_rgb_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec')
    parser.add_argument('--trg_cat', type=str, default='toothbrush')
    args = parser.parse_args()
    main(args)