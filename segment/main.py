from segment_anything import SamPredictor, sam_model_registry
import argparse, os
from PIL import Image
import numpy as np
from segment_anything import SamAutomaticMaskGenerator


def main(args):


    print(f'step 1. prepare model')

    model_type = "vit_h"
    path_to_checkpoint= r'/home/dreamyou070/pretrained_stable_diffusion/sam_vit_h_4b8939.pth'
    sam = sam_model_registry[model_type](checkpoint=path_to_checkpoint)
    predictor = SamPredictor(sam)


    print(f'step 2. prepare images')
    base_folder = args.base_folder
    cats = os.listdir(base_folder)

    for cat in cats:
        cat_dir = os.path.join(base_folder, f'{cat}')
        normalities = os.listdir(cat_dir)
        for normality in normalities :
            normality_dir = os.path.join(cat_dir, normality)
            xray_folder = os.path.join(normality_dir, 'xray')

            dir1 = os.path.join(normality_dir, 'mask_1')
            dir2 = os.path.join(normality_dir, 'mask_2')
            dir3 = os.path.join(normality_dir, 'mask_3')
            os.makedirs(dir1, exist_ok=True)
            os.makedirs(dir2, exist_ok=True)
            os.makedirs(dir3, exist_ok=True)

            images = os.listdir(xray_folder)
            for img in images :
                img_dir = os.path.join(xray_folder,img)
                pil_img = Image.open(img_dir)
                org_h, org_w = pil_img.size

                np_img = np.array(pil_img)

                # [1] setting the image
                predictor.set_image(np_img)

                # [2]
                h, w, c = np_img.shape
                input_point = np.array([[0, 0], ])
                # [int(h/2),int(w/2)]])
                input_label = np.array([1])  # 1 indicates a foreground point
                masks, scores, logits = predictor.predict(point_coords=input_point,
                                                          point_labels=input_label,
                                                          multimask_output=True, )
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    np_mask = (mask * 1)
                    np_mask = np.where(np_mask == 1, 0, 1) * 255
                    sam_result_pil = Image.fromarray(np_mask.astype(np.uint8))
                    sam_result_pil = sam_result_pil.resize((org_h, org_w))
                    save_dir = os.path.join(normality_dir, f'mask_{i+1}/{img}')
                    sam_result_pil.save(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/NFBS_Dataset_SY')
    parser.add_argument('--trg_cat', type=str, default='transistor')
    args = parser.parse_args()
    main(args)