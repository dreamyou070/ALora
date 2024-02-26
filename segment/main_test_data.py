from segment_anything import SamPredictor, sam_model_registry
import argparse, os
from PIL import Image
import numpy as np
from segment_anything import SamAutomaticMaskGenerator
from rembg import remove

def remove_background(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)

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
        if cat == args.trg_cat:

            cat_dir = os.path.join(base_folder, f'{cat}')
            test_dir = os.path.join(cat_dir, 'test')
            defects_folders = os.listdir(test_dir)

            for defect_folder in defects_folders:
                defect_dir = os.path.join(test_dir, f'{defect_folder}')

                rgb_org_folder = os.path.join(defect_dir, 'rgb_org')
                images = os.listdir(rgb_org_folder)

                backremove_sub_folder = os.path.join(defect_dir, 'backremove_sub')
                os.makedirs(backremove_sub_folder, exist_ok=True)


                object_mask_folder = os.path.join(defect_dir, 'object_mask_second')
                os.makedirs(object_mask_folder, exist_ok=True)
                mask_1_folder = os.path.join(object_mask_folder, 'mask_1')
                os.makedirs(mask_1_folder, exist_ok=True)
                mask_2_folder = os.path.join(object_mask_folder, 'mask_2')
                os.makedirs(mask_2_folder, exist_ok=True)
                mask_3_folder = os.path.join(object_mask_folder, 'mask_3')
                os.makedirs(mask_3_folder, exist_ok=True)

                for image in images:

                    # [1] background remove
                    img_dir = os.path.join(rgb_org_folder, image)
                    sub_dir = os.path.join(backremove_sub_folder, image)
                    remove_background(img_dir, sub_dir)
                    Image.open(sub_dir).convert('RGB').save(sub_dir)

                    # [2] segment anything
                    pil_img = Image.open(sub_dir).convert('RGB')
                    org_h, org_w = pil_img.size
                    np_img = np.array(pil_img)

                    # [1] setting the image
                    predictor.set_image(np_img)

                    # [2]
                    h, w, c = np_img.shape
                    input_point = np.array([[0, 0],])
                                            #[int(h/2),int(w/2)]])
                    input_label = np.array([1])  # 1 indicates a foreground point
                    masks, scores, logits = predictor.predict(point_coords=input_point,
                                                              point_labels=input_label,
                                                              multimask_output=True, )
                    for i, (mask, score) in enumerate(zip(masks, scores)):
                        np_mask = (mask * 1)
                        np_mask = np.where(np_mask == 1, 0, 1) * 255
                        sam_result_pil = Image.fromarray(np_mask.astype(np.uint8))
                        sam_result_pil = sam_result_pil.resize((org_h, org_w))
                        mask_save_dir = os.path.join(object_mask_folder, f'mask_{i+1}/{image}')
                        sam_result_pil.save(mask_save_dir)

                    # [2]
                    #os.rename(img_dir, os.path.join(rgb_org_folder, image))

                    # [3]
                    #org_np = np.array(pil_img)
                    #mask = np.array(Image.open(mask_save_dir).convert('L'))
                    #mask = np.where(mask > 0, 1, 0)  # .expand_dims(2).repeat(3, axis=2)
                    #mask = mask[:, :, np.newaxis].repeat(3, axis=2)
                    #new_img = org_np * mask
                    #new_img = Image.fromarray(new_img.astype(np.uint8))
                    #new_img.save(os.path.join(rgb_folder, image))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec')
    parser.add_argument('--trg_cat', type=str, default='screw')
    args = parser.parse_args()
    main(args)