import os
from PIL import Image
import numpy as np

base_folder = 'turft/raw/cropped_sample'
train_folder = os.path.join(base_folder, 'test')
train_normal_folder  = os.path.join(train_folder, 'normal')
img_folder = os.path.join(train_normal_folder, 'rgb')
mask_folder = os.path.join(train_normal_folder, 'gt')


train_anormal_folder = os.path.join(train_folder, 'anormal')
anomal_img_folder = os.path.join(train_anormal_folder, 'rgb')
anomal_mask_folder = os.path.join(train_anormal_folder, 'gt')
os.makedirs(anomal_img_folder, exist_ok = True)
os.makedirs(anomal_mask_folder, exist_ok = True)

imgs = os.listdir(img_folder)

for img in imgs :

    img_path = os.path.join(img_folder, img)
    mask_path = os.path.join(mask_folder, img)
    mask_pil = Image.open(mask_path).convert('L')
    mask_sum = np.array(mask_pil).sum()
    # ------------------------------------------------------------------------
    if mask_sum != 0 :
        save_folder = train_anormal_folder
        new_img_path = os.path.join(anomal_img_folder, img)
        new_mask_path = os.path.join(anomal_mask_folder, img)
        os.rename(img_path, new_img_path)
        os.rename(mask_path, new_mask_path)
        """
        # ------------------------------------------------------------------------
        pil_img = Image.open(img_path).convert('L')
        np_img = np.array(pil_img)
        hist, bins = np.histogram(np_img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
    
        # History Equalization 공식
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        # Mask처리를 했던 부분을 다시 0으로 변환
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        img2 = cdf[np_img]
        pil_img2 = Image.fromarray(img2)
        pil_img2.save(os.path.join(save_folder,f'rgb/{img}'))
        mask_pil.save(os.path.join(save_folder,f'gt/{img}'))
        """