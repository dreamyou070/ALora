import os
from PIL import Image, ImageFilter
import numpy as np

base_folder = 'turft'
train_folder = os.path.join(base_folder, 'train')
test_folder = os.path.join(base_folder, 'test')

train_normal_folder = os.path.join(train_folder, 'normal')
train_normal_rgb_folder = os.path.join(train_normal_folder,'rgb')
train_normal_gt_folder = os.path.join(train_normal_folder,'gt')
images = os.listdir(train_normal_rgb_folder)
for i, image in enumerate(images) :
    if i < 1 :
        image_path = os.path.join(train_normal_rgb_folder, image)
        mask_path = os.path.join(train_normal_gt_folder, image)
        mask_pil = Image.open(mask_path).convert('L')
        #print(mask_pil.size)
        print(image)

        #
        pil_img = Image.open(image_path).convert('L')
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
        #pil_img2.show()
        sharpened_pil = pil_img2.filter(ImageFilter.SHARPEN)#.resize((512,512),Image.BICUBIC)
        sharpened_pil = sharpened_pil.filter(ImageFilter.SHARPEN)#.resize((512,512),Image.BICUBIC)
        print(sharpened_pil.size)

        sharpened_pil.show()
