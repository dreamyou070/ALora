import os
import numpy as np
from torch.utils.data import Dataset
import torch
import glob
from data.perlin import rand_perlin_2d_np
from PIL import Image
from torchvision import transforms
import cv2
import imgaug.augmenters as iaa


perlin_scale = 3
min_perlin_scale = 2
perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
perlin_noise = rand_perlin_2d_np((512,512),
                                 (perlin_scalex, perlin_scaley))
threshold = 0.5
# perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
# 0 and more than 0.5
perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
# smoothing
perlin_thr = cv2.GaussianBlur(perlin_thr, (3, 3), 0)
Image.fromarray((perlin_thr * 255).astype(np.uint8)).show()