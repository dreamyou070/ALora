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
from transformers import CLIPVisionModel, AutoProcessor

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

#img_dir = '/home/dreamyou070/MyData/anomaly_detection/MVTec/transistor/train/good/rgb/000.png'
#image = Image.open(img_dir).convert('RGB')
#inputs = processor(images=image, return_tensors="pt").to(device)
#outputs = model(**inputs)
#last_hidden_state = outputs.last_hidden_state
#pooled_output = outputs.pooler_output  # pooled CLS states
#print(f'clip last hidden state : {last_hidden_state.shape}')
#print(f'clip pooled output : {pooled_output.shape}')


def passing_mvtec_argument(args):
    global argument
    global anomal_p
    global do_rot_augment

    argument = args
    anomal_p = args.anomal_p
    do_rot_augment = args.do_rot_augment

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample




class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self,
                 root_dir,
                 anomaly_source_path,
                 resize_shape=None,
                 tokenizer=None,
                 caption : str = None,
                 use_perlin: bool = False,
                 anomal_only_on_object : bool = True,
                 anomal_training : bool = False,
                 latent_res : int = 64,
                 do_anomal_sample : bool = True,
                 use_object_mask : bool = True) :

        # [1] base image
        self.root_dir = root_dir
        image_paths = []
        folder_path = os.path.join(root_dir, 'good')
        rgb_folder = os.path.join(folder_path, "rgb")
        images = os.listdir(rgb_folder)
        for image in images:
            image_path = os.path.join(rgb_folder, image)
            image_paths.append(image_path)

        self.resize_shape=resize_shape
        if do_anomal_sample :
            assert anomaly_source_path is not None, "anomaly_source_path should be given"

        if anomaly_source_path is not None:
            self.anomaly_source_paths = []
            for ext in ["png", "jpg"]:
                self.anomaly_source_paths.extend(sorted(glob.glob(anomaly_source_path + f"/*/*/*.{ext}")))
        else :
            self.anomaly_source_paths = []

        self.caption = caption
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5]),])
        self.use_perlin = use_perlin
        if do_rot_augment :
            self.rot_augmenters = [iaa.Affine(rotate=(0, 0)),
                                   iaa.Affine(rotate=(180, 180)),
                                   iaa.Affine(rotate=(90, 90)),
                                   iaa.Affine(rotate=(270, 270)),
                                   iaa.flip.Fliplr(p=1),]
        else :
            self.rot_augmenters = [iaa.Affine(rotate=(0, 0))]
        num_repeat = len(self.rot_augmenters) # 4
        self.image_paths = [image_path for image_path in image_paths for i in range(num_repeat)]

        self.anomal_only_on_object = anomal_only_on_object
        self.anomal_training = anomal_training
        self.latent_res = latent_res

        background_base_dir = os.path.join(folder_path, "background")
        self.background_paths = []
        for ext in ["png", "jpg"]:
            self.background_paths.extend(sorted(glob.glob(background_base_dir + f"/*.{ext}")))

        self.use_object_mask = use_object_mask

    def __len__(self):
        if len(self.anomaly_source_paths) > 0 :
            return max(len(self.image_paths), len(self.anomaly_source_paths))
        else:
            return len(self.image_paths)

    def torch_to_pil(self, torch_img):
        # torch_img = [3, H, W], from -1 to 1
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil = Image.fromarray(np_img)
    def get_img_name(self, img_path):
        rgb_folder, name = os.path.split(img_path)
        net_name, ext = os.path.splitext(name)
        #class_folder, rgb = os.path.split(rgb_folder)
        #return name, class_folder
        return net_name

    def get_object_mask_dir(self, img_path):
        parent, name = os.path.split(img_path)
        parent, _ = os.path.split(parent)
        object_mask_dir = os.path.join(parent, f"object_mask/{name}")
        return object_mask_dir

    def randAugmenter(self, idx):
        rot_aug_ind = idx % len(self.rot_augmenters)
        rot_aug = self.rot_augmenters[rot_aug_ind]
        aug = iaa.Sequential([rot_aug])
        return aug

    def get_input_ids(self, caption):
        tokenizer_output = self.tokenizer(caption, padding="max_length", truncation=True,return_tensors="pt")
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        return input_ids, attention_mask

    def augment_image(self, image, anomaly_source_img,
                      min_perlin_scale, max_perlin_scale,
                      min_beta_scale, max_beta_scale,
                      object_position, trg_beta):

        # [2] perlin noise
        while True :

            while True :
                # [1] size of noise :big perlin scale means smaller noise
                perlin_scalex = 2 ** (torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0])
                perlin_scaley = 2 ** (torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0])
                perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                threshold = 0.3
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                # smoothing
                perlin_thr = cv2.GaussianBlur(perlin_thr, (3,3), 0)
                # only on object
                total_object_pixel = int(self.resize_shape[0] * self.resize_shape[1])
                if object_position is not None:
                    total_object_pixel = np.sum(object_position)
                    perlin_thr = perlin_thr * object_position
                binary_2D_mask = (np.where(perlin_thr == 0, 0, 1)).astype(np.float32)  # [512,512,3]
                if np.sum(binary_2D_mask) > anomal_p * total_object_pixel :
                    break
            blur_3D_mask = np.expand_dims(perlin_thr, axis=2)  # [512,512,3]

            if trg_beta is None :
                while True :
                    # [1] how transparent the noise
                    beta = torch.rand(1).numpy()[0]
                    if max_beta_scale > beta > min_beta_scale :
                        break
            else:
                beta = trg_beta

            # big beta = transparent
            A = beta * image + (1 - beta) * anomaly_source_img.astype(np.float32) # merged
            augmented_image = (image * (1 - blur_3D_mask) + A * blur_3D_mask).astype(np.float32)
            anomal_img = np.array(Image.fromarray(augmented_image.astype(np.uint8)), np.uint8)
            binary_2d_pil = Image.fromarray((binary_2D_mask * 255).astype(np.uint8)).convert('L').resize((64, 64))
            anomal_mask_torch = torch.where((torch.tensor(np.array(binary_2d_pil)) / 255) > 0.5, 1, 0)
            if anomal_mask_torch.sum() > 0:
                break
        return anomal_img, anomal_mask_torch

    def load_image(self, image_path, trg_h, trg_w, type='RGB'):
        image = Image.open(image_path)
        if type == 'RGB' :
            if not image.mode == "RGB":
                pil_image = image.convert("RGB")
        elif type == 'L':
            if not image.mode == "L":
                pil_image = image.convert("L")
        if trg_h and trg_w:
            pil_image = image.resize((trg_w, trg_h), Image.BICUBIC)
        img = np.array(pil_image, np.uint8)
        return pil_image, img

    def __getitem__(self, idx):

        # [0] augmenter
        aug = self.randAugmenter(idx)

        # [1] base
        img_idx = idx % len(self.image_paths)
        img_path = self.image_paths[img_idx]
        pil_img, img = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1])  # np.array,
        img = aug(image=img)
        name = self.get_img_name(img_path)

        # [2] object mask dir
        if self.use_object_mask :
            object_mask_dir = self.get_object_mask_dir(img_path)
            _, object_img = self.load_image(object_mask_dir, self.latent_res, self.latent_res, type='L')
            object_img = aug(image=object_img)
            object_mask_np = np.where((np.array(object_img, np.uint8) / 255) == 0, 0, 1)  # object = 1
            object_mask = torch.tensor(object_mask_np)  # shape = [64,64], 0 = background, 1 = object
        else :
            object_mask = torch.zeros(self.latent_res, self.latent_res)
        zero_position = torch.zeros(self.latent_res, self.latent_res)

        # [3] anomaly
        anomal_img = img
        anomal_mask_torch = object_mask
        anomal_name = 'none'
        if len(self.anomaly_source_paths) > 0:
            anomal_src_idx = idx % len(self.anomaly_source_paths)
            anomal_dir = self.anomaly_source_paths[anomal_src_idx]
            parent, a_name = os.path.split(anomal_dir)
            a_name = a_name.split('.')[0]
            anomal_class = os.path.split(parent)[1]
            anomal_name = f'{anomal_class}_{a_name}'
            object_position = None
            if self.anomal_only_on_object:
                _, obj = self.load_image(object_mask_dir, self.resize_shape[0], self.resize_shape[1], type='L')
                object_img_aug = aug(image= obj)
                object_position = np.where((np.array(object_img_aug)) == 0, 0, 1)             # [512,512]

            # [4.1] anomal img
            _, anomaly_source_img = self.load_image(self.anomaly_source_paths[anomal_src_idx], self.resize_shape[0], self.resize_shape[1])
            anomal_img, anomal_mask_torch = self.augment_image(img, # 512
                                                               anomaly_source_img, # 512
                                                               argument.anomal_min_perlin_scale, argument.anomal_max_perlin_scale,
                                                               argument.anomal_min_beta_scale, argument.anomal_max_beta_scale,
                                                               object_position=object_position,
                                                               trg_beta = argument.anomal_trg_beta) # [512,512,3], [512,512]

            # [4] background
            back_idx = idx % len(self.background_paths)
            back_path = self.background_paths[back_idx]
            background_img_pil, background_img = self.load_image(back_path, self.resize_shape[0], self.resize_shape[1])
            back_anomal_img, back_anomal_mask_torch = self.augment_image(img,
                                                                         aug(image=background_img),
                                                                         argument.back_min_perlin_scale, argument.back_max_perlin_scale,
                                                                         argument.back_min_beta_scale, argument.back_max_beta_scale,
                                                                         object_position=object_position,
                                                                         trg_beta=argument.back_trg_beta)  # [512,512,3], [512,512]
        else :
            anomal_img = img
            anomal_mask_torch = object_mask.unsqueeze(0)
            back_anomal_img = img
            back_anomal_mask_torch = object_mask.unsqueeze(0)

        # [5] rotate image
        # import random
        # p = random.random()
        rorate_angle = 180
        rotate_img = Image.open(img_path).resize((self.resize_shape[0], self.resize_shape[1])).rotate(
            rorate_angle)  # PIL image
        rotate_np = np.array(rotate_img)  # np.array

        sub_mask_pil = Image.open(object_mask_dir).convert('L').resize((self.latent_res, self.latent_res)).rotate(
            rorate_angle)
        sub_mask_np = np.array(sub_mask_pil)
        final_mask = (np.where((object_mask_np + sub_mask_np) > 0, 255, 0) / 255)
        rotate_mask = torch.tensor(final_mask)  # shape = [64,64], 0 = background, 1 = object
        rotate_mask = torch.where(rotate_mask>0.5, 1, 0).to(dtype=zero_position.dtype)

        # --------------------------------------------------------------------------------------------------------------------- #
        # [6] unmatching anomal
        import random
        partial_anomal_img = img.copy()
        object_pil_img = Image.open(object_mask_dir).convert('L').resize((self.resize_shape[0], self.resize_shape[1]), Image.BICUBIC)
        patch_size = 128
        patch_index_h_list = [i for i in range(int(self.resize_shape[0] / patch_size))][2:]
        patch_index_w_list = [i for i in range(int(self.resize_shape[1] / patch_size))][1:-1]
        object_np = np.array(object_pil_img) / 255
        object_np = np.where(object_np != 0, 1, 0)
        h_idx = random.choice(patch_index_h_list)
        w_idx = random.choice(patch_index_w_list)
        crop_h_start, crop_h_end = patch_size * h_idx, patch_size * (h_idx + 1)
        crop_w_start, crop_w_end = patch_size * w_idx, patch_size * (w_idx + 1)
        partial_anomal_img[crop_h_start:crop_h_end, crop_w_start:crop_w_end, :] = background_img[crop_h_start:crop_h_end,crop_w_start:crop_w_end, :]
        crop_position_np = np.zeros((self.resize_shape[0],self.resize_shape[1]))
        crop_position_np[crop_h_start:crop_h_end, crop_w_start:crop_w_end] = 1
        anomal_pil_np = np.where(((object_np == 1) & (crop_position_np == 1)), 1, 0)
        partial_anomal_mask_pil = Image.fromarray((anomal_pil_np * 255).astype(np.uint8)).convert('L').resize((self.latent_res,self.latent_res),)
        partial_anomal_mask_np = np.where((np.array(partial_anomal_mask_pil, np.uint8) / 255) == 0, 0, 1)  # object = 1
        partial_anomal_mask = torch.tensor(partial_anomal_mask_np)  # shape = [64,64], 0 = background, 1 = object

        # --------------------------------------------------------------------------------------------------------------------- #
        # [7] random rot
        angles = [45,90,135,-45,-90,-135]
        angle = angles[idx % len(angles)]
        # [7.1] src
        rot_pil_img = pil_img.rotate(angle)
        rot_np_img = np.array(rot_pil_img)
        # [7.2] object
        object_pil_img = Image.open(object_mask_dir).convert('L').resize((self.resize_shape[0],self.resize_shape[1]), Image.BICUBIC)
        rot_object_pil_img = object_pil_img.rotate(angle).convert('RGB')
        rot_np_object_img = np.array(rot_object_pil_img)
        rot_np_object_img = np.where(rot_np_object_img == 0, 0, 1)
        # [7.3]
        rot_object_np = rot_np_img * rot_np_object_img
        object_pil = Image.fromarray(rot_object_np.astype(np.uint8))
        # [4] get background
        background_np_img = np.array(background_img_pil) * (1 - rot_np_object_img)
        random_rot_np = (rot_object_np + background_np_img).astype(np.uint8)

        anomal_mask = rot_np_object_img[:, :, 0] + object_pil_img
        random_rot_mask_pil = Image.fromarray(np.where(anomal_mask == 0, 0, 255).astype(np.uint8)).resize((self.latent_res, self.latent_res))
        random_rot_mask_np = np.where((np.array(random_rot_mask_pil, np.uint8) / 255) == 0, 0, 1)
        random_rot_mask = torch.tensor(random_rot_mask_np)

        if self.tokenizer is not None:
            input_ids, attention_mask = self.get_input_ids(self.caption)  # input_ids = [77]
        else:
            input_ids = torch.tensor([0])

        return {'image': self.transform(img),  # original image
                "zero_mask": zero_position.unsqueeze(0),  # [1, 64, 64]

                'anomal_image': self.transform(anomal_img),
                "anomal_mask": anomal_mask_torch,

                #'bg_anomal_image': self.transform(back_anomal_img),  # masked image
                #'bg_anomal_mask': back_anomal_mask_torch,

                'bg_anomal_image': self.transform(partial_anomal_img),
                'bg_anomal_mask': partial_anomal_mask.unsqueeze(0) ,

                'rotate_image': self.transform(rotate_np),
                'rotate_mask': rotate_mask.unsqueeze(0),

                'random_rot_image' : self.transform(random_rot_np),
                'random_rot_mask' : random_rot_mask.unsqueeze(0) ,

                'empty_image': self.transform(background_img),
                'empty_mask': object_mask.unsqueeze(0),

                'idx': idx,
                'input_ids': input_ids.squeeze(0),
                'caption': self.caption,
                'image_name': name,
                'anomal_name': anomal_name,
                }

