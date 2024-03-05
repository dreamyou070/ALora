import os

test_folder = 'turft/raw/cropped_test'
os.makedirs(test_folder, exist_ok = True)

train_folder = 'turft/raw/cropped_train'
folders = os.listdir(train_folder)

for folder in folders :
    folder_dir = os.path.join(train_folder, folder)
    test_folder_dir = os.path.join(test_folder, folder)
    os.makedirs(test_folder_dir, exist_ok = True)

    rgb_folder = os.path.join(folder_dir, 'rgb')
    gt_folder = os.path.join(folder_dir, 'gt')
    test_rgb_folder = os.path.join(test_folder_dir, 'rgb')
    test_gt_folder = os.path.join(test_folder_dir, 'gt')
    os.makedirs(test_rgb_folder, exist_ok = True)
    os.makedirs(test_gt_folder, exist_ok=True)

    imgs = os.listdir(rgb_folder)
    total_num = len(imgs)
    test_num = int(total_num * 0.3)

    for i in range(total_num) :
        if i < test_num :
            org_rgb_dir = os.path.join(rgb_folder, imgs[i] )
            org_gt_dir = os.path.join(gt_folder, imgs[i] )
            new_rgb_dir = os.path.join(test_rgb_folder, imgs[i] )
            new_gt_dir = os.path.join(test_gt_folder, imgs[i])

            os.rename(org_rgb_dir, new_rgb_dir)
            os.rename(org_gt_dir, new_gt_dir)
