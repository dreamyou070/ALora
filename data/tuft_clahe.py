import cv2, os
def equalize_clahe_image(image_path):
    img = cv2.imread(image_path, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    cl = clahe.apply(img)
    #cv2.imwrite(image_path, cl)
    return cl

raw_folder = 'turft/raw/Radiographs'
equalize_clahe_image_folder = 'turft/raw/equalize_clahe'
os.makedirs(equalize_clahe_image_folder, exist_ok = True)
imgs = os.listdir(raw_folder)
for img in imgs :
    img_path = os.path.join(raw_folder, img)
    changed_img = equalize_clahe_image(img_path)
    cv2.imwrite(os.path.join(equalize_clahe_image_folder, img), changed_img)