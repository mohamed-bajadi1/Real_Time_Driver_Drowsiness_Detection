import os
import cv2
from sklearn.model_selection import train_test_split


# define directories
base_dir = 'DataSet/eyes/CEW'
base_out_dir= 'data/eyes'
categories = ['Close', 'Open']
splits = ['train', 'val', 'test']

# -- image size --
IMG_SIZE = (64, 64)

# create directories for splits
for split in splits:
    for category in categories:
        os.makedirs(os.path.join(base_out_dir, split, category), exist_ok=True)

# create function to process , resize and split images
def preprocess_images(base_dir, category):
    image_paths = os.listdir(os.path.join(base_dir, category))
    images = []
    for image_name in image_paths:
        img_path = os.path.join(base_dir , category, image_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        images.append((img, image_name))

    train_val , test = train_test_split(images, test_size=0.2, random_state=42)
    train , val = train_test_split(train_val, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

    for split, split_name in zip([train, val, test], splits):
        for img, img_name in split:
            cv2.imwrite(os.path.join(base_out_dir, split_name, category, img_name), img)

    print(f"Processed {len(images)} images for category: {category}")


for category in categories:
    preprocess_images(base_dir, category)

print("="*30 +" phase 1 done" + "="*30)


base_dir = 'DataSet/eyes/MRL'
# define a function to resize mrl data and add it to the preprocessed CEW data
def preprocess_mrl(base_dir, category, split):
    input_dir = os.path.join(base_dir, split, category)
    output_dir = os.path.join(base_out_dir, split, category)
    os.makedirs(output_dir, exist_ok=True)
    images = os.listdir(input_dir)
    for image_name in images:
        image_path = os.path.join(input_dir, image_name)
        # Skip hidden files or non-image files
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        img = cv2.imread(image_path)

        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)

        save_path = os.path.join(output_dir, image_name)
        cv2.imwrite(save_path, img)

        print(f"{split}/{category} ajouté ✔")

    print("\n🎉 MRL ajouté au dataset CEW avec succès !")


for split in splits:
    for category in categories:
        preprocess_mrl(base_dir , category , split)

print("="*30 +" phase 2 done" + "="*30)

