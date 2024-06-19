import os
import random
import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.resize(img ,(128,128))
    return img


def save_image(img, output_path):
    cv2.imwrite(output_path, img)
    
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(input_folder, output_folder, num_images=3000):
    all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    random.shuffle(all_files)
    selected_files = all_files[:num_images]

    train_split = int(num_images * 0.7)
    val_split = int(num_images * 0.15)

    train_files = selected_files[:train_split]
    val_files = selected_files[train_split:train_split + val_split]
    test_files = selected_files[train_split + val_split:]

    train_dir = os.path.join(output_folder, 'train', 'orange')
    val_dir = os.path.join(output_folder, 'val', 'orange')
    test_dir = os.path.join(output_folder, 'test', 'orange')

    create_directory(train_dir)
    create_directory(val_dir)
    create_directory(test_dir)
    
   
    for file in train_files:
        processed_img = preprocess_image(file)
        save_image(processed_img, os.path.join(train_dir, os.path.basename(file)))

    for file in val_files:
        processed_img = preprocess_image(file)
        save_image(processed_img, os.path.join(train_dir, os.path.basename(file)))

    for file in test_files:
        processed_img = preprocess_image(file)
        save_image(processed_img, os.path.join(train_dir, os.path.basename(file)))

if __name__ == "__main__":
    input_folder = r"D:\Desktop\Fruit-262\orange"
    output_folder = r"D:\Desktop\fruit\dataset"
    main(input_folder, output_folder)
