
import collections
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from image_pre import preprocess_image
import logging


#按文件夹对应标签批次取出数据

logging.basicConfig(level=logging.INFO)

class FruitDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.classes = sorted(os.listdir(data_folder))
        
        logging.info(f"Classes: {self.classes}")
        self.label_to_class = {label: idx for idx, label in enumerate(self.classes)}
        
        logging.info(f"Label to Class mapping: {self.label_to_class}")
        
        self.images = []
        self.labels = []
       
        transform = ToTensor()
        self.transform = transform
        
        label_counter = collections.Counter()
        
        for label in self.classes:
            label_folder = os.path.join(data_folder, label)
            if os.path.isdir(label_folder):
                for image_name in os.listdir(label_folder):
                    image_path = os.path.join(label_folder, image_name)
                    image_path = image_path.replace("\\", "/")
                    self.images.append(image_path)
                    self.labels.append(self.label_to_class[label])
                    label_counter[label] += 1
        
        for label, count in label_counter.items():
            print(f"Class '{label}' count: {count}")
                    
    def __len__(self):
        return len(self.images)
    
    
    def __getitem__(self, idx):
        try:
            image_path = self.images[idx]
            preprocess_fruit_image = preprocess_image(image_path)
            if preprocess_fruit_image is not None:
                if self.transform:
                    image = self.transform(preprocess_fruit_image)

                label = int(self.labels[idx])
                sample = {'image': image, 'label': label}  
                #print(f"Image Path: {image_path}, Label: {label}")             
                return sample
            else :
                return None
        
        except Exception as e:
            print('异常原因{}'.format(e))
            return None
        
def custom_collate_fn(batch):
    batch = [sample for sample in batch if sample is not None]

    return torch.utils.data.dataloader.default_collate(batch)




if __name__ == "__main__":
    
   
    train_data_folder = r"D:\Desktop\fruit\dataset\train"
    train_dataset = FruitDataset(data_folder=train_data_folder)
    train_dataloader = DataLoader(train_dataset, batch_size = 32 ,shuffle=True, collate_fn=custom_collate_fn)

    for batch in train_dataloader:
        print(batch)