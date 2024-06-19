import torch
from PIL import Image
from torchvision.transforms import ToTensor
from cnn_model import FruitCNNmodel
from image_pre import preprocess_image
import time

transform = ToTensor()
model = FruitCNNmodel()
model.load_state_dict(torch.load(r'D:\Desktop\fruit\Fruit_model.pth')) #模型路径
model.eval()
#device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


# 图片处理识别
def image_fruit_predict(image_path):
    start_time = time.time()  
    if image_path is not None:
        preprocess_image_fruit = preprocess_image(image_path)
        print(preprocess_image_fruit)
        if preprocess_image_fruit is not None:
            img_tensor = transform(Image.fromarray(preprocess_image_fruit)).unsqueeze(0)
            predicted_class = predict(img_tensor)
        else:
            predicted_class = torch.tensor([-1])
    end_time = time.time()
    print(f'水果识别时间: {(end_time - start_time) * 1000:.2f} ms')
    return predicted_class


# 预测函数
def predict(img_tensor):
    
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_class = torch.max(output, 1)
   # print(f'Predicted Fruit Class: {predicted_class.item()}')
    return predicted_class

 #类别匹配
def map_fruit_class(predicted_class):
    Fruit_mapping = {
        -1:'None',
        0: 'apple',
        1: 'banana',
        2: 'cherry',
        3: 'orange',
        4: 'pear',
        5: 'strawberry'
    }
    mapped_fruit_class = Fruit_mapping.get(predicted_class.item())

    return mapped_fruit_class


if __name__ == "__main__":
    
    # 图片
    image_path = r"D:\Desktop\fruit\dataset\train\apple\15.jpg"
    predicted_class = image_fruit_predict(image_path)
    print(predicted_class)
    mapped_fruit_class = map_fruit_class(predicted_class)
    
    print(f'Predicted Fruit Class: {mapped_fruit_class}')

   
