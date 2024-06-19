from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from get_data import FruitDataset ,custom_collate_fn 
from cnn_model import FruitCNNmodel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns


def test_model(model, test_dataloader, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            print(predicted)
            
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
    accuracy = correct_predictions / total_samples
    print(f'Test Accuracy: {accuracy * 100:.6f}%')
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_predicted)
    print(len(all_labels))
    print(conf_matrix)
    # 创建热力图可视化混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['apple', 'banana', 'cherry', 'orange', 'pear', 'strawberry'], yticklabels=['apple', 'banana', 'cherry', 'orange', 'pear', 'strawberry'])
    plt.ylabel('Predicted labels')
    plt.xlabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(r'D:\Desktop\fruit\result\confusion_matrix.png')
    plt.show()
    
    TP = np.sum(np.diag(conf_matrix)) 

    with open("./confusion_matrix_values.txt", "a") as f:
        f.write("True Positives: {}\n".format(TP))
        f.write("conf_matrix: {}\n".format(conf_matrix))

    precision = precision_score(all_labels, all_predicted, average='weighted')
    recall = recall_score(all_labels, all_predicted, average='weighted')
    f1 = f1_score(all_labels, all_predicted, average='weighted')
    
    print('函数','精确率:',precision,'召回率：',recall,'f1:',f1)
    

if __name__ == '__main__':
   
    model = FruitCNNmodel()
    model.load_state_dict(torch.load(r'D:\Desktop\fruit\Fruit_model.pth'))#模型路径
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_data_folder = r"D:\Desktop\fruit\dataset\test"
    test_dataset = FruitDataset(data_folder=test_data_folder)
    print('开始取出数据')
    test_dataloader = DataLoader(test_dataset, batch_size=10, collate_fn=custom_collate_fn)
    
    # 测试模型
    test_model(model, test_dataloader, device)
