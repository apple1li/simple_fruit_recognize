from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from get_data import FruitDataset, custom_collate_fn
from cnn_model import FruitCNNmodel
import time
from tqdm import tqdm

val_losses = []
val_accuracies = []
train_losses = []

# 验证集验证
def evaluate_model(val_dataloader, model, loss_fn, device):
    model.eval()    
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
                
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('验证正确：',correct,'总数：',total)
    
    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)
        
    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)
    return accuracy,val_loss

# 模型训练
def training(train_dataloader, model, loss_fn, optimizer, device, epochs=50):
    for epoch in range(1, epochs + 1):
        start_time = time.time() 
        model.train()
        train_loss = 0.0
        with tqdm(total=len(train_dataloader), unit_scale=True, desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for batch, data in enumerate(train_dataloader):
                images, labels = data['image'].to(device), data['label'].to(device)
            
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
            
                train_loss += loss.item()
                
                pbar.set_postfix({'Train Loss': train_loss / (batch + 1),
                                  'Train Time': time.time() - start_time})
                pbar.update()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        
    # 每个epoch结束后进行验证
        val_accuracy ,val_loss = evaluate_model(val_dataloader, model, loss_fn, device)
        end_time = time.time()
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f},Val Loss: {val_loss:.6f}, Validation Accuracy: {val_accuracy:.2f}%, Time: {end_time - start_time:.2f}s")
    
    # # 保存模型
    torch.save(model.state_dict(), 'Fruit_model.pth')
       

if __name__ == '__main__':
    
    model = FruitCNNmodel()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    model.to(device)

    train_data_folder = r"D:\Desktop\fruit\dataset\train"
    val_data_folder = r"D:\Desktop\fruit\dataset\val"
    
    train_dataset = FruitDataset(data_folder=train_data_folder)
    val_dataset = FruitDataset(data_folder=val_data_folder)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    training(train_dataloader, model, loss_fn, optimizer, device)

