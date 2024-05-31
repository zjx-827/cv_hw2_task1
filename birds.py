path ="/root/autodl-tmp/CUB_200_2011/"
ROOT_TRAIN = path + 'images/train/'
ROOT_TEST = path + 'images/test/'
BATCH_SIZE = 16



import os
import time
import copy
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings("ignore", category=UserWarning)

# 定义数据转换
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # 随机裁剪并调整大小
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(), # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
    ]),
    'test': transforms.Compose([
        transforms.Resize(256), # 调整大小
        transforms.CenterCrop(224), # 中心裁剪
        transforms.ToTensor(), # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
    ]),
}

# 设置数据目录
data_dir = "/root/autodl-tmp/CUB_200_2011/dataset/"

# 加载数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}

# 创建数据加载器
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
               for x in ['train', 'test']}

# 获取数据集大小和类别名称
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # writer = SummaryWriter(log_dir='runs')
    writer = SummaryWriter(log_dir='runs')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 在TensorBoard中记录损失和准确率
            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:
                writer.add_scalar('Loss/test', epoch_loss, epoch)
                writer.add_scalar('Accuracy/test', epoch_acc, epoch)

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    torch.save(best_model_wts, 'best_model_no_pretrain.pth')
    model.load_state_dict(best_model_wts)
    writer.close()
    return model


def visualize_model(model, dataloaders, device, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                # 将图像转换为NumPy数组，并缩放到[0, 1]范围，然后转换为(H, W, C)格式
                image_np = torchvision.utils.make_grid(inputs.cpu().data[j]).numpy().transpose((1, 2, 0))
                # 缩放图像数据到[0, 1]范围
                image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
                plt.imshow(image_np)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ 
        Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    model_ft, input_size = initialize_model("resnet", 200, feature_extract=True, use_pretrained=False)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # 观察只有最后一层参数被优化
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)  # 使用Adam优化器


    # # 每7个epoch衰减学习率
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.5)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)
    visualize_model(model_ft, dataloaders, device, class_names)
    # plt.ioff()
    # plt.show()
