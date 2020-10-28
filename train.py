import torch
from efficientnet_pytorch.model import EfficientNet
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from utils.datasets import CustomDataset
from torch.utils.data import DataLoader
import numpy as np


def accuracy(output, target):
    output = np.array(output.cpu().numpy())
    target = np.array(target.cpu().numpy())
    prec = 0
    for i in range(output.shape[0]):
        pos = np.unravel_index(np.argmax(output[i]), output.shape)
        pre_label = pos[1]
        if pre_label == target[i]:
            prec += 1

    prec /= target.size
    prec *= 100
    return prec


device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
EPOCHS = 200

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_data = CustomDataset(root='data/',
                           datatxt='train.txt',
                           transform=transform)

train_loader = DataLoader(dataset = train_data, batch_size=16, shuffle=True)



pth_map = {
    'efficientnet-b0': 'efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'efficientnet-b7-dcc49843.pth',
}
model = EfficientNet.from_pretrained('efficientnet-b0')

# 离线加载模型
# model = EfficientNet.from_name('efficientnet-b0')
# net_weight = 'eff_weights/'+pth_map['efficientnet-b0']
# state_dict = torch.load(net_weight)
# model_ft.load_state_dict(state_dict)




for param in model.parameters():
    param.requires_grad = True

num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, 4)

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(EPOCHS):

    for i, (images, labels) in enumerate(train_loader, 0):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        prec = accuracy(outputs.data, labels)

        if i + 1 == len(train_loader):
            print('after {} epoch, {}th batchsize, prec: {}%, loss: {}, input: {}, output: {}'.format(epoch+1, i+1, prec,
                                                                                           loss, images.size(), outputs.size()))



PATH = 'ljfl.pth'
torch.save(model.state_dict(), PATH)


