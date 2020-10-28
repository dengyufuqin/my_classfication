import torch
from efficientnet_pytorch.model import EfficientNet
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2
from PIL import Image


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = EfficientNet.from_name('efficientnet-b0')

num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, 4)

net_weight = 'weights/ljfl.pth'
state_dict = torch.load(net_weight)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



def detect_mantong(image):
    image = Image.fromarray(image)
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        print(model(image))
        outputs = model(image)
        _, predict = torch.max(outputs.data, 1)
    predict = predict.cpu().numpy()[0]
    if predict == 1:
        predict = 0
    elif predict == 0:
        predict = 1
    elif predict == 2:
        predict = 3
    else:
        predict = 2

    return predict

if __name__ == '__main__':
    image = cv2.imread('data/full/full0.jpg')

    print(detect_mantong(image))
