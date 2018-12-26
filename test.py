import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable
from utils.transforms import resize
from datasets import ListDataset
from models import SSD300, SSDBoxCoder

from PIL import Image, ImageDraw
import os
import sys


print('Loading model..')
net = SSD300(num_classes=2)
#print(net)
model = torch.load('./checkpoint/train_synthtext_detection_2.pth')
net.load_state_dict(model['net'])
net.eval()

print('Loading image..')
img = Image.open('/usr/local/share/data/SynthText/160/stage_40_64.jpg')
ow = oh = 300
img = img.resize((ow,oh))

print('Predicting..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
x = transform(img)
x = Variable(x, volatile=True)
loc_preds, cls_preds = net(x.unsqueeze(0))

print('Decoding..')
box_coder = SSDBoxCoder(net)
boxes, labels, scores = box_coder.decode(
    loc_preds.data.squeeze(), F.softmax(cls_preds.squeeze(), dim=1).data,
    score_thresh=0.5)
print(labels)
print(scores)

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='green')
img.save('./result/stage_40_64.jpg')




