from __future__ import print_function

import os
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable

from models import SSD300,SSDBoxCoder
from datasets import ListDataset

from utils.loss import SSDLoss
from utils.transforms import resize, random_flip, random_paste, random_crop, random_distort

parser = argparse.ArgumentParser(description='PyTorch Textboxes Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
#parser.add_argument('--lr_decay', default = 1e-1, type = float, help = 'learning rate decay after 40k iter')
parser.add_argument('--resume', '-r', default = False, help='resume from checkpoint')
parser.add_argument('--model', default='./checkpoint/vgg16_reducedfc.pth', type=str, help='initialized vgg16 model path')
parser.add_argument('--checkpoint', default='./checkpoint/synthtext_detection.pth', type=str, help='checkpoint path')
parser.add_argument('--train_model', default='./checkpoint/train_synthtext_detection.pth', type=str, help='train checkpoint path')
args = parser.parse_args()

# Model
print('==> Building model..')
# net = SSD512(num_classes=21)
net = SSD300(num_classes=2)

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

if args.resume:
    print('==> Resuming from train_model..')
    '''
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch'] + 1
    '''
    train_model = torch.load(args.train_model)
    net.load_state_dict(train_model['net'])
    best_loss = train_model['loss']
    start_epoch = train_model['epoch']

else:
    # initialize the parameters
    net.apply(weights_init)
    # load vgg16 parameters
    model_dict = net.state_dict()
    pretrained_dict = torch.load(args.model)
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    
    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last epoch

# Dataset
print('==> Preparing dataset..')
box_coder = SSDBoxCoder(net)
img_size = 300
def transform_train(img, boxes, labels):
    img = random_distort(img)
    if random.random() < 0.5:
        img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123,116,103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=(img_size,img_size), random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

trainset = ListDataset(root='/usr/local/share/data/SynthText/',
                       list_file= 'data_train.txt',
                       transform=transform_train)

def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size,img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

testset = ListDataset(root='/usr/local/share/data/SynthText/',
                      list_file='data_val.txt',
                      transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

net.cuda()
#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True

# loss function and optimizer
criterion = SSDLoss(num_classes=2)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], train_loss/(batch_idx+1), batch_idx+1, len(trainloader)))

        if (batch_idx + 1) % 100 == 0:
            print('Saving..')
            batch_loss = train_loss/(batch_idx+1)
            state = {
                'net': net.state_dict(),
                'loss': batch_loss,
                'epoch': epoch,
            }
            if not os.path.isdir(os.path.dirname(args.train_model)):
                os.mkdir(os.path.dirname(args.train_model))
            torch.save(state, args.train_model)

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], test_loss/(batch_idx+1), batch_idx+1, len(testloader)))


    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.dirname(args.checkpoint)):
            os.mkdir(os.path.dirname(args.checkpoint))
        torch.save(state, args.checkpoint)
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
