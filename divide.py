import numpy as np
import os
import sys
import random

with open('data.txt') as f:
    lines = f.readlines()
    num_imgs = len(lines)
random.shuffle(lines)
print(num_imgs)

train_data = lines[:int(num_imgs*0.98)]
val_data = lines[int(num_imgs*0.98):int(num_imgs*0.99)]
test_data = lines[int(num_imgs*0.99):]
#print(len(train_data))
#print(len(val_data))
#print(test_data[0])
f1 = open('data_train.txt', 'w')
f2 = open('data_val.txt', 'w')
f3 = open('data_test.txt', 'w')

print('train data')
for i in range(len(train_data)):
    f1.write(train_data[i])
f1.close()

print('val data')
for i in range(len(val_data)):
    f2.write(val_data[i])
f2.close()

print('test data')
for i in range(len(test_data)):
    f3.write(test_data[i])
f3.close()

print('finished.')