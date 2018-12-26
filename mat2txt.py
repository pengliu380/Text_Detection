import scipy.io as sio 
import matplotlib.pyplot as plt 
import numpy as np
#mat文件名
matfile = '/usr/local/share/data/SynthText/gt.mat'
data = sio.loadmat(matfile)

#print(len(data['imnames'][0]))
#print(data['imnames'][0][0][0])
#print(data['wordBB'][0][0].shape)
#print(data['wordBB'][0][92][0][0])
#print(type(data['wordBB'][0][0][0][0][0]))


f = open('data.txt', 'a')
print('start write')
for i in range(len(data['imnames'][0])):
    s = str(data['imnames'][0][i][0])
    f.write(s + ' ')
    if isinstance(data['wordBB'][0][i][0][0], np.ndarray):
        for j in range(len(data['wordBB'][0][i][0][0])):
            f.write(str(data['wordBB'][0][i][0][0][j]) + ' ')
            f.write(str(data['wordBB'][0][i][1][0][j]) + ' ')
            f.write(str(data['wordBB'][0][i][0][2][j]) + ' ')
            f.write(str(data['wordBB'][0][i][1][2][j]) + ' ')
    else:
        f.write(str(data['wordBB'][0][i][0][0]) + ' ')
        f.write(str(data['wordBB'][0][i][1][0]) + ' ')
        f.write(str(data['wordBB'][0][i][0][2]) + ' ')
        f.write(str(data['wordBB'][0][i][1][2]) + ' ')
    f.write('\n')
f.close()
print('finished')
