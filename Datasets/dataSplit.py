#! /usr/bin/python3

import os, os.path, shutil
import numpy as np


#short part about splitting the data

dataTotalDirectory = 'D:/Justyna/all/AGH/FIB/AHLT/lab/data/Train/TotalTrain/'

#xmls = listdir(dataTotalDirectory)

allFileNames = os.listdir(dataTotalDirectory)
np.random.shuffle(allFileNames)
print(allFileNames)


train_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*0.8)])
print(len(train_FileNames))
print(len(test_FileNames))


for name in train_FileNames:
    shutil.copy(dataTotalDirectory+name, 'D:/Justyna/all/AGH/FIB/AHLT/lab/data/Train/Train80')

for name in test_FileNames:
    shutil.copy(dataTotalDirectory+name, 'D:/Justyna/all/AGH/FIB/AHLT/lab/data/Train/Test20')



