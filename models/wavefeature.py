#导入库

import numpy as np
import xlrd
import random
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
import pywt
import pywt.data
import pandas as pd
# 载入数据
# np.random.seed(1337)
table = xlrd.open_workbook('./fcao.xlsx').sheets()[0]  # 第一个数据表
#定义数据转换函数
def getData(table):
    row = table.nrows
    col = table.ncols
    datamatrix = np.zeros((row,col))
    for x in range(row):
        rows = np.matrix(table.row_values(x))
        datamatrix[x, :] = rows  # 按行存入矩阵中
    return datamatrix

def single_wal(x,n):
    wp = pywt.WaveletPacket(data=x, wavelet='db3',maxlevel=n)
    re = []  #第n层所有节点的分解系数
    # 这个i是节点啊，aaa  aad 之类 , 自然顺序的频率带
    # 'aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd'
    for i in [node.path for node in wp.get_level(n, 'natural')]:
        re.append(wp[i].data)
    #第n层能量特征
    energy = []
    for i in re:
        energy.append(pow(np.linalg.norm(i,ord=None),2))
    return energy/sum(energy),re
def get_Wave(data):
    MatrixOfWave = []
    for i in range(8):
        tmp = sig_wal(data[i],3)
        MatrixOfWave.append(tmp)
    return np.array(MatrixOfWave).T
datamatrix = getData(table)
processdata=datamatrix[:,0:600]
processdata = processdata.reshape(processdata.shape[0],60,10)
re = []  # 第n层所有节点的分解系数
for i in range(processdata.shape[0]):
    for j in range(10):
        data1 = processdata[i,...,j]
        wp = pywt.WaveletPacket(data=data1, wavelet='db3', maxlevel=3)
        # 这个i是节点啊，aaa  aad 之类 , 自然顺序的频率带
        # 'aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd'
        for k in ['aaa', 'aad','daa', 'dad']:
            re.append(wp[k].data)

re = np.array(re)
re =re.reshape(2916,440)

zero =np.zeros((2916,600))
zero[:,0:440]=re[:,:]
data = pd.DataFrame(zero)

writer = pd.ExcelWriter('A.xlsx')		# 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()

writer.close()

