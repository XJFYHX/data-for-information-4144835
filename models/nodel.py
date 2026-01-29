#导入库
import numpy as np
from sklearn import metrics
import xlrd
import random
import matplotlib.pyplot as plt
#from attention import MySelfAttention,MyMultiHeadAttention
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader,Dataset
# 载入数据
# np.random.seed(1337)
table = xlrd.open_workbook('fcao.xlsx').sheets()[0]  # 第一个数据表
#定义数据转换函数
def getData(table):
    row = table.nrows
    col = table.ncols
    datamatrix = np.zeros((row,col))
    for x in range(row):
        rows = np.matrix(table.row_values(x))
        datamatrix[x, :] = rows  # 按行存入矩阵中
    return datamatrix

# train_x= datamatrix[0:1000, 0:600]  # 200*480
# train_y= datamatrix1[0:1000, 600]  # 200*1
# test_x = datamatrix[1000:1200, 0:600] # (822-200)*100
# test_y  = datamatrix1[1000:1200, 600]  #




timestep_size = 60
feature_size = 7

def getdata(s):
    table = xlrd.open_workbook(s).sheets()[0]  # 第一个数据表

    # 定义数据转换函数

    datamatrix = getData(table)

    # slect = random.sample(range(1,1440),120)

    slect = [488, 735, 877, 1191, 841, 634, 679, 1313, 931, 509, 1293, 418, 1054, 582,
             27, 1229, 792, 433, 747, 982, 338, 860, 989, 453, 951, 1239, 721, 1068, 778,
             246, 1398, 1085, 1154, 1115, 224, 1412, 425, 119, 1276, 1257, 2, 609, 529,
             1204, 1095, 563, 1033, 758, 362, 782, 574, 687, 940, 987, 1010, 306, 864,
             266, 52, 724, 737, 670, 601, 1001, 309, 280, 1029, 87, 1030, 95, 716, 205,
             37, 749, 827, 655, 1190, 83, 585, 40, 31, 1019, 1112, 725, 532, 109, 566,
             1134, 207, 1118, 1258, 1050, 1016, 908, 185, 612, 611, 517, 763, 214, 1277,
             475, 1069, 1148, 286, 983, 638, 91, 685, 640, 512, 1251, 391, 49, 1322, 904,
             1073, 531, 229, 732, 581, 961, 454, 1407, 417, 1250, 1037, 252, 1212, 793,
             72, 801, 64, 486, 191, 127, 802, 316, 1274, 1023, 298, 849, 592, 530, 1367,
             1060, 881, 1374, 756, 446, 258, 1292, 1317, 890, 1234, 210, 397, 1278, 1334,
             755, 681, 707, 97, 490, 567, 1242, 923, 1377, 811, 1279, 542, 1331, 180, 1221,
             68, 1387, 1031, 924, 203, 672, 195, 1416, 752, 104, 511, 1108, 1196, 1020, 817,
             136, 77, 626, 101, 851, 106, 594, 1420, 678, 622, 187]

    '''slect = random.sample(range(1,2900),500)'''

    min_max_scaler = preprocessing.MinMaxScaler()
    datamatrix1 = min_max_scaler.fit_transform(datamatrix)
    data1 = datamatrix[0:2900, 0:420]
    # label = datamatrix1[0:1440, 600]
    label = datamatrix[0:2900, 600]
    # 划分训练集和测试集，测试集共120组
    test_x = data1[slect]  # 200*480
    test_y = label[slect]  # 200*1
    train_x = np.delete(data1, slect, 0)  # (822-200)*100
    train_y = np.delete(label, slect, 0)  #
    train_X = train_x.reshape((train_x.shape[0], timestep_size, feature_size))
    test_X = test_x.reshape((test_x.shape[0], timestep_size, feature_size))
    return train_X,train_y,test_X,test_y

train_X,train_y,test_X,test_y = getdata("fcao.xlsx")
train_X2,_,test_X2,_ = getdata("A.xlsx")
# 定义残差块网络TC
# 定义残差块网络TCN

#定义注意力机制
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.tanh = nn.Tanh()
        self.liner = nn.Linear(64,64)
        self.liner1 = nn.Linear(64, 64)
        self.liner2 = nn.Linear(64, 64)
        self.softmax = nn.Softmax(dim=1)
        self.Norm = nn.LayerNorm(64)
        # nn.init.uniform(self.liner.weight)
        # nn.init.uniform(self.liner.bias)
        # nn.init.uniform(self.liner1.weight)
        # nn.init.uniform(self.liner1.bias)
        # nn.init.uniform(self.liner2.weight)
        # nn.init.uniform(self.liner2.bias)
    def forward(self,x):
        x = x.transpose(2, 1)

        q = self.liner(x)
        k = self.liner1(x)
        v = self.liner2(x)
        a = self.softmax(q@k.transpose(2,1))
        o = self.Norm(a @ v + x)
        return o.transpose(2, 1)

class Models(nn.Module):
    def __init__(self):
        super(Models, self).__init__()
        self.conv1 = nn.Conv1d(feature_size, 64, kernel_size=15,padding=7).double()
        self.attn1 = Attention().double()
        self.conv2 = nn.Conv1d(64, 64, kernel_size=11, padding=10,dilation=2).double()
        self.attn2 = Attention().double()
        self.conv3 = nn.Conv1d(64, 64, kernel_size=7, padding=12,dilation=4).double()
        self.attn3 = Attention().double()
        dim = 60*(67+7)
        self.Norm = nn.LayerNorm(dim)
        self.conv = nn.Linear(dim,64).double()
        self.conv4 = nn.Linear(64, 1).double()
    def forward(self,x, x2):
        x = x.transpose(2, 1)
        d2 = x2.transpose(2, 1)
        c = x.shape[-1]
        x1 = self.conv1(x)

        x2 = self.attn1(x1)
        x3 = self.conv2(F.relu(x2))
        x4 = self.attn2(x3)
        x5 = self.conv3(F.relu(x4))
        x6 = F.relu(self.attn3(x5))

        m = x.mean(dim=1).view(-1,1,c)
        s = x.std(dim=1).view(-1,1,c)
        a = (x - x.mean(dim=1).view(-1,1,c)).abs().mean(dim=1).view(-1,1,c)
        x6 = torch.cat([x6,m,s,a, d2],dim=1)
        x6 = x6.transpose(2, 1)
        x6 = x6.reshape(-1, 1, x6.shape[-1] * x6.shape[-2])
        return self.conv4(self.conv(self.Norm(x6))).view(-1,1)

class dataset():
    def __init__(self,data,data2,label):
        self.train = data
        self.train2 = data2
        self.label = label

    def __getitem__(self, index):
        train = self.train[index,...]
        train2 = self.train2[index, ...]
        label = self.label[index,...]
        return {"train":train, "label":label,"train2":train2}

    def __len__(self):
        return self.train.shape[0]


mseloss = torch.nn.MSELoss()
traindata = DataLoader(dataset(train_X,train_X2,train_y), batch_size=128)
testdata = DataLoader(dataset(test_X,test_X2, test_y), batch_size=1)

model = Models().cuda()
# 定义模型结构
adam = torch.optim.AdamW(model.parameters(),lr=0.005)

for index in  range(1):
    print('step:{}'.format(index))
    for data in traindata:
        train = data["train"].cuda()
        train2 = data["train2"].cuda()
        label = data["label"].cuda()
        out = model(train,train2)
        loss = mseloss(out,label.view(-1,1))
        adam.zero_grad()
        loss.backward()
        print(loss.item())

ls = []
output = []
for data in testdata:
    train = data["train"].cuda()
    train2 = data["train2"].cuda()
    label = data["label"].cuda().view(-1,1)
    out = model(train,train2)
    output.append(out)
    loss = mseloss(out,label)
    ls.append(loss.cpu().data.numpy())
output = torch.cat(output,dim=0).view(-1,1).cpu().data.numpy()
# 编译模型
# 训练模型
#test_prediction = test_prediction* (max11 - min11) + min11#+np.random.uniform(-0.1,0.1,(200,1))
#test_y =test_y* (max11 - min11) + min11
#打印预测结果
#print(test_prediction)
#绘制训练损失图像
plt.figure(1)
plt.plot(ls)
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend('train', loc='upper left')
plt.show()
test_mse = metrics.mean_squared_error(test_y, output) # 均方根误差
test_rmse = np.sqrt(test_mse)  # 均方根误差
test_mae = metrics.mean_absolute_error(test_y, output) #np.mean(np.abs(test_prediction - test_y))  # 平均绝对误差
#test_mre = np.mean(np.divide(np.abs(test_prediction - test_y), test_prediction))  # 平均相对误差
r_2 = r2_score(test_y,output)
#print(" test_mse %g,test_rmse %g, test_mae %g,test_mre %g,r2 %g" % (test_mse,test_rmse, test_mae, test_mre,r_2))
print(" test_rmse %g, test_mae %g, r2 %g" % (test_rmse, test_mae ,r_2))

plt.figure(2)
dim2 = len(test_y)
x_n1 = range(0, dim2)
#* (max11 - min11) + min11
plt.plot(x_n1, test_y , label='real', color='b',marker='*', linewidth=1)
plt.plot(x_n1, output , label='predicted', color='r',marker='+',linewidth=1)
plt.xlabel('Samples')
plt.ylabel('f_Cao')
# plt.ylim(min11 - 20, max11 + 20)
plt.legend()
plt.show()
