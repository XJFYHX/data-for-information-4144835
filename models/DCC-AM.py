#导入库
import keras.backend
import numpy as np
from sklearn import metrics
import xlrd
import random
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
#from attention import MySelfAttention,MyMultiHeadAttention
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.layers import *
from keras.models import *
from keras.losses import *
from keras.optimizers import Adam ,SGD,RMSprop
from keras.callbacks import EarlyStopping
from keras.layers.merge import concatenate
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
datamatrix = getData(table)
max11, min11 = max(datamatrix[:, 600]), min(datamatrix[:, 600])


#slect = random.sample(range(1,1440),120)

'''
slect = [3,9,24,26,28,53,56,58,78,82,84,85,86,91,104,123,131,134,137,140,141,149,
         164,175,194,198,216,224,229,232,243,246,247,253,261,263,264,268,269,
         293,295,304,305,311,319,322,336,345,351,352,360,371,373,376,388,
         401,402,403,418,428,432,444,463,465,466,469,470,472,475,479,481,484,485,486,
         511,512,524,537,541,544,546,559,568,573,574,575,576,580,581,589,596,
         600,631,634,646,665,675,677,679,698,712,727,749,750,756,764,
         768,771,787,788]
'''
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
#label = datamatrix1[0:1440, 600]
label = datamatrix[0:2900, 600]
# 划分训练集和测试集，测试集共120组
test_x = data1[slect]  # 200*480
test_y = label[slect]  # 200*1
train_x = np.delete(data1, slect, 0)  # (822-200)*100
train_y = np.delete(label, slect, 0)  #
# train_x= datamatrix[0:1000, 0:600]  # 200*480
# train_y= datamatrix1[0:1000, 600]  # 200*1
# test_x = datamatrix[1000:1200, 0:600] # (822-200)*100
# test_y  = datamatrix1[1000:1200, 600]  #




timestep_size = 60
feature_size = 7
train_X = train_x.reshape((train_x.shape[0], timestep_size, feature_size))
test_X = test_x.reshape((test_x.shape[0], timestep_size, feature_size))
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        print("inputs.shape", inputs.shape)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        print("x.shape", x.shape)
        # x.shape = (batch_size, seq_len, time_steps)
        t = K.tanh(K.dot(x, self.W) + self.b)
        print(t.shape)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        print("a.shape", a.shape)
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        print(outputs.shape)
        # outputs = K.sum(outputs, axis=1)
        # print(outputs.shape)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0],input_shape[1], input_shape[2]

# 定义残差块网络TCN
def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate,activation='relu')(x)

    o = AttentionLayer()(r)
    o = Activation('relu')(o)  # 激活函数
    return o
# 定义残差块网络TCN
def ResBlock1(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate,activation='relu')(x)
    # 第一卷积
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)  # 第二卷积
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  # shortcut（捷径）
    o = add([r, shortcut])
    o = AttentionLayer()(o)
    o = Activation('relu')(o)  # 激活函数
    return o
#定义注意力机制

# 定义模型结构
inputs = Input(shape=(timestep_size, feature_size))


x = ResBlock(inputs, filters=64 , kernel_size=16, dilation_rate=1)
x = ResBlock(x, filters=64, kernel_size=12 , dilation_rate=1)
x = ResBlock(x, filters=64, kernel_size=8 , dilation_rate=1)
#x=LSTM(64,return_sequences=True)(x)

# x = Conv1D(inputs, filters=64 , kernel_size=16)
# x = Conv1D(x, filters=64, kernel_size=12 )
# x = Conv1D(x, filters=64, kernel_size=8 )
x = Flatten()(x)
x = Dense(64)(x)

y = Dense(1)(x)
model = Model(input=inputs, output=y)
# 查看网络结构
model.summary()
# 编译模型
adam = Adam(0.0005)
model.compile(optimizer='adam', loss='mse')
# 训练模型
history = model.fit(train_X, train_y, batch_size=128, nb_epoch=30, verbose=2, validation_split=0)
# 评估模型
#model.save('TCN_Att.h5')
loss1 = model.evaluate(test_X, test_y)
print('loss:', loss1)

test_prediction = model.predict(test_X)
#test_prediction = test_prediction* (max11 - min11) + min11#+np.random.uniform(-0.1,0.1,(200,1))
#test_y =test_y* (max11 - min11) + min11
#打印预测结果
print(test_prediction)
#绘制训练损失图像
plt.figure(1)
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend('train', loc='upper left')
plt.show()


test_mse = metrics.mean_squared_error(test_y, test_prediction) # 均方根误差
test_rmse = np.sqrt(test_mse)  # 均方根误差
test_mae = metrics.mean_absolute_error(test_y, test_prediction) #np.mean(np.abs(test_prediction - test_y))  # 平均绝对误差
#test_mre = np.mean(np.divide(np.abs(test_prediction - test_y), test_prediction))  # 平均相对误差
r_2 = r2_score(test_y,test_prediction)
#print(" test_mse %g,test_rmse %g, test_mae %g,test_mre %g,r2 %g" % (test_mse,test_rmse, test_mae, test_mre,r_2))
print(" test_rmse %g, test_mae %g, r2 %g" % (test_rmse, test_mae ,r_2))

plt.figure(2)
dim2 = len(test_y)
x_n1 = range(0, dim2)
#* (max11 - min11) + min11
plt.plot(x_n1, test_y , label='real', color='b',marker='*', linewidth=1)
plt.plot(x_n1, test_prediction , label='predicted', color='r',marker='+',linewidth=1)
plt.xlabel('Samples')
plt.ylabel('f_Cao')
# plt.ylim(min11 - 20, max11 + 20)
plt.legend()
plt.show()
