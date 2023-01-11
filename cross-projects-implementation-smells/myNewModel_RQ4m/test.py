import numpy as np
_train_labels = [0,1,2,3,4,5,6,7,8,9,9]
_train_labels = np.array(_train_labels)
train_labels = np.zeros((_train_labels.shape[0],11))
train_labels[np.arange(_train_labels.shape[0]),_train_labels] = 1

print(train_labels)

print(type(train_labels.tolist()))


from sklearn import preprocessing
def mm():
    """
    按列，一列表示一个特征
    归一化处理，指定范围缩放
    :return:
    """
    # feature_range 指定缩放后的值在2到3之间
    mm = preprocessing.MinMaxScaler(feature_range=(2, 3))
    data = mm.fit_transform([[90, 2, 10, 40],
                             [60, 4, 15, 45],
                             [75, 3, 13, 46]])
    print(data)

mm()

def stand():
    """
    按列，一列表示一个特征
    标准化缩放，化到均值为0，标准差为1的区间
    :return:
    """
    std = preprocessing.StandardScaler()
    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])
    print(data)

stand()
