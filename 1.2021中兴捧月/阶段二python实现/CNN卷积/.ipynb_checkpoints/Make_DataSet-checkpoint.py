# Learner: 王振强
# Learn Time: 2021/5/7 17:26
import pandas as pd
import numpy as np
"""
    制作训练用训练集+测试集
    1.星期one-hot编码
    2.滑动平均项作为特征
    3.剔除全为nan以及数据量小于91个的数据集,不足以构成训练集
    4.保存得到的训练集(23820, 365, 14)(23820, 91)两个数组
"""
# 读取数据
train_data = pd.read_csv('dataset/All_dataset.csv')
# 删除原有week列
train_data.drop('weekday', axis=1, inplace=True)
# 找到需要预测id列表
Test_list_ID = pd.unique(train_data['id'])

# 存储训练集
train_dataset_X = []
train_dataset_Y = []
# 存储训练集标签
test_dataset = []
for i,now_id in enumerate(Test_list_ID):
    # 取出某一传感器数据
    train_temp = train_data[train_data['id']==now_id]
    # 删除id,date无关序列
    train_temb = train_temp.copy()
    train_temb.drop('id', axis=1, inplace=True)
    train_temb.drop('date', axis=1, inplace=True)

    # 计算nan占比
    index_nan = train_temb['value'].isnull()
    m = len(train_temb)
    P_nan = sum(index_nan.astype(int))/m

    # 将dataframe转化为array
    array_temp = np.array(train_temb)

    if P_nan==1:
        # 全为nan不作为训练集
        pass
    elif m<=91:
        # 数据量不足以构成训练集的标签与训练数据也筛选除去
        pass
    elif m<456:
        # 数据量小于365+91=456需要补0
        xx_temp = np.array(array_temp[:-91,:])
        # 补456-m排0
        zero_temp = np.zeros((365-xx_temp.shape[0],xx_temp.shape[1]))
        # print(zero_temp.shape)
        train_x = np.vstack((zero_temp,xx_temp))
        # print(train_x.shape)
        train_y = np.array(array_temp[-91::, 0])

        train_dataset_X.append(train_x)
        train_dataset_Y.append(train_y)
    else:
        # 剩下数据可以正常提取为训练集
        train_x = np.array(array_temp[-456:-91,:])
        train_y = np.array(array_temp[-91::,0])
        train_dataset_X.append(train_x)
        train_dataset_Y.append(train_y)
    print(i)

# 保存高维数组
np.save(file='dataset/train_X.npy', arr=np.array(train_dataset_X))
np.save(file='dataset/train_Y.npy', arr=np.array(train_dataset_Y))

XXX = np.load(file='dataset/train_X.npy')
YYY = np.load(file='dataset/train_Y.npy')
print(np.array(XXX).shape)
print(np.array(YYY).shape)


















































