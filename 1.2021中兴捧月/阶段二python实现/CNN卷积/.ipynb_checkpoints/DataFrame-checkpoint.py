# Learner: 王振强
# Learn Time: 2021/5/3 14:31
# Learner: 王振强
# Learn Time: 2021/4/26 18:44
import pandas as pd
import numpy as np

"""
    制作训练用训练集+测试集
    1.星期one-hot编码
    2.滑动平均项作为特征
    3.剔除全为nan以及数据量小于91个的数据集,不足以构成训练集

"""

# 读取数据
train_data = pd.read_csv('dataset/train_clean.csv')
train_data['date'] = pd.to_datetime(train_data['date'])

# 找到需要预测id列表
Test_list_ID = pd.unique(train_data['id'])

save_file = 'dataset/All_dataset.csv'
for i,now_id in enumerate(Test_list_ID):
    # 取出某一传感器数据
    train_temp = train_data[train_data['id']==now_id]
    train_temb = train_temp.copy()
    # 取出value并转换为Series
    value_temp = train_temb['value']
    value = pd.Series(value_temp)

    # 计算nan占比
    index_nan = train_temb['value'].isnull()
    m = len(train_temb)
    P_nan = sum(index_nan.astype(int))/m

    if P_nan == 1:
        # 数据集全为nan
        train_temb.loc[:,'2_avg'] = np.nan
        train_temb.loc[:,'3_avg'] = np.nan
        train_temb.loc[:,'4_avg'] = np.nan
        train_temb.loc[:,'5_avg'] = np.nan
        train_temb.loc[:,'6_avg'] = np.nan
        train_temb.loc[:,'7_avg'] = np.nan

    elif m<7:
        # 有小100个传感器数据量过少
        # 滑动平均最多到m次
        if m == 1:
            train_temb.loc[:, '2_avg'] = value
            train_temb.loc[:, '3_avg'] = value
            train_temb.loc[:, '4_avg'] = value
            train_temb.loc[:, '5_avg'] = value
            train_temb.loc[:, '6_avg'] = value
            train_temb.loc[:, '7_avg'] = value
        elif m == 2:
            train_temb.loc[:,'2_avg'] = value.rolling(2).mean()
            train_temb.loc[:,'3_avg'] = value.rolling(2).mean()
            train_temb.loc[:,'4_avg'] = value.rolling(2).mean()
            train_temb.loc[:,'5_avg'] = value.rolling(2).mean()
            train_temb.loc[:,'6_avg'] = value.rolling(2).mean()
            train_temb.loc[:,'7_avg'] = value.rolling(2).mean()
        elif m == 3:
            train_temb.loc[:, '2_avg'] = value.rolling(2).mean()
            train_temb.loc[:, '3_avg'] = value.rolling(3).mean()
            train_temb.loc[:, '4_avg'] = value.rolling(3).mean()
            train_temb.loc[:, '5_avg'] = value.rolling(3).mean()
            train_temb.loc[:, '6_avg'] = value.rolling(3).mean()
            train_temb.loc[:, '7_avg'] = value.rolling(3).mean()
        elif m == 4:
            train_temb.loc[:,'2_avg'] = value.rolling(2).mean()
            train_temb.loc[:,'3_avg'] = value.rolling(3).mean()
            train_temb.loc[:,'4_avg'] = value.rolling(4).mean()
            train_temb.loc[:,'5_avg'] = value.rolling(4).mean()
            train_temb.loc[:,'6_avg'] = value.rolling(4).mean()
            train_temb.loc[:,'7_avg'] = value.rolling(4).mean()
        elif m == 5:
            train_temb.loc[:,'2_avg'] = value.rolling(2).mean()
            train_temb.loc[:,'3_avg'] = value.rolling(3).mean()
            train_temb.loc[:,'4_avg'] = value.rolling(4).mean()
            train_temb.loc[:,'5_avg'] = value.rolling(5).mean()
            train_temb.loc[:,'6_avg'] = value.rolling(5).mean()
            train_temb.loc[:,'7_avg'] = value.rolling(5).mean()
        elif m == 6:
            train_temb.loc[:,'2_avg'] = value.rolling(2).mean()
            train_temb.loc[:,'3_avg'] = value.rolling(3).mean()
            train_temb.loc[:,'4_avg'] = value.rolling(4).mean()
            train_temb.loc[:,'5_avg'] = value.rolling(5).mean()
            train_temb.loc[:,'6_avg'] = value.rolling(6).mean()
            train_temb.loc[:,'7_avg'] = value.rolling(6).mean()
    else:
        train_temb.loc[:,'2_avg'] = value.rolling(2).mean()
        train_temb.loc[:,'3_avg'] = value.rolling(3).mean()
        train_temb.loc[:,'4_avg'] = value.rolling(4).mean()
        train_temb.loc[:,'5_avg'] = value.rolling(5).mean()
        train_temb.loc[:,'6_avg'] = value.rolling(6).mean()
        train_temb.loc[:,'7_avg'] = value.rolling(7).mean()

    """归一化"""
    train_temb.loc[:, 'value'] = train_temb['value'].apply(np.log10)*0.1+0.8
    train_temb.loc[:, '2_avg'] = train_temb['2_avg'].apply(np.log10)*0.1+0.8
    train_temb.loc[:, '3_avg'] = train_temb['3_avg'].apply(np.log10)*0.1+0.8
    train_temb.loc[:, '4_avg'] = train_temb['4_avg'].apply(np.log10)*0.1+0.8
    train_temb.loc[:, '5_avg'] = train_temb['5_avg'].apply(np.log10)*0.1+0.8
    train_temb.loc[:, '6_avg'] = train_temb['6_avg'].apply(np.log10)*0.1+0.8
    train_temb.loc[:, '7_avg'] = train_temb['7_avg'].apply(np.log10)*0.1+0.8
    # 保留5位小数
    train_temb = train_temb.round(5)
    # 移动平均产生的nan用下面数据向上填充
    train_temb.fillna(method='bfill', axis=0, inplace=True)

    """one-hot编码"""
    week = train_temb['weekday']
    bins = [-1, 0, 1, 2, 3, 4, 5, 6]
    # 数据量小于7时候也可以正常编码
    weeks = pd.cut(week, bins)
    one_hot = pd.get_dummies(weeks)
    train_temb.loc[:, 'week1'] = one_hot.loc[:,0]
    train_temb.loc[:, 'week2'] = one_hot.loc[:,1]
    train_temb.loc[:, 'week3'] = one_hot.loc[:,2]
    train_temb.loc[:, 'week4'] = one_hot.loc[:,3]
    train_temb.loc[:, 'week5'] = one_hot.loc[:,4]
    train_temb.loc[:, 'week6'] = one_hot.loc[:,5]
    train_temb.loc[:, 'week7'] = one_hot.loc[:,6]

    # 删除原有week列
    # train_temb.drop('weekday', axis=1, inplace=True)

    """分割训练集测试集"""

    # 保存
    if i == 0:  # 第一次追加表头
        train_temb.to_csv(save_file, mode='a', index=None,na_rep='nan')
    else:
        train_temb.to_csv(save_file,mode='a',index=None,header=0,na_rep='nan')
    #if i % 100 == 0:
    print(i)

# 保存训练集与测试集
# test_data.to_csv('dataset/train_dataset.txt',sep=',',index=False,na_rep='nan')


















































