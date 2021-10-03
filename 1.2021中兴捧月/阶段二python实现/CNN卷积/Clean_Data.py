# Learner: 王振强
# Learn Time: 2021/5/6 20:47

import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor

"""
    0.将测试集中出现的传感器数据集从训练集中挑出来
    1.特征中加入星期几
    2.将测试集中的ID从训练集中筛选出来，其余无用ID剔除
    3.将训练集中大于100的value置为nan,等于0的value置为10-8,小于0的数不存在故不需考虑
    4.使用7天间隔数据+随机森林回归补全nan
"""

# 读取数据
train_data = pd.read_csv('train_step2.csv')
train_data['date'] = pd.to_datetime(train_data['date'])
print(train_data.dtypes)

"""1.加入星期几特征'weekday'"""
# dt.quarter季度,dt.month月份,dt.year年份
train_data['weekday'] = train_data['date'].dt.weekday

# 改变DateFrame的值出现警告,消除警告
# pd.set_option('mode.chained_assignment', None)
"""2.1将大于100的异常值置为nan"""
train_data.loc[train_data.value > 100,'value'] = np.nan
"""2.2将等于0的值置为1e-8"""
train_data.loc[train_data.value == 0,'value'] = 1e-8

"""3.填充nan"""
test_data = pd.read_csv('test_step2.csv')
# 找到需要预测id列表
Test_list_ID = pd.unique(test_data['id'])
# 清洗数据保存目录
save_file = 'dataset/train_clean.csv'
for i,now_id in enumerate(Test_list_ID):
    train_temp = train_data[train_data['id'] == now_id]
    # 解决警告
    train_temb = train_temp.copy()

    """用前后7天数据填充"""
    # 计算nan占比
    index_nan = train_temb['value'].isnull()
    m = len(train_temb)
    P_nan = sum(index_nan.astype(int))/m

    if P_nan>0:
        n = math.ceil(len(train_temp) / 7)
    else:
        n = 0
    for j in range(n):
        train_temb.loc[:,'-7day'] = train_temb['value'].shift(7)
        train_temb.loc[:,'+7day'] = train_temb['value'].shift(-7)
        # 将-7day列放在value列前面
        train_array = train_temb.loc[:,['-7day','value','+7day']]
        # 缺失值填充为7天前的数据,bfill下一个数据填充
        train_array.fillna(method='ffill',axis=1,inplace=True)
        train_array.fillna(method='bfill',axis=1,inplace=True)

        # print(i)
        # 加入id,date,weekday列
        train_temb.loc[:,'value'] = train_array['value']
        train_temb.drop('-7day',axis=1,inplace=True)
        train_temb.drop('+7day',axis=1,inplace=True)

    """随机森林回归填充"""
    # 计算nan占比
    indexS_nan = train_temb['value'].isnull()
    ms = len(train_temb)
    PS_nan = sum(indexS_nan.astype(int))/ms
    if PS_nan>0 and PS_nan<1:
        """若还有nan用随机森林"""
        # 不含缺失值以及不能用于训练的数据的其它所有列
        df_full = train_temb.drop(labels='value', axis=1)
        df_full = df_full.drop(labels='id', axis=1)
        df_full = df_full.drop(labels='date', axis=1)
        # 含缺失值的那一列
        df_nan = train_temb.loc[:, 'value']
        # 区别我们的训练集和测试集
        Y_train = df_nan[df_nan.notnull()]
        Y_test = df_nan[df_nan.isnull()]
        X_train = df_full.loc[Y_train.index]
        X_test = df_full.loc[Y_test.index]

        # 用随机森林回归来填补缺失值
        rfc = RandomForestRegressor(n_estimators=100)
        # id和datetime不能训练
        rfc = rfc.fit(X_train, Y_train)
        Y_predict = rfc.predict(X_test)

        df_nnan = df_nan.copy()
        df_nnan[df_nnan.isnull()] = Y_predict

        train_temb.loc[:, 'value'] = df_nnan
    else:
        pass

    # 保存
    if i == 0:  # 第一次追加表头
        train_temb.to_csv(save_file, mode='a', index=None,na_rep='nan')
    else:
        train_temb.to_csv(save_file,mode='a',index=None,header=0,na_rep='nan')
    #if i % 100 == 0:
    print(i)









