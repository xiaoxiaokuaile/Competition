# Learner: 王振强
# Learn Time: 2021/5/8 0:09
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

# 加载模型
model = load_model('./dataset/train_model.h5')
# 读取数据
train_data = pd.read_csv('dataset/All_dataset.csv')
test_data = pd.read_csv('test_step2.csv')

# 删除date,weekday列,保留id查找传感器
train_data.drop('date', axis=1, inplace=True)
train_data.drop('weekday', axis=1, inplace=True)
# 找到需要预测id列表
Test_list_ID = pd.unique(test_data['id'])

for i,now_id in enumerate(Test_list_ID):
    # 取出某一传感器数据
    train_temp = train_data[train_data['id']==now_id]
    # 拷贝数据并删除Id
    train_temb = train_temp.copy()
    train_temb.drop('id', axis=1, inplace=True)

    # 计算nan占比
    index_nan = train_temb['value'].isnull()
    m = len(train_temb)
    P_nan = sum(index_nan.astype(int))/m

    # 将dataframe转化为array
    array_temp = np.array(train_temb)

    if P_nan == 1:
        # 数据全为nan,将预测值设置为nan
        yhat = np.full((1,91),np.nan)
    elif m<365:
        # 补365-m排0
        zero_temp = np.zeros((365-array_temp.shape[0],array_temp.shape[1]))
        Pre_x = np.vstack((zero_temp,array_temp))  # (365,14)
        # 添加维度
        Pre_xx = Pre_x[np.newaxis,:,:]
        # print(Pre_x.shape)
        # 预测
        yhat = model.predict(Pre_xx, verbose=0)
        yhat = np.array(yhat) # (1,91)
        # print(yhat)
        # print(yhat.shape)
    else:
        # 正常截取预测输入
        Pre_x = np.array(array_temp[-365:,:])  # (365,14)
        # 添加维度
        Pre_xx = Pre_x[np.newaxis,:,:]
        # 预测
        yhat = model.predict(Pre_xx, verbose=0)
        yhat = np.array(yhat)

    # 将预测得到超限的数据设限到[0,1]之间
    yhat = np.clip(yhat,a_min=0,a_max=1)
    yhat = np.array(yhat)
    # 将0设为Nan,因为0还原之后没分可能性很大
    yhat[yhat == 0] = np.nan

    # 还原数据(0,100]
    y_old = 10**((yhat-0.8)*10)
    # 写入test,同时将预测数据维度(1,91)变为(91,)
    test_data.loc[test_data['id']==now_id,'value'] = np.squeeze(y_old, 0)

    if i % 100 == 0:
        print(i,'id:',now_id,'该传感器已知数据数目:',len(train_temp),'nan占比',P_nan)

# 保存
test_data.to_csv('dataset/result.txt',sep=',',index=False,na_rep='nan')







