# Learner: 王振强
# Learn Time: 2021/4/26 18:44
import pandas as pd
import numpy as np

# 读取数据
train_data = pd.read_csv('dataset/train_clean.csv')
test_data = pd.read_csv('test_step2.csv')
# print(train_data.head())
# print(test_data.head())
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

# 对value取对数
train_data['value'] = np.log1p(train_data['value'])

# 添加一列
test_data['value'] = ''
print(train_data.dtypes)
print(test_data.dtypes)

# 找到需要预测id列表
Test_list_ID = pd.unique(test_data['id'])

min_mean_Index = 5  # 最小移动平均数
max_mean_Index = 15  # 最大移动平均数
threshold = 10  # 判定数据量过少阈值
num_no_pred = 0  # 记录无法预测传感器个数

for i,now_id in enumerate(Test_list_ID):
    # 取出某一传感器数据
    train_temp = train_data[train_data['id']==now_id]
    # 求训练集大小
    m = len(train_temp)
    # 计算nan占比
    index_nan = train_temp['value'].isnull()
    P_nan = sum(index_nan.astype(int))/m
    not_nan = train_temp['value'][index_nan==False]

    # 替换nan为mean
    if P_nan==1:  # 有全为nan的数据
        mean_temp = np.nan
    else:
        mean_temp = sum(not_nan)/len(not_nan)
    train_temp[train_temp['value'] == np.nan]['value'] = mean_temp

    # 找到测试集该传感器数据
    test_temp = test_data[test_data['id']==now_id]

    is_nan = False  # 判定该传感器是否有nan值
    # 创建预测序列
    pre_line = [mean_temp]*len(test_temp)
    if P_nan>0.01:
        # 先判定如果nan占比超过某一阈值,就无法预测,预测值设置为nan
        pre_line = [np.nan]*len(test_temp)
        is_nan = True
        num_no_pred += 1
    elif P_nan != 0:
        is_nan = True
        # ===========================训练集后5个数有无nan========================
        pre_line = [np.nan]*len(test_temp)
        n = 5
        n_nan = train_temp['value'][m-n:m].isnull()
        have_nan = sum(n_nan.astype(int))  # 训练集后5个数有nan则have_nan>0
        # ====================================================================
        if have_nan>0:
            pre_line = [np.nan] * len(test_temp)
        else:   # 后5个数没有nan正常预测
            # 预测
            # 建立个中间序列
            t_pre_line = list([np.nan]*(n+len(test_temp)))
            t_pre_line[:n] = train_temp['value'][m-n:m]
            t_pre_line[n:] = list(pre_line)
            # 添加预测序列
            for kk in range(n,len(t_pre_line)):
                t_pre_line[kk] = sum(t_pre_line[kk-n:kk])/n
            pre_line = t_pre_line[n:]
    else:
        # 正常数据预测
        if mean_temp<0.01:    # 整体数据太小就不预测
            pre_line = [np.nan]*len(test_temp)
        elif m < threshold:  # 数据量太少就取均值
            pre_line = [mean_temp]*len(test_temp)
        else:
            # 若数据量小于最大移动平均数,最大移动平均数设定为数据量大小
            if m<max_mean_Index:
                max_mean_Index = m - 1
            # 生成移动平均步数序列
            n = list(range(min_mean_Index,max_mean_Index))
            s = [np.inf]*len(n)  # 保存训练方差
            for k in range(len(n)):
                # 初始化训练预测顺序
                yhat = [mean_temp]*(m-n[k])
                for j in range(m-n[k]):
                    yhat[j] = sum(train_temp['value'][j:j+n[k]])/n[k]
                s[k] = (sum(train_temp['value'][n[k]:m]-yhat[:])**2)**0.5/(m-n[k])
            # 找到使得方差最小的模型
            min_s = min(s)
            best_Index = s.index(min_s)
            # 若最小值不存在,取最小移动项数
            if best_Index is None:
                best_n = min_mean_Index
            else:
                best_n = n[best_Index]

            # 预测
            # 建立个中间序列
            t_pre_line = list(train_temp['value'][m-best_n:m]) + list(pre_line)
            # 添加预测序列
            for kk in range(best_n,len(t_pre_line)):
                t_pre_line[kk] = sum(t_pre_line[kk-best_n:kk])/best_n
            pre_line = t_pre_line[best_n:]

    #pre_line = np.exp(pre_line)-1
    pre_line = np.exp(pre_line)-1
    # 写入test,只能这么使用
    test_data.loc[test_data['id']==now_id,'value'] = pre_line

    if i % 100 == 0:
        print(i,'id:',now_id,'该传感器已知数据数目:',len(train_temp),'nan占比',P_nan,'均值',mean_temp)
    #Same_id =

# 保存
test_data.to_csv('dataset/result.txt',sep=',',index=False,na_rep='nan')

















































