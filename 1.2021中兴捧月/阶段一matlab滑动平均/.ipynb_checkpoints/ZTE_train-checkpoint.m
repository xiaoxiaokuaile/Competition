clc,clear
%% 读取数据
format rat
train_data = readtable('train_step1.csv'); 
test_data = readtable('test_step1.csv');

%% 分别读取数据
train_ID = train_data(:,1);
train_ID = table2array(train_ID); % ID转化为数组
train_Time = train_data(:,2);
train_Time = table2array(train_Time);
train_value = train_data(:,3);
train_value = table2array(train_value);
test_ID = test_data(:,1);
test_ID = table2array(test_ID);
test_Time = test_data(:,2);
test_Time = table2array(test_Time);
N = length(test_ID);
test_value = zeros(N,1);
%% 两个表ID列表数目相同18416
Test_list_ID = unique(train_ID(:,1));
Train_list_ID = unique(train_data(:,1));

%% 
num = 0; % 计数
num_fig = 1; % 显示图片窗口计数
for i = 1:length(Test_list_ID) 
    Total_ID = Test_list_ID(i,1);
    Same_ID = find(train_ID(:)==Total_ID);
    Index_train_min = min(Same_ID);
    Index_train_max = max(Same_ID);
    % 提取对应传感器的数据
    data_temp = train_value(Index_train_min:Index_train_max);  % 提取出该传感器已知value数据
    time_temp = train_Time(Index_train_min:Index_train_max);   % 提取出该传感器已知time数据
    % 找到该传感器在测试集中的对应位置
    Same_test_ID = find(test_ID(:)==Total_ID);
    length_test = length(Same_test_ID);  % 得到需要预测的数组长度
    Index_text_min = min(Same_test_ID);
    Index_text_max = max(Same_test_ID);
    % 求平均值
    mean_value = mean(data_temp);
    % 找到每个序列最优移动平均数
    m = length(data_temp); % 该传感器已知序列长度
    % 移动平均项数(回看下是否超出已有序列长度)
    n = 4:20;
    s = Inf(1,length(n));
    
    % 先判定,数据量太少就以均值作为预测结果
    if m<6
        pre_line = ones(length_test,1)*mean_value;
        num = num + 1;
    else
        for k=1:length(n)
        % 初始化训练预测数据
        yhat = ones(m-n(k),1)*mean_value;
        for j=1:m-n(k)
            yhat(j)=sum(data_temp(j:j+n(k)-1))/n(k); 
        end 
        s(k)=sqrt(mean((data_temp(n(k)+1:m,1)-yhat(1:end,1)).^2));
        end
        min_s = min(s);
        best_Index = find(s==min_s);
        best_n = n(best_Index(1));
        % 预测
        % 创建一个预测序列
        pre_line = ones(best_n+length_test,1)*mean_value;
        pre_line(1:best_n) = data_temp(m-best_n+1:m); % 将有用的数组序列加入预测序列
        % 添加预测序列
        for kk = best_n+1:length(pre_line)
            pre_line(kk) = sum(pre_line(kk-best_n:kk-1))/best_n;
        end
        pre_line = pre_line(best_n+1:end);  % 取出预测的序列
    end
    
    % 写入test
    test_value(Index_text_min:Index_text_max)=pre_line;

    disp([num2str(i),'同一传感器监控次数:',num2str(length(Same_ID)),'最佳移动平均数:',num2str(best_n),'数据量过少组数:',num2str(num)]);
    % 每隔3000代显示该传感器数据
    % if rem(i,3000)==0
    %    figure(num_fig);
    %    num_fig = num_fig + 1;
    %    plot(time_temp,data_temp)
    % end
end

%% 写入csv，txt
csvwrite('value.txt',test_value);





























