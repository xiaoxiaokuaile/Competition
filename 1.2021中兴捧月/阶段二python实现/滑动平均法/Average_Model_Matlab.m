clc,clear
%% 读取数据
format rat
train_data = readtable('train_step2.csv'); 
test_data = readtable('test_step2.csv');

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
test_value = zeros(N,1); % 初始化测试集结果
%% 需要预测ID列表,阶段2训练集有较多冗余数据
Test_list_ID = unique(test_ID(:,1));

%% 训练及预测
num = 0; % 计数
num_have_nan = 0; % 记录后5个有Nan传感器组数
num_no_pred = 0; % 记录无法预测个传感器个数
num_fig = 1; % 显示图片窗口计数
for i = 1:length(Test_list_ID) 
    Total_ID = Test_list_ID(i,1);
    Same_ID = find(train_ID(:)==Total_ID);
    Index_train_min = min(Same_ID);
    Index_train_max = max(Same_ID);
    % 提取对应传感器的数据
    data_temp = train_value(Index_train_min:Index_train_max);  % 提取出该传感器已知value数据
    time_temp = train_Time(Index_train_min:Index_train_max);
    
    % 将数据中为nan的数据替换为其它无损数据平均值
    find_nan = isnan(data_temp);
    % 计算Nan在数据中占比
    P_nan = sum(find_nan)/length(find_nan);
    % 找到不为nan所有数据
    not_nan = data_temp(find_nan==0); 
    mean_temp = sum(not_nan)/length(not_nan);   
    % 替换值为nan的数据为0
    % data_temp(find_nan==1) = 0;
    data_temp(find_nan==1) = mean_temp;
    
    % 找到该传感器在测试集中的对应位置
    Same_test_ID = find(test_ID(:)==Total_ID);
    length_test = length(Same_test_ID);  % 得到需要预测的数组长度
    Index_text_min = min(Same_test_ID);
    Index_text_max = max(Same_test_ID);
    % 求平均值
    mean_value = sum(data_temp)/length(data_temp);
    % 找到每个序列最优移动平均数
    m = length(data_temp); % 该传感器已知序列长度
    % 移动平均项数(回看下是否超出已有序列长度)
    min_mean_Index = 5;  % 最小移动平均项数
    max_mean_Index = 5; % 最大移动平均项数
    threshold = 6;       % 判定数据量过少阈值
    
    is_Nan = false;  % 判定该传感器是否有nan值
    
    
    if P_nan>0.01
        % 先判定如果nan占比超过某一阈值,就无法预测,预测值设置为nan
        pre_line = NaN(length_test,1);
        num_no_pred = num_no_pred + 1;
        is_Nan = true;
    elseif P_nan ~= 0
        is_Nan = true;
        % num_no_pred = num_no_pred + 1;
        % ==========计算训练集后5个数有无nan============
        n_nan = data_temp(m-n+1:m)==mean_temp;
        have_nan = sum(n_nan);  % 训练集后5个数有nan则have_nan大于0
        % =============================================
        % nan占比没有超过阈值就使用滑动平均法补全数据
        % 再判定,数据量太少就以nan作为预测结果
        if m<threshold
            pre_line = NaN(length_test,1);
            num = num + 1;
        elseif have_nan>0  % 有nan则预测值全置为nan
            pre_line = NaN(length_test,1);
            num_have_nan = num_have_nan + 1;
        else  % 无nan则正常预测
            % =====================补全数据,移动平均数就固定为5======================
            n = 5;
            for k=1:length(n)
                % 初始化训练预测数据
                yhat = ones(m-n,1)*mean_value;
                for j=1:m-n
                    yhat(j)=sum(data_temp(j:j+n-1))/n;
                    % 更新缺失部分数据为滑动平均预测值
                    if data_temp(j+n)==mean_temp
                        data_temp(j+n) = yhat(j);
                    end
                end 
            end
            
            % ==================================补全数据就使用滑动平均法====================================
            % 若数据量小于最大移动平均数,最大移动平均数设定为数据量大小
            if m<=max_mean_Index
                max_mean_Index = m-1;
            end
            n = min_mean_Index:max_mean_Index;
            s = Inf(1,length(n));    
            for k=1:length(n)
            % 初始化训练预测数据
            yhat = ones(m-n(k),1)*mean_value;
            for j=1:m-n(k)
                yhat(j)=sum(data_temp(j:j+n(k)-1))/n(k); 
            end 
            s(k)=sqrt(sum((data_temp(n(k)+1:m,1)-yhat(1:end,1)).^2)/(m-n(k)));
            end
            min_s = min(s);
            best_Index = find(s==min_s);
            % 若出现最小值不存在,最优平均移动项数设置为最小移动项数
            if isempty(best_Index)
                best_n = min_mean_Index;
            else
                best_n = n(best_Index(1));
            end
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
        % ==================================================================================================
    else
        % 再判定,数据量太少就以均值作为预测结果
        if m<threshold
            pre_line = ones(length_test,1)*mean_value;
            num = num + 1;
        else
            % 若数据量小于最大移动平均数,最大移动平均数设定为数据量大小
            if m<=max_mean_Index
                max_mean_Index = m-1;
            end
            n = min_mean_Index:max_mean_Index;
            s = Inf(1,length(n));    
            for k=1:length(n)
            % 初始化训练预测数据
            yhat = ones(m-n(k),1)*mean_value;
            for j=1:m-n(k)
                yhat(j)=sum(data_temp(j:j+n(k)-1))/n(k); 
            end 
            s(k)=sqrt(sum((data_temp(n(k)+1:m,1)-yhat(1:end,1)).^2)/(m-n(k)));
            end
            min_s = min(s);
            best_Index = find(s==min_s);
            % 若出现最小值不存在,最优平均移动项数设置为最小移动项数
            if isempty(best_Index)
                best_n = min_mean_Index;
            else
                best_n = n(best_Index(1));
            end
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
    end
    % 写入test
    test_value(Index_text_min:Index_text_max)=pre_line;

    disp(['第',num2str(i),'个传感器,同一传感器监控次数:',num2str(length(Same_ID)),'最佳移动平均数:',num2str(best_n),...
        '数据量过少组数:',num2str(num),'无法预测传感器个数:',num2str(num_no_pred),'训练集后5个数有nan传感器个数:',num2str(num_have_nan)]);
    
    % 每隔200个问题传感器显示一次问题数据,观察分析
    
    % if is_Nan == true&&rem(num_no_pred,200)==0
    %    figure(num_fig);
    %    num_fig = num_fig + 1;
    %    plot(time_temp,data_temp)
    % end
    
end

%% 写入csv，txt
csvwrite('value.txt',test_value);































