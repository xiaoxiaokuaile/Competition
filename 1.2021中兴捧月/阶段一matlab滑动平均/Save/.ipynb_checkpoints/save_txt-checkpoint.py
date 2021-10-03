# Learner: 王振强
# Learn Time: 2021/4/22 12:49

with open(r'C:/Users/44313/Desktop/ZTE/Save_result/第一次提交结果/test_step1.txt','r') as f1:
    Readdata1 = f1.readlines()
    Readdata1 = [line.strip('\n') for line in Readdata1]
    with open(r'C:\Users\44313\Desktop\ZTE\Save_result\第十二次结果/value.txt','r') as f2:
        Readdata2 = f2.readlines()
        print(len(Readdata1))
        for i in range(0,len(Readdata1)):
            Readdata1[i] = Readdata1[i] + ',' + Readdata2[i]

with open(r'C:\Users\44313\Desktop\ZTE\Save_result\第十二次结果/result12.txt','w') as f3:
    f3.writelines(Readdata1)











































