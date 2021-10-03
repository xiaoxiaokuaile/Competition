# 2021中兴捧月 迪杰斯特拉赛道

## 1. 思路简介
这是个时间序列预测题目，只有一条时间序列，没有其他特征，初赛分为阶段一和阶段二，这也是我第一次接触数据竞赛，当时这个比赛宣传海报放在学校食堂，看起来奖励诱人就去了解了这些呢东西。阶段一使用了matlab写了个滑动平均算法对这个单变量时间序列进行预测，阶段二则是有官方人员在赛题解读中提示说可以用LGB，CNN、以及挖取特征等方法解决，当时还并不晓得baseline这个神奇的东西，然后自己根据官方人员的讲解捕风捉影的写了个CNN预测的算法，最终得分只有30多(满分100)，所以就没有进入下一阶段，不过这场比赛学到了非常多的东西，无论知识还是技能上。

## 2. 实现步骤

### 2.1. 阶段一

阶段一的分数则是根据官方自己的基线为满分进行放大的，所以只要分数达到官方基线就可以满分，这一阶段我是用matlab写了一个动态的滑动平均算法，每个样本滑动平均的窗口大小根据自己的历史拟合程度动态确定，滑动窗口大小[2,20]，除了动态调整滑动窗口大小外，所有样本滑动窗口大小为5也是可以达到官方基线以上的分数.

```
具体代码见 “阶段一matlab滑动平均“ 文件夹
```



### 2.2. 阶段二

阶段二官方则完全放开分数，不再以基线放大了，第二阶段继续使用滑动平均法分数可以在63以上，不过这样不断试的话感觉太浪费时间，所以根据官方赛题解读又去尝CNN等方法实现时间序列预测，虽然结果不怎么样吧，但是懵懵懂懂学到了很多东西。
```
具体代码见 “阶段二python实现“ 文件夹
```



## 3. 参考文献

- A Multi-Horizon Quantile Recurent Forecaster
- Analyzing and Exploiting NARX Rceurrent Neural Networks for Long-Term Dependencies,11
- Deep AR:Probabilistic Forecasting with Autoregressive Recurrent Networks
- Modeling Long-and Short-Term Temporal Patterns with Deep Neural Networks
- An empirical evaluation of generic convolutional and recurrent networks for sequence modeling
- Temporal Pattern Attention for Multivariate Time Series Forecasting 
- Deep Factors for Forecasting
- Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

