---
title: 机器学习笔试整理
date: 2019-09-01 22:27:50
categories:
- PRML
tags:
photos:
author: 
---

## 生成式 & 判别式模型

 |生成式模型|判别式模型|
 |--|--|
 |根据概率乘出结果|给定输入，计算出结果|
 |$P(x|C_k)P(C_k)$ | $P(C_k|x)$|
 |考虑了类别本身的概率|x出现在哪一类的概率高就属于哪一类|

 常见的生成式模型（generative model）有: 
 > Gaussian mixture model and othertypes of mixture model（高斯混合及其他类型混合模型）  
 > Hidden Markov model（隐马尔可夫）  
 > NaiveBayes（朴素贝叶斯）  
 > AODE（平均单依赖估计）  
 > Latent Dirichlet allocation（LDA主题模型）  
 > Restricted Boltzmann Machine（限制波兹曼机）  

 常见的判别式模型（discriminative model）有：  
 > Logistic regression（logistical 回归）  
 > Linear discriminant analysis（线性判别分析）  
 > Supportvector machines（支持向量机）  
 > Boosting（集成学习）  
 > Conditional random fields（条件随机场）  
 > Linear regression（线性回归）  
 > Neural networks（神经网络）  

## 隐马尔可夫模型

 HMM的三个基本问题

 |问题|算法|已知|求解|
 |--|:--:|:--:|:--:|
 |概率计算问题|前后向算法|隐层，参数|观测|
 |学习问题|Baum-Welch模型，EM算法|观测|参数|
 |预测问题|Viterbi算法|观测|隐层|

## 线性回归模型

### R-squared

 衡量回归方程与真实样本输出之间的相似程度。越大越相似。
 $$
 R^{2}=1-\frac{\sum(y-\hat{y})^{2}}{\sum(y-\bar{y})^{2}}\approx {均方差MSE\over 方差Var}
 $$
 单独看 R-Squared，并不能推断出增加的特征是否有意义。通常来说，增加一个特征特征，R-Squared 可能变大也可能保持不变，两者不一定呈正相关。

 |item|value|
 |-|-|
 |data|5,6,7,8|
 |mean|(5 + 6 + 7 + 8) / 4 = 6.5|
 |predicts|4.5, 6.3, 7.2, 7.9|
 |MSE|(5 – 4.5) ^ 2 + (6 – 6.3) ^ 2 + (7 – 7.2) ^ 2 + (8 – 7.9) ^ 2|
 |Var|(5 – 6.5) ^ 2 + (6 – 6.5) ^ 2 + (7 – 6.5) ^ 2 + (8 – 6.5) ^ 2|

### Adjusted R-Squared
 其中，n 是样本数量，p 是特征数量。Adjusted R-Squared 抵消样本数量对 R-Squared 的影响，做到了真正的 0~1，越大越好。
 $$
 R^{2}_{adjusted}=1-\frac{\left(1-R^{2}\right)(n-1)}{n-p-1}
 $$
 增加一个特征变量，如果这个特征有意义，Adjusted R-Square 就会增大，若这个特征是冗余特征，Adjusted R-Squared 就会减小。  

 如果单变量线性回归，则使用 R-squared 评估，多变量，则使用 adjusted R-squared。

## 分类模型

### 指标

|item|expression|
|--|--|
|TP|将正类预测为正类数|
|FN|将正类预测为负类数|
|FP|将负类预测为正类数|
|TN|将负类预测为负类数|
|准确率|Accuracy= T / (T + F)|
|精准率 precision|P = TP / (TP + FP)|
|召回率 recall(TPR)|R = TP / (TP + FN)|
|FPR|FPR = FP / FP + TN|
|F1值|F1 = 2 P R / (P + R)|
|ROC 曲线|不同分类阈值时的 TPR 与 FPR|
|AUC|ROC曲线下的面积|

为了解决准确率和召回率冲突问题，引入了F1分数

![ROC curve](roc.png)  
理想情况下，TPR应该接近1，FPR应该接近0。  
故ROC曲线越靠拢(0,1)点，越偏离45度对角线越好。  

使用AUC值作为评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC更大的分类器效果更好。

### SVM

SVM的目标是找到使得训练数据尽可能分开且分类间隔最大的超平面，应该属于结构风险最小化。

可以通过正则化系数控制模型的复杂度，避免过拟合。

### 多分类问题

 针对不同的属性训练几个不同的弱分类器，然后将它们集成为一个强分类器。这里狱警、 小偷、送餐员 以及他某某，分别根据他们的特点设定依据，然后进行区分识别。

### 层次聚类问题
 创建一个层次等级以分解给定的数据集。监狱里的对象分别是狱警、小偷、送餐员、或者其 他，他们等级应该是平等的，所以不行。此方法分为自上而下（分解）和自下而上（合并）两种操作方式。

### k-中心点聚类问题

 挑选实际对象来代表簇，每个簇使用一个代表对象。它是围绕中心点划分的一种规则，

### 结构分析

 结构分析法是在统计分组的基础上，计算各组成部分所占比重，进而分析某一总体现象的内部结构特征、总体的性质、总体内部结构依时间推移而表现出的变化规律性的统计方法。结构分析法的基本表现形式，就是计算结构指标。