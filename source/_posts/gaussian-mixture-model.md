---
title: 高斯混合模型
date: 2019-09-22 22:19:10
categories: PRML
tags:
 - GMM
 - EM
---

顾名思义，每个成分都是高斯分布。

$$
\DeclareMathOperator{\Norm}{\mathcal{N}}
p(x|\boldsymbol{\mu}, \boldsymbol{\lambda}, \boldsymbol{\pi}) = \sum_{k=1}^{K} \pi_k \Norm(x|\mu_k, (\lambda_k)^{-1})
$$
* $\boldsymbol{\mu}$ is the vector of $K$ means
* $\boldsymbol{\lambda}$ is the vector of $K$ precisions
* $\boldsymbol{\pi}$ is the vector of $K$ weights such that $\sum_{k=1}^K \pi_k = 1$ 

这样看就是 K 个高斯的线性组合。对于每个观测数据 $x_i$ 而言，可以引入潜变量 $z_i$，表示 $x_i$ 属于哪一个高斯（1 到 K）。当给定 $z_i=k$ 时，$x_i\sim \Norm(x|\mu_k, (\lambda_k)^{-1})$，可以根据这个来采样。例如我们需要根据 GMM 采 N 个点，首先根据 $\boldsymbol{\pi}$ 得到每一个高斯有多少个点。在同一类中再根据对应的高斯分布采点。

## EM算法求解参数

1. 初始化参数 $\pi$, $\mu$, $\Sigma$
2. E step  
根据当前的参数计算后验：
$$
p(z_i = k \mid x_i, \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\lambda}) = \frac{\pi_k \Norm(x_i \mid \mu_k, \lambda_k)} {\sum_{j=0}^{K} \pi_j \Norm(x_i \mid \mu_j, \lambda_j)}
$$
3. M step  
最大化后验计算新的参数：
$$
\begin{aligned} \boldsymbol{\mu}_{k}^{n e w} &=\frac{1}{N_{k}} \sum_{n=1}^{N} p\left(z_{n k}\right) \boldsymbol{x}_{n} \\ \boldsymbol{\Sigma}_{k}^{n e w} &=\frac{1}{N_{k}} \sum_{n=1}^{N} p\left(z_{n k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}^{n e w}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}^{n e w}\right)^{T} \\ \pi_{k}^{n e w} &=\frac{N_{k}}{N} \end{aligned}
$$
其中
$$
N_{k}=\sum_{n=1}^{N} p\left(z_{n k}\right)
$$
4. 计算对数似然作为损失  
$$
\ln p(\boldsymbol{x} | \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})=\sum_{n=1}^{N} \ln \left\{\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x}_{k} | \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)\right\}
$$
5. 若收敛则结束，否则到 2.