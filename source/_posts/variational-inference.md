---
title: 变分推理
date: 2019-09-28 23:19:10
categories: PRML
tags:
 - VI
---

## Background
数据量大，如何挖掘其中的含义，找到其中的因果关系？

我们用机器学习和统计方法找到其中的联系。

流程一般是提出假设，挖掘出特征，得到预测，修改模型，提出假设……迭代进行。

概率机器学习的目标是学习出后验，即$p(z|x)={p(z,x) \over p(x)}$。其中联合概率可以用生成式得到，关于分母则需要边缘化隐变量，因此不易求得。

## Variational Inference

![](variational-inference/VI-family.png)

找到一个后验的近似分布 $q(z;v)$ 使得它们的 KL 散度尽量小。

先选择一个分布族，再在该分布族里优化参数 $v$，使得 KL 散度最小，此时是最优的后验分布 $q(z;v^*)$.

> 指数族分布  
$$
p(x)=h(x) \exp \left\{\eta^{\top} t(x)-a(\eta)\right\}
$$
- $\eta$ the natural parameter
- $t(x)$ the sufficient statistics
- $a(\eta)$ the log normalizer
- $h(x)$ the base density

### ELBO

KL 不易计算，因此 VI 优化 **证据下界（ELBO）** 来求解。

$$
\mathscr{L}(v)=\mathbb{E}_{q}[\log p(\beta, \mathbf{z}, \mathbf{x})]-\mathbb{E}_{q}[\log q(\beta, \mathbf{z} ; v)]
$$

## Reference
- [Variational Inference: Foundations and Modern Methods PDF](https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf)
- [Variational Inference: Foundations and Modern Methods VIDEO](https://www.bilibili.com/video/av43405716/)