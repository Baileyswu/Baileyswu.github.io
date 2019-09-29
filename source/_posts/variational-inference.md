---
title: 变分推理
date: 2019-09-28 23:19:10
categories: PRML
tags:
 - VI
 - 指数族分布

---

## Background
数据量大，如何挖掘其中的含义，找到其中的因果关系？

我们用机器学习和统计方法找到其中的联系。

流程一般是提出假设，挖掘出特征，得到预测，修改模型，提出假设……迭代进行。

**概率机器学习**的目标是学习出后验，即$p(z|x)={p(z,x) \over p(x)}$。其中联合概率可以用生成式得到，关于分母则需要边缘化隐变量，因此不易求得。

## Variational Inference

![](variational-inference/VI-family.png)

找到一个后验的近似分布 $q(z;v)$ 使得它们的 KL 散度尽量小。

先选择一个**分布族**，再在该分布族里优化参数 $v$，使得 KL 散度最小，此时是最优的后验分布 $q(z;v^*)$.

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

第一项相当于联合似然，要尽可能大。第二项则相当于变分分布的熵，使其 diffuse.

还有一种分解 ELBO 的方式：
$$
\mathscr{L}(v)=\mathbb{E}_{q}[\log p(\mathbf{x} | \beta, \mathbf{z})]-\operatorname{KL}(q(\beta, \mathbf{z} ; v) \| p(\beta, \mathbf{z}))
$$

第一项相当于似然，第二项是变分分布和先验的 KL 散度。

注意 ELBO 不一定是凸的。

### Mean-filed VI

![](variational-inference/Mean-field.png)

隐变量之间相互独立，则可以假设各自的分布。在优化其中一个参数时，固定住其他参数，进行优化。再迭代优化出所有参数。与 Gibbs Sampling 具有一些关联。

与 EM 类似，比较依赖于初始化参数。由于非凸，只能取到局部最优值。

### Stochastic VI

传统的 VI 不能处理大量数据。因此需要随机 VI 来处理。

思想：大数据采出子集，进行局部结构的优化，再更新全局结构。再进行下一次迭代。

原始的 ELBO 的 natural gradient 是这样的  
$$
\nabla_{\lambda}^{\mathrm{nat}} \mathscr{L}(\lambda)=\left(\alpha+\sum_{i=1}^{n} \mathbb{E}_{\phi_{i}^{*}}\left[t\left(Z_{i}, x_{i}\right)\right]\right)-\lambda
$$  
经过随机优化，得到 noisy natural gradient 
$$
\begin{aligned} j & \sim \text { Uniform }(1, \ldots, n) \\ \hat{\nabla}_{\lambda}^{\text {nat }} \mathscr{L}(\lambda) &=\alpha+n \mathbb{E}_{\phi_{j}^{*}}\left[t\left(Z_{j}, x_{j}\right)\right]-\lambda \end{aligned}
$$

这样一个数据点就可以更新整个自然梯度。需要满足的前提是 noisy natural gradient 是无偏的。

## Black box VI


## Reference
- [Variational Inference: Foundations and Modern Methods PDF](https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf)
- [Variational Inference: Foundations and Modern Methods VIDEO](https://www.bilibili.com/video/av43405716/)