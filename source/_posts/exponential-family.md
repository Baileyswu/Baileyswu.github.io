---
title: 指数族分布
date: 2019-11-25 16:37:59
categories: PRML
tags:
 - 指数族分布
 - 共轭
photos:
author: Danliwoo
---
指数族分布形式为
$$p(x | \eta) = h(x)g(\eta) \exp \{\eta^T u(x)\}~~~~~~~~(1)$$
总满足
$$g(\eta) \int h(x) \exp \{\eta^T u(x)\} dx = 1~~~~~~~~(2)$$

- 变量 $x$ 可以为离散或者连续的标量或向量
- $\eta$ the natural parameter 自然参数
- $u(x)$ the sufficient statistics 充分统计量
- $g(\eta)$ the normalizer 归一化系数
- $h(x)$ the base density

有时候也把 $g(\eta)$ 求对数放在指数项里。许多分布均为指数族分布，如二项分布、多项分布、高斯分布等，调整后可以化出以上的标准形式。因此只要讨论指数族分布的一般性质，可以将它应用到多种分布之中。

### 例子：一元高斯分布

$$
\begin{array}{l}
p\left(x | \mu, \sigma^{2}\right)&=&\frac{1}{\left(2 \pi \sigma^{2}\right)^{\frac{1}{2}}} \exp \left\{-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right\} \\ 
&=&\frac{1}{\left(2 \pi \sigma^{2}\right)^{\frac{1}{2}}} \exp \left\{-\frac{1}{2 \sigma^{2}} x^{2}+\frac{\mu}{\sigma^{2}} x-\frac{1}{2 \sigma^{2}} \mu^{2}\right\}
\end{array}
$$

$$
\begin{array}{c}
\eta&=&\left(\begin{array}{c}{\frac{\mu}{\sigma^{2}}} \\ \frac{-1}{2\sigma^{2}}\end{array}\right) \\
u(x)&=&\left(\begin{array}{c}{x} \\ {x^{2}}\end{array}\right) \\
h(x)&=&(2 \pi)^{-\frac{1}{2}} \\
g({\eta})&=&\left(-2 \eta_{2}\right)^{\frac{1}{2}} \exp \left(\frac{\eta_{1}^{2}}{4 \eta_{2}}\right)
\end{array}
$$

凑参数的顺序一般是：
1. 先分解出指数上 $u(x)$ 和对应的系数 $\eta$
2. 将非指数的系数中不包括 $\eta$ 的部分写成 $h(x)$
3. 剩余带 $\eta$ 的部分写入 $g(\eta)$
4. 归一化，将剩余的系数都乘入 $g(\eta)$

## 求期望

对 (2)式 关于 $\eta$ 求导，即可得
$$−\nabla \ln g(\eta) = E[u(x)]$$
（具体的推导可以看Reference）。因此如果可以算出归一化项 $g(\eta)$，那就可以用它的梯度来计算统计量的期望。  $u(x)$ 的协方差可以根据 $g(\eta)$ 的二阶导数表达，对于高阶矩的情形也类似。

## 充分统计量

当有多个独立样本时，对 (1)式 求最大似然下的参数
$$−\nabla \ln g(\eta _{M L} ) ={1\over N}\sum u(x_n )$$
可以根据充分统计量来计算归一化项的梯度。在伯努利分布中有 $u(x)=x$，在高斯分布里有 $u(x)=(x,x^2)^T$，只需计算出数据集中的这些量，就可以代替整个数据集去估计参数。因此称之为充分统计量。

## 共轭先验

一般情况下，对于一个给定的概率分布 $p(x|\mu)$，我们能够寻找一个先验 $p(\eta)$ 使其与似然函数共轭，从而 *后验分布的函数形式与先验分布相同*，因此使得贝叶斯分析得到了极大的简化。

多项式分布的参数的共轭先验是狄利克雷分布 (Dirichlet distribution)，而高斯分布的均值的共轭先验是另一个高斯分布。所有这些分布都是指数族 (exponential family) 分布的特例。

### 共轭贝叶斯推理

假设有先验形式为
$$p_0(x | \eta) = h(x)g_0(\eta_0) \exp \{\eta_0^T u(x)\}$$
似然形式为
$$p_l(x | \eta) = \exp \{\lambda^T u(x)\}$$
则后验的形式为
$$p(x | \eta) = h(x)g(\eta) \exp \{\eta^T u(x)\}$$
其中 $\eta=\eta_0+\lambda$，$g(\eta)$ 为凑出来的归一化项。




## Reference

Bishop CM (2006) In: Pattern Recognition and Machine Learning, Springer, chap 2.