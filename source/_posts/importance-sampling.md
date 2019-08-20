---
title: 重要性采样
date: 2019-03-21 19:28:16
categories: PRML
tags: 
 - sampling
 - Bayesian
author:
---

## 采样

求期望原本是个数学推导的过程。如果数学不好，推不出答案；或者问题压根没有闭式解，怎么样才能得到一个近似的结果呢？采样就可以做到。

采样，即人为地产生一些数据，使之服从目标分布。最简单的，如果想产生均匀分布，之间 `rand() % MAXSIZE` 就均匀地产生了 `[0, MAXSIZE)` 的随机数。

如果现在有一个任意的概率密度 $p(x)$，那怎么采样呢？比较一般的想法是，先求出 CDF，即积分得到分布函数 $F(x)$。它的值域一定在 $[0, 1]$。那我们从值域上均匀采样，再通过反函数求到对应的 $x=F^{-1}(y)$，也就能成功采样了。

![sample](sample.png)

学会了采样，求 $\mathbb{E}[f(x)]$ 就更加简单了：
$$
\mathbb{E}[f(x)] = \int_x f(x) p(x)dx = {1\over N}\sum_{x\sim p} f(x) = \overline {f(x)}
$$
即采样出服从 $p$ 的 $N$ 个 $x$，计算相应的 $f(x)$，再求平均。根据大数定理，平均值就是期望。

## 重要性采样

刚才的采样有一个非常重要的前提条件，那就是 CDF 可求，且可逆。如果遇到不会积分，分段函数，或者奇奇怪怪的情况，都会使得原来的做法太困难。

因此我们假设有一个我们易于掌控的概率函数 $q(x)$，它的 CDF 可求，且可逆，将原式化成：
$$
\mathbb{E}[f(x)]=\int_{x}\frac{p(x)}{q(x)} f(x)  q(x)d x
= {1\over N}\sum_{x\sim q} \frac{p(x)}{q(x)} f(x) = \overline {w(x) f(x)}
$$
其中$w(x)=\frac{p(x)}{q(x)}$。即采样出服从 $q$ 的 $N$ 个 $x$，计算相应的 $w(x)f(x)$，再求平均。

根据大数定理，平均值就是期望。

## 贝叶斯推理

在机器学习中，如果可以求出参数的期望，那么训练也就完成了。

$$
\mathbb{E}[\theta]=\int_{\theta|X}\theta~p(\theta|X) d\theta
$$

由于 
$$
p(\theta|X) = {p(X|\theta) p(\theta)\over p(X)}\propto p(X|\theta) p(\theta)
$$

我们通过 $p(X|\theta) p(\theta)$ 计算得到的 $p'$ 积分不为 1，而是一个难以计算的常数。真正的后验是$p$。设 $C = \int_{\theta} p' d\theta$，$p=p'/C$ 。

再通过重要性采样时，$\tilde w={p'\over q}=C{p\over q}$，一般地我们可以改写上面重要性采样的公式：

$$\begin{eqnarray}
\mathbb{E}[f(x)]&=&\int_x f(x) p(x)dx
=\int_{x} \frac{p'(x)}{C q(x)} f(x) q(x) d x \\
&=& {1\over C}{1\over N}\sum_{x\sim q} \frac{p'(x)}{q(x)} f(x)
= {1\over C}\overline {\tilde w(x) f(x)}
\end{eqnarray}$$

C 同样可以采样得到：
$$
C = \int_x p'(x) dx = \int_x {p'(x)\over q(x)} q(x) dx = {1\over N}\sum_{x\sim q} {p'(x)\over q(x)} = \overline{\tilde w(x)}
$$

因此
$$
\mathbb{E}[f(x)] = {\overline {\tilde w(x) f(x)}\over \overline{\tilde w(x)}}
$$

回到贝叶斯
> $$f(\theta) \leftarrow \theta$$
$$\tilde w(\theta) \leftarrow {p(X|\theta) p(\theta)\over q(\theta|X)}$$

$p(X|\theta)$ 是模型假设；$p(\theta)$ 和 $q(\theta)$ 是形式相同的两个分布。$q$ 可能会根据输入数据的特性来调整。

参见下面的例子：

![example](example.png)

## 避免溢出下界

可以先求对数，等需要的时候再算指数。

另一方面，在求平均数之前，势必已经算出了各个数字。如果每个数字都除掉了最大数，那么在最后的式子里不受影响。

$$
\mathbb{E}[f(x)] = {\overline {\exp(\log\tilde w(x)-M) f(x)} \over \overline{\exp(\log\tilde w(x)-M) }}
$$
$$M = max(\log\tilde w(x))$$

## q 的选取

$q$ 在 $p$ 高密度的地方也要尽量高密度，这样才能真实还原数据的性质。

我看的几个 exerise 里面 $q$ 都取和先验一样的形式。一般以 ESS 来衡量其稳定性：

$$
E S S=\sqrt{\frac{1}{N} \sum_{i=1}^{N}\left(\frac{\tilde{w}\left(X_{i}\right)}{\overline{w}}-1\right)^{2}}
$$

当 $q$ 本身的方差在合理的范围内，ESS 才会比较小。下图 c 衡量了方差。

![ESS-q](ESS-q.png)

## Reference

[Importance Sampling](http://dept.stat.lsa.umich.edu/~jasoneg/Stat406/lab7.pdf)