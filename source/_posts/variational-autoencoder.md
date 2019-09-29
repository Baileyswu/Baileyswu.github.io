---
title: 变分自编码
date: 2019-09-29 22:47:10
categories: PRML
tags:
 - VAE
 - VI
---

## 编码

编码是从高维信息降维得到低维信息的过程。

自编码器是输入 X 进行编码后得到 Y，再利用 Y 解码得到 X'.对比 X 与 X' 的误差，利用神经网络训练使得误差逐渐减小，达到非监督学习的目的。可以认为是非线性的 PCA。

## 解码

用深度生成式网络进行解码时，往往会假设
$$
\begin{array}{l}{p(\mathbf{z})=\operatorname{Normal}(0,1)} \\ {p(\mathbf{x} | \mathbf{z})=\operatorname{Normal}\left(\mu_{\beta}(\mathbf{z}), \sigma_{\beta}^{2}(\mathbf{z})\right)}\end{array}
$$
where
$$
\mu \text { and } \sigma^{2} \text { are deep networks with parameters } \beta
$$

