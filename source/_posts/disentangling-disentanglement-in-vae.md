---
title: "Disentangling Disentanglement in Variational Autoencoders"
date: 2019-11-20 20:08:04
tags: 
 - VAE
 - disentanglement
categories: PRML
---

对于深度生成模型而言，数据的可解释性具有重要的意义。
如果可以解释 VAE 在编码以后的成分，那么有选择性地改变隐层的部分数据，就可以指导性地生成不同的观测（图片、音频等）。
由于在隐层中的表示并不是我们想象中那么独立的，实际上需要做一些变换，使得不同的特征可以解开纠缠（Disentangling Disentanglement）。

## Main Points

这篇论文主要提出了以下几个观点：

- 分解（decomposition）得好需要满足两个方面
- 理论分析了 $\beta$-VAE 可以满足第一个方面，但不能满足第二个方面
- 一种新的 $\beta$-VAE 的构造可以同时满足这两个方面

## Decomposition: A Generalisation of Disentanglement

**分解**（Decomposition）是一种广义上的 **解纠缠**（Disentanglement）.

对于解纠缠来说，要求一个比隐层更为抽象的真实空间的各个 **generative factors (GF)** 都是相互独立的。这个条件比较苛刻。

对于分解而言，并不要求 GF 相互独立。只要它在隐层的空间中可以学习出一种隐含的结构就够了。这种结构一般都由先验来显式地表示。

> **Tips on GF**  
经过 VAE 的编码，可以生成我们的隐空间。在隐空间上，可以再抽象出一个 GF 空间。理想状况下，我们希望隐空间可以解释所有的 GF 空间。比如隐空间的向量 $h_1,h_2$ 在调整数值后可以改变袖子的长短。而袖子的长度即是一种 GF。而实际上，往往没有这么好的隐表示，通常都会有一个隐向量 $h_1$ 改变后，多个 GF 受到了影响。为了更接近我们的理想，需要对隐空间做一些结构化的假设，以学得更好的隐表示。

分解的表现如何，可以用两个方面来衡量：
- The latent encodings of data having an appropriate level of **Overlap** 
- The aggregate encoding of data conforming to a desired **Structure**, represented through the prior.

![src: github.com/iffsid/disentangling-disentanglement](disentangling-disentanglement-in-vae/two-factors.png)

这两个因素是相辅相成的。如果没有 overlap，分布之间太稀疏，将变为查表，导致隐层含有的信息太少。如果没有 structure，隐层将不会按照特定的结构化分解。

还有一个值得注意的地方是，在 VAE 中我们总是会假设变分分布是各向同性的高斯分布，但从实验结果来看这并不利于解纠缠。

## Decomposition Analysis of $\beta$-VAE

## Deconstruct $\beta$-VAE in a new way

## Reference 

- [Disentangling Disentanglement in Variational Autoencoders, ICML 2019](http://proceedings.mlr.press/v97/mathieu19a.html)