---
title: "超简单的标签云"
date: 2021-01-12 21:32:15
categories: 技术
tags:
 - tag
 - theme
photos:
author:
---

也许有人中意那种翻滚的标签云。但我的要求比较朴素，能够看到全部标签，并且将高频出现的放大即可。于是开启 `F12` 果然找到了`item.length` 就是某个 tag 出现的频次。

于是上了最简单的线性表达式——

```js
style="font-size: calc(<%-item.length%>px * 3 + 12px)"
```

甚至简单地令人失望……
<a class="link-item" title="Tags" href="/tag">
<i class="iconfont icon-tags">lovely cloud tag</i>
</a>