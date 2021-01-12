---
title: test equation
date: 2018-09-25 22:02:30
categories: 技术
tags: 
 - mathjax
 - swig
---

$x^3+y^3=z^3$

$$x^3+y^3=z^3$$

下面这个都不显示了

很关键的一点，选一个好一点的 mathjax CDN ， 格式什么的也会不一样，比如我现在这个公式就特别粗感觉……

```yml
math:
  enable: true
  engine: mathjax
  mathjax:
    # Use 2.7.1 as default, jsdelivr as default CDN, works everywhere even in China
    # cdn: //cdn.jsdelivr.net/npm/mathjax@2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML
    # For newMathJax CDN (cdnjs.cloudflare.com) with fallback to oldMathJax (cdn.mathjax.org).
    #cdn: //cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML
    # For direct link to MathJax.js with CloudFlare CDN (cdnjs.cloudflare.com).
    # cdn: //cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML
    # For automatic detect latest version link to MathJax.js and get from Bootcss.
    cdn: //cdn.bootcss.com/mathjax/2.7.1/latest.js?config=TeX-AMS-MML_HTMLorMML
```

根据自己主题的特性，看看怎么加合适吧。写我这个主题的人好像喜欢用 swig ， 貌似是一种可以调用 js 的高级语言，然后在合适的地方放上 `mathjax.swig`，只要保证后面配置的时候写对路径就行。最后一行看到了 `mathjax.cdn` 了吧，就是要用上面说的 `cdn` 。

```swig
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    extensions: ["tex2jax.js"],
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
      processEscapes: true
    },
    "HTML-CSS": { availableFonts: ["TeX"] }
  });
</script>

<script type="text/x-mathjax-config">
    MathJax.Hub.Queue(function() {
      var all = MathJax.Hub.getAllJax(), i;
        for (i=0; i < all.length; i += 1) {
          all[i].SourceElement().parentNode.className += ' has-jax';
        }
    });
</script>
<script type="text/javascript" src="{{ theme.math.mathjax.cdn }}"></script>
```

然后调用这个东西的 `math\index.swig` 长这样
```swig
{% if theme.math.enable %}
  {% set is_index_has_math = false %}

  {# At home, check if there has `mathjax: true` post #}
  {% if is_home() %}
    {% for post in page.posts %}
      {% if post.mathjax and not is_index_has_math %}
        {% set is_index_has_math = true %}
      {% endif %}
    {% endfor %}
  {% endif %}

  {% if not theme.math.per_page or (is_index_has_math or page.mathjax) %}
    {% if theme.math.engine == 'mathjax' %}
      {% include 'mathjax.swig' %}
    {% elif theme.math.engine == 'katex' %}
      {% include 'katex.swig' %}
   {% endif %}
  {% endif %}
{% endif %}
```

然后调用上面这个的 `post.swig` 长这样
```swig
{% extends 'includes/layout.swig' %}

{% block body %}
  {% include 'math/index.swig' %}
  <article id="post">
    <h1>{{ page.title || __('without_title') }}</h1>
    <p class="page-title-sub">
    ......
```

没错， `include` 的位置很关键，既不能太前面，也不能太后面。它会按顺序生成 `html`，每一句话就是一个小模块。所以说呀 `swig` 还是很高级的，跟它一个等级的还有 `ejs` 吧，大概。

然后在生成 `html` 之后，会先按照 `escape` 和 `em` 的规则自己替换一通转义符，导致后面有些公式符号也被替换了，`mathjax` 表示哭晕在厕所。所以要在自己的 `node_modules` 里面找到替换规则，换一个正则表达式，好跳过 "$$"、 "__" 这种公式必须的东西。

凭记忆找了一通，应该是这个！嗯！
```js
path: node_modules/kramed/lib/rules/inline.js

var inline = {
  math: /^\$\$\s*([\s\S]*?[^\$])\s*\$\$(?!\$)/,
  // escape: /^\\([\\`*{}\[\]()#$+\-.!_>])/,
  escape: /^\\([`*\[\]()#+\-.!_>])/,
  autolink: /^<([^ >]+(@|:\/)[^ >]+)>/,
  url: noop,
  html: /^<!--[\s\S]*?-->|^<(\w+(?!:\/|[^\w\s@]*@)\b)*?(?:"[^"]*"|'[^']*'|[^'">])*?>([\s\S]*?)?<\/\1>|^<(\w+(?!:\/|[^\w\s@]*@)\b)(?:"[^"]*"|'[^']*'|[^'">])*?>/,
  link: /^!?\[(inside)\]\(href\)/,
  reflink: /^!?\[(inside)\]\s*\[([^\]]*)\]/,
  nolink: /^!?\[((?:\[[^\]]*\]|[^\[\]])*)\]/,
  reffn: /^!?\[\^(inside)\]/,
  strong: /^__([\s\S]+?)__(?!_)|^\*\*([\s\S]+?)\*\*(?!\*)/,
  // em: /^\b_((?:__|[\s\S])+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
  em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
  code: /^(`+)\s*([\s\S]*?[^`])\s*\1(?!`)/,
  br: /^ {2,}\n(?!\s*$)/,
  del: noop,
  text: /^[\s\S]+?(?=[\\<!\[_*`$]| {2,}\n|$)/,
};
```

要是公式还是显示不出来，老乡学个前端吧-.-