---
title: 进击的DP
date: 2018-10-27 13:32:11
categories: ACM
tags:
 - DP
 - ACM
 - 背包
 - RMQ
---

## 直观的DP
[POJ 1163](http://poj.org/problem?id=1163) 


7  
3   8  
8   1   0  
2   7   4   4  
4   5   2   6   5  


正向思维，把每一层的所有状态全都算出来，再给下一层。题目是不难，但这个图是比较直观的反映了 DP ，解决问题时如果可以把层次关系描述出来可以有效帮助做题。

其中只有根是已知的，每个状态的转移方式也是已知的。  
从根到叶是**正向思维**。顺次可以得到所有的儿子的状态。当所有状态数量是有限的时候，正向的规模被限制，可以尝试求解。  
但有时只需要求其中一个儿子的状态，如果还是从根节点出发一一求出状态则显得有点浪费。所以此时应该**逆向思维**。  
当上一层对下一层状态如树状（类似于分治），则非常理想，从儿子看，他只需要知道父亲的状态，而不需知道伯伯的状态，依次到达根节点。但当状态有交叉时，则要求出相关联的伯伯的状态。**逆向求解有助于减少运算的规模**。

根据情况初始化。有时可能只能以角落（如第一行第一列的第一个格子）为第一层，有时可能以边界为第一层（比如第一行所有格子）。在层与层的转换时，注意剪枝，即本层要达到状态 s，则前一层必须达到 s', 达不到 s' 的状态可以不考虑。

[走格子1](https://vjudge.net/problem/CodeForces-429B)

一个人要从左上角往右下角走，每次只能向左或向下走一格；另一个人要从左下角往右上角走，每次只能向右或向上走一格。两条路径交叉于一点。求两条路径上的点数之和（不包括交叉点）。

<details>
  <summary>显示解答</summary>
![](http://ww1.sinaimg.cn/mw690/006aIx0Cgw1f673xu77irj30bp062wez.jpg)
```cpp
dp[i][j-1][0] + dp[i-1][j][1] + dp[i][j+1][2] + dp[i+1][j][3] 
dp[i][j-1][3] + dp[i-1][j][0] + dp[i][j+1][1] + dp[i+1][j][2]
```
</details>

[走格子2](https://vjudge.net/problem/HDU-5492)  

给一个 N * M (N, M < 31)的矩阵，从 (1,1) 到 (N,M) 经过的格点分值分别为 Ai，(0 <= Ai < 31) 路径只能向右或向下走,共N+M-1步)。求各种路径中方差最小的路径。

<details>
  <summary>显示解答</summary>
其中$A_{avg}={1\over {N+M-1}}\sum A_i$,  
则$\sum 2A_iA_{avg}=2(N+M-1)A_{avg}^2$
$$原式=(N+M−1)[(\sum A_i^2)-(N+M-1)A_{avg}^2]$$$$=(N+M−1)\sum A_i^2-(\sum A_i)^2$$

```cpp
dp[i][j][s]
dp[i-1][j][s-a[i][j]] + a[i][j] * a[i][j]
dp[i][j-1][s-a[i][j]] + a[i][j] * a[i][j]
```
</details>

![](https://gss0.baidu.com/94o3dSag_xI4khGko9WTAnF6hhy/zhidao/pic/item/55e736d12f2eb938b3e61ca1dd628535e4dd6fdd.jpg)

## 背包——DP中的经典
背包系列问题是DP入门的经典。一言以蔽之：
`n`件物品，第`i`件体积为`v[i]`, 价值为`w[i]`, 件数为`k[i]`
$F_i(j)：前i个物品放进容积为j的容器里的最大价值（逐步达到最大）$

### 0-1背包
**所有k[i]均为1**

<details>
  <summary>显示公式</summary>
$$F_i(j)=
\begin{cases}
F_{i-1}(j) &&\text{$V_i > j >= 0$}\\
max\{F_{i-1}(j),F_\color{red}{i-1}(j-V_i)+W_i\} &&\text{$C>=j>=V_i$}
\end{cases}
$$
</details>
<details>
  <summary>显示代码</summary>
```cpp
for(int i = 0;i < n;i++)
    for(int j = v[i];j <= C;j++)
        dp[j] = max(dp[j], dp[j-v[i]]+w[i]);
```
</details>
<details>
  <summary><b>滚动数组</b></summary>
```cpp
for(int i = 0;i < n;i++) {
    memcpy(temp, dp, sizeof(dp));
    for(int j = v[i];j <= C;j++)
        dp[j] = max(dp[j], temp[j-v[i]]+w[i]);
}
```
</details>
<details>
  <summary>显示代码</summary>
```cpp
for(int i = 0;i < n;i++)
    for(int j = C;j >= v[i];j--)
        dp[j] = max(dp[j], dp[j-v[i]]+w[i]);
```
</details>

### 完全背包
**所有k[i]=inf**
<details>
  <summary>显示公式</summary>
$$F_i(j)=
\begin{cases}
F_{i-1}(j) &&\text{$V_i > j >= 0$}\\
max\{F_{i-1}(j),F_\color{red}{i}(j-V_i)+W_i\} &&\text{$C>=j>=V_i$}
\end{cases}
$$
</details>
<details>
  <summary>显示代码</summary>
```cpp
for(int i = 0;i < n;i++)
    for(int j = v[i];j <= C;j++)
        dp[j] = max(dp[j], dp[j-v[i]]+w[i]);
```
</details>

### 多重背包
**k[i]>=1**
<details>
  <summary>显示解答</summary>
1. 化为0-1背包，则共有 $\sum k_i$ 件物品，当物品件数多时可能会超时。  
2. 二进制拆分。将k拆为 $1,2,4,...,2^p, k-2^{p+1}+1$,  
p 为最大的满足该式的值。可知它们可以组合出在 1 到 k 范围内的所有数字。则只要将一件物品拆分成这样的 p+2 件物品，化为 0-1 背包。**注意每件物品重量、价值等比例扩大。**
```cpp
for(int i = 0;i < n;i++)
{
    int m = 1;
    while(k[i] > 0)
    {
        if(m > k[i]) m = k[i];
        k[i] -= m;
        for(int j = C;j >= v[i];j--)
            dp[j] = max(dp[j], dp[j-v[i]]);
        m <<= 1;
    }
}
```
3. 单调队列优化
</details>

### 变形背包
- [倒水](https://vjudge.net/problem/CodeForces-730J)

有 n 个瓶子，各有水量和瓶体积。把水从一个瓶倒到另一个瓶。首先要使得最后不空的瓶子数最少，其次要倒水量最少。求瓶子数和倒水量。

<details>
  <summary>分析</summary>
1. 确定瓶子数。  
对瓶子的体积排序，前 km 个瓶子体积 V 恰好不小于总水量之和 wt ，则 km 即为最少的瓶子数。  
2. 确定倒水量  
`dp[i][j][k]`表示前 i 个瓶子选取 k 个（且第 i 个为所选第 k 个），使得 k 个瓶子体积和为 j ，可以容纳的最大水量。  
先求出在 `dp[n-1][wt~V][km]` 的 `max`，再用 `wt-max` 即答案。  

由于轮换，可降维到 `dp[j][k]`。  
另外通过 `reach[j][k]` 表示是否可到达该状态。
`j`, `k` 的两层循环位置可调换，答案不变。但是一种比另一种速度快一倍，这个问题组原课有解释。  
由于是0-1背包，须注意 `j`, `k` 是循环递减来遍历，否则就是完全背包了。  
</details>

## 状态压缩

当状态的内部顺序对于结果没有影响时，可以将多个状态进行状态压缩简化推理。

例题：在一个 n * n （n < 17）的矩阵里选择 n 个数字（两两不同行、列），求数字和最大是多少。


<details>
  <summary>分析</summary>
逐行考虑。在考虑第 i 行的第 j 列 是否要取时，先观察第 j 列 上是否已经取了数字。至于第 j 列上取的数字是第几行的，则不需要考虑。所有关于第 j 列上放了数字的状态，都可以压缩到一起。
</details>

<details>
  <summary>压缩</summary>
定义：逐行选择，并且用一个n位的二进制数表示各列的选择情况。比如00101表示已经选择了两行，第三列、第五列被选择了。   
a[i][j] 表示第i行、第j列的数值；   
F[s] 表示状态 s （用二进制表示）选取的最大值。

递推关系：每个二进制状态从前几个相关状态转换而来。比如01101由00101、01001、01100转化过来，即：   
```cpp
F[01101] = max(F[00101]+a[3][2], F[01001]+a[3][3], F[01100]+a[3][5])   
```
初始每个F都是0；   
最后要求的就是F[111111(n-1个1)]。   
s 可顺次枚举下去，不用担心子状态还未被计算。因为每个s某位少一个1时，必小于s，即该状态已经得到。

</details>

- [求三角形面积和的最大值](https://vjudge.net/problem/HDU-5135)

给 n（n<13）条边，每条边只能用一次，拼成多个三角形，求三角形面积和的最大值。
<details>
  <summary>分析</summary>
1~2^12 只有 4096，对每个数二进制分解，第 i 位为 1 则用这条边。 
预处理出所有的数位和为 3 的倍数的状态。 
</details>

## 区间DP

求取一个区间内的有效信息，可以借助于子区间得到。分割子区间的方式，既可以是多次分割取其中一种，也可以是就分割一次。类似于最前面提到的，有树状的，也有非树状的。

例如区间 [1,5] 上的数的最大公约数，只需要分割一次，求 gcd[gcd[1, 3], gcd[4, 5]] 即可。分割的子区间没有交集。


### 多分割

- [括号匹配1](http://poj.org/problem?id=2955)  

求最多的括号匹配数目。

<details>
  <summary>显示解答</summary>
枚举每种区间长度。
`dp[i][j]`代表`str[i...j]`区间内最多的合法括号数

状态转移方程：

```cpp
if((str[i]=='(' && str[j]==')') || (str[i]=='[' && str[j]==']'))
     dp[i][j] = dp[i+1][j-1] + 2;
for(int k = i;k < j;k++)
    dp[i][j] = max{ dp[i][k] + dp[k+1][j] };
```
</details>


- [括号匹配2](http://poj.org/problem?id=1141)  

补全最少的括号使得所有括号匹配。

<details>
  <summary>显示解答</summary>
  `v[i][j]`记录`dp[i][j]`取最大值时的情况：  
-1表示`str[i]`与`str[j]`相互匹配；其余表示`dp[i][j]`取最大值时的k值，`str[i]`与`str[k]`相互匹配。

打印过程递归：

```cpp
void draw(int x, int y){
    if(x > y) return;
    if(x == y){
        if(str[x] == '(' || str[x] == ')') printf("()");
        else printf("[]");
        return;
    }
    if(v[x][y] == -1){
        printf("%c", str[x]);
        draw(x+1, y-1);
        printf("%c", str[y]);
        return;
    }
    draw(x, v[x][y]);
    draw(v[x][y]+1, y);
}
```
</details>

### 单一分割

#### 线段树

略略略

#### RMQ 

查询O(1)

- [区间的最大公约数](http://acm.hdu.edu.cn/showproblem.php?pid=5726)

给N($N\le 100000$)个数，Q($Q\le 100000$)个询问，每次查询输出区间的最大公约数，以及最大公约数为这个数的区间数目。

<details>
  <summary>显示解答</summary>
  查询次数很多，要做预处理，用map<最大公约数,区间数>存下来，实现O(1)的查询。  
预处理发现对于同一左端点的区间而言，右端点越靠右，区间gcd单调递减。因此可以固定左端点，二分右端点，找到gcd突变的右端点，确定对于同一gcd的区间数目。为了查询得更快，用ST表(RMQ)存下区间gcd，只要O(1)即可查询得到区间gcd。

对于这题还不够，需要继续进行优化。
</details>

- 建表

$dp[i][j]$表示长度为$2^i$的区间$[i,i+2^j-1]$范围内的$gcd$  
$$dp[i][j]=gcd(dp[i][j-1], dp[i+2^j][j-1])$$  
即区间$[i,i+2^{j-1}-1]与[i+2^{j-1},i+2^j-1]$共同确定了区间$[i,i+2^j-1]$的$gcd$  

```cpp
for(int i = 0;i < n;i++){
        dp[i][0] = a[i];
}
for(int j = 1;(1<<j) <= n;j++){
    for(int i = 0;i + (1<<j) - 1 < n;i++){
        dp[i][j] = __gcd(dp[i][j-1], dp[i+(1<<j-1)][j-1]);
    }
}
```
- 询问

$[l,r]$由$[l,...]与[...,r]$共同决定。两者必须共同覆盖了整个区间。区间长度为$2^k$，其中$k=log_2(r-l+1)$

```cpp
int ask(int l, int r){
    int k = log(1.0*(r-l+1))/log(2);
    return __gcd(dp[l][k], dp[r-(1<<k)+1][k]);
}
```

## 树状DP

树已经构造好了，如何利用父亲节点的已知信息进行DP。遍历更新深度时即是一种应用：
```cpp
dp[i] = dp[parent[i]] + 1
```

- [深度的期望](https://vjudge.net/problem/CodeForces-697D)

- [树上转移](https://vjudge.net/problem/CodeForces-697C)


## 与贪心的关系

贪心需要论证，保证一步优，步步优。如果要证明贪心是有问题的，举一个 DP 的反例即可。

如果 DP 的过程中发现了每次都选了局部最优，则很有可能该问题是可以贪心解决的。