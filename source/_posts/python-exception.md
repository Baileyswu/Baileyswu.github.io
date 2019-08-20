---
title: Python 异常与断言
date: 2019-01-06 16:25:04
categories: 技术
tags:
 - Python
 - Exception
author:
---

## 为什么

编写自动化验证程序可以有效加快产品迭代和开发。

## 断言

测试程序时经常使用的是 assert，示例如下

```py
a = 2
b = 0
assert b != 0
print(a/b)
==================================================
Traceback (most recent call last):
  File "exp.py", line 4, in <module>
    assert b!= 0
AssertionError
```

测试不成功时，会触发 `AssertionError`，中断测试。

## 异常

在产品上线时，光是检测出问题是不够的。还需要在程序异常时，提供必要的解决方案，使得程序可以绕过错误继续运行下去。

```py
try:
    print(a/b)
except Exception as e:
    print("There is an exception")
    print(e)
else:
    print("Lucky no exception")
finally:
    print("Finally run out")
print("Assure never die")
==================================================
There is an Exception
division by zero
Finally run out
Assure never die
```

异常有多种类别，这个异常就是 `ZeroDivisionError`，因此可以具体到
```py
...
except ZeroDivisionError as e:
    print("There is a ZeroDivisionError")
    print(e)
except Exception as e:
    print("There is an exception")
    print(e)
...
===================================================
There is a ZeroDivisionError
division by zero
Finally run out
Assure never die
```

此时多个 `except` 并列，起到 `if ...` `else if ...` 的效果。

如果没有抛出异常，还可以跑 `else` 下面的语句。

```py
a = 2
b = 1
===================================================
2.0
Lucky no exception
Finally run out
Assure never die
```

还可以自己定义异常，暂时还没用到……

## 区别

两者都可以在测试中用到。断言的写法比较简洁，但只会提示错误，然后 go die。后者可以在异常发生时，继续处理程序。