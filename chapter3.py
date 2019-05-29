# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline 

# %% [markdown] {"toc-hr-collapsed": false}
# # 神经网络
#
# 结构包含：
# 1. 输入层
# 2. 中间层，也称隐藏层
# 3. 输出层

# %% [markdown] {"toc-hr-collapsed": true}
# ## 感知机与神经网络 
# - 感知机现在找到符合预期的输入与输出的权重是由人工进行的;可以使用神经网络能够从数据中学习到适合的权重。
# - 将输入信号总和转换成输出的函数一般称为激活函数。感知机和神经网络主要区别激活函数的不同。
# - 朴素感知机指单层网络，使用阶跃函数的模型。
# - 多层感知机就指神经网络， 即使用 simgoid等平滑激活函数的网络。
#
# 引入激活(阶跃)函数 $h(x)$ 得到感知机简化式：
# $$ y = h(b+w_1 x_1+w_2 x_2) $$

# %% [markdown] {"toc-hr-collapsed": false}
#
#
# ### 阶跃函数
# $$ h(x)= \begin{cases} 0\quad x\leq0 \\ 1\quad x>0\end{cases} $$

# %%
import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    ''' 阶跃函数实现演示
        x: a array of np
    '''
    return np.array(x>0, dtype=np.int)

print(step_function(np.array([-1, 2, 3])))

# %%
x = np.arange(-5, 5, 0.1)
y = step_function(x)
plt.ylim(-0.1, 1.1)
plt.plot(x, y)
plt.show()


# %% [markdown] {"toc-hr-collapsed": false}
# ### simgoid 函数
# > simgoid 是神经网络一种经常使用激活函数；
#
# $$ h(x) =  \frac{\mathrm{1} }{\mathrm{1} + e^{-x} } $$

# %%
def simgoid_func(x):
    return 1/(1+np.exp(-x))


# %%
x = np.arange(-5, 5, 0.1)
y = simgoid_func(x)
plt.ylim(-0.1, 1.1)
plt.plot(x, y)
plt.show()

# %%
