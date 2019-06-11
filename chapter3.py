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
def sigmoid_func(x):
    return 1/(1+np.exp(-x))


# %%
x = np.arange(-5, 5, 0.1)
y = sigmoid_func(x)
plt.ylim(-0.1, 1.1)
plt.plot(x, y)
plt.show()

# %% [markdown]
# ### 神经网络的内积
# 使用矩阵运算 numpy.dot 一次性完成计算实现神经网络

# %%
# 忽略 偏置和激活函数, 一次性计算Y结果
X = np.array([1, 2])
W = np.array([[1,3,5], [2,4,6]])
print(X.shape)
print(W.shape)
Y = np.dot(X, W)
print(Y)


# %% [markdown]
# 加上加权后表示如下
#
# $$ A^{(1)}=W^{(1)}X^{(1)}+B^{(1)} $$
# 并且偏置（加权）数量取决于后一层的数量, 如后一层有3个神经元
#
# $$ B^{(1)}=(b_1^{(1)},b_2^{(1)},b_3^{(1)}) $$
#
# ### 恒等函数
# 恒等函数(sigma)为一无任何作用的函数：它总是传回和其引数相同的值。这里将作为输出层的激活函数 $\sigma()$ 
#
# > 一般 回归使用 sigma恒等函数；二元分类使用sigmoid函数；多元分类使用softmax函数。
#

# %%
def identity_function(x):
    return x

# 总结实现3层神经网络
def init_network():
    network = {}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['W2']=np.array([[0.1, 0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])
    return network

def forward(network, x):
    """传递处理
    arg:
        network: 权重和偏置参数;a numpy.array
        x: 输入; a numpy.array
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    A1 = np.dot(x, W1)+b1
    Z1 = sigmoid_func(A1)
    
    A2 = np.dot(Z1, W2)+b2
    Z2 = sigmoid_func(A2)
    
    A3 = np.dot(Z2, W3) + b3
    Y = identity_function(A3)
    return Y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

# %%
