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
# - 感知机现在找到符合预期的输入与输出的权重是由人工进行的;可以使用神经网络能够从数据中学习（训练）到适合的权重。
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
import os, sys
# sources 本书练习源码目录
src = os.path.join(os.pardir, "sources")
sys.path.append(src)
from dataset.mnist import load_mnist
import pickle


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
# > 机器学习的问题大致分为回归问题和分类问题。
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


# %% [markdown]
# ## 输出层设计
#
# ### softmax函数
#
#
# $$ y_k = \frac{e^{a_k}}{ \sum_n^{k=1}e^{a_i}} $$
# $y_k$ 表示计算第看个神经元的输出，分母是所有输入信号的指数（e是纳皮尔常数2.7182...）和，可以看出输出层受到各个输入信号的影响。
# 公式在运算时存在溢出的风险如 $e^{20000}$,所以为了避免在分子和分母同时乘以常数。改进如下：
#
# $$
# y_k = \frac{e^{a_k}}{ \sum_n^{k=1}e^{a_i}} = \frac{Ce^{a_k}}{C \sum_n^{k=1}e^{a_i}}= \frac{e^{(a_k+C')}}{ \sum_n^{k=1}e^{(a_i+C')}}
# $$
#
# 1. softmax函数输出是0 到 1之间的实数。
# 2. 输出总和等于1。
# 3. 可以把softmax函数的输出解释为*概率*。
# 4. 使用softmax函数不会改变神经元的位置。
# 5. 输出层的softmax可以被省略。
# 6. 分类问题的，输出层的神经元数量是设定为类别数量。
#

# %%
# softmax 实现，为防止溢出，通过减去输入信号的最大值c；
def softmax(a):
    """
        arg:
        a: a numpy.array 输入
    """
    axis =None
    if a.ndim == 2:
        a=a.T # 行列倒置
        axis=0
    c = np.max(a, axis=axis)
    # y = np.exp(a - c) / np.sum(exp(a-c))
    exp_a = np.exp(a-c)
    return (exp_a/np.sum(exp_a, axis=axis)).T

a = np.array([[0.100, 0.200, 1.900], [0.150, 0.300, 2.900]])
print(softmax(a))


# %% [markdown]
# ## 手写数字识别
#
# > 使用mnist手写数字图像集进行实验
#
# 知识点：
# * 求解机器学习问题的步骤分学习(训练)和推理两个阶段
# * 推理处理
# * 识别精度(accuracy)；评价推理处理的识别精度
# * 正规化和预处理；实验中对图像素除以255，使得数据值在0.0～1.0的范围内
# * 输入数据的集合称为批
#

# %%
def get_data():
    # 获取测试数据集， 并正规货预处理
    _, (x_test, t_test) = load_mnist(normalize=True, flatten=True)
    print(x_test.shape)
    print(t_test.shape)
    return x_test, t_test

def load_network():
    with open('sample_weight.pkl', 'rb') as fn:
        network = pickle.load(fn)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    print(W1.shape)
    print(b1.shape)
    A1 = np.dot(x, W1)+b1
    Z1 = sigmoid_func(A1)
    
    A2 = np.dot(Z1, W2)+b2
    Z2 = sigmoid_func(A2)
    
    A3 = np.dot(Z2, W3) + b3
    
    print(A3.shape)
    return softmax(A3)

network = load_network()
x_test, t_test = get_data()

ret = np.argmax(predict(network, x_test), axis=1)
accuracy_cnt = np.sum(ret == t_test)
print("accuracy: ", float(accuracy_cnt)/len(x_test))

# %%
