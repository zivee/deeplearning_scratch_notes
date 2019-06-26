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

# %% [markdown]
# # 神经网络的学习
#
# 本章目的
#
# * 损失函数
# * 梯度法
# * 用python实现对MNIST的手写数字数据集的学习
#
# ## 从数据中学习
#
# ### 感知机收敛定理 Novikoff
# 通过有限次数的学习，线性可分问题可解的。
#
# 学习方案
# * 机器学习：需要在转换为向量时使用的人工考虑合适的特征量，并用机器学习技术学习这些特征量的模式。
# * 深度学习(神经网络)也称为端到端的机器学习的；优点是对所有的问题都可以用同样的流程来解决。
#
# ### 训练数据和测试数据
#
# * 机器学习，把数据分为训练数据和测试数据（监督数据）两部分分别进行学习和验证等，为了正确评价模型的泛化能力。
# * 泛化能力是指处理未被观察数据的能力，泛化能力是机器学习的终极目标。
# * 机器学习应避免对数据集的过拟合的问题。
#
# ## 损失函数
#
# 损失函数(loss function)可以是任意函数，一般使用均方误差和交叉熵误差等；损失函数是神经网络性能的恶劣程度的指标，即指神经网络对监督数据有多大程度的不拟合。
#
# ### 均方误差 (mean squared error)
# $$ \mathit{E} = \frac{1}{2}\sum_k(y_k-t_k)^2 $$
#
# > $y_k$ 是表示神经网络的输出，$t_k$ 表示监督数据，k表示数据的维数

# %%
import numpy as np
import os, sys
# sources 本书练习源码目录
src = os.path.join(os.pardir, "sources")
sys.path.append(src)
from dataset.mnist import load_mnist


# %%
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

if __name__ == "__main__":
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    err = mean_squared_error(y, t)
    print(err)


# %% [markdown]
# ### 交叉熵误差 (cross entropy error)
# $$ \mathit{E} = -\sum_{k}t_{k}\log y_k$$
# > log是以e为底的自然对数 $\log_e$, ，$t_k$ 中正确解标签的索引为1，其他均为0（one-hot表示）,所以交叉熵误差是由正确标签所对应的输出y决定的

# %%
def cross_entropy_error(y, t, one_hot=True):
    """ 交叉熵误差函数，支持mini-batch
        Args:
            y: A 2d numpy.array，训练输出
            t: A 2d numpy.array, 监督数据
            one_hot: t是否为one_hot表示
        Returns:
            A float of error
    """
    # 适应mini-batch, 对单个数据reshape
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    # 加入微小值防止负无穷大
    delta = 1e-7
    batch_size = y.shape[0]
    if one_hot:
        return -np.sum(t * np.log(y+delta)) / batch_size
    else:
        # t为直接的标签，可以y[np.arange(batch_size), t] 获得正确解标签对应的神经网络的输出
        return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
    
if __name__ == "__main__":
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    err = cross_entropy_error(y, t)
    print(err)

# %% [markdown]
# ### mini-batch学习
#
# 计算损失函数时必须将所有的训练数据作为对象，但把所有数据作为对象计算损失函是不现实的，神经网络的学习会从训练数据中选择小批量(mini-batch)数据进行学习；这种方式称为**mini-batch学习**。
#
# 以交叉熵误差位列：
# $$ \mathit{E} = -\frac{1}{N}\sum_{n}\sum_{k}t_{nk}\log y_{nk}$$
# > 假设N个的数据集， $t_{nk}$指第n个数据的第k个监督数据；最后除以N 进行正规化，求得单个数据的**平均损失误差**

# %%
if __name__ == "__main__":
    (x_train, t_train), _ = load_mnist(normalize=True, one_hot_label=True)
    print("训练数据的形状", x_train.shape, t_train.shape)
    batch_size=10
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    print("随机索引", batch_mask)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print("随机选择mini-batch作为全体训练数据的近视值", x_batch.shape, t_batch.shape)


# %% [markdown]
#  > 在进行神经网络的学习时，不能将识别精度作为指标。因为如果以 识别精度为指标，则参数的导数(梯度)在绝大多数地方都会变为0。它的值 也是不连续地、突然地变化。作为激活函数的阶跃函数也有同样的情况。而sigmoid函数，不仅函数的输出（竖轴的值）是连续变化的，曲线的斜率（导数） 也是连续变化的。
#  
#  ## 数值微分

# %%
