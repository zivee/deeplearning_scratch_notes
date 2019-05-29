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

# %% [markdown]
# # 感知机
#
# 感知机接收多个输入信号， 输出一个信号。输入的信号被送往神经元时， 会被分别乘以固定的权重。神经元会计算传送过来的信号总和， 只有当这个中和超过了某个界限值(*阀值*)时， 才会输出1。这也称为_神经元被激活_。
#
# 感知机的运行原理，就是式：
# $$ y= \begin{cases} 0\quad(w_1x_1+w_2x_2 \leq\theta) \\ 1\quad(w_1x_1+w_2x_2 >\theta)\end{cases} $$
#

# %% [markdown]
# ## 与门实现
#
# 阀值取反为b 称为偏值并决定了神经元激活的容易程度， w1 和 w2为权重说明各个输入的重要性。异或门因为感知机局限性无法实现， 感知机局限性就在于它只能表示用直线分割的空间。
#
# > 感知机支持 与门， 与非门， 或门 是具有相同构造的感知机， 区别只在于权重参数的值。
#
# $$ y= \begin{cases} 0\quad(b+w_1x_1+w_2x_2 \leq0) \\ 1\quad(b+w_1x_1+w_2x_2 >0)\end{cases} $$

# %%
import numpy as np


# b 称为偏值并决定了神经元激活的难易成都， w1 和 w2为权重说明各个输入的重要性。
def AND(x1, x2):
    w1, w2 = 0.5, 0.5
    b = -0.7
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
print(AND(1, 1))
print(AND(1, 0))
print(AND(0, 0))

# %%
def NAND(x1, x2):
    w1, w2 = -0.5, -0.5
    b = 0.7
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
print(NAND(1, 1))
print(NAND(1, 0))
print(NAND(0, 0))

# %%
def OR(x1, x2):
    w1, w2 = 0.5, 0.5
    b = -0.2
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
print(OR(1, 1))
print(OR(1, 0))
print(OR(0, 0))


# %% [markdown]
# ## 多层感知机
#
# 感知机可以进行叠加，叠加多层的感知机也称为多层感知机；由于单层感知机的局限性（无法分离非线性的空间）无法实现异或门，将组合多层感知机实现异或门。
#
# ### 异或门的实现

# %%
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    
    return AND(s1, s2)


print(XOR(1, 1))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(0, 0))

# %% [markdown]
# 通过叠加层（加深层），感知机能进行更加灵活的表示。

# %% [markdown]
# ## 重点
#
# 1. 感知机算法；
# 1. 单层感知机，权重和偏置设为参数；
# 1. 多层感知机，可以表示非线性空间。
# 1. 多层感知机表示计算机
