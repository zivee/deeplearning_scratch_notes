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
# # 第一章
#
# > 首行加入__%matplotlib inline__ sets the backend of matplotlib to the 'inline' backend
#
# 本章主要介绍python基础知识， 介绍2个主要的库：
#
# 1. Numpy
# 2. Matplotlib
#
# ## Numpy
# 这里介绍使用numpy用于数组、矩阵的算术运算；数学上称一维数组为向量， 二维数组为矩阵， 并一般化后的向量和矩阵等统称为张量 tensor。
# numpy的广播功能，实现不同形状的数组之间进行运算；
#
# ## Matplotlib
# 使用matplotlib库用于实验中，对数据进行可视化和图形的绘制；

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# %%
# 生成数据
x = np.arange(0, 6, 0.1)  # 以0.1为单位，生成0到6的数据
y = np.sin(x)

# 绘制图形
plt.plot(x, y)
plt.show()

# %%
print(os.path.realpath('./')) 

# %%
img = imread('../sources/dataset/lena.png')
plt.imshow(img)
plt.show()
