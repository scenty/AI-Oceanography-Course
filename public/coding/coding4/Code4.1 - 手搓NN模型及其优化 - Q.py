# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 14:13:39 2023

@author: Scenty
"""
import numpy as np
import time
import netCDF4 as nc
from matplotlib.pyplot import *

# 生成数据集的函数
def get_data():
    #read dataset 1
    Dataset=nc.Dataset(r'../Data/ERA5_bulk_wave.nc')
    swh=Dataset['swh'][:]
    lon=Dataset['longitude'][:]
    lat=Dataset['latitude'][:]
    Dataset.close()
    gswh=np.ma.mean(swh,axis=(1,2)).filled(np.nan)
    year=np.arange(1940+1/24,2026+5/24,1/12)
    map_aspect = 1/np.cos(np.deg2rad(np.mean(lat)))
    #read dataset 2
    Dataset=nc.Dataset(r'../Data/ERA5_bulk_wind.nc')
    u10=Dataset['u10'][:]
    v10=Dataset['v10'][:]
    ws=np.sqrt(u10**2+v10**2)
    ws=ws[:,::2,::2]
    ws=np.ma.masked_where(np.ma.getmaskarray(swh), ws)
    Dataset.close()
    gws=np.ma.mean(ws,axis=(1,2)).filled(np.nan)
    
    #Hint: 到这一步，将数据准备完毕
    x=(gws-np.mean(gws))/4
    y=gswh-np.mean(gswh)
    plot(x,y,'b.')
    title('dSWH vs dWS')
    ylabel('dSWH (m)')
    xlabel('dWS (m s^-1)')
    show()

    return x,y


# Q1: 完善 Sigmoid 激活函数
def sigmoid(x):
    return 

# Q2: 完善 ReLU 激活函数
def ReLU(x):
    return 

x = np.arange(-1,1,.01)
w = 100
b = -50
plot(x,sigmoid(x * w + b))

# 生成数据集
x_all, y_all = get_data()


# 作业用真实数据！
X, Y = x_all[100:],y_all[100:] 
X_test, Y_test = x_all[:100],y_all[:100]

# 初始化神经网络参数
# 第一层（隐藏层）
w1_1 = np.random.rand()  # 第一个神经元的权重
b1_1 = np.random.rand()  # 第一个神经元的偏置
w1_2 = np.random.rand()  # 第二个神经元的权重
b1_2 = np.random.rand()  # 第二个神经元的偏置
w1_3 = np.random.rand()  # 第三个神经元的权重
b1_3 = np.random.rand()  # 第三个神经元的偏置
# 第二层（输出层）
w2_1 = np.random.rand()  # 第一个输入的权重
w2_2 = np.random.rand()  # 第二个输入的权重
w2_3 = np.random.rand()  # 第三个输入的权重
b2_1 = np.random.rand()  # 偏置

#Q3: 完成前向传播代码
def forward_propagation(X):
    # 隐藏层
    z1_1 =
    g1_1 = 
    z1_2 =
    g1_2 = 
    z1_3 = 
    g1_3 = 
    # 输出层
    z2_1 = 
    yhat = 

    return yhat, z2_1, g1_3, z1_3, g1_2, z1_2, g1_1, z1_1


# 设置超参数：学习率
n = len(X)
alpha = 0.1
beta  = 0.01/n
# 设置超参数：迭代次数
epochs = 25000
J_all = []
# 迭代训练模型
start_time = time.time()  # 记录训练开始时间
for m in range(epochs):
    # 遍历数据集中的每个样本
    for i in range(n):
        x = X[i]
        y = Y[i]

        # 向前传播
        yhat, z2_1, g1_3, z1_3, g1_2, z1_2, g1_1, z1_1 = forward_propagation(x)

        # 完成反向传播代码

        # 计算误差对各个参数的偏导数
        dJdyhat = -2 * (y - yhat)
        dyhatdz2_1 = 
        dz2_1dw2_1 = 
        dz2_1dw2_2 = 
        dz2_1dw2_3 = 
        
        dJdw2_1 = 
        dJdw2_2 = 
        dJdw2_3 =

        dz2_1db2_1 = 
        dJdb2_1 = dJdyhat * dyhatdz2_1 * dz2_1db2_1 

        dz2_1dg1_1 = 
        dz2_1dg1_2 = 
        dz2_1dg1_3 = 
        
        dg1_1dz1_1 = 
        dg1_2dz1_2 = 
        dg1_3dz1_3 = 
        dz1_1dw1_1 = 
        dz1_2dw1_2 = 
        dz1_3dw1_3 = 

        dJdw1_1 =  
        dJdw1_2 = 
        dJdw1_3 = 

        dz1_1db1_1 = 1
        dJdb1_1 = 

        dz1_2db1_2 = 1
        dJdb1_2 =  

        dz1_3db1_3 = 1
        dJdb1_3 = 


        # 更新参数
        w2_1 -= alpha * dJdw2_1
        w2_2 -= alpha * dJdw2_2
        w2_3 -= alpha * dJdw2_3
        b2_1 -= alpha * dJdb2_1

        w1_1 -= alpha * dJdw1_1
        w1_2 -= alpha * dJdw1_2
        w1_3 -= alpha * dJdw1_3
        b1_1 -= alpha * dJdb1_1
        b1_2 -= alpha * dJdb1_2
        b1_3 -= alpha * dJdb1_3
    # 动态绘制（不要在JupyterLab中运行）
    
    yhat, z2_1, g1_3, z1_3, g1_2, z1_2, g1_1, z1_1 = forward_propagation(x_all)
    J =  np.mean( (yhat - y)**2 )
    J_all.append(J)
    if m % 1000 == 0:
        yhat, z2_1, g1_3, z1_3, g1_2, z1_2, g1_1, z1_1 = forward_propagation(X)
        clf()
        scatter(X, Y)
        plot(X, yhat)
        pause(0.01)
# 显示训练耗时
end_time = time.time()  # 记录训练结束时间
train_time = end_time - start_time  # 计算训练耗时
print('Training time:', train_time, 'seconds')  # 输出训练耗时

# 输出训练结果
print("学习 {} 次结果 w1 = {}, b1 = {}, w2 = {}, b2 = {}".format(
    n * (m + 1), (w1_1, w1_2, w1_3), (b1_1, b1_2, b1_3),
    (w2_1, w2_2, w2_3), b2_1))

# 计算模型的预测结果并可视化

yhat, z2_1, g1_3, z1_3, g1_2, z1_2, g1_1, z1_1 = forward_propagation(x_all)
plot(X, Y,'b.')
plot(X_test, Y_test,'r.')
plot(x_all, yhat,'k')
show()