# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 14:13:39 2023

@author: Scenty
"""
import numpy as np
import netCDF4 as nc
from matplotlib.pyplot import *
from scipy.io import loadmat
import matplotlib.cm as cm
import cmocean.cm as cmo

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


#read dataset 3
Dataset=loadmat(r'../Data/coastlines.mat')
clon=Dataset['coastlon'][:]
clat=Dataset['coastlat'][:]
Dataset.clear()



pcolor(lon,lat,swh[6],vmin=0,vmax=1,cmap=cmo.thermal);colorbar()
plot(clon,clat,color='k',linewidth=.3)
xlim((100,120))
ylim((0,50))
gca().set_aspect(map_aspect)
title('Regional SWH at Jul. 1940')
show()


pcolor(lon,lat,ws[6],vmin=-2,vmax=30,cmap=cmo.thermal);colorbar()
plot(clon,clat,color='k',linewidth=.3)
xlim((100,120))
ylim((0,50))
gca().set_aspect(map_aspect)
title('Regional Wind Speed at Jul. 1940')
show()

pcolor(lon,lat,swh[1026],vmin=0,vmax=1,cmap=cmo.thermal);colorbar()
plot(clon,clat,color='k',linewidth=.3)
xlim((100,120))
ylim((0,50))
gca().set_aspect(map_aspect)
title('Regional SWH at Jul. 2025')
show()


pcolor(lon,lat,ws[1026],vmin=-2,vmax=30,cmap=cmo.thermal);colorbar()
plot(clon,clat,color='k',linewidth=.3)
xlim((100,120))
ylim((0,50))
gca().set_aspect(map_aspect)
title('Regional Wind Speed at Jul. 2025')
show()


#
# for step in range(300):
#     plot(gws,gswh,'b.')
#     plot(gws[step:step+12],gswh[step:step+12],'r-')
#     title('12-month SWH vs Wind Speed starting at '+format(year[step],'.1f'))
#     ylabel('Regional mean SWH (m)')
#     xlabel('Regional mean Wind Speed (m s^-1)')
#     show()


#Hint: 到这一步，将数据准备完毕
x=(gws-np.mean(gws))/4
y=gswh-np.mean(gswh)
plot(x,y,'b.')
title('dSWH vs dWS')
ylabel('dSWH (m)')
xlabel('dWS (m s^-1)')
show()

#two-parameter fit ( y = theta1 * x + theta0)
#方法1:格点搜索

#方法2:数学推导(最小二乘拟合，得到真值)
theta1_true,theta0_true = np.polyfit(x,y,1)

#方法3:gradient descent
#重新定义网格范围，为了可视化效果
[t0,t1]=np.meshgrid(np.arange(-1,1,0.01),np.arange(-1,1,0.01))
J_mesh = np.sum ( ( t1[np.newaxis] * x[:,np.newaxis,np.newaxis] + t0[np.newaxis] - y[:,np.newaxis,np.newaxis]) ** 2, axis=0)/2/len(y)
#grid search
M,N = t0.shape
theta1 = 0.6
theta0 = 0.16

#初始化参数
alpha = 1.0

theta1_all=[]
theta0_all=[]
J_all=[]

for ii in range(1000):
    
    #Q：此处，补充计算J及其梯度的代码，并实现梯度下降，<10 行
    #将初始值代入模型传播
    #计算代价函数
    #计算参数的梯度
    #更新参数
    #print(str(theta1))
    #print(str(theta0))
    theta0_all.append(theta0)
    theta1_all.append(theta1)
    J_all.append(J)
    #以下为可视化：
    fig = figure(figsize=(14,6))
    subplot(121)
    plot(x,y,'b.')
    plot(x,x*theta1+theta0,'r-')
    text(0.3,0.06,'Our model: theta0 = '+format(theta0,'.3f')+' theta1 = '+format(theta1,'.3f'))
    text(0.3,0.0,'Theory: theta0 = '+format(theta0_true,'.3f')+' theta1= '+format(theta1_true,'.3f'))
    subplot(122)
    pcolor(t0,t1,J_mesh,cmap=cm.coolwarm);colorbar()
    plot(theta0,theta1,'rx')
    title('J='+format(J,'.6f'))
    show()
