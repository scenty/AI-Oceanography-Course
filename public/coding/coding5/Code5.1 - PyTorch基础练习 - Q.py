from scipy.io import loadmat
from matplotlib.pyplot import *
import time 

#数组基础
#Q: 导入pytorch

#Q: 创建从0到9的一维数字数组

#Q: 创建二维数组，分别为[ [1,2,3], [4,5,6]]

#Q: 查看数组的数据类型、维度数量、总元素数量

#Q: 对以下矩阵进行四种以上不同的切片操作
#numpy方法
#x = np.random.random(size=[100,100,20])
#y = np.random.random(size=[100,100,20])
#torch方法


#Q：进行多种操作
#最大最小值、range
#x.max()-x.min() #numpy方法
#torch方法

#variable coef
#np.std(x)/x.mean() #numpy方法
#torch方法

#median
#np.median(x) #numpy方法
#torch方法

#Q: 矩阵的转置transpose/高维转置

#Q：矩阵的变形reshape

#Q：矩阵的交换轴swapaxies

#Q：矩阵扁平化flatten

#Q：矩阵维度压缩squeeze
sst=sst.reshape([312, 1 ,130, 360])
sst=sst[:,:,:,np.newaxis,:]
sst=np.squeeze(sst,1)
sst=np.squeeze(sst)

#Q：矩阵拼接
np.concatenate((sst,sst),axis=2) #numpy方法

#Q: 将sst矩阵中大于20的元素变为nan
    
#Nan
sst[sst>20]=np.nan #numpy方法
mask=~np.isnan(np.sum(sst,axis=0)) #numpy方法
