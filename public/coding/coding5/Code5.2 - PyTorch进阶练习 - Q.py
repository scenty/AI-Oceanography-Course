import numpy as np
import time
import torch

#Vectorization 
#Q1: 考虑一个数组a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14],如何生成一个数组c = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ...,[11,12,13,14]]?
#not vectorized 
#vectorized/function
print("\n===== Q1 =====")
a = np.arange(1, 15)
# not vectorized
c_list = []
for i in range(len(a) - 3):
    c_list.append(a[i:i + 4])
c_not_vec = np.array(c_list)
print("not vectorized:\n", c_not_vec)

# vectorized
c = a[np.arange(11)[:, None] + np.arange(4)]
print("vectorized:\n", c)

# PyTorch练习位（保持同样输入输出形状）
# import torch
a_t = torch.arange(1, 15)
print(_t)

#Q2: skip
#Q3: 将二维数组Z进行翻转，实现 Z[i,j] = Z_new[j,i]
#wrong answer, why?
print("\n===== Q3 =====")
Z = np.random.randint(0, 9, (5, 5))
print("original Z:\n", Z)

# wrong answer原因：原地覆盖（in-place）会把还没读取到的元素提前改掉
Z_wrong = Z.copy()
for i in range(5):
    for j in range(5):
        Z_wrong[i, j] = Z_wrong[j, i]
print("wrong (in-place overwrite):\n", Z_wrong)

# not vectorized（正确：用临时副本）
Z_ok_loop = Z.copy()
Z_tmp = Z_ok_loop.copy()
for i in range(5):
    for j in range(5):
        Z_ok_loop[i, j] = Z_tmp[j, i]
print("loop with temp copy:\n", Z_ok_loop)

# vectorized
Z_vec = Z.T
print("vectorized transpose:\n", Z_vec)

# PyTorch练习位
# import torch
# Z_t = torch.randint(0, 9, (5, 5))
#not vectorized

#vectorized


#Q4:创建一个 10x10 的随机数组并找到它的最大值和最小值 
print("\n===== Q4 =====")
R = np.random.random((10, 10))
print("min:", R.min(), "max:", R.max())
# PyTorch练习位


#Q5: 从数组中提取所有的奇数
print("\n===== Q5 =====")
arr = np.arange(1, 21)
odds = arr[arr % 2 == 1]
print("arr:", arr)
print("odds:", odds)

# PyTorch练习位
arr_t = torch.arange(1, 21)


#Q6: 将arr中的所有奇数替换为-1
#function
#注意！如果: out = arr + 0
print("\n===== Q6 =====")
def replace_odds_with_minus_one(x):
    out = x + 0  # 副本，避免改到原数组
    out[out % 2 == 1] = -1
    return out

arr2 = np.arange(1, 21)
print("before:", arr2)
print("after :", replace_odds_with_minus_one(arr2))
print("original unchanged:", arr2)

# PyTorch练习位


#Q7：将一维数组转换为2行的2维数组
#给定np.arange(10)：
# > array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#希望得到：
# > array([[0, 1, 2, 3, 4],
# >        [5, 6, 7, 8, 9]])
print("\n===== Q7 =====")
v = np.arange(10)
v2 = v.reshape(2, 5)
print(v2)
# PyTorch练习位


#Q8: 垂直堆叠数组a和数组b
#给定a和b
#期望的输出：
# > array([[0, 1, 2, 3, 4],
# >        [5, 6, 7, 8, 9],
# >        [1, 1, 1, 1, 1],
# >        [1, 1, 1, 1, 1]])
# Answers
# Method 1:
# Method 2:
# Method 3:
print("\n===== Q8 =====")
a = np.arange(10).reshape(2, 5)
b = np.ones((2, 5), dtype=int)

stack1 = np.vstack([a, b])
print("Method 1 vstack:\n", stack1)

stack2 = np.concatenate([a, b], axis=0)
print("Method 2 concatenate:\n", stack2)

stack3 = np.r_[a, b]
print("Method 3 r_:\n", stack3)

# PyTorch练习位



#Q9：生成矩阵：array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
print("\n===== Q9 =====")
q9 = np.r_[np.repeat([1, 2, 3], 3), np.tile([1, 2, 3], 3)]
print(q9)

# PyTorch练习位


#Q10
#Q11: 从矩阵中提取范围内的所有数字
a = np.array([2, 6, 1, 9, 10, 3, 27])
# Method 1
# Method 2:
# Method 3:
print("\n===== Q11 =====")
lo, hi = 5, 10

m1 = a[(a >= lo) & (a <= hi)]
print("Method 1 mask:", m1)

idx = np.where((a >= lo) & (a <= hi))
m2 = a[idx]
print("Method 2 where:", m2)

m3 = a[np.logical_and(a >= lo, a <= hi)]
print("Method 3 logical_and:", m3)

# PyTorch练习位



#Q12：交换数组的1和2列
# Input
arr = np.arange(9).reshape(3,3)
# Solution
print("\n===== Q12 =====")
arr_swapped = arr.copy()
arr_swapped[:, [0, 1]] = arr_swapped[:, [1, 0]]
print("before:\n", arr)
print("after:\n", arr_swapped)

# PyTorch练习位



#Q13：反转二维数组的行
print("\n===== Q13 =====")
M = np.arange(1, 13).reshape(3, 4)
print("before:\n", M)
print("after (rows reversed):\n", M[::-1, :])

# PyTorch练习位
#Method1
#Method2

#Q15: vectorization的优势
print("\n===== Q15 =====")

#vectorization 1
sst = np.random.randint(0, 30, (60, 60, 10)).astype(np.int64)
start=time.time()
for ii in range(sst.shape[0]):
    for jj in range(sst.shape[1]):
        for kk in range(sst.shape[2]):
            sst[ii][jj][kk]=sst[ii][jj][kk]+ii*jj*kk
print('Time: ',str(time.time()-start))

start=time.time()
I=np.arange(sst.shape[0])[:,np.newaxis,np.newaxis]
J=np.arange(sst.shape[1])[np.newaxis,:,np.newaxis]
K=np.arange(sst.shape[2])[np.newaxis,np.newaxis,:]
tmp=I*J*K
sst=sst+tmp
print('Time: ',str(time.time()-start))

#vectorization 2
# goal: find sst<15, if so, sst+=1
# Method 1: vectorize
# function to find the square
def find_square(x):
    if x < 15:
        return x + 1
    else:
        return x
# passing an array to a vectorized function
start = time.time()
vectorized_function = np.vectorize(find_square)
result = vectorized_function(sst)
print('Time: ',str(time.time()-start))
print(result)
# vectorize() to vectorize the function find_square()
#Method 2: use mask to subset
start = time.time()
result=sst+0 ####!!!!!!
result[np.where(sst<15)]=result[np.where(sst<15)]+1
print('Time: ',str(time.time()-start))
print(result)

# PyTorch练习位（mask 方式非常接近 NumPy）
# import torch
# sst_t = torch.randint(0, 30, (60, 60, 10))
# out = sst_t.clone()
# out[sst_t < 15] += 1
