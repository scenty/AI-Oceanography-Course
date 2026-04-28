# %% [markdown]
# # AI 海洋学：课前热身编程练习 (50 Quizzes)
# 本练习旨在帮助你快速复习 Python、NumPy 和 Matplotlib 的核心操作，
# 并掌握向量化编程思维，为后续运用深度学习模型处理复杂的海洋与气候数据打下基础。
# 
# 请在带有 `# TODO:` 的注释下方编写你的代码。

# %% [markdown]
# ## 单元一：Python 基本操作 (Quiz 1 - 10)
# 本单元复习基础的数据结构、循环控制与函数封装。

# %%
# Quiz 1: 打印一条欢迎信息："欢迎来到 AI 海洋学课程！"
# TODO:
print('欢迎来到AI海洋海洋学课程！')

# Quiz 2: 创建一个包含三个测站名称的列表，命名为 `stations`，元素为 'St_A', 'St_B', 'St_C'。
# TODO:
stations = ['St_A','St_B','St_C']

# Quiz 3: 将“嘉庚”号（'Jiageng'）作为移动测站，追加到 `stations` 列表的末尾。
# TODO:
stations.append('Jiageng')

# Quiz 4: 创建一个字典 `station_coords`，键为测站名 'St_A' 和 'St_B'，值为包含经纬度的元组，例如 (113.5, 22.1) 和 (114.0, 21.8) (珠江口附近坐标)。
# TODO:
station_coords = {'St_A': (113.5, 22.1), 'St_B': (114.0, 21.8)}

# Quiz 5: 从字典 `station_coords` 中提取 'St_A' 的坐标并打印。
# TODO:
print(station_coords['St_A'])

# Quiz 6: 使用 for 循环遍历 `stations` 列表，打印每个测站的名称。
# TODO:
for station in stations:
    print(station)

# Quiz 7: 编写一个函数 `celsius_to_kelvin(c)`，将摄氏度转换为开尔文温度 (K = C + 273.15)。
# TODO:
def celsius_to_kelvin(c):
    return c + 273.15

# Quiz 8: 定义一个包含三个海表温度(SST)的列表 `sst_c = [24.5, 28.1, 15.0]`，使用 for 循环和上一步的函数，将其全部转换为开尔文并存入新列表 `sst_k`。
# TODO:
sst_c = [24.5, 28.1, 15.0]
sst_k = [celsius_to_kelvin(c) for c in sst_c]
print(sst_k)

# Quiz 9: 使用列表推导式 (List Comprehension)，从 `sst_c` 中筛选出大于 25°C 的温度值。
# TODO:
sst_c = [24.5, 28.1, 15.0]
sst_c_gt_25 = [c for c in sst_c if c > 25]
print(sst_c_gt_25)

# Quiz 10: 使用 f-string 格式化字符串，打印输出："测站 St_A 的当前水温为 24.5 °C"。
# TODO:
print(f"测站 St_A 的当前水温为 {sst_c[0]} °C")

# %% [markdown]
# ## 单元二：NumPy 基础操作 (Quiz 11 - 25)
# 本单元重点在于多维数组的创建、索引、切片及基本运算。

# %%
# Quiz 11: 导入 numpy 库，并简写为 np。
# TODO:
import numpy as np

# Quiz 12: 创建一个包含 5 个不同深度盐度值 (如 33.1, 33.5, 34.0, 34.2, 34.5) 的 1D NumPy 数组 `salinity`。
# TODO:
salinity = np.array([33.1, 33.5, 34.0, 34.2, 34.5])

# Quiz 13: 创建一个 3x3 的 2D NumPy 数组 `sst_grid`，模拟一小块海域的网格化海表温度。
# TODO:
sst_grid = np.array([[24.5, 25.0, 25.5], [25.0, 25.5, 26.0], [25.5, 26.0, 26.5]])

# Quiz 14: 创建一个 10x10 的全 0 数组 `depth_mask`，数据类型指定为 float32。
# TODO:
depth_mask = np.zeros((10, 10), dtype=np.float32)

# Quiz 15: 创建一个 5x5 的全 1 数组 `land_mask`，数据类型指定为 int8。
# TODO:
land_mask = np.ones((5, 5), dtype=np.int8)

# Quiz 16: 使用 np.arange 创建一个水深序列 `depths`，从 0 开始，到 1000 结束（不包含1000），步长为 50。
# TODO:
depths = np.arange(0, 1000, 50)

# Quiz 17: 使用 np.linspace 创建一个纬度序列 `lats`，在 10 到 20 之间均匀分布 50 个点。
# TODO:
lats = np.linspace(10, 20, 50)

# Quiz 18: 打印 `sst_grid` 数组的维度(ndim)、形状(shape)和元素总数(size)。
# TODO:
print(sst_grid.ndim, sst_grid.shape, sst_grid.size)

# Quiz 19: 假设有一个包含 12 个月平均温度的 1D 数组 (长度为12)，请将其 reshape 为 3x4 的数组（代表3个季度，每个季度4个月...虽不符合实际四季，但作为矩阵变换练习）。
# temp_12 = np.arange(15, 27)
# TODO:
temp_12 = np.arange(15, 27)

# Quiz 20: 索引：提取 `salinity` 数组中的第 3 个元素（注意索引从0开始）。
# TODO:


# Quiz 21: 索引：提取 `sst_grid` 数组中心位置 (第2行，第2列) 的元素。
# TODO:


# Quiz 22: 切片：提取 `salinity` 数组的前 3 个元素。
# TODO:


# Quiz 23: 切片：提取 `sst_grid` 数组的第 1 列所有数据。
# TODO:


# Quiz 24: 标量运算：假设全球变暖导致温度上升 1.5 度，给 `sst_grid` 中的每个元素都加上 1.5，结果存入 `sst_warmed`。
# TODO:
# 矢量化 vectorization:

# Quiz 25: 数组运算：计算 `sst_warmed` 与原始 `sst_grid` 之间的温差（温度异常计算基础）。
# TODO:
# 矢量化 vectorization:

# %% [markdown]
# ## 单元三：其他工具（Matplotlib）基础操作 (Quiz 26 - 35)
# 本单元复习科学可视化的基础，特别是温盐图、剖面图和空间网格的绘制。

# %%
# Quiz 26: 导入 matplotlib.pyplot 并简写为 plt。
# TODO:


# 准备绘图数据
# depths_plot = np.array([0, 50, 100, 200, 500, 1000])
# temps_plot = np.array([25.0, 24.5, 22.0, 18.0, 10.0, 4.0])

# Quiz 27: 绘制一个简单的折线图，x 轴为温度，y 轴为深度（模拟海洋温度垂直剖面）。
# TODO:


# Quiz 28: 为上述图表添加标题："Temperature Depth Profile"。
# TODO:


# Quiz 29: 为 x 轴添加标签 "Temperature (C)"，为 y 轴添加标签 "Depth (m)"。
# TODO:


# Quiz 30: 在海洋学中，深度通常向下增加。请将 y 轴反转 (提示: gca().invert_yaxis())，最后调用 plt.show() 显示图像。
# TODO:


# 准备温盐散点数据
# T = np.random.uniform(10, 30, 50)
# S = np.random.uniform(33, 35, 50)

# Quiz 31: 绘制温度(T)和盐度(S)的散点图 (T-S Diagram)，x轴为盐度，y轴为温度。
# TODO:


# Quiz 32: 绘制一组随机生成的海表高度异常(SLA)数据的直方图，设置 bins=20。
# sla_data = np.random.randn(1000) * 0.1
# TODO:


# 准备 2D 网格数据
# sst_field = np.random.rand(20, 20) * 10 + 20

# Quiz 33: 使用 plt.imshow() 将 2D 数组 `sst_field` 绘制为热力图 (伪彩色图)，设置 colormap 为 'coolwarm'。
# 
# TODO:


# Quiz 34: 为上一步的热力图添加颜色条 (Colorbar)，并标明单位 "SST (°C)"。
# TODO:


# Quiz 35: 将绘制好的热力图保存为当前目录下的高清图片文件 'sst_spatial_map.png'，设置 dpi=300。
# TODO:


# %% [markdown]
# ## 单元四：NumPy 矢量化进阶操作 (Quiz 36 - 50)
# 抛弃 for 循环，使用 NumPy 提供的矢量化操作来高效处理大规模矩阵，这在海洋数据预处理和深度学习张量运算中至关重要。

# %%
# 生成测试数据：模拟 100x100 的海表温度场 (含部分极端值和缺失值)
sst_large = np.random.normal(loc=25, scale=3, size=(100, 100))
sst_large[10:15, 10:15] = np.nan # 模拟云遮挡导致的缺失数据

# Quiz 36: 计算 `sst_large` 数组的全局均值（暂不处理 NaN）。你会发现结果是 nan。
# TODO:
print(np.mean(sst_large))

# Quiz 37: 计算 `sst_large` 中非 NaN 部分的全局均值。(提示：使用 np.nanmean)
# TODO:
print(np.nanmean(sst_large))

# Quiz 38: 找出整个温度场中的最大值和最小值 (同样需要忽略 NaN)。
# TODO:
print(np.max(sst_large), np.min(sst_large))

# Quiz 39: 计算整个温度场的标准差，以评估空间温度的离散程度。
# TODO:
print(np.std(sst_large))

# Quiz 40: 降维聚合：计算每一列的平均值（模拟纬向平均 Zonal Mean），返回一个长度为 100 的 1D 数组。
# TODO:
print(np.mean(sst_large, axis=0))

# Quiz 41: 布尔掩码 (Boolean Masking)：创建一个掩码矩阵 `is_heatwave`，判断 `sst_large` 中哪些像素的温度大于 30°C。
# TODO:
is_heatwave = sst_large > 30

# Quiz 42: 使用掩码提取出所有发生热浪 (>30°C) 的具体温度值，结果将是一个 1D 数组。
# TODO:
print(sst_large[is_heatwave])

# Quiz 43: 条件替换 (np.where)：将 `sst_large` 中所有小于 20°C 的值替换为 0，大于等于 20°C 的保留原值。
# TODO:
sst_large = np.where(sst_large < 20, 0, sst_large)

# 准备广播机制 (Broadcasting) 数据
# daily_sst = np.random.rand(30, 50, 50) * 5 + 20  # 30天，50x50的空间网格
# clim_mean = np.random.rand(50, 50) + 22         # 气候态平均矩阵，50x50

# Quiz 44: 广播机制：在不使用 for 循环的情况下，计算每一天、每个格点的温度距平 (Anomaly)，即从 `daily_sst` 中减去 `clim_mean`。
# TODO:


# Quiz 45: 统计运算：计算上一步算出的三维距平矩阵中，有多少个网格点的距平值大于 2.0 °C。(提示：(条件).sum())
# TODO:


# Quiz 46: 梯度计算：使用 np.gradient 计算 2D 温度场 `clim_mean` 在 x 和 y 方向的梯度（模拟寻找海洋锋面）。
# TODO:


# Quiz 47: 缺失值识别：找出一个布尔数组，标明 `sst_large` 中哪些位置是 NaN (缺失观测)。
# TODO:


# Quiz 48: 缺失值填充：将 `sst_large` 中的所有 NaN 替换为该矩阵的有效全局均值 (第37题的结果)。(提示：可以将掩码和索引结合使用，或者使用 np.nan_to_num)
# TODO:


# Quiz 49: 矩阵乘法：提取 `clim_mean` 的前 3x3 子矩阵 A，与自身转置 A.T 进行点乘操作 (Dot Product)。这是神经网络中线性层的基础运算。
# TODO:


# Quiz 50: 高级索引 (Fancy Indexing)：假设你有几个特定浮标的坐标索引 `lats_idx = [5, 12, 45]`，`lons_idx = [10, 20, 30]`，请从 `clim_mean` 中一次性提取这三个离散位置的温度值。
# TODO:

# 恭喜完成！
