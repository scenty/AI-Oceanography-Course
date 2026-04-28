"""
预提供功能：数据加载、气候态计算、风速合成
待完成标记：# TODO [任务编号] - 任务描述
"""

import xarray as xr
import numpy as np
from matplotlib.pyplot import *


#ds = xr.open_dataset("ERA5_bulk_wave.grib", engine="cfgrib")
#ds.to_netcdf("D:/ERA5_bulk_wave.nc", engine="netcdf4")

# 预提供：数据加载与预处理
ds = xr.open_dataset(r"ERA5_bulk_wind.nc")
ds2 = xr.open_dataset(r"ERA5_bulk_wave.nc")
# 可以读取
u10 = ds['u10'].values
v10 = ds['v10'].values
ws =...
# 也可以直接计算






