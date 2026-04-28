# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import netCDF4 as nc
from matplotlib.pyplot import *
import torch
import torch.nn as nn


def get_data():
    Dataset = nc.Dataset(r'../Data/ERA5_bulk_wave.nc')
    swh = Dataset['swh'][:]
    lon = Dataset['longitude'][:]
    lat = Dataset['latitude'][:]
    Dataset.close()
    gswh = np.ma.mean(swh, axis=(1, 2)).filled(np.nan)

    Dataset = nc.Dataset(r'../Data/ERA5_bulk_wind.nc')
    u10 = Dataset['u10'][:]
    v10 = Dataset['v10'][:]
    ws = np.sqrt(u10**2 + v10**2)
    ws = ws[:, ::2, ::2]
    ws = np.ma.masked_where(np.ma.getmaskarray(swh), ws)
    Dataset.close()
    gws = np.ma.mean(ws, axis=(1, 2)).filled(np.nan)

    x = (gws - np.mean(gws)) / 4
    y = gswh - np.mean(gswh)
    mn, mx = y.min(), y.max()
    y = (y - mn) / (mx - mn)

    return x, y


x_all, y_all = get_data()

x_all_t = torch.tensor(x_all, dtype=torch.float32).view(-1, 1)
y_all_t = torch.tensor(y_all, dtype=torch.float32).view(-1, 1)

X = x_all_t[100:]
Y = y_all_t[100:]
X_test = x_all_t[:100]
Y_test = y_all_t[:100]

model = nn.Sequential(
    nn.Linear(1, 3),
    nn.Sigmoid(),
    nn.Linear(3, 1),
    nn.Sigmoid()
)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 20000
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    yhat = model(x_all_t).numpy()

plot(X, Y, 'b.')
plot(X_test, Y_test, 'r.')
plot(x_all, yhat, 'k')
show()
