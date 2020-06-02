"""
数据分析中包含了几种对结果的图像化代码内容
包括了用热力图描述12个光电场的空间相关系数
用折线图描述场景等
"""

import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#热力图
#真实数据
'''with open('datasets/spatial 1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
a = np.array(rows, dtype=float)
a=np.corrcoef(a,rowvar=0)
print(np.shape(a))
sns.set()
# data:数据 square:是否是正方形 vmax:最大值 vmin:最小值 robust:排除极端值影响
#sns.heatmap(data=grid, square=True, vmax=20, vmin=0, robust=True)
ax = sns.heatmap(a, center=0.2, annot=False, vmax=1, vmin=0, robust=True)
#plt.savefig('res.png', dpi=300)
# 标题
plt.title("Spatial correlation of 12 real scenarios of photovoltaic power plants")
plt.show()

#生成数据
with open('generated_iteration-1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
a = np.array(rows, dtype=float)
a=np.corrcoef(a)
print(np.shape(a))
sns.set()
ax = sns.heatmap(a, center=0.2, annot=False, vmax=1, vmin=0, robust=True)
# 标题
plt.title("forecast scenarios of photovoltaic power plants")
plt.show()'''

with open('generated_iteration-1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
data = np.array(rows, dtype=float)
with open('predict dataset\oneday_ahead_datasets -test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
data1 = np.array(rows, dtype=float)
with open('datasets\spatial 1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
data2 = np.array(rows, dtype=float)
for i in range(12):
 plt.plot(data[i,288:], color='r')
plt.plot(data1[288:,11],color='b')
plt.plot(data2[288:,11],color='g')
plt.title("Scenario Forecasting", fontsize=24)
plt.xlabel("time/5min", fontsize=14)
plt.ylabel("power/MW", fontsize=14)
plt.show()