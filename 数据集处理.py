"""
对来自NREL的光伏发电数据集进行预处理
"""

import csv
import numpy as np

rows1=[]
rows1 = np.array(rows1).reshape(105120, -1)
with open('initial data/ahead_data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
rows = np.array(rows, dtype=float)
print(np.shape(rows))
for i in range(np.shape(rows)[1]):
  row= rows[:,i]
  row = np.array(row).reshape(105120,1)
  print(np.shape(row))
  row =row/ np.max(row) * 8
  rows1=np.hstack((rows1, row))
with open('predict dataset/oneday_ahead_datasets.csv' , 'w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows1)
