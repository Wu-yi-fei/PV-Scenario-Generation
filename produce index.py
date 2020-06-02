"""
生成某区间内不重复的N个随机数表的方法

"""

import random;
import numpy as np
import csv

# 1、利用递归生成
resultList = [];  # 用于存放结果的List鼠标
A = 0;  # 最小随机数
B = 9464  # 最大随机数
N = 9464
resultList=random.sample(range(A,B),N)
print(resultList)
index = np.reshape(resultList, (-1, 1))
with open('predict dataset/index.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(index)


