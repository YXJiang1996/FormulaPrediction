import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import torch
import copy
from sklearn.manifold import TSNE
from matplotlib import cm
from sklearn.decomposition import PCA
from winter_try.utils import conc2ref_km, ciede2000_color_diff

#
# a1 = [10000, 0, 0, 0, 0, 0, 20000, 0, 0, 0, 50000]
# a2 = [20000, 0, 0, 0, 0, 0, 10000, 0, 0, 0, 70000]
# a3 = [30000, 0, 0, 0, 0, 0, 50000, 0, 0, 0, 20000]
# a4 = [40000, 0, 0, 0, 0, 0, 30000, 0, 0, 0, 40000]
# b1 = [0, 0, 0, 20000, 0, 0, 20000, 0, 0, 0, 50000]
# b2 = [0, 0, 0, 30000, 0, 0, 10000, 0, 0, 0, 70000]
# b3 = [0, 0, 0, 20000, 0, 0, 50000, 0, 0, 0, 20000]
# b4 = [0, 0, 0, 50000, 0, 0, 30000, 0, 0, 0, 40000]
# c1 = [0, 30000, 0, 0, 0, 20000, 0, 0, 0, 0, 50000]
# c2 = [0, 20000, 0, 0, 0, 40000, 0, 0, 0, 0, 70000]
# c3 = [0, 80000, 0, 0, 0, 40000, 0, 0, 0, 0, 20000]
# c4 = [0, 60000, 0, 0, 0, 40000, 0, 0, 0, 0, 40000]
# d1 = [10000, 0, 0, 60000, 0, 0, 0, 0, 80000, 0, 0]
# d2 = [20000, 0, 0, 20000, 0, 0, 0, 0, 70000, 0, 0]
# d3 = [30000, 0, 0, 70000, 0, 0, 0, 0, 20000, 0, 0]
# list = []
# list.append(a1)
# list.append(a2)
# list.append(a3)
# list.append(a4)
# list.append(b1)
# list.append(b2)
# list.append(b3)
# list.append(b4)
# list.append(c1)
# list.append(c2)
# list.append(c3)
# list.append(c4)
# list.append(d1)
# list.append(d2)
# list.append(d3)
# target=[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3]
#
# # 将生成的配方降维可视化
# def tsne_plot(predict_formula):
#     target = []  # 每种配方应该是什么颜色
#     next_color = 0  # 下一种画图颜色的编号
#     color_dict = {}  # 用于保存配方和颜色的对应关系
#     for n in range(predict_formula.shape[0]):
#         temp_formula = copy.copy(predict_formula[n, :])
#         temp_formula[temp_formula != 0] = 1
#         dict_key = tuple(temp_formula)
#         if dict_key in color_dict:
#             target.append(color_dict[dict_key])
#         else:
#             color_dict[dict_key] = next_color
#             target.append(next_color)
#             next_color += 1
#     # 二维可视化
#     # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#     # low_dim_embs = tsne.fit_transform(predict_formula)
#     # plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], c=target)
#     # plt.show()
#     # 三维可视化
#     tsne = TSNE(perplexity=30, n_components=3, init='pca', n_iter=5000)
#     low_dim_embs = tsne.fit_transform(predict_formula)
#     fig=plt.figure()
#     ax=Axes3D(fig)
#     ax.scatter(low_dim_embs[:,0],low_dim_embs[:,1],low_dim_embs[:,2],c=target)
#     plt.show()
#
# tsne_plot(np.array(list))


# dic = {'a': 3, 'b': 6, 'c': 1, 'd': 4}
# dic = dict(sorted(dic.items(), key=lambda item: item[1]))
# count=0
# for key,value in dic.items():
#     if count>=3:
#         break
#     count+=1
#     print("{}:{}".format(key,value))
list=[[],[],[],[],[],[],[],[],[],[]]
list[0].append([1,2,3])
list[1].append([2,3,4])
print(list)
for item in list:
    print(item)