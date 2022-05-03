from tkinter import font
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("csv/100Hz_normal_2022_03_11_14_34_38.csv")
# print(data)
# print("\f")

# 單跑一次，並評估平均值，且預測
kmeansModel = KMeans(n_clusters=3, random_state=70)
clusters_pred = kmeansModel.fit_predict(data)

# 每個點到其他叢集的質心的距離之和
# print(kmeansModel.inertia_)

# print("\f")

# 特徵的中心點
# print(kmeansModel.cluster_centers_)

hcluster = cluster.AgglomerativeClustering(
    linkage="word", affinity="euclidean", n_clusters=3)

hcluster.fix(data)
cluster_label = hcluster.labels_
print(cluster_label)
print("------------")

plt.scatter(kmeansModel.cluster_centers_[0], kmeansModel.cluster_centers_[2],
            c="g")

# 測試 K = 1~9，選擇迅速下降轉為平緩的點
# kmeans_list = [KMeans(n_clusters=k, random_state=80).fit(data)
#                for k in range(1, 10)]
# inertias = [model.inertia_ for model in kmeans_list]

# print("\f")
# print(inertains)

# plt.figure(figsize=(8, 3.5))
# plt.plot(range(1, 10), inertias, "bo-")
# plt.xlabel("$K$", fontsize=14)
# plt.ylabel("Inertia", fontsize=14)
# plt.axis([1, 9, 0, 20000000000000])
# plt.show()


# silhouette_score = [silhouette_score(data, model.labels_)
#                     for model in kmeans_list[1:]]

# print("\f")
# print(silhouette_score)

# sns.lmplot("channel_0", hue="channel_0", data=data, fit_reg=False)
