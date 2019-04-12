#!/usr/bin/env python
"""
@author:HaipengDuan
@time:2019/4/12 21:12
"""
#-*-coding:utf-8-#-
import numpy as np
from sklearn.cluster import KMeans
from CommunityDetection import NCut


def node_kmeans(file, k):
    np_node = np.delete(NCut.sort_output(file), 0, axis=1)
    # print(np_node)
    # labels = KMeans(n_clusters=k)
    # np_file_out = np.loadtxt(file, dtype=float, skiprows=1)
    # labels = KMeans(n_clusters=k).fit_predict(np_node)
    labels = [str(i) for i in KMeans(n_clusters=k).fit_predict(np_node)]
    print(labels)
    for i in range(len(labels)):
        # print(labels[i])
        if labels[i] == '0':
            labels[i] = 'r'
        elif labels[i] == '1':
            labels[i] = 'b'
        elif labels[i] == '2':
            labels[i] = 'g'
        elif labels[i] == '3':
            labels[i] = 'y'
    print(labels)

    return labels

