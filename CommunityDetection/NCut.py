#!/usr/bin/env python
"""
@author:HaipengDuan
@time:2019/4/5 17:55
"""
# -*-coding:utf-8-#-
import numpy as np
import math
from DeepWalk import graph
from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist

def sort_output(file):
        """

        :param file: output文件
        :return: 按节点序号排号的矩阵
        """
        np_file_out = np.loadtxt(file, dtype=float, skiprows=1)
        # file = open(file)
        # file_output = file.read().split("\n")
        # for i in range(len(file_output)):
        #     file_output[i] = list(file_output[i].split())
        # np_file_out = np.delete(np.array(file_output), 0, axis=0)
        # np_file_out = np.delete(np.array(file_output), -1, axis=0)
        np_file_out = np_file_out[np_file_out[:, 0].argsort()]
        # print(np_file_out)
        # print(len(np_file_out))
        return np_file_out


def build_W_D_L(G, file, k):
        """

        :param file: output文件
        :param k: 聚类个数
        :return: 聚类结果
        """
        np_node = sort_output(file)
        # print(np_node)
        w = np.zeros((len(np_node), len(np_node)))
        for i in range(len(np_node)):
                # print(i)
                for j in range(i+1):
                        # print(i)
                        # print(j)
                        stemp = np.vstack([np_node[i, 1:], np_node[j, 1:]])
                        # print(np_node[j, 1:])
                        w[i, j] = 1 - pdist(stemp, 'cosine')
                        w[j, i] = w[i, j]
        # print(w[11])
        # print(w)

        # list_d = []
        # for i in range(len(G)):
        #         list_d.append(len(G[i+1]))
        # s = np.array(list_d)
        s = np.sum(w, axis=1)

        print(s)
        s1 = np.power(s, -0.5)
        print(s1)
        # for i in range(len(np_node)):
        #         d[i, i] = s[i]
        d = np.diag(s1)
        print(d)
        print(w)
        l = np.dot(np.dot(d, (d - w)), d)

        eigvals, eigvecs = LA.eig(l)

        indices = np.argsort(eigvals)[:k]

        k_small_eigenvectors = normalize(eigvecs[:, indices])
        print(k_small_eigenvectors)

        label = KMeans(n_clusters=k).fit_predict(k_small_eigenvectors)
        print(label)





