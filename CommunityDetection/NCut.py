#!/usr/bin/env python
"""
@author:HaipengDuan
@time:2019/4/5 17:55
"""
# -*-coding:utf-8-#-
import numpy as np
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
        print("Cutting...")
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
                        # if i != j:
                        #         w[i, j] = 1 - pdist(stemp, 'cosine')
                        #         w[j, i] = w[i, j]
                        # else:
                        #         w[i, j] = 0
        # print(w[11])
        print(w)

        list_d = []
        for i in range(len(G)):
                list_d.append(len(G[i+1]))
        s = np.array(list_d)
        # print(s)

        d = np.diag(np.power(s, -0.5))
        # d = np.diag(np.power(np.sum(w, axis=1), -0.5))
        print(d)
        # print(w)
        l = np.dot(np.dot(d, (d - w)), d)
        # l = np.eye(len(G)) - np.dot(np.dot(d, w), d)
        # print(l)
        eigvals, eigvecs = LA.eig(l)
        # print(eigvals)
        indices = np.argsort(eigvals)[:k]

        k_small_eigenvectors = normalize(eigvecs[:, indices])
        # print(k_small_eigenvectors)

        labels = [str(i) for i in KMeans(n_clusters=k).fit_predict(k_small_eigenvectors)]
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





