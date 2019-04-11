#!/usr/bin/env python
"""
@author:HaipengDuan
@time:2019/3/31 17:02
"""
#-*-coding:utf-8-#-

from io import open
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from itertools import product, permutations
from scipy.io import loadmat
from scipy.sparse import issparse
import networkx as nx
import matplotlib.pyplot as plt


class Graph(defaultdict):

    def __init__(self):
        """
        构建图为字典，key为节点，value为一个list
        """
        # super(Graph, self).__init__(list)
        super().__init__(list)

    def nodes(self):
        """
        :return: 图中所有节点
        """
        return self.keys()

    def adjacency_iter(self):
        """
        :return: 字典列表的迭代
        """
        # return self.iteritems()
        return self.items()

    def subgraph(self, nodes={}):
        """
        :param nodes:
        :return: 返回子图
        """
        subgraph = Graph()

        for n in nodes:
            if n in self:
                """
                n是否在图的字典中
                """
                subgraph[n] = [x for x in self[n] if x in nodes]
                """
                把nodes中的节点以及对应的边加入子图中
                """
        return subgraph

    def make_undirected(self):
        """
        将有向图变为无向图
        :return:
        """
        # t0 = time()

        for v in self.keys():
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        # t1 = time()
        # logger.info('make_directed: added missing edges {}s'.format(t1-t0))

        self.make_consistent()
        return self

    def make_consistent(self):
        """
        去掉转换成无向图时出现的重复
        将list内部节点从小到大排序
        :return:
        """
        # t0 = time()
        for k in iterkeys(self):
            """
            去重，排序，重新赋值
            """
            self[k] = list(sorted(set(self[k])))

        # t1 = time()
        # logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):
        """
        去掉自循环
        :return:
        """
        removed = 0
        # t0 = time()

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1

        # t1 = time()
        #
        # logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
        return self

    def check_self_loops(self):
        """
        检查是否含有自回路
        :return:
        """
        for x in self:
            for y in self[x]:
                if x == y:
                    return True
        return False

    def has_edge(self, v1, v2):
        """
        判断两个节点是否有边
        :param v1:
        :param v2:
        :return:
        """
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None):
        """
        求节点对应的度数
        :param nodes: node为节点时，输出该节点的度，若是list，则返回字典，key为节点，value为度
        :return:
        """
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def order(self):
        """"Returns the number of nodes in the graph"""
        return len(self)

    def number_of_edges(self):
        """"Returns the number of nodes in the graph"""
        # 无向图边的个数，若是有向图则去掉除以2
        return sum([self.degree(x) for x in self.keys()]) / 2

    def number_of_nodes(self):
        """"Returns the number of nodes in the graph"""
        return self.order()

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.

            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = self
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            # 定位到随机游走到的最后一个节点。
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]


# TODO add build_walks in here


def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
    """
    对一个图生成语料库
    :param G:
    :param num_paths:
    :param path_length:
    :param alpha:
    :param rand:
    :return:
    """
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        """
        循环num_paths次，即对一个起点游走num_path次
        """
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))

    return walks


def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                               rand=random.Random(0)):
    """
    迭代的生成语料库
    :param G:
    :param num_paths:
    :param path_length:
    :param alpha:
    :param rand:
    :return:
    """
    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def clique(size):
    return from_adjlist(permutations(range(1, size + 1)))


def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"""
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def parse_adjacencylist(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            introw = [int(x) for x in l.strip().split()]
            row = [introw[0]]
            row.extend(set(sorted(introw[1:])))
            adjlist.extend([row])

    return adjlist


def parse_adjacencylist_unchecked(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            adjlist.extend([[int(x) for x in l.strip().split()]])

    return adjlist


def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):
    if unchecked:
        parse_func = parse_adjacencylist_unchecked
        convert_func = from_adjlist_unchecked
    else:
        parse_func = parse_adjacencylist
        convert_func = from_adjlist

    adjlist = []

    total = 0
    with open(file_) as f:
        for idx, adj_chunk in enumerate(map(parse_func, grouper(int(chunksize), f))):
            adjlist.extend(adj_chunk)
            total += len(adj_chunk)

    G = convert_func(adjlist)

    if undirected:
        G = G.make_undirected()

    return G


def load_edgelist(file_, undirected=True):
    G = Graph()
    with open(file_) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = int(x)
            y = int(y)
            G[x].append(y)
            if undirected:
                G[y].append(x)

    G.make_consistent()
    return G


def load_matfile(file_, variable_name="network", undirected=True):
    mat_varables = loadmat(file_)
    mat_matrix = mat_varables[variable_name]

    return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
        raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]

        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        # g_see = nx.Graph
        # g_see.add_node(node)
        neighbors = row[1:]
        # for i in range(len(neighbors)):
        #     g_see.add_edge(node, neighbors[i])
        # nx.draw(g_see)
        # plt.savefig("原始输入图")
        # plt.show()
        G[node] = neighbors

    return G

