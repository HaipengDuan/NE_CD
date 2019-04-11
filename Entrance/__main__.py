#!/usr/bin/env python
"""
@author:HaipengDuan
@time:2019/3/31 17:21
"""
#-*-coding:utf-8-#-

import sys
import random
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from DeepWalk import graph
from DeepWalk import walks as serialized_walks
from CommunityDetection import NCut
from gensim.models import Word2Vec
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def process(args):

    if args.format == "adjlist":
        G = graph.load_adjacencylist(args.input, undirected=args.undirected)
    elif args.format == "edgelist":
        G = graph.load_edgelist(args.input, undirected=args.undirected)
    elif args.format == "mat":
        G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

    print("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * args.number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * args.walk_length

    print("Data size (walks*length): {}".format(data_size))

    # 为了便于研究，直接将游走序列写入磁盘。
    # plt.style.use('ggplot')
    # fig, (ax0, ax1) = plt.subplots(ncols=2)
    # ax0.scatter()
    g_see = nx.read_adjlist("../example_graphs/karate.adjlist", nodetype=int)
    # label = dict((i, i) for i in G.nodes())
    # nx.draw_networkx_labels(g_see, pos=nx.spring_layout(g_see), labels=label)
    nx.draw(g_see, pos=nx.spring_layout(g_see), with_labels=True)
    plt.savefig("原始数据图.png")
    plt.show()

    print("Walking...")
    walks_filebase = args.output + ".walks"
    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                                      path_length=args.walk_length, alpha=0,
                                                      rand=random.Random(args.seed),
                                                      num_workers=args.workers)
    walks = serialized_walks.WalksCorpus(walk_files)

    print("Training...")
    model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1,
                     workers=args.workers)

    model.wv.save_word2vec_format(args.output)
    labels = NCut.build_W_D_L(G, args.output, 3)
    # labelss = labels.labels_
    # plt.scatter(G[0], G[0], c=labels)
    # print(labelss)

    plt.show()


def main():
    parser = ArgumentParser("deepwalk",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                        help="drop a debugger if an exception is raised.")

    # 默认输入的图为adjlist
    parser.add_argument('--format', default='adjlist',
                        help='File format of input file')

    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')

    parser.add_argument("-l", "--log", dest="log", default="INFO",
                        help="log verbosity level")

    # 针对matlab文件.mat设置的，是这个文件的内置的邻接矩阵名字
    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')

    # 节点使用总次数的计数
    parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                        help='Size to start dumping walks to disk, instead of keeping them in memory.')

    # 每个节点的随机游走次数，默认10次
    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')

    parser.add_argument('--output', required=True,
                        help='Output representation file')

    # 每个节点要学习的潜在维数
    parser.add_argument('--representation-size', default=64, type=int,
                        help='Number of latent dimensions to learn for each node.')

    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for random walk generator.')

    # 图视为无向图，默认为是
    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')

    # 利用顶点度估计随机游走中节点的频率，此选项比计算词汇表快，默认不用
    parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                        help='Use vertex degree to estimate the frequency of nodes '
                        'in the random walks. This option is faster than '
                        'calculating the vocabulary.')

    # 每个节点的游走路路径长度，默认长度40
    parser.add_argument('--walk-length', default=40, type=int,
                        help='Length of the random walk started at each node')

    # skipgram中的窗口宽度，默认为5
    parser.add_argument('--window-size', default=5, type=int,
                        help='Window size of skipgram model.')

    # 并行进程的数量，默认为1
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of parallel processes.')

    # args = parser.parse_args() 接送实际使用命令行时的输入
    args = parser.parse_args("--input ../example_graphs/karate.adjlist "
                             "--output ./output".split())
    # numeric_level = getattr(logging, args.log.upper(), None)
    # logging.basicConfig(format=LOGFORMAT)
    # logger.setLevel(numeric_level)
    #
    # if args.debug:
    #     sys.excepthook = debug
    print(args)

    process(args)


if __name__ == "__main__":
    sys.exit(main())
