#!/usr/bin/env python
"""
@author:HaipengDuan
@time:2019/3/31 17:03
"""
#-*-coding:utf-8-#-

from io import open
from os import path
from multiprocessing import cpu_count
import random
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

# 字典类型
from collections import Counter

from six.moves import zip

from DeepWalk import graph

__current_graph = None

# count_words与count_textfiles多线程实现节点出现次数的统计
def count_words(file):
    """ Counts the word frequences in a list of sentences.
    file即随机游走后生成的walks，一次游走的结果为一行
    Note:
      This is a helper function for parallel execution of `Vocabulary.from_text`
      method.
    输出：
      输出为各个节点出现的次数
    """
    c = Counter()
    with open(file, 'r') as f:
        for l in f:
            words = l.strip().split()
            c.update(words)
    return c


def count_textfiles(files, workers=1):
    """
    多线程技术 ProcessPoolExecutor方法
    executor.map(fun, input_files)
    ==
    result=[]
    fot item in input_files:
      re = fun(item)
      result.append(re)
    """
    c = Counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for c_ in executor.map(count_words, files):
            c.update(c_)
    return c


def count_lines(f):
    if path.isfile(f):
        num_lines = sum(1 for line in open(f))
        return num_lines
    else:
        return 0


def _write_walks_to_disk(args):
    """
    后续多进程有调用
    :param args: ppw, path_length, alpha, random.Random(rand.randint(0, 2 ** 31)), file_
    ppw: paths number for a single worker（路径号）
    path_length: 随机游走的路径长度
    alpha: 以alpha的概率停止继续走，回到路径的起点重新走
    random.Random(rand.randint(0, 2 ** 31)): 随机数种子，这个种子是在rand种子基础上建立的种子
    file_: './output.walks.0' 随机游走路径存储的文件

    :return:
    """
    num_paths, path_length, alpha, rand, f = args
    G = __current_graph
    # t_0 = time()
    with open(f, 'w') as fout:
        for walk in graph.build_deepwalk_corpus_iter(G=G, num_paths=num_paths, path_length=path_length,
                                                     alpha=alpha, rand=rand):
            fout.write(u"{}\n".format(u" ".join(v for v in walk)))
    # logger.debug("Generated new file {}, it took {} seconds".format(f, time() - t_0))
    return f


def write_walks_to_disk(G, filebase, num_paths, path_length, alpha=0, rand=random.Random(0), num_workers=cpu_count(),
                        always_rebuild=True):
    """
    :param G:
    :param filebase: './output.walks'
    :param num_paths:  对一个起点游走的次数
    :param path_length: 随机游走的步长
    :param alpha: 概率
    :param rand: 随机数种子
    :param num_workers: 进程处理器个数
    :param always_rebuild:
    :return:
    """
    global __current_graph
    __current_graph = G

    # files_list 存储一系列文件名
    files_list = ["{}.{}".format(filebase, str(x)) for x in list(range(num_workers))]
    print("num_paths is:", num_paths)
    print([str(x) for x in list(range(num_paths))])
    print("file_lists:", files_list)
    # print(G)
    expected_size = len(G)
    args_list = []
    files = []

    # 对不同采样轮数与处理器个数的关系进行分情况处理
    if num_paths <= num_workers:
        paths_per_worker = [1 for x in range(num_paths)]
    else:
        paths_per_worker = [len(list(filter(lambda z: z is not None, [y for y in x])))
                            for x in graph.grouper(int(num_paths / num_workers)+1, range(1, num_paths+1))]
    # print(paths_per_worker)
    """
    grouper(arg1,arg2,arg3)
    arg1:int
    arg2:可迭代对象
    arg3：默认参数
    example:
    grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')
    grouper(3, range(1,7+1))-->(1,2,3),(4,5,6),(7,None,None)
    """

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for size, file_, ppw in zip(executor.map(count_lines, files_list), files_list, paths_per_worker):
            if always_rebuild or size != (ppw*expected_size):
                args_list.append((ppw, path_length, alpha, random.Random(rand.randint(0, 2**31)), file_))
            else:
                files.append(file_)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for file_ in executor.map(_write_walks_to_disk, args_list):
            files.append(file_)

    return files


class WalksCorpus(object):
    def __init__(self, file_list):
        self.file_list = file_list

    def __iter__(self):
        for file in self.file_list:
            with open(file, 'r') as f:
                for line in f:
                    yield line.split()
