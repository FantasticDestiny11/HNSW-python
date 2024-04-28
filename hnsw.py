# -*- coding: utf-8 -*-
"""
这段代码定义了一个HNSW（可扩展的高维数据结构）类，用于存储和搜索大规模高维数据，
如在相似性搜索中使用。
主要功能包括添加元素、平衡添加元素、搜索最近邻居等。该类支持两种不同的距离计算方法（L2距离和余弦距离），并且支持向量化和非向量化两种计算方式。

在初始化类时，用户可以指定距离类型（L2距离或余弦距离）、
参数m（每个节点保留的最大连接数量）、
参数ef（每个节点在搜索时参考的最大候选节点数量）、
参数m0（首层节点保留的最大连接数量）、
启发式搜索的开关以及向量化的开关。

类中的add方法用于向数据结构中添加新元素，balanced_add方法用于平衡添加元素以确保数据结构的性质，search方法用于搜索距离指定查询点最近的k个点。

代码使用了heap队列和其他数据结构来高效地搜索和更新数据结构中的节点，包括选择最近邻居节点的方法、向节点添加连接的方法等。
"""

from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from math import log2
from operator import itemgetter
from random import random

import numpy as np


class HNSW(object):
    # self._graphs[level][i] contains a {j: dist} dictionary,
    # where j is a neighbor of i and dist is distance

    def l2_distance(self, a, b):
        # 计算l2距离（欧几里得距离）
        return np.linalg.norm(a - b)

    def cosine_distance(self, a, b):
        # 计算余弦距离（余弦相似度）
        try:
            return np.dot(a, b)/(np.linalg.norm(a)*(np.linalg.norm(b)))
        except ValueError:
            print(a)
            print(b)
        

    def _distance(self, x, y):
        # 向距离函数计算距离
        return self.distance_func(x, [y])[0]

    def vectorized_distance_(self, x, ys):
        # pprint.pprint([self.distance_func(x, y) for y in ys])
        return [self.distance_func(x, y) for y in ys]

    def __init__(self, distance_type, m=5, ef=200, m0=None, heuristic=True, vectorized=False):
        # 初始化相关变量
        self.data = []
        if distance_type == "l2":
            # l2 distance
            distance_func = self.l2_distance
        elif distance_type == "cosine":
            # 余弦距离
            distance_func = self.cosine_distance
        else:
            raise TypeError('Please check your distance type!')

        self.distance_func = distance_func

        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = distance_func
        else:
            self.distance = distance_func
            self.vectorized_distance = self.vectorized_distance_

        self._m = m
        self._ef = ef
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log2(m)
        self._graphs = []
        self._enter_point = None

        self._select = (
            self._select_heuristic if heuristic else self._select_naive)

    def add(self, elem, ef=None):
        """
        向HNSW图中添加元素

        Args:
            elem: 要添加的元素
            ef: 搜索时使用的最大候选项数量，默认为初始化时设置的值

        Returns:
            None
        """
        # 如果 ef 为 None，则使用初始化时的默认值
        if ef is None:
            ef = self._ef

        # 获取距离函数、数据、图结构、入口点、m
        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m

        # 确定要插入的元素的层级
        level = int(-log2(random()) * self._level_mult) + 1
        # print("level: %d" % level)

        # 将元素插入数据列表中，并记录其索引
        idx = len(data)
        data.append(elem)

        # 如果存在入口点，即HNSW图不为空，我们有一个入口点
        if point is not None:
            # 对于我们不必插入 elem 的所有级别，我们搜索最近的邻居
            dist = distance(elem, data[point])
            for layer in reversed(graphs[level:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
            # 在这些级别中，我们必须插入 elem; ep 是入口点的堆。
            ep = [(-dist, point)]
            # pprint.pprint(ep)
            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0
                # 遍历图并更新 ep 为我们找到的最近节点
                ep = self._search_graph(elem, ep, layer, ef)
                # 在 g[idx] 中插入最佳邻居
                layer[idx] = layer_idx = {}
                self._select(layer_idx, ep, level_m, layer, heap=True)
                # 插入到新节点的反向链接
                for j, dist in layer_idx.items():
                    self._select(layer[j], (idx, dist), level_m, layer)
                    # assert len(g[j]) <= level_m
                # assert all(e in g for _, e in ep)
        # 对于新的级别，创建一个空图
        for i in range(len(graphs), level):
            graphs.append({idx: {}})
            self._enter_point = idx

    def balanced_add(self, elem, ef=None):
        """
        在HNSW图中平衡地添加元素

        Args:
            elem: 要添加的元素
            ef: 搜索时使用的最大候选项数量，默认为初始化时设置的值

        Returns:
            None
        """
        # 如果 ef 为 None，则使用初始化时的默认值
        if ef is None:
            ef = self._ef

        # 获取距离函数、数据、图结构、入口点、m、m0
        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m
        m0 = self._m0

        # 将元素添加到数据列表中，并记录其索引
        idx = len(data)
        data.append(elem)

        # 如果存在入口点，即HNSW图不为空，我们有一个入口点
        if point is not None:
            dist = distance(elem, data[point])
            pd = [(point, dist)]
            # pprint.pprint(len(graphs))
            for layer in reversed(graphs[1:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
                pd.append((point, dist))
            for level, layer in enumerate(graphs):
                # print('\n')
                # pprint.pprint(layer)
                level_m = m0 if level == 0 else m
                # 获取候选项
                candidates = self._search_graph(
                    elem, [(-dist, point)], layer, ef)
                layer[idx] = layer_idx = {}
                # 选择最佳邻居
                self._select(layer_idx, candidates, level_m, layer, heap=True)
                # 添加反向边
                for j, dist in layer_idx.items():
                    self._select(layer[j], [idx, dist], level_m, layer)
                    assert len(layer[j]) <= level_m
                # 如果当前层的节点数小于最大允许节点数，直接返回
                if len(layer_idx) < level_m:
                    return
                # 如果当前层不是最后一层，且下一层存在与当前层相邻的节点，则直接返回
                if level < len(graphs) - 1:
                    if any(p in graphs[level + 1] for p in layer_idx): 
                        return
                # 弹出上一层的节点及其距离
                point, dist = pd.pop()
        # 创建一个新图并设置入口点为当前节点
        graphs.append({idx: {}})
        self._enter_point = idx

    def search(self, q, k=None, ef=None):
        """
        查找与 q 最接近的 k 个点。

        Args:
            q: 查询点
            k: 要返回的最接近的点的数量，默认为 None，表示返回所有找到的点
            ef: 在底层级别查找时使用的最大候选项数量，默认为初始化时设置的值

        Returns:
            如果 k 不为 None，则返回一个列表，其中包含 k 个最接近的点的索引和距离，按距离从远到近排序；
            如果 k 为 None，则返回一个列表，其中包含所有找到的最接近的点的索引和距离，按距离从远到近排序。
        """

        # 获取距离函数、图结构、入口点
        distance = self.distance
        graphs = self._graphs
        point = self._enter_point

        # 如果图为空，则抛出异常
        if ef is None:
            ef = self._ef

        if point is None:
            raise ValueError("Empty graph")

        # 计算查询点到入口点的距离
        dist = distance(q, self.data[point])
        # 从顶层向第二层查找最近的邻居
        for layer in reversed(graphs[1:]):
            point, dist = self._search_graph_ef1(q, point, dist, layer)
        # 在底层级别查找 ef 个邻居
        ep = self._search_graph(q, [(-dist, point)], graphs[0], ef)

        # 如果指定了返回的最近点的数量 k，则返回前 k 个最接近的点；否则，返回所有找到的点
        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)

        # 将距离转换为负数，并返回结果
        return [(idx, -md) for md, idx in ep]

    def _search_graph_ef1(self, q, entry, dist, layer):
        """
        在HNSW图的单个层中使用 EF=1 进行搜索，查找最接近的邻居。

        Args:
            q: 查询点
            entry: 入口点
            dist: 入口点与查询点的距离
            layer: 当前层的图结构

        Returns:
            返回距离查询点最近的邻居的索引和距离。
        """
        # 获取向量化距离函数和数据
        vectorized_distance = self.vectorized_distance
        data = self.data

        # 初始化最近邻居和最近距离
        best = entry
        best_dist = dist
        # 初始化候选列表，将入口点添加到候选列表中
        candidates = [(dist, entry)]
        # 初始化已访问集合，并将入口点添加到其中
        visited = set([entry])

        # 循环直到候选列表为空
        while candidates:
            # 从候选列表中弹出一个候选点及其距离
            dist, c = heappop(candidates)
            # 如果当前候选点的距离超过了当前最近距离，则退出循环
            if dist > best_dist:
                break
            # 获取当前候选点的未访问邻居
            edges = [e for e in layer[c] if e not in visited]
            # 将这些邻居添加到已访问集合中
            visited.update(edges)
            # 计算当前候选点的未访问邻居与查询点的距离
            dists = vectorized_distance(q, [data[e] for e in edges])
            # 遍历所有未访问邻居，并更新最近邻居和最近距离
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    # 将未访问邻居添加到候选列表中
                    heappush(candidates, (dist, e))
                    # break
        # 返回距离查询点最近的邻居的索引和距离
        return best, best_dist

    def _search_graph(self, q, ep, layer, ef):
        """
        在HNSW图的单个层中进行搜索，使用给定的最大候选项数量 EF。

        Args:
            q: 查询点
            ep: 入口点的候选列表，每个元素是一个二元组，包含距离和索引
            layer: 当前层的图结构
            ef: 最大候选项数量

        Returns:
            返回距离查询点最近的候选项列表。
        """
        # 获取向量化距离函数和数据
        vectorized_distance = self.vectorized_distance
        data = self.data

        # 初始化候选列表，并将入口点的候选列表转换为堆
        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        # 初始化已访问集合，并将入口点的索引添加到其中
        visited = set(p for _, p in ep)

        # 循环直到候选列表为空或距离已经超过最近邻居
        while candidates:
            # 从候选列表中弹出一个候选点及其距离
            dist, c = heappop(candidates)
            # 获取入口点候选列表中距离最近点的距离作为参考距离
            mref = ep[0][0]
            # 如果当前候选点的距离超过了参考距离，则退出循环
            if dist > -mref:
                break
            # 获取当前候选点的未访问邻居
            edges = [e for e in layer[c] if e not in visited]
            # 将这些邻居添加到已访问集合中
            visited.update(edges)
            # 计算当前候选点的未访问邻居与查询点的距离
            dists = vectorized_distance(q, [data[e] for e in edges])
            # 遍历所有未访问邻居，并更新候选列表和入口点的候选列表
            for e, dist in zip(edges, dists):
                mdist = -dist
                # 如果入口点的候选列表未满，则直接添加到候选列表和入口点的候选列表中
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    # 更新参考距离
                    mref = ep[0][0]
                # 如果入口点的候选列表已满且当前距离大于参考距离，则替换入口点的候选列表中的最远点
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    # 更新参考距离
                    mref = ep[0][0]

        # 返回入口点的候选列表
        return ep


    def _select_naive(self, d, to_insert, m, layer, heap=False):
        """
        在图的层中执行简单的节点选择操作，将新节点插入到节点字典中。

        Args:
            d: 节点字典，存储节点索引和与查询点的距离
            to_insert: 要插入的新节点的列表，每个元素是一个二元组，包含距离和节点索引
            m: 最大节点数量
            layer: 当前层的图结构
            heap: 是否以堆方式存储节点字典，默认为 False

        Returns:
            无返回值，直接修改节点字典。
        """
        # 如果不是以堆方式存储，则直接执行简单的选择逻辑
        if not heap:
            idx, dist = to_insert
            # 断言新节点不在节点字典中
            assert idx not in d
            # 如果节点字典未满，则直接插入新节点
            if len(d) < m:
                d[idx] = dist
            # 如果节点字典已满，则替换距离最远的节点
            else:
                max_idx, max_dist = max(d.items(), key=itemgetter(1))
                if dist < max_dist:
                    del d[max_idx]
                    d[idx] = dist
            return

        # 如果以堆方式存储，则执行更复杂的选择逻辑
        # 断言要插入的节点不在节点字典中
        assert not any(idx in d for _, idx in to_insert)
        # 对要插入的节点按距离从小到大排序，取出最小的 m 个节点
        to_insert = nlargest(m, to_insert)  # smallest m distances
        # 计算要检查的未选中节点数量
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        # 将未选中节点和已选中节点分开
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        # 如果有未选中节点需要检查，则找出节点字典中距离最远的节点
        if to_check > 0:
            checked_del = nlargest(to_check, d.items(), key=itemgetter(1))
        else:
            checked_del = []
        # 将未选中节点插入节点字典中
        for md, idx in to_insert:
            d[idx] = -md
        # 将新插入的节点与节点字典中距离最远的节点进行比较，如果有更近的节点，则替换之
        zipped = zip(checked_ins, checked_del)
        for (md_new, idx_new), (idx_old, d_old) in zipped:
            if d_old <= -md_new:
                break
            del d[idx_old]
            d[idx_new] = -md_new
            # 断言节点字典的大小不超过最大节点数量
            assert len(d) == m


    def _select_heuristic(self, d, to_insert, m, g, heap=False):
        """
        在图的层中执行启发式的节点选择操作，将新节点插入到节点字典中。

        Args:
            d: 节点字典，存储节点索引和与查询点的距离
            to_insert: 要插入的新节点的列表，每个元素是一个二元组，包含距离和节点索引
            m: 最大节点数量
            g: 图的层结构
            heap: 是否以堆方式存储节点字典，默认为 False

        Returns:
            无返回值，直接修改节点字典。
        """
        # 获取节点字典中每个节点的邻居字典
        nb_dicts = [g[idx] for idx in d]

        # 定义优先级函数，根据节点与其邻居的距离优先级
        def prioritize(idx, dist):
            return any(nd.get(idx, float('inf')) < dist for nd in nb_dicts), dist, idx

        # 如果不是以堆方式存储，则将要插入的节点转换为带有优先级的列表
        if not heap:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
        else:
            to_insert = nsmallest(m, (prioritize(idx, -mdist)
                                      for mdist, idx in to_insert))

        # 断言要插入的节点列表不为空且未在节点字典中
        assert len(to_insert) > 0
        assert not any(idx in d for _, _, idx in to_insert)

        # 计算要检查的未选中节点数量
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        # 将要插入的节点和已选中节点分开
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        # 如果有已选中节点需要检查，则找出节点字典中距离最远的节点
        if to_check > 0:
            checked_del = nlargest(to_check, (prioritize(idx, dist)
                                              for idx, dist in d.items()))
        else:
            checked_del = []
        # 将要插入的节点插入节点字典中
        for _, dist, idx in to_insert:
            d[idx] = dist
        # 将新插入的节点与节点字典中距离最远的节点进行比较，如果有更近的节点，则替换之
        zipped = zip(checked_ins, checked_del)
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
            if (p_old, d_old) <= (p_new, d_new):
                break
            del d[idx_old]
            d[idx_new] = d_new
            # 断言节点字典的大小不超过最大节点数量
            assert len(d) == m


    def __getitem__(self, idx):

        for g in self._graphs:
            try:
                yield from g[idx].items()
            except KeyError:
                return