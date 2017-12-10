#!/usr/bin/env python2
'''
used to solve the influence maximization problem
'''
from __future__ import division
import random
import logging
import heapq
from collections import defaultdict


class Solver(object):
    '''
    Just a solver
    '''

    def __init__(self, graph, time_limit, random_seed, num_k):
        if random_seed is not None:
            random.seed(random_seed)
        self.num_k = num_k
        self.time_limit = time_limit
        self.graph = graph
        self.heuristic = True

    def solve_ic(self):
        '''
        control for the ic model
        '''
        if self.heuristic:
            # use degree discount heuristics
            seedset = self.degree_discount()
        else:
            # use CELF
            seedset = self.ic_celf()
        logging.debug(seedset)

    def solve_lt(self):
        '''
        control for the LT model
        '''
        if self.heuristic:
            # use degree discount heuristics
            seedset = self.degree_discount()
        else:
            # use CELF
            seedset = self.lt_celf()
        logging.debug(seedset)

    def ic_celf(self):
        '''
        implementation of CELF
        note that the first element in the tuple is negative form of the the spread contribution
        '''
        state_list = list()
        cur_spread = 0
        cur_set = set()
        # init the heap, note this is a minheap
        for nodeid in self.graph.vertices():
            new_spread = self.ic_evaluate(set.union(cur_set, {nodeid}))
            state_list.append((-new_spread, nodeid))
        heapq.heapify(state_list)
        inserted_node = heapq.heappop(state_list)
        cur_set.add(inserted_node[1])
        cur_spread = -inserted_node[0]
        cur_max = 1
        count = 0
        while len(cur_set) < self.num_k:
            next_node = heapq.heappop(state_list)
            if next_node[0] < cur_max:
                count += 1
                new_spread = self.ic_evaluate(
                    set.union(cur_set, {next_node[1]}))
                diff = new_spread - cur_spread
                next_node = (-diff, next_node[1])
                if next_node[0] < cur_max:
                    cur_max = next_node[0]
                heapq.heappush(state_list, next_node)
            else:
                inserted_node = heapq.heappop(state_list)
                cur_set.add(inserted_node[1])
                cur_spread += -inserted_node[0]
                cur_max = 1
        logging.debug(cur_spread)
        return cur_set

    def lt_celf(self):
        '''
        implementation of CELF
        note that the first element in the tuple is negative form of the the spread contribution
        '''
        state_list = list()
        cur_spread = 0
        cur_set = set()
        # init the heap, note this is a minheap
        for nodeid in self.graph.vertices():
            new_spread = self.lt_evaluate(set.union(cur_set, {nodeid}))
            state_list.append((-new_spread, nodeid))
        heapq.heapify(state_list)
        inserted_node = heapq.heappop(state_list)
        cur_set.add(inserted_node[1])
        cur_spread = -inserted_node[0]
        cur_max = 1
        count = 0
        while len(cur_set) < self.num_k:
            next_node = heapq.heappop(state_list)
            if next_node[0] < cur_max:
                count += 1
                new_spread = self.lt_evaluate(
                    set.union(cur_set, {next_node[1]}))
                diff = new_spread - cur_spread
                next_node = (-diff, next_node[1])
                if next_node[0] < cur_max:
                    cur_max = next_node[0]
                heapq.heappush(state_list, next_node)
            else:
                inserted_node = heapq.heappop(state_list)
                cur_set.add(inserted_node[1])
                cur_spread += -inserted_node[0]
                cur_max = 1
        logging.debug(cur_spread)
        return cur_set

    def lt_evaluate(self, seeds):
        pass

    def ic_evaluate(self, seeds):
        cnt = 0
        for _ in range(10000):
            activated = set()
            next_layer = seeds
            while next_layer:
                new_layer = set()
                for node in next_layer:
                    for linked_node, value in self.graph[node].iteritems():
                        rnd = random.random()
                        if linked_node not in activated and rnd < value['weight']:
                            new_layer.add(linked_node)
                activated = set.union(activated, next_layer)
                next_layer = new_layer
            cnt += len(activated)
        return cnt / 10000

    def degree_discount(self):
        '''
        A heristic method of get the set
        '''
        seed_set = set()
        full_set = set(self.graph.vertices())
        degree = defaultdict(dict)
        discount_degree = defaultdict(dict)
        t_selected = defaultdict(dict)
        for vertex in full_set:
            degree[vertex] = len(self.graph[vertex])
            discount_degree[vertex] = degree[vertex]
            t_selected[vertex] = 0
        for _ in range(self.num_k):
            max_vertex = sorted(discount_degree.iteritems(), \
                key=lambda (k, v): v, reverse=True)[0][0]
            del discount_degree[max_vertex]
            seed_set.add(max_vertex)
            full_set.remove(max_vertex)
            for vertex in self.graph[max_vertex].keys():
                if vertex in full_set:
                    t_selected[vertex] += 1
                    # discount_degree[vertex] = degree[vertex] - t_selected[vertex]
                    # how much is the p? average?
                    discount_degree[vertex] = degree[vertex] - 2 * t_selected[vertex] -\
                        (degree[vertex] - t_selected[vertex]) * \
                        t_selected[vertex] * 0.75
                    # print (vertex, discount_degree[vertex])
        return seed_set
