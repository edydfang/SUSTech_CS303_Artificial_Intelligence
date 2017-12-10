#!/usr/bin/env python2
'''
used to solve the influence maximization problem
'''
from __future__ import division
import random
import logging
import heapq
import time
import os
from multiprocessing import Pool, Queue
from collections import defaultdict
from utils.imp_evaluator import Evaluator

N_PROCESSORS = 2


class Solver(object):
    '''
    Just a solver
    '''

    def __init__(self, graph, model, time_limit, random_seed, num_k):
        if random_seed is not None:
            random.seed(random_seed)
        self.num_k = num_k
        self.time_limit = time_limit
        self.graph = graph
        self.model = model
        # use degree discount heuristics
        if time_limit != -1:
            self.seedset_heristics = self.degree_discount()
        # initialize the process pool
        self.workers = list()
        self.total_sim_round = 10000
        self.sim_round_process = int(self.total_sim_round / N_PROCESSORS)
        self.avg_time = None

        for idx in range(N_PROCESSORS):
            new_task_queue = Queue()
            new_result_queue = Queue()
            evaluator = Evaluator(
                graph, model, new_task_queue, new_result_queue, random_seed + str(idx))
            self.workers.append((evaluator, new_task_queue, new_result_queue))
            evaluator.start()

    def solve(self):
        '''
        unique solver
        '''
        seedset = self.solve_celf()
        for seed in seedset:
            print(seed)
        for worker in self.workers:
            worker[1].put((-1, None, None))
        for worker in self.workers:
            worker[0].join()

    def solve_celf(self):
        '''
        implementation of CELF
        note that the first element in the tuple is negative form of the the spread contribution
        '''
        state_list = list()
        cur_spread = 0
        cur_set = set()
        # init the heap, note this is a minheap
        for nodeid in self.graph.vertices():
            new_spread = self.seed_evaluate(set.union(cur_set, {nodeid}))
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
                new_spread = self.seed_evaluate(
                    set.union(cur_set, {next_node[1]}))
                diff = new_spread - cur_spread
                next_node = (-diff, next_node[1])
                if next_node[0] < cur_max:
                    cur_max = next_node[0]
                heapq.heappush(state_list, next_node)
            else:
                inserted_node = next_node
                cur_set.add(inserted_node[1])
                cur_spread += -inserted_node[0]
                cur_max = 1
        logging.debug(cur_spread)
        return cur_set

    def seed_evaluate(self, seeds):
        '''
        evaluate based on linear threshold model
        '''
        cnt = 0
        # start = time.time()
        for idx, worker in enumerate(self.workers):
            worker[1].put((idx, seeds, self.sim_round_process))
        for worker in self.workers:
            cnt += worker[2].get()[1]
        # endtime = time.time()
        # logging.debug(endtime - start)
        return cnt / len(self.workers)

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
            max_vertex = sorted(discount_degree.iteritems(),
                                key=lambda (k, v): v, reverse=True)[0][0]
            del discount_degree[max_vertex]
            seed_set.add(max_vertex)
            full_set.remove(max_vertex)
            for vertex in self.graph[max_vertex].keys():
                if vertex in full_set:
                    t_selected[vertex] += 1
                    # how much is the p? average?
                    discount_degree[vertex] = degree[vertex] - 2 * t_selected[vertex] -\
                        (degree[vertex] - t_selected[vertex]) * \
                        t_selected[vertex] * 0.75
        return seed_set
