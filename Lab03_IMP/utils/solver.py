'''
used to solve the influence maximization problem
'''

import random
import logging
import Queue as Q

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
        self.heuristic = False

    def solve_ic(self):
        '''
        control for the ic model
        '''
        if self.heuristic:
            # use degree discount heuristics
            pass
        else:
            # use CELF
            pass
        logging.debug("ic")

    def solve_lt(self):
        '''
        control for the LT model
        '''
        if self.heuristic:
            # use degree discount heuristics
            pass
        else:
            # use CELF
            pass
        logging.debug("lt")

    def ic_celf(self):
        '''
        implementation of CELF
        '''
        pq = Q.PriorityQueue()
        nodes = set(self.graph.vertices())
        for idx in range(self.num_k):
            pass

