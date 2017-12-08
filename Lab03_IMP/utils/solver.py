'''
used to solve the influence maximization problem
'''

import random
import logging

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
        logging.debug("ic")

    def solve_lt(self):
        '''
        control for the LT model
        '''
        logging.debug("lt")

