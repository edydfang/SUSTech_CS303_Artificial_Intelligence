'''
ISE
'''
from __future__ import division
import random
import logging


class Estimater(object):
    '''
    Class to solve the problem 01
    '''

    def __init__(self, graph, seeds, model):
        model_map = {'IC': self.ic_simulate, 'LT': self.lt_simulate}
        self.graph = graph
        self.seeds = seeds
        self.nodes = self.graph.vertices()
        self.model = model_map[model]

    def estimate(self):
        '''
        return the influenece valuence
        '''
        sim_round = int((len(self.nodes) / len(self.seeds)) * 1000)
        sum_activated = 0
        for _ in range(sim_round):
            estimated_set = self.model()
            sum_activated += len(estimated_set)
        logging.debug(sum_activated / sim_round)
        # print(sum_activated/sim_round)

    def ic_simulate(self):
        '''
        use the independent Cascade model
        '''
        activated = set()
        next_layer = self.seeds
        while next_layer:
            new_layer = set()
            for node in next_layer:
                for linked_node, value in self.graph[node].iteritems():
                    rnd = random.random()
                    if linked_node not in activated and rnd < value['weight']:
                        new_layer.add(linked_node)
            activated = set.union(activated, next_layer)
            # print(activated)
            next_layer = new_layer
        return activated

    def lt_simulate(self):
        '''
        use the independent Cascade model
        '''
        activated = self.seeds
        threshold = dict()
        for node in self.nodes:
            threshold[node] = random.random()
        changed = True
        while changed:
            changed = False
            inactive = set.difference(set(self.nodes), activated)
            for node in inactive:
                indicator = 0
                for linked_node, value in self.graph.inverse[node].iteritems():
                    if linked_node in activated:
                        indicator += value['weight']
                if indicator > threshold[node]:
                    activated.add(node)
                    changed = True
        return activated
