#!/usr/bin/env python2
'''
This program is used as subprocess for the solver to get the Monto-Carlo results
'''
import random
import logging
from multiprocessing import Process


class Evaluator(Process):
    '''
    evaluator for multi-processing
    ::params: task_queue (id, seeds, round)
    result_queue (id, result)
    id = -1 => exit
    '''

    def __init__(self, graph, model, task_queue, result_queue):
        '''
        input the global paras
        '''
        super(Evaluator, self).__init__(target=self.start)
        self.graph = graph
        self.model = model
        self.task_queue = task_queue
        self.result_queue = result_queue
        model_map = {'IC': self.ic_evaluate, 'LT': self.lt_evaluate}
        self.evaluate = model_map[model]

    def run(self):
        '''
        start to retrive tasks from task_queue
        '''
        while True:
            # logging.debug("getting tasks")
            new_task = self.task_queue.get()
            if new_task[0] == -1:
                exit(0)
            result = self.evaluate(new_task[1], new_task[2])
            self.result_queue.put((new_task[0], result))

    def ic_evaluate(self, seeds, sim_round):
        '''
        evaluate based on independent cascade model
        '''
        cnt = 0
        for _ in range(sim_round):
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
        return cnt / sim_round

    def lt_evaluate(self, seeds, sim_round):
        cnt = 0
        for _ in range(sim_round):
            activated = seeds
            threshold = dict()
            for node in self.graph.vertices():
                threshold[node] = random.random()
            changed = True
            while changed:
                changed = False
                inactive = set.difference(
                    set(self.graph.vertices()), activated)
                for node in inactive:
                    indicator = 0
                    for linked_node in self.graph.inverse[node].keys():
                        if linked_node in activated:
                            indicator += self.graph.inverse[node][linked_node]['weight']
                    if indicator > threshold[node]:
                        activated.add(node)
                        changed = True
            cnt += len(activated)
        return cnt / sim_round
