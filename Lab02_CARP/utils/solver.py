'''
Solver for CARP
'''
from collections import defaultdict
from itertools import tee, izip
import random
from random import shuffle
import copy
import os
from utils.digraph import DiGraph
from utils.graph import Graph
from multiprocessing import Process, Queue

class Solver(object):
    '''
    A solver class
    '''

    def __init__(self, graph, spec, time_limit, seed, solution_receiver):
        self.solution_receiver = solution_receiver
        self.x_bsf = (-1, float('inf'))
        self.gf = graph
        self.time_limit = time_limit
        self.seed = seed
        self.capacity = int(spec['CAPACITY'])
        self.depot = int(spec['DEPOT'])
        self.time_limit = time_limit
        if self.seed:
            random.seed(self.seed)

    def update_bsf(self, solution_id, fitness):
        '''
        get the new solution known to main process
        '''
        if self.x_bsf[1] > fitness:
            self.x_bsf = (solution_id, fitness)
            self.solution_receiver.put([solution_id, fitness])
            #print(self.x_bsf)


    def solve(self):
        print('Run child process %s...' % os.getpid())
        init_solution1 = self.path_scanning()
        init_solution2 = self.augment_merge()
        p1 = self.ps_to_chromesome(init_solution1)
        p2 = self.am_to_chromesome(init_solution2)
        ps_random = 18
        ps = 20
        # seq = list(range(20))
        P = defaultdict(dict)
        MAX_ITERATION = 500
        MAX_RESTART = 10
        n_iteration = 0
        
        P[0]['chromesome'] = p1
        P[0]['partition'] = self.chromesome_partition(p1)
        fitness = self.solution_verify(P[0]['partition'])[2]
        P[0]['fitness'] = fitness
        self.update_bsf(0,fitness)
        '''
        if x_bsf[1] > fitness:
            x_bsf = (0, fitness)
            print(x_bsf)
        '''
        P[1]['chromesome'] = p2
        P[1]['partition'] = self.chromesome_partition(p1)
        fitness = self.solution_verify(P[1]['partition'])[2]
        P[1]['fitness'] = fitness
        self.update_bsf(1,fitness)
        # if x_bsf[1] > fitness:
        #     x_bsf = (1, fitness)
        #     print(x_bsf)
        for x in range(2, 2 + ps_random):
            p = self.ps_to_chromesome(self.path_scanning())
            # p = self.random_init()
            P[x]['chromesome'] = p
            P[x]['partition'] = self.chromesome_partition(p)
            P[x]['fitness'] = self.solution_verify(P[x]['partition'])[2]
            self.update_bsf(x,P[x]['fitness'])
            # if x_bsf[1] > P[x]['fitness']:
            #     x_bsf = (x, P[x]['fitness'])
            #     print(x_bsf)
        while n_iteration < MAX_ITERATION:
            a, b = random.sample(xrange(ps), 2)
            if P[a]['fitness'] < P[b]['fitness']:
                tmp, b = b, a
                a = tmp
            child = Solver.cxOrdered(P[a]['chromesome'], P[b]['chromesome'])
            child = random.choice(child)
            new_partition = self.chromesome_partition(child)
            new_fitness = self.solution_verify(new_partition)[2]
            # print(new_fitness)
            if new_fitness < P[a]['fitness']:
                P[a]['chromesome'] = child
                P[a]['partition'] = new_partition
                P[a]['fitness'] = new_fitness
                # print(new_fitness)
                self.update_bsf(a,new_fitness)
                # if new_fitness < x_bsf[1]:
                #     print(x_bsf)
                #     x_bsf = (a, new_fitness)
                if child == P[b]['chromesome']:
                    self.mutation(P[a])
            n_iteration += 1
        return
        # print(P)

    def mutation(self, idv):
        a, b = random.sample(xrange(len(idv['chromesome'])), 2)
        print('mutation', a, b)
        tmp = idv['chromesome'][a]
        idv['chromesome'][a] = idv['chromesome'][b]
        idv['chromesome'][b] = tmp
        idv['partition'] = self.chromesome_partition(idv['chromesome'])
        idv['fitness'] = self.solution_verify(idv['partition'])[2]
        return idv

    def random_init(self):
        tasks = self.gf.get_tasks_unique()
        solution = list(tasks)
        shuffle(solution)

        def flip(task):
            if random.random() < 0.5:
                return (task[1], task[0])
            else:
                return task
        solution = map(flip, solution)
        return solution

    def which_better(self, u1, u2, load):
        r_cq1 = self.gf[u1[0]][u1[1]]['cost'] / self.gf[u1[0]][u1[1]]['demand']
        r_cq2 = self.gf[u2[0]][u2[1]]['cost'] / self.gf[u2[0]][u2[1]]['demand']
        return_cost1 = self.gf.get_shortest_path(u1[1], self.depot)[1]
        return_cost2 = self.gf.get_shortest_path(u2[1], self.depot)[1]
        # print(u1,u2)
        '''
        if load < capacity/2:
            if r_cq1 > r_cq2:
                return u1
            elif r_cq1 < r_cq2:
                return u2
            elif return_cost1 > return_cost2:
                return u1
            elif return_cost1 < return_cost2:
                return u2
            else:     
                return random.choice([u1, u2])
        else:
            if r_cq1 < r_cq2:
                return u1
            elif r_cq1 > r_cq2:
                return u2
            elif return_cost1 < return_cost2:
                return u1
            elif return_cost1 > return_cost2:
                return u2
            else:
                return random.choice([u1, u2])
        '''
        # random better
        rule = random.choice([0, 1, 2, 3, 4])
        if rule == 0:
            if r_cq1 > r_cq2:
                return u1
            elif r_cq1 < r_cq2:
                return u2
            else:
                return random.choice([u1, u2])
        elif rule == 1:
            if return_cost1 > return_cost2:
                return u1
            elif return_cost1 < return_cost2:
                return u2
            else:
                return random.choice([u1, u2])
        elif rule == 2:
            if r_cq1 < r_cq2:
                return u1
            elif r_cq1 > r_cq2:
                return u2
            else:
                return random.choice([u1, u2])
        elif rule == 3:
            if return_cost1 < return_cost2:
                return u1
            elif return_cost1 > return_cost2:
                return u2
            else:
                return random.choice([u1, u2])
        elif rule == 4:
            if load < self.capacity / 2:
                if return_cost1 > return_cost2:
                    return u1
                elif return_cost1 < return_cost2:
                    return u2
                else:
                    return random.choice([u1, u2])
            else:
                if return_cost1 < return_cost2:
                    return u1
                elif return_cost1 > return_cost2:
                    return u2
                else:
                    return random.choice([u1, u2])

    def ps_to_chromesome(self, solution):
        '''
        input pathscanning solution
        output chromesome rep
        '''
        result = list()
        for k, v in solution.iteritems():
            result += v
        return result

    def path_scanning(self):
        '''
        get init solutions from path scanning
        '''
        k = 0
        R = defaultdict(dict)
        load = defaultdict(dict)
        cost = defaultdict(dict)
        free_task = set(self.gf.get_tasks())
        while len(free_task) > 0:
            k += 1
            R[k] = list()
            load[k], cost[k] = 0, 0
            end = self.depot
            u = None
            while True:
                if len(free_task) == 0:
                    break
                d_min = float('inf')
                for f_task in free_task:
                    if self.gf[f_task[0]][f_task[1]]['demand'] + load[k] > self.capacity:
                        continue
                    if u == None:
                        u = f_task
                        d_min = self.gf.get_shortest_path(end, f_task[0])[1]
                    d_tmp = self.gf.get_shortest_path(end, f_task[0])[1]
                    # print(d_tmp,d_min, end,f_task, u)
                    if d_tmp < d_min:
                        d_min = d_tmp
                        u = f_task
                    elif d_tmp == d_min:
                        d_min = d_tmp
                        # print(u)
                        u = self.which_better(u, f_task, load[k])
                        # print(u)
                if d_min == float('inf'):
                    break
                R[k].append(u)
                free_task.remove(u)
                free_task.remove((u[1], u[0]))
                cost[k] += self.gf[u[0]][u[1]]['cost'] + d_min
                load[k] += self.gf[u[0]][u[1]]['demand']
                end = u[1]
            cost[k] += self.gf.get_shortest_path(u[1], self.depot)[1]
        # return R, load, cost
        return R

    def concate_circles(self, circle1, circle2, idx1, idx2):
        """
        params: circle: the path
        return [circle_list, overpapping_best, cost_largest]
        """
        overlapping_best = (0, 0)
        cost_largest = 0  # largest saving cost
        for p1, val1 in enumerate(circle1[-1:-1 - idx1:-1]):
            for p2, val2 in enumerate(circle2[0:idx2]):
                if val1 == val2:
                    cost_save_1 = Graph.calculate_path_cost(
                        self.gf, circle1[-1:-1 - p1 - 1])
                    cost_save_2 = Graph.calculate_path_cost(
                        self.gf, circle2[0:p2 + 1])
                    save_total = cost_save_1 + cost_save_2
                    if cost_largest < save_total:
                        overlapping_best = (p1, p2)
                        cost_largest = save_total
    #     if cost_largest is 0:
    #         print("gg", circle1, circle2, idx1, idx2)
        return circle1[:-1 - overlapping_best[0]] + circle2[overlapping_best[1]:], overlapping_best, cost_largest

    def get_first_last_req(self, circle):
        """
        params circle: dict
        """
        first, last = 0, 0
        for idx, val in enumerate(circle['circle'][-1::-1]):
            required = False
            for required_e in circle['aq_set']:
                if val in required_e:
                    last = idx
                    # print(val, required_e)
                    required = True
                    break
            if required:
                break
        for idx, val in enumerate(circle['circle']):
            required = False
            for required_e in circle['aq_set']:
                if val in required_e:
                    first = idx
                    required = True
                    break
            if required:
                break
        return first, last

    def merge_circles(self, circle1, circle2):
        if circle1['load'] + circle2['load'] > self.capacity:
            return None
        idx1 = self.get_first_last_req(circle1)
        idx2 = self.get_first_last_req(circle2)
        # print(idx1,idx2)
        results = list()
        results.append(self.concate_circles(
            circle1['circle'], circle2['circle'], idx1[1], idx2[0]))
        results.append(self.concate_circles(
            circle1['circle'][::-1], circle2['circle'], idx1[0], idx2[0]))
        results.append(self.concate_circles(
            circle2['circle'], circle1['circle'], idx2[1], idx1[0]))
        results.append(self.concate_circles(
            circle2['circle'], circle1['circle'][::-1], idx2[1], idx1[1]))
        max_save = 0
        new_circle = None
        for result in results:
            if result[2] >= max_save:
                new_circle = result[0]
                max_save = result[2]
        if new_circle == None:
            # print(results)
            return None
        return {'circle': new_circle, 'cost': circle1['cost'] + circle2['cost'] - max_save, 'saving': max_save,
                'load': circle1['load'] + circle2['load'], 'aq_set': circle1['aq_set'].union(circle2['aq_set'])}

    @staticmethod
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)

    @staticmethod
    def is_inclueded(path, target):
        inv_target = (target[1], target[0])
        for edge in Solver.pairwise(path):
            # print(target, edge)
            if target == edge or inv_target == edge:
                return True
        return False

    def augment_merge(self):
        # init routes
        k = 0
        free_tasks = set(self.gf.get_tasks_unique())
        R = defaultdict(dict)
        # cost = defaultdict(dict)
        for f_task in free_tasks:
            x = self.gf.get_shortest_path(self.depot, f_task[0])
            y = self.gf.get_shortest_path(f_task[1], self.depot)
            if x[1] == 0:
                x = ([self.depot, ], y[1])
            if y[1] == 0:
                y[0] = ([self.depot, ], x[1])
            R[k]['circle'] = x[0] + y[0]
            R[k]['cost'] = x[1] + y[1]
            R[k]['load'] = self.gf[f_task[0]][f_task[1]]['demand']
            R[k]['aq_set'] = set()
            R[k]['aq_set'].add(f_task)
            #print(f_task, R[k], cost[k])
            k += 1
        # augment phase
        R_sorted = sorted(R.iteritems(), key=lambda (k, v): len(v['circle']), reverse=True)
        deleted = set()
        for idx, circle in enumerate(R_sorted):
            for smaller_circle in R_sorted[idx + 1:]:
                if smaller_circle[0] in deleted:
                    continue
                if circle[1]['load'] + smaller_circle[1]['load'] > self.capacity:
                    continue
                # print(smaller_circle[1][1]['aq_set'])
                flag_include = True
                for edge in smaller_circle[1]['aq_set']:
                    # print(circle)
                    if not self.is_inclueded(circle[1]['circle'], edge):
                        flag_include = False
                if flag_include:
                    deleted.add(smaller_circle[0])
                    circle[1]['aq_set'] = circle[1]['aq_set'].union(
                        smaller_circle[1]['aq_set'])
                    circle[1]['load'] += smaller_circle[1]['load']
                    #print("[oK]", circle, smaller_circle)
        R_aug = list()
        for circle in R_sorted:
            if circle[0] not in deleted:
                circle[1]['delete'] = False
                R_aug.append(circle[1])
        # merge phase
        while True:
            merge_res = list()
            merge_next = list()
            for idx1, circle1 in enumerate(R_aug):
                for idx2, circle2 in enumerate(R_aug[idx1 + 1:]):
                    #print(self.gf, circle1, circle2)
                    tmp = self.merge_circles(circle1, circle2)
                    if tmp:
                        merge_res.append((tmp, idx1, idx1 + idx2 + 1))
                        #print(tmp, idx1, idx2+idx1+1, circle1, circle2)
            if len(merge_res) is 0:
                break
            merge_res = sorted(merge_res, key=lambda (tmp, idx1, idx2): tmp['saving'], reverse=True)
            for merger in merge_res:
                if not R_aug[merger[1]]['delete'] and not R_aug[merger[2]]['delete']:
                    # print(merger[1],merger[2])
                    R_aug[merger[1]]['delete'] = True
                    R_aug[merger[2]]['delete'] = True
                    del(merger[0]['saving'])
                    merger[0]['delete'] = False
                    merge_next.append(merger[0])
            for circle in R_aug:
                if not circle['delete']:
                    # print(circle)
                    merge_next.append(circle)
            R_aug = merge_next
        R_final = R_aug
        return R_final

    def am_to_chromesome(self, am_solution):
        result = list()
        for item in am_solution:
            circle = item['circle']
            for p in Solver.pairwise(circle):
                tmp = self.gf.get_unique_edge(p)
                if tmp in item['aq_set'] and p not in result and p[::-1] not in result:
                    result.append(p)
        return result

    def chromesome_partition(self, chromesome):
        '''
        transform the solution from chromesome format to partitioned solution optimally
        input: chromesome and graph
        output:
        '''
        # generate Auxiliary graph
        aux_graph = DiGraph()
        start_cost = list()
        end_cost = list()
        arc_cost = dict()
        aq_edges = list()
        pre_task = -1
        for idx, task in enumerate(chromesome):
            start_cost.append(
                self.gf.get_shortest_path(self.depot, task[0])[1])
            aq_edges.append(
                (self.gf[task[0]][task[1]]['cost'], self.gf[task[0]][task[1]]['demand']))
            end_cost.append(self.gf.get_shortest_path(task[1], self.gf)[1])
            if pre_task != -1:
                arc_cost[(idx - 1, idx)
                         ] = self.gf.get_shortest_path(pre_task, task[0])[1]
            pre_task = task[1]

        # print(start_cost, end_cost, arc_cost, edge_cost)
        for node1 in range(len(chromesome)):
            cost = start_cost[node1]
            load = 0
            for node2 in range(node1 + 1, len(chromesome) + 1):
                if load + aq_edges[node2 - 1][1] > self.capacity:
                    break
                load += aq_edges[node2 - 1][1]
                cost += aq_edges[node2 - 1][0]
                aux_graph.add_weighted_edge(
                    (node1, node2), cost + end_cost[node2 - 1])
                if node2 != len(chromesome):
                    cost += arc_cost[(node2 - 1, node2)]
                #print(node1, node2)
                # aux_graph.add_weighted_edge()

        result = aux_graph.get_shortest_path((0, len(chromesome)))
        partitioned = list()
        for pair in self.pairwise(result[0]):
            partitioned.append(chromesome[pair[0]:pair[1]])
        return partitioned

    @staticmethod
    def cxOrdered(x1, x2):
        """Executes an ordered crossover (OX) on the input
        individuals. The two individuals are modified in place. This crossover
        expects :term:`sequence` individuals of indices, the result for any other
        type of individuals is unpredictable.

        :param x1: The first individual participating in the crossover.
        :param x2: The second individual participating in the crossover.
        :returns: A tuple of two individuals.
        Moreover, this crossover generates holes in the input
        individuals. A hole is created when an attribute of an individual is
        between the two crossover points of the other individual. Then it rotates
        the element so that all holes are between the crossover points and fills
        them with the removed elements in order. For more details see
        [Goldberg1989]_.

        This function uses the :func:`~random.sample` function from the python base
        :mod:`random` module.

        .. [Goldberg1989] Goldberg. Genetic algorithms in search, 
        optimization and machine learning. Addison Wesley, 1989
        """
        ind1, ind2 = copy.copy(x1), copy.copy(x2)
        size = min(len(ind1), len(ind2))
        a, b = random.sample(xrange(size), 2)
        # print(a,b)
        if a > b:
            a, b = b, a

        holes1, holes2 = set(), set()
        for i in range(size):
            if i < a or i > b:
                holes1.add(Graph.get_unique_edge(ind2[i]))
                holes2.add(Graph.get_unique_edge(ind1[i]))
    #             holes1[ind2[i]] = False
    #             holes2[ind1[i]] = False

        # We must keep the original values somewhere before scrambling everything
        temp1, temp2 = ind1, ind2
        k1, k2 = b + 1, b + 1
        for i in range(size):
            if Graph.get_unique_edge(temp1[(i + b + 1) % size]) in holes1:
                ind1[k1 % size] = temp1[(i + b + 1) % size]
                k1 += 1

            if Graph.get_unique_edge(temp2[(i + b + 1) % size]) in holes2:
                ind2[k2 % size] = temp2[(i + b + 1) % size]
                k2 += 1

        # Swap the content between a and b (included)
        for i in range(a, b + 1):
            ind1[i], ind2[i] = ind2[i], ind1[i]

        return ind1, ind2

    def solution_verify(self, partitioned_solution):
        loadlist = list()
        cost_total = 0
        costlist = list()
        for route in partitioned_solution:
            # costlist.append(route)
            load = 0
            prev_task = None
            cost_total += self.gf.get_shortest_path(route[0][0], self.depot)[1]
            costlist.append(cost_total)
            for task in route:
                # print(task)
                if prev_task is not None:
                    cost_total += self.gf.get_shortest_path(
                        prev_task[1], task[0])[1]
                    costlist.append(cost_total)
                load += self.gf[task[0]][task[1]]['demand']
                cost_total += self.gf[task[0]][task[1]]['cost']
                costlist.append(cost_total)
                prev_task = task
            loadlist.append(load)
            cost_total += self.gf.get_shortest_path(
                route[-1][1], self.depot)[1]
            costlist.append(cost_total)
        #print(sum(loadlist))
        return loadlist, costlist, cost_total, sum(loadlist)
