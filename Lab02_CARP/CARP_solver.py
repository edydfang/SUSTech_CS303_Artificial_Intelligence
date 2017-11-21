#!/usr/bin/env python2
# -*- coding=utf-8 -*-
'''
This Program is used to solve the CARP problem in a limited time.
'''
import argparse
import numpy as np
from utils.solver import Solver


def main():
    '''
    main entrance
    '''
    parser = argparse.ArgumentParser(
        description='Find Solutions for CARP Problem\nCoded by Edward FANG')
    parser.add_argument('instance', type=argparse.FileType('r'),
                        help='filename for CARP instance')
    parser.add_argument('-t', metavar='termination', type=int,
                        help='termination time limit', required=True)
    parser.add_argument('-s', metavar='random seed',
                        help='random seed for stochastic algorithm')
    args = parser.parse_args()
    time_limit = args.t
    seed = args.s
    instance_file = args.instance
    # print(time_limit, seed)
    spec, data = read_instance_file(instance_file)
    solver1 = Solver(data, spec, time_limit, seed)
    solver1.solve()
    # print(spec, data)
    # args.instance[0], args.s[0], args.t[0]

def read_instance_file(fd):
    '''
    ::param: filename: string, filename that indicates the location of instance data file
    ::return value: (specification, data)
    :: specification: dict, specification of the instance
    :: data: the numpy array with a list of edges and their cost, demand
    :: data: [vertex1 vertex2 cost demand]
    '''
    content = fd.readlines()
    content = [x.strip() for x in content] 
    specification = dict()
    for i in range(8):
        line = content[i].split(':')
        specification[line[0].strip()] = line[1].strip()
    # print(specification)
    data = list()
    for line in content[9:-1]:
        tmp = line.split()
        data.append([int(x.strip()) for x in tmp])
    data = np.array(data)
    fd.close()
    return specification, data


if __name__ == '__main__':
    main()
