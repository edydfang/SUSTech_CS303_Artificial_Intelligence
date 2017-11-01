#!/usr/bin/env python2
# -*- coding=utf-8 -*-
'''
This Program is used to solve the CARP problem in a limited time.
'''
import argparse


def main():
    '''
    main entrance
    '''
    parser = argparse.ArgumentParser(
        description='Find Solutions for CARP Problem\nCoded by Edward FANG')
    parser.add_argument('instance', nargs=1, type=argparse.FileType('r'),
                        help='filename for CARP instance')
    parser.add_argument('-t', nargs=1, metavar='termination', type=int,
                        help='termination time limit', required=True)
    parser.add_argument('-s', nargs=1, metavar='random seed',
                        help='random seed for stochastic algorithm')
    args = parser.parse_args()
    print(args)
    # args.instance[0], args.s[0], args.t[0]

if __name__ == '__main__':
    main()
