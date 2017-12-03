#!/usr/bin/env python2
# -*- coding=utf-8 -*-
'''
Entry file for Influence Spread Estimating ISE

i. Input: the format of the estimator call should be as follows:
python ISE.py –i <social network> -s <seed set> -m <diffusion
model> -b <termination type> -t <time budget> -r <random seed>
test python I

<social network> is the absolute path of the social network file
<seed set> is the absolute path of the seed set file
<diffusion model> can only be IC or LT
<termination type> specifies the termination manner and the value can
only be 0 or 1. If it is set to 0, the termination condition is as the same
defined in your algorithm. Otherwise, the maximal time budget specifies
the termination condition of your algorithm.
<time budget> is a positive number which indicates how many seconds
(in Wall clock time, range: [60s, 1200s]) your algorithm can spend on
this instance. If the <termination type> is 0, it still needs to accept -t
<time budget>, but can just ignore it while estimating.
<random seed> specifies the random seed used in this run. In case that
your solver is stochastic, the random seed controls all the stochastic
behaviors of your solver, such that the same random seeds will make
your solver produce the same results. If your solver is deterministic, it
still needs to accept –r <random seed>, but can just ignore them while
estimating.

ii. Output: the value of the estimated influence spread.
'''
import argparse
import utils.digraph


def main():
    '''
    decode the parameters
    '''
    parser = argparse.ArgumentParser(
        description='Program for Influence Spread Computation\nCoded by Edward FANG')
    parser.add_argument('-i', metavar='social_network', type=argparse.FileType('r'),
                        required=True, help='the absolute path of the social network file')
    parser.add_argument('-s', metavar='seed_set', type=argparse.FileType('r'),
                        required=True, help='the absolute path of the seed set file')
    parser.add_argument('-m', metavar='diffusion_model',
                        required=True, help='can only be IC or LT')
    parser.add_argument('-b', metavar='termination_type', type=int,
                        required=True, help='specifies the termination manner')
    parser.add_argument('-t', metavar='time_budget', type=int,
                        help='allowed time for tha algorithm, in second')
    parser.add_argument('-r', metavar='random_seed',
                        help='random seed for stochastic algorithm')
    args = parser.parse_args()
    
    print(args)


class InfluenceNetwork()

if __name__ == '__main__':
    main()
