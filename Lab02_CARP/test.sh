#!/bin/bash
time ./CARP_solver.py CARP_samples/gdb1.dat -t 15 -s 68852874
time ./CARP_solver.py CARP_samples/gdb1.dat -t 30 -s 68852874
time ./CARP_solver.py CARP_samples/gdb1.dat -t 60 -s 68852874
time ./CARP_solver.py CARP_samples/gdb1.dat -t 300 -s 68852874
time ./CARP_solver.py CARP_samples/gdb1.dat -t 600 -s 68852874
time ./CARP_solver.py CARP_samples/egl-s1-A.dat -t 15 -s 68852874
time ./CARP_solver.py CARP_samples/egl-s1-A.dat -t 30 -s 68852874
time ./CARP_solver.py CARP_samples/egl-s1-A.dat -t 60 -s 68852874
time ./CARP_solver.py CARP_samples/egl-s1-A.dat -t 300 -s 68852874
time ./CARP_solver.py CARP_samples/egl-s1-A.dat -t 600 -s 68852874