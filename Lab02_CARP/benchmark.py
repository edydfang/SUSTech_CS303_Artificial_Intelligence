import subprocess
import time
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

instances = [ 'gdb1', 'gdb10', 'val1A', 'val4A', 'val7A', 'egl-e1-A','egl-s1-A']
times = [ '10', '20', '30', '60', '90']
seed = '45328751642'

for instance in instances:
	for t in times:
		in_file = dir_path + '/CARP_samples/%s.dat' % instance
		out_file = open('./output/%s-%s.txt' % (instance,t), 'a')
		#print(dir_path)
		process = subprocess.Popen(['python2', dir_path + '/CARP_solver.py', in_file, '-t', t, '-s', seed], stdout=out_file)
		time_start = time.time()
		process.wait()
		time_end = time.time()
		print in_file, t, time_end - time_start
