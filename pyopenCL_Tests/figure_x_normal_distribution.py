import pyopencl as cl
import pyopencl.array 
import numpy
import numpy.linalg as la
import time



HARD_LOCATIONS = 2**20
EXPECTED_NUM_HARD_LOCATIONS = 1250
HASH_TABLE_SIZE =  25033 # 12043 # 25033 #12043  #Must be a prime number in the OpenCL code

# HASH_TABLE_SIZE must be prime.  The higher it is, the more bandwidth, but way less collisions.
# How to use if in __local memory, then join up to __global memory?

print "HASH_TABLE_SIZE=", HASH_TABLE_SIZE  #WHAT IS THE OPTIMUM HASH_TABLE_SIZE??

ACCESS_RADIUS_THRESHOLD = 104 #COMPUTE EXPECTED NUMBER OF num_ACTIVE_locations_found HARD LOCATIONS


import SDM_OpenCL_scan
from SDM_OpenCL_scan import *

Results = numpy.zeros(HASH_TABLE_SIZE).astype(numpy.uint32)

start = time.time()
num_times = 1
for x in range(num_times):
	
	bitstring_gpu = Get_Bitstring_GPU_Buffer(ctx)
	
	err = prg.clear_bin_active_indexes_gpu(queue, (HASH_TABLE_SIZE,), None, bin_active_index_gpu).wait()

	err = prg.get_active_hard_locations(queue, (HARD_LOCATIONS,), None, memory_addresses_gpu, bitstring_gpu, distances_gpu, bin_active_index_gpu).wait()  
	if err: print 'Error --> ',err

	
	hash_table_active_index = Get_Hash_Table() #THIS IS SLOWING EVERYTHING!


	err = cl.enqueue_read_buffer(queue, bin_active_index_gpu, hash_table_active_index).wait()
	if err: print 'Error in retrieving hash_table_active_index? --> ',err

	# Removing zeros from hash_table_active_index, 2 options:
	# option 1: use numpy masked arrays
	# hash_table_active_index = numpy.ma.masked_equal(hash_table_active_index,0).compressed()
	# option 2: a[a != 0]
	hash_table_active_index = hash_table_active_index[hash_table_active_index!=0]
	

	#print hash_table_active_index
	num_active_locations_found = numpy.size(hash_table_active_index)
	#print  "num_active_locations_found=", num_active_locations_found
	Results[x] = num_active_locations_found
	
	time_elapsed = (time.time()-start)
	#if (x%1==0): print x, time_elapsed, "\n\n"

time_elapsed = (time.time()-start)

mean  = Results[Results !=0].mean()

print Results[Results !=0].min(), " the minimum of HLs found should be 1001"
print mean, "the mean of HLs found should be 1119.077"
print Results[Results !=0].max(), "the max of HLs found should be 1249"


print numpy.size(hash_table_active_index)

print hash_table_active_index

print hamming_distances[hash_table_active_index]

print '\nTime to compute some Hamming distances', num_times,'times:', time_elapsed

sum = numpy.sum(Results)


# multiple-hashing in OpenCL code
usual_result = 2238155
print '\n Sum of num_active_locations_found locations = ', sum, "error =", sum-usual_result, "\n"
print 'expected HLs missed per scan is', (usual_result-sum)/num_times