import pyopencl as cl
import pyopencl.array 
import numpy
import numpy.linalg as la
import time



HARD_LOCATIONS = 2**20
BUFFER_SIZE_EXPECTED_ACTIVE_HARD_LOCATIONS = 1300  #Compute analytically; prove it's safe... 

HASH_TABLE_SIZE =  25033 
HASH_TABLE_SIZE_FILE = \
"#define HASH_TABLE_SIZE 25033 \n\
#define HASH_TABLE_SIZE2 25032 \n\
#define HASH_TABLE_SIZE3 25031 \n\
#define HASH_TABLE_SIZE4 25030 \n\
#define HASH_TABLE_SIZE5 25029 \n\
#define HASH_TABLE_SIZE6 25028 \n\
#define HASH_TABLE_SIZE7 25027 \n\
"


# HASH_TABLE_SIZE must be prime.  The higher it is, the more bandwidth, but way less collisions.  It should also be "far" from a power of 2.


print "HASH_TABLE_SIZE=", HASH_TABLE_SIZE  #WHAT IS THE OPTIMUM HASH_TABLE_SIZE??

ACCESS_RADIUS_THRESHOLD = 104 #COMPUTE EXPECTED NUMBER OF num_ACTIVE_locations_found HARD LOCATIONS



numpy.random.seed(seed=12345678)  

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mem_flags = cl.mem_flags


def Get_Hash_Table():
	hash_table_active_index = numpy.zeros(HASH_TABLE_SIZE).astype(numpy.int32) 
	return hash_table_active_index

def Get_Hash_Table_GPU_Buffer(ctx):
	hash_table_active_index = Get_Hash_Table()
	hash_table_gpu = cl.Buffer(ctx, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=hash_table_active_index)
	return hash_table_gpu

def Get_Hamming_Distances():
	hamming_distances = numpy.zeros(HARD_LOCATIONS).astype(numpy.uint32) #32 BITS??????
	return hamming_distances

def Get_Distances_GPU_Buffer(ctx):
	Distances = Get_Hamming_Distances() 
	hamming_distances_gpu = cl.Buffer(ctx, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=Distances) 
	return hamming_distances_gpu

def Get_Random_Bitstring():
	bitstrings = numpy.random.random_integers(0,2**32,size=8).astype(numpy.uint32)
	return bitstrings


def Get_Bitstring_GPU_Buffer(ctx):
	bitstring = Get_Random_Bitstring()
	bitstring_gpu = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=bitstring)
	return bitstring_gpu


'''
def Get_num_times_Random_Bitstrings():
	for x in range(num_times):
		bitstrings[x] = numpy.random.random_integers(0,2**32,size=8).astype(numpy.uint32)
	return bitstrings
'''

'''
def Get_num_times_Bitstrings_GPU_Buffer(ctx):
	for x in range(num_times):
		bitstrings = numpy.random.random_integers(0,2**32,size=8*num_times).astype(numpy.uint32)
		#bitstrings.shape = (8,num_times)
		bitstring_gpu = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=bitstrings)
	return bitstring_gpu
'''


def Create_Memory_Addresses():
	print 'creating memory memory_addresses...'
	memory_addresses = numpy.random.random_integers(0,2**32,size=(HARD_LOCATIONS)*8).astype(numpy.uint32)
	return memory_addresses


def Get_Text_code(filename):
	with open (filename, "r") as myfile:
	    data = myfile.read()
	    return HASH_TABLE_SIZE_FILE + data 



def Get_Active_Locations(bitstring, ctx):
	err = prg.clear_hash_table_gpu(queue, (HASH_TABLE_SIZE,), None, hash_table_gpu.data).wait()
 
	err = prg.get_active_hard_locations_no_dist_buffer(queue, (HARD_LOCATIONS,), None, memory_addresses_gpu, bitstring_gpu, hash_table_gpu.data).wait()  
	if err: print 'Error --> ',err
	err = cl.enqueue_read_buffer(queue, hash_table_gpu.data , hash_table_active_index).wait()
	if err: print 'Error in retrieving hash_table_active_index? --> ',err
	active_hard_locations = hash_table_active_index[hash_table_active_index!=0]  ## THIS HAS TO BE DONE ON THE GPU, MORON!  If it's sorted, you can get the num of items and copy only those...
	'''
	ALSO, when GID=0, you are counting the hard locations there.  HL[0] IS ALWAYS ACTIVE.
	'''

	return active_hard_locations



#from pyopencl.algorithm import copy_if  
import my_pyopencl_algorithm
from my_pyopencl_algorithm import copy_if  

def Get_Active_Locations2(bitstring, ctx):
	

	err = prg.compute_distances(queue, (HARD_LOCATIONS,), None, memory_addresses_gpu, bitstring_gpu, distances_gpu.data).wait()  
	if err: print 'Error --> ',err

	from pyopencl.algorithm import copy_if
	active_hard_locations_on_device, count_active_on_device, evt = copy_if(distances_gpu, "ary[i] < 104")
	active_hard_locations = active_hard_locations_on_device.get()

	active_hard_locations = count_active_on_device.get()

	return active_hard_locations




def Get_Active_Locations3(ctx):
	hash_table_gpu = cl_array.zeros(queue, (HASH_TABLE_SIZE,), dtype=numpy.int32)
 
	prg.get_active_hard_locations_32bit(queue, (HARD_LOCATIONS,), None, memory_addresses_gpu.data, bitstring_gpu, distances_gpu, hash_table_gpu.data ).wait()
	#if err: print 'Error --> ',err

	active_hard_locations_gpu, count_active_gpu, event = my_pyopencl_algorithm.copy_if_2(hash_table_gpu, "ary[i] > 0")  

	Num_HLs = int(count_active_gpu.get())


	#prg.get_HL_distances_from_gpu3(queue, (Num_HLs,), None, active_hard_locations_gpu.data, hash_table_gpu.data, distances_gpu, final_locations_gpu.data, final_distances_gpu.data)
	prg.get_HL_distances_from_gpu5(queue, (Num_HLs,), None, active_hard_locations_gpu.data, distances_gpu, final_locations_gpu.data, final_distances_gpu.data)

	return Num_HLs, final_locations_gpu.get()[:Num_HLs], final_distances_gpu.get()[:Num_HLs] 




def Get_Active_Locations4(bitstring, ctx):
	hash_table_gpu = cl_array.zeros(queue, (HASH_TABLE_SIZE,), dtype=numpy.int32)
 
	prg.get_active_hard_locations_32bit(queue, (HARD_LOCATIONS,), None, memory_addresses_gpu.data, bitstring_gpu, distances_gpu, hash_table_gpu.data ).wait()
	#if err: print 'Error --> ',err

	active_hard_locations_gpu, count_active_gpu, event = copy_if(hash_table_gpu, "ary[i] > 0")  

	Num_HLs = int(count_active_gpu.get())

	final_locations_gpu = cl_array.zeros(queue, (Num_HLs,), dtype=numpy.int32)
	final_distances_gpu = cl_array.zeros(queue, (Num_HLs,), dtype=numpy.int32)


	prg.get_HL_distances_from_gpu(queue, (Num_HLs,), None, active_hard_locations_gpu.data, hash_table_gpu.data, distances_gpu)

	prg.copy_final_results(queue, (Num_HLs,), None, final_locations_gpu.data, active_hard_locations_gpu.data, final_distances_gpu.data, hash_table_gpu.data)
	# err = 
	return Num_HLs, final_locations_gpu.get(), final_distances_gpu.get()





def Get_Active_Locations5(bitstring_gpu, ctx):
	hash_table_gpu = cl_array.zeros(queue, (HASH_TABLE_SIZE,), dtype=numpy.int32)
 
	prg.get_active_hard_locations_32bit(queue, (HARD_LOCATIONS,), None, memory_addresses_gpu.data, bitstring_gpu[:8,].data, distances_gpu, hash_table_gpu.data ).wait()
	#if err: print 'Error --> ',err

	active_hard_locations_gpu, count_active_gpu, event = my_pyopencl_algorithm.copy_if_2(hash_table_gpu, "ary[i] > 0")  

	Num_HLs = int(count_active_gpu.get())


	#prg.get_HL_distances_from_gpu3(queue, (Num_HLs,), None, active_hard_locations_gpu.data, hash_table_gpu.data, distances_gpu, final_locations_gpu.data, final_distances_gpu.data)
	prg.get_HL_distances_from_gpu5(queue, (Num_HLs,), None, active_hard_locations_gpu.data, distances_gpu, final_locations_gpu.data, final_distances_gpu.data)

	return Num_HLs, final_locations_gpu.get()[:Num_HLs], final_distances_gpu.get()[:Num_HLs] 


criteria = 104

OpenCL_code = Get_Text_code ('GPU_Code_OpenCLv1_2.cl')

import os
import pyopencl.array as cl_array

final_locations_gpu = cl_array.zeros(queue, (BUFFER_SIZE_EXPECTED_ACTIVE_HARD_LOCATIONS,), dtype=numpy.int32)
final_distances_gpu = cl_array.zeros(queue, (BUFFER_SIZE_EXPECTED_ACTIVE_HARD_LOCATIONS,), dtype=numpy.int32)


os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

print 'sending memory_addresses to compute device...'
memory_addresses_gpu = cl_array.to_device(queue, Create_Memory_Addresses()) 

distances_host = Get_Hamming_Distances

distances_gpu = Get_Distances_GPU_Buffer(ctx)


prg = cl.Program(ctx, OpenCL_code).build()

hash_table_active_index = Get_Hash_Table()
hamming_distances = Get_Hamming_Distances()


print "\n"


num_times = 4000
Results_and_Statistics = numpy.zeros(num_times+1).astype(numpy.uint32) 
usual_result = 2238155  # for 2000 runs of 2^20 hard locations


print '\n\n===================================================='

#from pyopencl.clrandom import rand as clrand

#distances_gpu = pyopencl.array.arange(queue, HARD_LOCATIONS, dtype=numpy.int32)

#from pyopencl.clrandom import rand as clrand

#bitstring_gpu = Get_num_times_Bitstrings_GPU_Buffer(ctx)


hash_table_gpu = cl_array.zeros(queue, (HASH_TABLE_SIZE,), dtype=numpy.int32)

start = time.time()
for x in range(num_times):
	
	bitstring_gpu = Get_Bitstring_GPU_Buffer(ctx)  #Optimize THIS!
	
	num_active_hard_locations, active_hard_locations, distances = Get_Active_Locations3(ctx) 

	Results_and_Statistics[x] = num_active_hard_locations
	
time_elapsed = (time.time()-start)


print Results_and_Statistics[Results_and_Statistics !=0].min(), " the minimum of HLs found should be 1001"
print Results_and_Statistics[Results_and_Statistics !=0].mean(), "the mean of HLs found should be 1119.077"
print Results_and_Statistics[Results_and_Statistics !=0].max(), "the max of HLs found should be 1249"

print num_active_hard_locations
print active_hard_locations
print distances

print '\n Seconds to Scan 2^20 Hard Locations', num_times,'times:', time_elapsed

sum = numpy.sum(Results_and_Statistics)

# COPY_IF in OpenCL code
print '\n Sum of num_active_locations_found locations = ', sum, "error =", sum-usual_result, "\n"
print 'expected HLs missed per scan is', (usual_result-sum)/num_times

print '====================================================\n\n\n\n'

#print hamming_distances[hash_table_active_index[hash_table_active_index!=0]]
#HEY!  You've never got those distances back... not even saved them in GPU memory... moron...