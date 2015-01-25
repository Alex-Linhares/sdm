import pyopencl as cl
import pyopencl.array 
import numpy
import numpy.linalg as la
import time



HARD_LOCATIONS = 2**20
EXPECTED_NUM_HARD_LOCATIONS = 1250  #Compute analytically; prove it's safe... 


HASH_TABLE_SIZE =  1489 
HASH_TABLE_SIZE_FILE = \
"#define HASH_TABLE_SIZE 1489 \n\
#define HASH_TABLE_SIZE2 1488 \n\
#define HASH_TABLE_SIZE3 1487 \n\
#define HASH_TABLE_SIZE4 1486 \n\
#define HASH_TABLE_SIZE5 1485 \n\
#define HASH_TABLE_SIZE6 1484 \n\
#define HASH_TABLE_SIZE7 1483 \n\
"

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
	hash_table_active_index = numpy.zeros(HASH_TABLE_SIZE).astype(numpy.uint32) 
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

def Create_Memory_Addresses():
	memory_addresses = numpy.random.random_integers(0,2**32,size=(HARD_LOCATIONS)*8).astype(numpy.uint32)
	return memory_addresses

def Get_Memory_Addresses_Buffer(ctx):
	memory_addresses = Create_Memory_Addresses() 	
	memory_addresses_gpu = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=memory_addresses)
	return memory_addresses_gpu

def Get_Text_code(filename):
	with open (filename, "r") as myfile:
	    data = myfile.read()
	    return HASH_TABLE_SIZE_FILE + data 


hash_table_gpu = Get_Hash_Table_GPU_Buffer(ctx)
memory_addresses_gpu = Get_Memory_Addresses_Buffer(ctx)
distances_gpu = Get_Distances_GPU_Buffer(ctx)

def Get_Active_Locations(bitstring, ctx):
	err = prg.clear_hash_table_gpu(queue, (HASH_TABLE_SIZE,), None, hash_table_gpu).wait()
	#err = prg.get_active_hard_locations(queue, (HARD_LOCATIONS,), None, memory_addresses_gpu, bitstring_gpu, distances_gpu, hash_table_gpu).wait()  
	err = prg.get_active_hard_locations_no_dist_buffer(queue, (HARD_LOCATIONS,), None, memory_addresses_gpu, bitstring_gpu, hash_table_gpu).wait()  
	if err: print 'Error --> ',err
	err = cl.enqueue_read_buffer(queue, hash_table_gpu, hash_table_active_index).wait()
	if err: print 'Error in retrieving hash_table_active_index? --> ',err

	active_hard_locations = hash_table_active_index[hash_table_active_index!=0]  ## THIS HAS TO BE DONE ON THE GPU, MORON!  If it's sorted, you can get the num of items and copy only those...
	return active_hard_locations




OpenCL_code = Get_Text_code ('GPU_Code_OpenCLv1_2.cl')

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

start = time.time()

prg = cl.Program(ctx, OpenCL_code).build()

hash_table_active_index = Get_Hash_Table()
hamming_distances = Get_Hamming_Distances()


print "\n"

num_times = 2000
Results_and_Statistics = numpy.zeros(num_times+1).astype(numpy.uint32) 

start = time.time()
for x in range(num_times):
	bitstring_gpu = Get_Bitstring_GPU_Buffer(ctx)
	active_hard_locations = Get_Active_Locations(bitstring_gpu, ctx)
	Results_and_Statistics[x] = numpy.size(active_hard_locations)
	
time_elapsed = (time.time()-start)


print Results_and_Statistics[Results_and_Statistics !=0].min(), " the minimum of HLs found should be 1001"
print Results_and_Statistics[Results_and_Statistics !=0].mean(), "the mean of HLs found should be 1119.077"
print Results_and_Statistics[Results_and_Statistics !=0].max(), "the max of HLs found should be 1249"

print numpy.size(active_hard_locations)
print active_hard_locations


#print hamming_distances[hash_table_active_index[hash_table_active_index!=0]]
#HEY!  You've never got those distances back... not even saved them in GPU memory... moron...

print '\n Seconds to Scan 2^20 Hard Locations', num_times,'times:', time_elapsed

sum = numpy.sum(Results_and_Statistics)


# multiple-hashing in OpenCL code
usual_result = 2238155
print '\n Sum of num_active_locations_found locations = ', sum, "error =", sum-usual_result, "\n"
print 'expected HLs missed per scan is', (usual_result-sum)/num_times

