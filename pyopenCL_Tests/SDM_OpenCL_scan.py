import pyopencl as cl
import pyopencl.array 
import numpy
import numpy.linalg as la
import time



HARD_LOCATIONS = 2**20
EXPECTED_NUM_HARD_LOCATIONS = 1250
HASH_TABLE_SIZE =  25033 # 12043 # 25033 #12043  #Must be a prime number in the OpenCL code

# HASH_TABLE_SIZE must be prime.  The higher it is, the more bandwidth, but way less collisions.


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
	bin_active_index_gpu = cl.Buffer(ctx, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=hash_table_active_index)
	return bin_active_index_gpu

def Get_Hamming_Distances():
	hamming_distances = numpy.zeros(HARD_LOCATIONS).astype(numpy.uint32) 
	return hamming_distances

def Get_Distances_GPU_Buffer(ctx):
	Distances = Get_Hamming_Distances() 
	hamming_distances_gpu = cl.Buffer(ctx, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=Distances) 
	return hamming_distances_gpu

def Get_Random_Bitstring():
	bitstring = numpy.random.random_integers(0,2**32,size=8).astype(numpy.uint32)
	return bitstring


def Get_Bitstring_GPU_Buffer(ctx):
	bitstring = Get_Random_Bitstring()
	bitstring_gpu = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=bitstring)
	return bitstring_gpu

def Create_Memory_Addresses():
	memory_addresses = numpy.random.random_integers(0,2**32,size=(2**20)*8).astype(numpy.uint32)
	return memory_addresses

def Get_Memory_Addresses_Buffer(ctx):
	memory_addresses = Create_Memory_Addresses() 	
	memory_addresses_gpu = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=memory_addresses)
	return memory_addresses_gpu

def Get_Text_code(filename):
	with open (filename, "r") as myfile:
	    data = myfile.read()
	    return data


def Get_Active_Locations(bitstring, ctx):
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
	return hash_table_active_index




#bin_active_index_gpu = Get_Bin_Active_Indexes_GPU_Buffer(ctx)

bin_active_index_gpu = Get_Hash_Table_GPU_Buffer(ctx)

memory_addresses_gpu = Get_Memory_Addresses_Buffer(ctx)
distances_gpu = Get_Distances_GPU_Buffer(ctx)

OpenCL_code = Get_Text_code ('GPU_Code_OpenCLv1_2.cl')

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

start = time.time()

prg = cl.Program(ctx, OpenCL_code).build()

hash_table_active_index = Get_Hash_Table()
hamming_distances = Get_Hamming_Distances()
